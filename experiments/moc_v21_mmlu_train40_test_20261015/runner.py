"""Approved small-training adaptation, frozen val and test; exact resumable units."""
import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from common import ROOT,CONFIG,rows,save,append,imports,BudgetExceeded,InvalidUnit,AuthorReplay,QUESTION
from domain_runtime import parse,target
from runtime_controls import stable_ids
from resume_cache import restore_main_cache
from moc_adaptive_sampling.paper_baselines import PaperRecipe,make_graph
from moc_adaptive_sampling.counterfactual import messages_for,replay_intervention
from moc_adaptive_sampling.bundles import evidence_groups,group_features
from moc_adaptive_sampling.stateful import structural_priors,count_context
from moc_adaptive_sampling.sampler import unit_vector
from moc_adaptive_sampling.expected_utility import ExpectedGroupUtilityModel
from moc_adaptive_sampling.expected_set import ExpectedSetSampler,ExpectedSetConfig
from moc_adaptive_sampling.adapter import MOCSamplingAdapter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reset_seed():
    import torch
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
def graph_for(api,backend,density,arm):
    reset_seed()
    with stable_ids():g=make_graph(api,PaperRecipe(domain=CONFIG['domain'],edge_density=density,use_cot=False),arm,roles=CONFIG['roles'])
    for node in [*g.nodes.values(),g.decision_node]:node.llm=backend.llm(node.llm)
    return g
def usage(backend,start):
    t=backend.cache.totals(start)
    return dict(t,main_calls=t['logical_calls'],main_model_tokens=t['logical_model_tokens'],merger_calls=0,merger_model_tokens=0)
def recoverable(error,backend):
    if backend.pending_requests:raise RuntimeError('unresolved request costs; preserve reservation') from error
    if isinstance(error,BudgetExceeded) or getattr(backend,'limit_reached',False):return False
    if isinstance(error, InvalidUnit):return True
    text=str(error).lower()
    known=('output cap reached','code timeout','prompt would exceed','incomplete backend response',
           'ordinary-node failure','incomplete graph outputs','fixed-material author graph',
           'original merger failure')
    return any(marker in text for marker in known)

async def collect(args):
    out=ROOT/'training'/f'shard{args.shard}';out.mkdir(parents=True,exist_ok=True)
    api,b,module,final=imports(args.main_host,out,'training');restore_main_cache(b,out)
    from src.llm.profile_embedding import get_sentence_embedding as encode
    inputs=rows(ROOT/'data/train_inputs.jsonl');gold={r['question_id']:target(r['answer']) for r in rows(ROOT/'data/train_gold_LOCAL_ONLY.jsonl')}
    done={r['question_id']:r for r in rows(out/'questions.jsonl')} if (out/'questions.jsonl').exists() else {}
    assert len(done)==len(rows(out/'questions.jsonl')) if (out/'questions.jsonl').exists() else True
    material_done={r['question_id']:r for r in rows(out/'materials.jsonl')} if (out/'materials.jsonl').exists() else {}
    with b.controls(module,final):
        for index,row in enumerate(inputs):
            if index%CONFIG['workers']!=args.shard or row['question_id'] in done:continue
            qid=row['question_id'];token=QUESTION.set(qid);start=len(b.cache.events)
            b.context=dict(question_id=qid,phase='training');density=(.3,.7)[index%2]
            try:
                if qid not in material_done:
                    g=graph_for(api,b,density,'vanilla_mas')
                    r=await g.arun({'task':row['task']},num_rounds=1,max_tries=1,max_time=300)
                    if len(r[0])!=1 or any(not n.outputs for n in g.nodes.values()) or b.cache.totals(start)['logical_calls']!=8:
                        raise InvalidUnit('incomplete fixed-material author graph')
                    material=dict(question_id=qid,density=density,task=row['task'],
                        materials={i:dict(role=n.role,text=str(n.outputs[-1]),sketch=unit_vector(encode(str(n.outputs[-1]))).tolist()) for i,n in g.nodes.items()},
                        predecessors={i:[p.id for p in n.spatial_predecessors] for i,n in g.nodes.items()},
                        baseline_usage=dict(final_order=list(g.nodes),decision_id=g.decision_node.id,
                                            fixed_graph_logical_tokens=b.cache.totals(start)['logical_model_tokens']))
                    append(out/'materials.jsonl',material);material_done[qid]=material
                material=material_done[qid]
                # Two receiver slots, rotating native nonroot roles without outcomes.
                candidates={}
                for receiver,node in material['materials'].items():
                    messages=messages_for(material,receiver);inventory={m.source:m for m in messages}
                    fitting=[g for g in evidence_groups(messages,material['predecessors'],2)
                             if count_context(b.counter,[inventory[s] for s in g.sources])<=1024]
                    if fitting:candidates[receiver]=(node['role'],messages,fitting)
                role_order=list(dict.fromkeys(n['role'] for i,n in material['materials'].items() if material['predecessors'][i]))
                pending=[];episodes=[]
                for slot in range(2):
                    desired=role_order[(index*2+slot)%len(role_order)]
                    proposals=[]
                    for receiver,(role,messages,groups) in candidates.items():
                        if role!=desired:continue
                        inv={m.source:m for m in messages}
                        for group in groups:
                            rank=(len(group.sources),max(inv[s].hop for s in group.sources),hashlib.sha256((qid+receiver+str(group.sources)).encode()).hexdigest())
                            proposals.append((rank,receiver,messages,group))
                    eid=qid+':'+str(slot)
                    if not proposals:
                        episodes.append(dict(episode_id=eid,receiver_role=desired,status='no_group_fits'));continue
                    _,receiver,messages,group=max(proposals,key=lambda x:x[0]);inv={m.source:m for m in messages}
                    variants={'empty':[],'a':[group.sources[0]]}
                    if len(group.sources)==2:variants.update(b=[group.sources[1]],ab=list(group.sources))
                    episode=dict(episode_id=eid,receiver_id=receiver,receiver_role=desired,density=density,variants=variants,path=list(group.path))
                    append(out/'episodes_planned.jsonl',dict(question_id=qid,**episode))
                    replay=AuthorReplay(api,b,material);replay.messages=messages;outcomes={}
                    names=list(variants);rotation=int(hashlib.sha256(eid.encode()).hexdigest(),16)%len(names)
                    for name in names[rotation:]+names[:rotation]:
                        local_start=len(b.cache.events);b.context.update(episode_id=eid,variant=name)
                        result=await replay_intervention(material,receiver,variants[name],replay.execute,replay.finalize)
                        outcome=dict(variant=name,final_output=result['final_output'],receiver_output=result['receiver_output'],
                            final_correct=int(parse(result['final_output'])==gold[qid]),receiver_correct=int(parse(result['receiver_output'])==gold[qid]),
                            affected_nodes=result['affected_nodes'],**usage(b,local_start))
                        outcomes[name]=outcome;append(out/'conditions_raw.jsonl',dict(question_id=qid,episode_id=eid,**outcome))
                    base=unit_vector(encode(f"Task:\n{row['task']}\nReceiver role:\n{desired}"))
                    priors,_=structural_priors(receiver,messages,material['predecessors'])
                    # One-shot deployed policy: only empty->action supervision.
                    for name,sources in variants.items():
                        if name=='empty':continue
                        actual=next(g for g in evidence_groups(messages,material['predecessors'],2) if set(g.sources)==set(sources))
                        x,info=group_features(base,[],actual,messages,priors,material['predecessors'],desired,b.counter,1024)
                        pending.append(dict(question_id=qid,episode_id=eid,left='empty',right=name,features=x.tolist(),feature_info=info,
                            downstream_delta=outcomes[name]['final_correct']-outcomes['empty']['final_correct']))
                    episodes.append(dict(episode,conditions=outcomes,status='ok'))
                if not pending:raise InvalidUnit('no complete eligible training episode')
                item=dict(question_id=qid,status='ok',labels=pending,episodes=episodes,**usage(b,start))
            except Exception as error:
                if not recoverable(error,b):raise
                item=dict(question_id=qid,status='excluded',reason=str(error),labels=[],**usage(b,start))
            finally:QUESTION.reset(token)
            append(out/'questions.jsonl',item);done[qid]=item
            save(out/'progress.json',dict(terminal_questions=len(done),valid_questions=sum(r['status']=='ok' for r in done.values()),target_questions=len(inputs)//2))
            print('TRAIN',len(done),qid,item['status'],flush=True)
    save(out/'finished.json',dict(status='complete',terminal_questions=len(done)))

def fit():
    if (ROOT/'policy_lock.json').exists():
        lock=json.loads((ROOT/'policy_lock.json').read_text());assert sha(ROOT/'group_utility_model.json')==lock['model_sha256'];return
    q=[r for p in (ROOT/'training').glob('shard*/questions.jsonl') for r in rows(p)]
    allowed={r['question_id'] for r in rows(ROOT/'data/train_inputs.jsonl')}
    assert len(q)==len(allowed) and {r['question_id'] for r in q}==allowed
    labels=[r for question in q if question['status']=='ok' for r in question['labels']]
    model=ExpectedGroupUtilityModel.fit([r['features'] for r in labels],[r['downstream_delta'] for r in labels],
        [r['question_id'] for r in labels],[r['feature_info'] for r in labels],episode_ids=[r['episode_id'] for r in labels],
        mode='compact',regularization=10,bootstrap_replicates=32,seed=42)
    model.write(ROOT/'group_utility_model.json')
    save(ROOT/'training_report.json',dict(planned_questions=len(q),effective_questions=sum(x['status']=='ok' for x in q),
        excluded=[{'question_id':x['question_id'],'reason':x.get('reason')} for x in q if x['status']!='ok'],
        labels=len(labels),coverage=sorted(model.coverage),metadata=model.metadata))
    save(ROOT/'policy_lock.json',dict(frozen_before_val_and_test=True,model_sha256=sha(ROOT/'group_utility_model.json'),
        training_commits_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'training').glob('shard*/questions.jsonl')},
        no_val_or_test_fitting=True,cost_penalty=.02,communication_tokens=1024,max_group_size=2))

async def evaluate(args):
    phase=args.phase;out=ROOT/phase/f'shard{args.shard}';out.mkdir(parents=True,exist_ok=True)
    api,b,module,final=imports(args.main_host,out,phase);restore_main_cache(b,out)
    from src.llm.profile_embedding import get_sentence_embedding as encode
    from merger_backend import MergerBackend
    merger=MergerBackend(args.merger_host,out,b) if args.arm=='moc' else None
    lock=json.loads((ROOT/'policy_lock.json').read_text());assert lock['frozen_before_val_and_test']
    path=ROOT/('frozen_aqua_model.json' if args.arm=='frozen_transfer' else 'group_utility_model.json')
    model=None if merger else ExpectedGroupUtilityModel.load(path,sha(path) if args.arm=='frozen_transfer' else lock['model_sha256'])
    sampler=ExpectedSetSampler(ExpectedSetConfig(communication_tokens=1024,max_group_size=2,cost_penalty=.02,local_state_weight=0),b.counter,model) if model else None
    done={(r['question_id'],r['arm'],r['density']):r for r in rows(out/'results.jsonl')} if (out/'results.jsonl').exists() else {}
    inputs=rows(ROOT/'data'/f'{phase}_inputs.jsonl')
    with b.controls(module,final),merger.route() if merger else nullcontext():
        for index,row in enumerate(inputs):
            if index%CONFIG['workers']!=args.shard:continue
            qid=row['question_id'];token=QUESTION.set(qid)
            try:
                for density in (.3,.7):
                    key=qid,args.arm,density
                    if key in done:continue
                    b.context=dict(question_id=qid,arm=args.arm,density=density,phase=phase)
                    start=len(b.cache.events);gs=len(merger.cache.events) if merger else 0;fs=len(merger.failures) if merger else 0
                    adapter=None
                    try:
                        g=graph_for(api,b,density,'moc' if merger else 'adaptive_sampling')
                        if sampler:adapter=MOCSamplingAdapter(g,sampler,encode,counter_name=b.counter.name,analyze_agent=CONFIG['domain']=='mmlu')
                        result=await g.arun({'task':row['task']},num_rounds=1,max_tries=1,max_time=300)
                        if merger and len(merger.failures)>fs:raise InvalidUnit('original merger failure: '+str(merger.failures[fs:]))
                        if len(result[0])!=1 or any(not n.outputs for n in g.nodes.values()):raise InvalidUnit('incomplete graph outputs')
                        stat=usage(b,start)
                        if stat['main_calls']!=8:raise InvalidUnit('expected seven native agents and final aggregation')
                        item=dict(question_id=qid,arm=args.arm,density=density,status='ok',final_output=result[0][0],answer=parse(result[0][0]),sampling=adapter.ledger if adapter else None,**stat)
                    except Exception as error:
                        if merger and merger.pending_requests:raise RuntimeError('unsettled merger cost') from error
                        if not recoverable(error,b):raise
                        item=dict(question_id=qid,arm=args.arm,density=density,status='invalid',reason=str(error),**usage(b,start))
                    finally:
                        if adapter:adapter.detach()
                    if merger:
                        t=merger.cache.totals(gs)
                        for k,v in t.items():item[k]+=v
                        item.update(merger_calls=t['logical_calls'],merger_model_tokens=t['logical_model_tokens'])
                        if item['status']=='ok' and t['logical_calls']%5:raise RuntimeError('incomplete original five-candidate merger')
                    append(out/'results.jsonl',item);done[key]=item
                    save(out/'progress.json',dict(terminal_conditions=len(done),last_question=qid,arm=args.arm))
                    print(phase,args.arm,len(done),qid,density,item['status'],flush=True)
            finally:QUESTION.reset(token)
    save(out/f'finished_{args.arm}.json',dict(status='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['training','fit','val','test'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--main-host');p.add_argument('--merger-host');p.add_argument('--arm',choices=['adapted','frozen_transfer','moc'],default='adapted')
    a=p.parse_args()
    if a.phase=='fit':fit()
    else:asyncio.run(collect(a) if a.phase=='training' else evaluate(a))
