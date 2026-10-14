"""Outcome-blind train/calibration manifest from completed v1 fixed material."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
WORKSPACE=REPO.parent
OLD=WORKSPACE/'experiments/moc_stateful_sampling_val_20261014'
sys.path.insert(0,str(REPO/'coding_base'))
sys.path.insert(0,str(WORKSPACE/'maintenance/task_runtime/moc_sampling_20261014'))
from moc_adaptive_sampling.sampler import Message
from moc_adaptive_sampling.bundles import evidence_groups
from moc_adaptive_sampling.stateful import count_context
from moc_adaptive_sampling.features import LocalTokenizerCounter

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')

def levels(receiver,predecessors):
    seen,frontier={receiver},[receiver]
    result={}
    for hop in range(1,5):
        following=[]
        for node in frontier:
            for parent in predecessors[node]:
                if parent not in seen:
                    seen.add(parent); following.append(parent)
        if not following:
            break
        result[hop]=following
        frontier=following
    return result

def messages_for(material,receiver):
    found=levels(receiver,material['predecessors'])
    messages=[]
    for hop in sorted(found,reverse=True):
        for source in found[hop]:
            m=material['materials'][source]
            messages.append(Message(source,m['role'],hop,len(messages),m['text'],m['sketch']))
    return messages

def main():
    if (ROOT/'experiment_lock.json').exists():
        raise RuntimeError('already frozen; do not regenerate selection')
    config=json.loads((OLD/'config.json').read_text(encoding='utf-8'))
    counter=LocalTokenizerCounter(WORKSPACE/'experiments/moc_index_pull_20261013/tokenizer/tokenizer.json',config['tokenizer_sha256'])
    committed=[q for p in (OLD/'remote_snapshot/training').glob('shard*/questions.jsonl') for q in rows(p)]
    valid={q['question_id']:q for q in committed if q['status']=='ok'}
    materials={m['question_id']:m for p in (OLD/'remote_snapshot/training').glob('shard*/materials.jsonl') for m in rows(p)}
    labels={q['question_id']:q for q in rows(OLD/'data/train_labels.jsonl')}
    # Same 47 valid groups and original fit/calibration boundary as v1.
    chosen=sorted(valid,key=lambda q:valid[q]['question_index'])
    assert len(chosen)==47 and sum(valid[q]['question_index']<36 for q in chosen)==35
    roles=('Inspector','Math Solver','Programming Expert','Mathematical Analyst')
    episodes=[]
    for qid in chosen:
        material=materials[qid]
        index=valid[qid]['question_index']
        role_indices=((index//2)%4, ((index//2)+2)%4)
        for slot,role_index in enumerate(role_indices):
            target_role=roles[role_index]
            candidates=[]
            for receiver,node in material['materials'].items():
                if node['role']!=target_role:
                    continue
                messages=messages_for(material,receiver)
                inventory={m.source:m for m in messages}
                groups=evidence_groups(messages,material['predecessors'],max_group_size=2)
                fitting=[g for g in groups if count_context(counter,[inventory[s] for s in g.sources])<=1024]
                for group in fitting:
                    rank=(len(group.sources),max(inventory[s].hop for s in group.sources),
                          sum(inventory[s].hop for s in group.sources),
                          hashlib.sha256((qid+receiver+','.join(group.sources)).encode()).hexdigest())
                    candidates.append((rank,receiver,group,messages))
            common={'episode_id':f'q{index:02d}_r{slot}', 'question_id':qid,'question_index':index,
                    'fit_or_calibration':'fit' if index<36 else 'calibration', 'receiver_role':target_role}
            if not candidates:
                episodes.append(dict(common,status='no_group_fits'))
                continue
            _,receiver,group,messages=max(candidates,key=lambda x:x[0])
            ids=list(group.sources)
            variants={'empty':[], 'a':[ids[0]]}
            if len(ids)==2:
                variants.update(b=[ids[1]],ab=ids)
            episodes.append(dict(common,status='planned',density=material['density'],receiver_id=receiver,
                                 group_sources=ids,path=list(group.path),variants=variants,
                                 candidate_hops={m.source:m.hop for m in messages},
                                 communication_tokens={name:count_context(counter,[m for m in messages if m.source in selected])
                                                       for name,selected in variants.items()}))
    data=ROOT/'data'
    data.mkdir(exist_ok=True)
    for name,values in [('fixed_materials_LOCAL_ONLY.jsonl',[materials[q] for q in chosen]),
                        ('train_gold_LOCAL_ONLY.jsonl',[labels[q] for q in chosen])]:
        (data/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8',newline='\n')
    source_meta={}
    cache=[]
    graph_usage={}
    for p in (OLD/'remote_snapshot/training').glob('shard*/trace.jsonl'):
        events=rows(p)
        source_meta[p.parent.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'relative_old_path':str(p.relative_to(OLD)).replace('\\','/')}
        active=None
        for event in events:
            if event['event_type']=='question_started' and event['question_id'] in chosen:
                active={'qid':event['question_id'],'decision_id':event['decision_id'],'calls':[]}
            elif active and event['event_type']=='llm_call':
                active['calls'].append(event)
            elif active and event['event_type']=='question_finished':
                qid=active['qid']
                if event['question_id']!=qid:
                    raise ValueError('interleaved question trace')
                node_usage={call['receiver_id']:call['response']['prompt_eval_count']+call['response']['eval_count']
                            for call in active['calls']}
                if len(node_usage)!=8:
                    raise ValueError('original one-round graph must have seven nodes plus FinalRefer')
                graph_usage[qid]={'by_node':node_usage,'decision_id':active['decision_id'],
                                  'final_order':event['final_predecessors'],'original_final_output':event['final_answers'][0]}
                active=None
        for index,event in enumerate(events):
            if event['event_type']!='llm_call' or event.get('question_id') not in chosen:
                continue
            if event['response'].get('done_reason')=='length':
                continue
            copied={'request':event['request'],'response':event['response'],
                    'provenance':{'origin':'frozen_v1_train','shard':p.parent.name,'source_sha256':source_meta[p.parent.name]['sha256'],
                                  'event_index':event['event_index'],'question_id':event['question_id']}}
            if index+1<len(events) and events[index+1]['event_type']=='code_execution_finished':
                copied['tool_result']=events[index+1]['result']
            cache.append(copied)
    for qid in chosen:
        materials[qid]['baseline_usage']=graph_usage[qid]
    # Re-save augmented fixed material; original files are never changed.
    (data/'fixed_materials_LOCAL_ONLY.jsonl').write_text(''.join(json.dumps(materials[q],ensure_ascii=False)+'\n' for q in chosen),encoding='utf-8',newline='\n')
    (data/'historical_calls_LOCAL_ONLY.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cache),encoding='utf-8',newline='\n')
    # Val inputs carry no gold. Existing baseline results are referenced only;
    # the new validation arm has no baseline generation loop.
    (data/'val_inputs.jsonl').write_bytes((OLD/'data/val_inputs.jsonl').read_bytes())
    val_cache=[]
    for p in (OLD/'remote_snapshot/runs').glob('shard*/trace.jsonl'):
        events=rows(p)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        for index,event in enumerate(events):
            if event['event_type']!='llm_call' or event['request']['model']!=config['main_model'] or event['response'].get('done_reason')=='length':
                continue
            record={'request':event['request'],'response':event['response'],
                    'provenance':{'origin':'frozen_v1_val_call_no_gold','source_sha256':digest,
                                  'shard':p.parent.name,'event_index':event['event_index'],'question_id':event['question_id']}}
            if index+1<len(events) and events[index+1]['event_type']=='code_execution_finished':
                record['tool_result']=events[index+1]['result']
            val_cache.append(record)
    (data/'val_historical_calls_LOCAL_ONLY.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in val_cache),encoding='utf-8',newline='\n')
    role_metadata=json.loads((OLD/'remote_snapshot/runs/shard0/metadata.json').read_text(encoding='utf-8'))
    save(ROOT/'val_role_snapshot.json',{'roles':role_metadata['roles'],'baseline_commit':config['author_commit'],
                                      'source_sha256':hashlib.sha256((OLD/'remote_snapshot/runs/shard0/metadata.json').read_bytes()).hexdigest()})
    save(ROOT/'training_manifest.json',{'selection':'same 47 completed original train groups, two outcome-blind receiver roles each',
                                    'question_groups':len(chosen),'episodes':episodes,
                                    'historical_sources':source_meta,'historical_calls':len(cache),
                                    'fit_questions':35,'calibration_questions':12,
                                    'train_only_selection':True,'val_used_for_selection':False,'test_used':False})
    print(json.dumps({'questions':len(chosen),'planned_episodes':sum(x['status']=='planned' for x in episodes),
                      'quartets':sum(len(x.get('variants',{}))==4 for x in episodes),
                      'dyads':sum(len(x.get('variants',{}))==2 for x in episodes),
                      'historical_calls':len(cache),'roles':[x['receiver_role'] for x in episodes]}))

if __name__=='__main__':
    main()
