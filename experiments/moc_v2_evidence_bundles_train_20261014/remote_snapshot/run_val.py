"""Only the frozen v2 graph arm on existing val60; no baseline generation."""
import argparse
import asyncio
import hashlib
import json
import random
import numpy as np

from common import ROOT,CONFIG,rows,save,append,imports,BudgetExceeded,InvalidUnit,QUESTION
from runtime_controls import stable_ids
from moc_adaptive_sampling.bundles import BundleConfig,BundleSampler
from moc_adaptive_sampling.uncertainty import GroupUtilityModel
from moc_adaptive_sampling.adapter import MOCSamplingAdapter
from moc_adaptive_sampling.paper_baselines import PaperRecipe,make_graph

async def run(args):
    lock=json.loads((ROOT/'policy_lock.json').read_text())
    assert lock['frozen_before_val']
    assert hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()==lock['config_sha256']
    assert hashlib.sha256((ROOT/'data/val_inputs.jsonl').read_bytes()).hexdigest()==lock['val_inputs_sha256']
    model=GroupUtilityModel.load(ROOT/'group_utility_model.json',lock['model_sha256'])
    out=ROOT/'val'/f'shard{args.shard}'
    if (out/'results.jsonl').exists() or (out/'calls_raw.jsonl').exists():
        raise RuntimeError('existing val attempt; no repeated completed conditions')
    api,backend,math_module,final_class=imports(args.main_host,out,'val')
    from src.llm.profile_embedding import get_sentence_embedding
    roles=json.loads((ROOT/'val_role_snapshot.json').read_text())['roles']
    inputs=rows(ROOT/'data/val_inputs.jsonl')
    assigned=[r for i,r in enumerate(inputs) if i%CONFIG['workers']==args.shard]
    params=CONFIG['model']
    sampler=BundleSampler(BundleConfig(communication_tokens=CONFIG['communication_tokens'],max_group_size=params['max_group_size'],
                                       cost_penalty=params['cost_penalty'],local_state_weight=0.),backend.counter,model)
    completed=invalid=0
    with backend.controls(math_module,final_class):
        for row in assigned:
            qid=row['question_id']; token=QUESTION.set(qid)
            task='Question: '+row['question']+'\n'+''.join(o+'\n' for o in row['options'])
            for density in (.3,.7):
                backend.context={'question_id':qid,'density':density}
                start=len(backend.cache.events)
                adapter=None
                try:
                    random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
                    import torch
                    torch.manual_seed(CONFIG['seed'])
                    with stable_ids():
                        graph=make_graph(api,PaperRecipe(edge_density=density,use_cot=CONFIG['use_cot']),'adaptive_sampling',roles=roles)
                    for node in [*graph.nodes.values(),graph.decision_node]:
                        node.llm=backend.llm(node.llm)
                    adapter=MOCSamplingAdapter(graph,sampler,get_sentence_embedding,counter_name=backend.counter.name)
                    # v1's author's complete arun remains the evaluation path.
                    result=await graph.arun({'task':task},num_rounds=1,max_tries=1,max_time=300)
                    if len(result[0])!=1 or any(not n.outputs for n in graph.nodes.values()):
                        raise InvalidUnit('incomplete graph outputs')
                    prompt=graph.decision_node.prompt_set
                    answer=prompt.postprocess_answer(prompt,result[0][0])
                    stats=backend.cache.totals(start)
                    if stats['logical_calls']!=8:
                        raise InvalidUnit('expected seven ordinary calls plus FinalRefer')
                    item=dict(question_id=qid,density=density,arm='v2_bundles',status='ok',answer=answer,
                              final_output=result[0][0],**stats,sampling=adapter.ledger)
                    completed+=1
                except BudgetExceeded:
                    raise
                except Exception as error:
                    reasons=[str(error),*(adapter.ledger.get('failures',[]) if adapter else [])]
                    if any('BudgetExceeded' in reason for reason in reasons):
                        raise BudgetExceeded('call/token/STOP limit reached inside author graph')
                    integrity=('output cap reached','code timeout','prompt would exceed safe context envelope',
                               'incomplete backend response','incomplete graph outputs','expected seven ordinary calls')
                    if not isinstance(error,InvalidUnit) and not any(token in reason for token in integrity for reason in reasons):
                        raise
                    item=dict(question_id=qid,density=density,arm='v2_bundles',status='invalid',reason=str(error),
                              **backend.cache.totals(start)); invalid+=1
                finally:
                    if adapter:
                        adapter.detach()
                append(out/'results.jsonl',item)
                save(out/'progress.json',dict(completed_conditions=completed,invalid_conditions=invalid,
                     processed_conditions=completed+invalid,target_conditions=len(assigned)*2,last_question=qid,**backend.cache.totals()))
                print(f'VAL processed={completed+invalid}/{len(assigned)*2} valid={completed} density={density}',flush=True)
            QUESTION.reset(token)
    save(out/'finished.json',dict(status='complete',completed_conditions=completed,invalid_conditions=invalid))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    parser.add_argument('--main-host',required=True)
    asyncio.run(run(parser.parse_args()))
