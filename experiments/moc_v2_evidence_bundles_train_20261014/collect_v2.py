"""Two receiver roles per original train question; commit complete question labels."""
import argparse
import asyncio
import hashlib
import time
import numpy as np

from common import (ROOT,CONFIG,AuthorReplay,BudgetExceeded,InvalidUnit,imports,
                    rows,save,append,QUESTION)
from moc_adaptive_sampling.counterfactual import replay_intervention,factorial_effects,messages_for
from moc_adaptive_sampling.bundles import EvidenceGroup,group_features
from moc_adaptive_sampling.stateful import structural_priors
from moc_adaptive_sampling.sampler import unit_vector

async def run(args):
    out=ROOT/'training'/f'shard{args.shard}'
    api,backend,math_module,final_class=imports(args.main_host,out,'training')
    from src.llm.profile_embedding import get_sentence_embedding
    manifest=__import__('json').loads((ROOT/'training_manifest.json').read_text())
    materials={m['question_id']:m for m in rows(ROOT/'data/fixed_materials_LOCAL_ONLY.jsonl')}
    gold={r['question_id']:r['correct'] for r in rows(ROOT/'data/train_gold_LOCAL_ONLY.jsonl')}
    ordered=sorted(materials,key=lambda q:next(e['question_index'] for e in manifest['episodes'] if e['question_id']==q))
    assigned=[q for i,q in enumerate(ordered) if i%CONFIG['workers']==args.shard]
    if (out/'questions.jsonl').exists() or (out/'calls_raw.jsonl').exists():
        raise RuntimeError('existing attempt; do not repeat calls silently')
    complete=excluded=0
    with backend.controls(math_module,final_class):
        for qid in assigned:
            started=time.monotonic(); pending=[]; outcomes=[]
            question_token=QUESTION.set(qid)
            backend.context={'question_id':qid}
            start=len(backend.cache.events)
            episodes=[e for e in manifest['episodes'] if e['question_id']==qid]
            try:
                for episode in episodes:
                    if episode['status']!='planned':
                        continue
                    replay=AuthorReplay(api,backend,materials[qid])
                    replay.messages=messages_for(materials[qid],episode['receiver_id'])
                    by_source={m.source:m for m in replay.messages}
                    priors,_=structural_priors(episode['receiver_id'],replay.messages,materials[qid]['predecessors'])
                    base=unit_vector(get_sentence_embedding(f"Task:\n{materials[qid]['task']}\nReceiver role:\n{episode['receiver_role']}"))
                    variants={}
                    # Rotate inference order independent of answers; cache shares
                    # exact repeated requests, never condition identities.
                    names=list(episode['variants'])
                    rotate=int(hashlib.sha256(episode['episode_id'].encode()).hexdigest(),16)%len(names)
                    for variant in names[rotate:]+names[:rotate]:
                        backend.context.update(episode_id=episode['episode_id'],variant=variant)
                        local_start=len(backend.cache.events)
                        result=await replay_intervention(materials[qid],episode['receiver_id'],
                                  episode['variants'][variant],replay.execute,replay.finalize)
                        item={'variant':variant,'receiver_answer':replay.parse(result['receiver_output']),
                              'final_answer':replay.parse(result['final_output']),
                              'receiver_output':result['receiver_output'],'final_output':result['final_output'],
                              'affected_nodes':result['affected_nodes'],
                              'full_logical_model_tokens':replay.full_logical_tokens(result['affected_nodes'],local_start),
                              **backend.cache.totals(local_start)}
                        variants[variant]=item
                        append(out/'conditions_raw.jsonl',dict(question_id=qid,episode_id=episode['episode_id'],**item))
                    transitions=[('empty','a')]
                    if 'ab' in variants:
                        transitions += [('empty','b'),('empty','ab'),('a','ab'),('b','ab')]
                    for left,right in transitions:
                        selected=[by_source[s] for s in episode['variants'][left]]
                        group=EvidenceGroup(tuple(episode['variants'][right]),tuple(episode['path']))
                        x,info=group_features(base,selected,group,replay.messages,priors,materials[qid]['predecessors'],
                                             episode['receiver_role'],backend.counter,CONFIG['communication_tokens'])
                        local_delta=int(variants[right]['receiver_answer']==gold[qid])-int(variants[left]['receiver_answer']==gold[qid])
                        final_delta=int(variants[right]['final_answer']==gold[qid])-int(variants[left]['final_answer']==gold[qid])
                        pending.append(dict(question_id=qid,question_index=episode['question_index'],
                                            episode_id=episode['episode_id'],fit_or_calibration=episode['fit_or_calibration'],
                                            receiver_role=episode['receiver_role'],receiver_id=episode['receiver_id'],
                                            left=left,right=right,features=x.tolist(),feature_info=info,
                                            local_delta=local_delta,downstream_delta=final_delta))
                    effects={scope:factorial_effects({k:int(v[scope+'_answer']==gold[qid]) for k,v in variants.items()})
                             for scope in ('receiver','final')} if 'ab' in variants else {}
                    outcomes.append(dict(episode_id=episode['episode_id'],receiver_role=episode['receiver_role'],
                                         density=episode['density'],variants=variants,effects=effects))
                if not pending:
                    raise InvalidUnit('no complete training comparisons fit')
                item=dict(question_id=qid,question_index=episodes[0]['question_index'],status='ok',
                          label_count=len(pending),episodes=outcomes,seconds=time.monotonic()-started,
                          **backend.cache.totals(start))
                # One JSON line is the atomic training unit. Orphan condition
                # rows cannot be consumed by fitting after an interrupted unit.
                append(out/'questions.jsonl',dict(item,labels=pending))
                complete+=1
                print(f'TRAIN complete={complete} excluded={excluded} qid={qid}',flush=True)
            except BudgetExceeded:
                raise
            except (InvalidUnit,ValueError) as error:
                item=dict(question_id=qid,status='excluded',reason=str(error),
                          seconds=time.monotonic()-started,**backend.cache.totals(start))
                append(out/'excluded_questions.jsonl',item); excluded+=1
                print(f'TRAIN_EXCLUDED qid={qid} reason={error}',flush=True)
            finally:
                QUESTION.reset(question_token)
                save(out/'progress.json',dict(completed_questions=complete,excluded_questions=excluded,
                      target_questions=len(assigned),last_question=qid,**backend.cache.totals()))
    save(out/'finished.json',dict(status='complete',completed_questions=complete,excluded_questions=excluded))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--shard',type=int,choices=(0,1),required=True)
    parser.add_argument('--main-host',required=True)
    asyncio.run(run(parser.parse_args()))
