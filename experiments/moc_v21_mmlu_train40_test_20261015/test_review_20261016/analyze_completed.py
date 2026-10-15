"""Local-only completed MMLU audit and paired scoring; never changes a run."""
import ast
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
RAW=OUT/'records'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tokens(x):return x['response']['prompt_eval_count']+x['response']['eval_count']
manifest=read(OUT/'member_sha256.json')
assert read(OUT/'download_verification.json')['all_hashes_match']
for n,v in manifest.items():assert sha(RAW/n)==v['sha256'],n
lock=read(RAW/'experiment_lock.json');assert lock==read(RUN/'experiment_lock.json')
for n,h in {**lock['source_hashes'],**lock['data_hashes']}.items():assert sha(RAW/n)==h,n
assert sha(RUN/'data/test_gold_LOCAL_ONLY.jsonl')==lock['local_gold_sha256']['test']
assert not (RAW/'data/test_gold_LOCAL_ONLY.jsonl').exists()
policy=read(RAW/'policy_lock.json');assert policy['no_val_or_test_fitting'] and policy['frozen_before_val_and_test']
assert sha(RAW/'group_utility_model.json')==policy['model_sha256']
for n,h in policy['training_commits_sha256'].items():assert sha(RAW/n)==h
source=RUN.parents[1]/'coding_base/MOC-official/datasets/data_process.py'
author=source.parents[1]
assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()==lock['author_commit']
assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain'],text=True).strip()
tree=ast.parse(source.read_text(encoding='utf-8'))
node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MMLUDataset'))
node.bases=[ast.Name(id='object',ctx=ast.Load())]
node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('postprocess_answer','record_to_target_answer')]
for n in ast.walk(node):
    if isinstance(n,ast.FunctionDef):
        n.returns=None
        for a in n.args.args:a.annotation=None
ns={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns)
cls=ns['MMLUDataset'];parser=cls.__new__(cls)
gold={x['question_id']:cls.record_to_target_answer({'correct_answer':x['answer']}) for x in rows(RUN/'data/test_gold_LOCAL_ONLY.jsonl')}
results=[x for p in (RAW/'test').glob('shard*/results.jsonl') for x in rows(p)]
keys=[(x['question_id'],x['arm'],x['density']) for x in results]
assert len(keys)==len(set(keys))==1140
assert set(keys)=={(q,a,d) for q in gold for a in ('moc','adapted') for d in (.3,.7)}
for x in results:
    if x['status']=='ok':
        assert parser.postprocess_answer(x['final_output'])==x['answer']
        assert x['main_calls']==8 and x['merger_calls']%5==0
        assert x['logical_model_tokens']==x['main_model_tokens']+x['merger_model_tokens']
    x['correct']=int(x['status']=='ok' and x['answer']==gold[x['question_id']])
budget=read(RAW/'budget.json');assert budget['reserved_tokens']==budget['unknown_cost_upper_bound_tokens']==0
calls={stage:[x for p in (RAW/stage).glob('shard*/*calls_raw.jsonl') for x in rows(p)] for stage in ('training','test')}
warm=rows(RAW/'startup_calls_raw.jsonl')
allraw=calls['training']+calls['test'];new=[x for x in allraw if not x['cache_hit']]+warm
assert len(new)==budget['new_calls'] and sum(map(tokens,new))==budget['actual_tokens']
assert sum(map(tokens,calls['test']))==sum(x['logical_model_tokens'] for x in results)
assert sum(tokens(x) for x in calls['test'] if not x['cache_hit'])==sum(x['new_model_tokens'] for x in results)
trace=[x for stage in ('training','test') for p in (RAW/stage).glob('shard*/trace.jsonl') for x in rows(p)]
started={x['reservation_id'] for x in trace if x['event_type']=='backend_request_started'}
settled={x['reservation_id'] for x in trace if x['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown')}
assert not started-settled
lengths=[dict(stage=x.get('stage'),arm=x.get('arm'),cache_hit=x.get('cache_hit'),done_reason=x['response'].get('done_reason')) for x in allraw if x['response'].get('done_reason') in ('length','max_tokens')]
assert not any(x['cache_hit'] for x in lengths),'truncated response in successful cache'
assert read(RAW/'status.json')['phase']=='complete' and read(RAW/'status.json')['cleanup_complete']
assert read(RAW/'cleanup.json')['cleanup_complete']
summary=dict(status='completed_local_scoring',questions=285,conditions=1140,arms={},paired={},
    training=read(RAW/'training_report.json'),budget=budget,one_time_cost={},output_cap_responses=lengths)
for stage,group in calls.items():
    summary['one_time_cost'][stage]=dict(actual_tokens=sum(tokens(x) for x in group if not x['cache_hit']),
        new_calls=sum(not x['cache_hit'] for x in group),logical_tokens=sum(map(tokens,group)),cache_hits=sum(x['cache_hit'] for x in group))
summary['one_time_cost']['warmup']=dict(actual_tokens=sum(map(tokens,warm)),new_calls=len(warm))
summary['one_time_cost']['failed_discarded_actual_tokens']=budget['actual_tokens']-sum(x['new_model_tokens'] for x in results)-sum(x['new_model_tokens'] for p in (RAW/'training').glob('shard*/questions.jsonl') for x in rows(p))-sum(map(tokens,warm))
for d in (.3,.7):
    groups={}
    for a in ('moc','adapted'):
        g=sorted((x for x in results if x['arm']==a and x['density']==d),key=lambda x:x['question_id']);groups[a]=g
        selections=[s for x in g if x.get('sampling') for s in x['sampling']['selections']]
        selected_hops=[]
        for s in selections:
            hs={m['source']:m['hop'] for m in s['candidates']}
            selected_hops.extend(hs[k] for k in s['source_ids'])
        info=dict(n=len(g),correct=sum(x['correct'] for x in g),accuracy=sum(x['correct'] for x in g)/285,
            invalid=sum(x['status']!='ok' for x in g))
        for k in ('logical_model_tokens','main_model_tokens','merger_model_tokens','new_model_tokens','logical_calls','main_calls','merger_calls','cache_hits'):
            info[k]=sum(x[k] for x in g)
        info.update(selection_sizes=dict(collections.Counter(len(s['source_ids']) for s in selections)),
            stop_reasons=dict(collections.Counter(s['stop_reason'] for s in selections)),
            candidate_hops=dict(collections.Counter(m['hop'] for s in selections for m in s['candidates'])),
            selected_hops=dict(collections.Counter(selected_hops)),
            encoding_seconds=sum(x['sampling']['encoding_seconds'] for x in g if x.get('sampling')),
            encoding_input_tokens=sum(x['sampling']['encoding_input_tokens'] for x in g if x.get('sampling')),
            routing_seconds=sum(x['sampling']['routing_seconds'] for x in g if x.get('sampling')))
        summary['arms'][f'{a}|{d}']=info
    old,newg=groups['moc'],groups['adapted'];diff=[b['correct']-a['correct'] for a,b in zip(old,newg)]
    w,l=diff.count(1),diff.count(-1);n=w+l;rng=random.Random(42)
    boot=sorted(sum(rng.choices(diff,k=285))/285 for _ in range(20000))
    summary['paired'][str(d)]=dict(gains=w,losses=l,accuracy_difference=sum(diff)/285,
        bootstrap95=[boot[499],boot[19499]],bootstrap_replicates=20000,seed=42,
        mcnemar_exact_two_sided=min(1,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1,
        logical_token_change=sum(x['logical_model_tokens'] for x in newg)/sum(x['logical_model_tokens'] for x in old)-1)
audit=dict(status='passed',members_verified=len(manifest),source_data_policy_verified=True,local_gold_hash_verified=True,
    author_commit_verified=lock['author_commit'],author_worktree_clean=True,
    original_parser_source=str(source),original_parser_sha256=sha(source),original_parser_rechecked=True,
    exact_coverage=True,duplicates=0,invalid=sum(x['status']!='ok' for x in results),budget=budget,
    raw_actual_and_calls_match=True,logical_test_tokens_match_raw=True,pending_trace_reservations=0,
    output_cap_responses=lengths,cleanup_verified=True,archive_sha256=read(OUT/'archive_receipt.json')['archive_sha256'])
(OUT/'test_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(RUN/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(arms=summary['arms'],paired=summary['paired'],cost=summary['one_time_cost'],output_cap_count=len(lengths),audit=audit['status']),ensure_ascii=True))
