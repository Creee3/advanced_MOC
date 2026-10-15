"""Local completed-val scoring using the original GSM8K parser; no test gold."""
import ast
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tarfile

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(ROOT/'download_manifest.json')
assert sha(ROOT/'val_records.tar.gz')==m['archive_sha256']
with tarfile.open(ROOT/'val_records.tar.gz') as t:t.extractall(ROOT/'records',filter='data')
for name,h in m['files'].items():assert sha(ROOT/'records'/name)==h,name
lock=read(RUN/'experiment_lock.json')
assert read(ROOT/'records/experiment_lock.json')==lock
assert sha(RUN/'data/val_gold_LOCAL_ONLY.jsonl')==lock['local_gold_sha256']['val']
policy=read(ROOT/'records/policy_lock.json')
assert policy['frozen_before_val_and_test'] and policy['no_val_or_test_fitting']
assert sha(ROOT/'records/group_utility_model.json')==policy['model_sha256']

source=RUN.parents[1]/'coding_base/MOC-official/datasets/data_process.py'
tree=ast.parse(source.read_text(encoding='utf-8'));classes=[]
for name in ('MathDataset','GSM8KDataset'):
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name))
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in ('_clean_numeric_result','postprocess_answer','record_to_target_answer')]
    if name=='MathDataset':node.bases=[ast.Name(id='object',ctx=ast.Load())]
    for n in ast.walk(node):
        if isinstance(n,ast.FunctionDef):
            n.returns=None
            for a in n.args.args:a.annotation=None
    classes.append(node)
ns={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(source),'exec'),ns)
cls=ns['GSM8KDataset'];parser=cls.__new__(cls)
gold={r['question_id']:cls.record_to_target_answer(r) for r in rows(RUN/'data/val_gold_LOCAL_ONLY.jsonl')}
rs=[r for p in (ROOT/'records/val').glob('shard*/results.jsonl') for r in rows(p)]
keys=[(r['question_id'],r['arm'],r['density']) for r in rs]
assert len(keys)==len(set(keys))==480
assert set(keys)=={(q,a,d) for q in gold for a in ('frozen_transfer','adapted') for d in (.3,.7)}
for r in rs:
    if r['status']=='ok':assert parser.postprocess_answer(r['final_output'])==r['answer']
    r['correct']=int(r['status']=='ok' and r['answer']==gold[r['question_id']])
summary=dict(n_questions=len(gold),n_conditions=len(rs),all_downloaded_hashes_verified=True,
             original_parser_rechecked=True,training=m['training'],arms={},paired_adapted_vs_transfer={})
for d in (.3,.7):
    groups={}
    for a in ('frozen_transfer','adapted'):
        g=sorted([r for r in rs if r['arm']==a and r['density']==d],key=lambda r:r['question_id']);groups[a]=g
        sels=[s for r in g for s in r['sampling']['selections']]
        summary['arms'][f'{a}|{d}']=dict(n=len(g),correct=sum(r['correct'] for r in g),accuracy=sum(r['correct'] for r in g)/len(g),
            invalid=sum(r['status']!='ok' for r in g),logical_model_tokens=sum(r['logical_model_tokens'] for r in g),
            new_model_tokens=sum(r['new_model_tokens'] for r in g),
            selection_sizes=dict(collections.Counter(len(s['source_ids']) for s in sels)),
            stop_reasons=dict(collections.Counter(s['stop_reason'] for s in sels)),
            encoding_seconds=sum(r['sampling']['encoding_seconds'] for r in g),
            routing_seconds=sum(r['sampling']['routing_seconds'] for r in g))
    old,new=groups['frozen_transfer'],groups['adapted']
    assert [r['question_id'] for r in old]==[r['question_id'] for r in new]
    diff=[b['correct']-a['correct'] for a,b in zip(old,new)];w,l=diff.count(1),diff.count(-1);n=w+l
    rng=random.Random(42);boots=sorted(sum(rng.choices(diff,k=len(diff)))/len(diff) for _ in range(20000))
    summary['paired_adapted_vs_transfer'][str(d)]=dict(gains=w,losses=l,accuracy_difference=sum(diff)/len(diff),
        bootstrap95=[boots[499],boots[19499]],replicates=20000,seed=42,
        mcnemar_exact_two_sided=min(1,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1,
        logical_token_change=sum(r['logical_model_tokens'] for r in new)/sum(r['logical_model_tokens'] for r in old)-1)
(ROOT/'val_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True,indent=2))
