"""Audit completed original-MOC supplement and compare the three frozen val arms."""
import collections
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import random
import runpy

ROOT = Path(__file__).resolve().parent
SUPPLEMENT = ROOT.parents[1]/'moc_mmlu_pro_val_baseline_20261015'
RAW = SUPPLEMENT/'remote_snapshot/records'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

with contextlib.redirect_stdout(io.StringIO()):
    parent = runpy.run_path(str(ROOT/'analyze_val.py'))
gold, methods = parent['gold'], parent['methods']
manifest = read(SUPPLEMENT/'remote_snapshot/manifest.json')
for name, digest in manifest['files'].items():
    assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == digest, name
lock = read(RAW/'experiment_lock.json')
for name,digest in {**lock['source_hashes'], **lock['data_hashes']}.items():
    assert hashlib.sha256((RAW/name).read_bytes()).hexdigest() == digest, name
assert (RAW/'data/val_inputs.jsonl').read_bytes() == (ROOT.parent/'data/val_inputs.jsonl').read_bytes()
assert lock['local_gold_sha256']['val'] == hashlib.sha256((ROOT.parent/'data/val_gold_LOCAL_ONLY.jsonl').read_bytes()).hexdigest()
budget = read(RAW/'budget.json')
assert read(RAW/'status.json')['phase'] == 'complete' and read(RAW/'cleanup.json')['cleanup_complete']
assert budget['reserved_tokens'] == 0 and budget['unknown_cost_upper_bound_tokens'] == 0
moc = [r for p in (RAW/'val').glob('shard*/results.jsonl') for r in rows(p)]
keys = [(r['question_id'], r['arm'], r['density']) for r in moc]
assert len(keys) == len(set(keys)) == 112
assert set(keys) == {(q,'moc',d) for q in gold for d in (.3,.7)}
for r in moc:
    if r['status'] == 'ok':
        assert methods['postprocess_answer'](None,r['final_output']) == r['answer']
        assert r['main_calls'] == 8 and r['merger_calls'] % 5 == 0
        assert r['logical_model_tokens'] == r['main_model_tokens']+r['merger_model_tokens']
    r['correct'] = int(r['status'] == 'ok' and r['answer'] == gold[r['question_id']])

calls = [r for name in ('calls_raw.jsonl','merger_calls_raw.jsonl')
         for p in (RAW/'val').glob('shard*/'+name) for r in rows(p)]
startup = rows(RAW/'startup_calls_raw.jsonl')
new = [r for r in calls if not r['cache_hit']]+startup
def tokens(r):
    return r['response']['prompt_eval_count']+r['response']['eval_count']
assert sum(tokens(r) for r in new) == budget['actual_tokens']
assert len(new) == budget['new_calls']
assert sum(tokens(r) for r in calls) == sum(r['logical_model_tokens'] for r in moc)
assert sum(not r['cache_hit'] for r in calls) == sum(r['new_calls'] for r in moc)
assert sum(tokens(r) for r in calls if not r['cache_hit']) == sum(r['new_model_tokens'] for r in moc)
lengths = [dict(question_id=r.get('question_id'),model=r['request']['model'],cache_hit=r.get('cache_hit'))
           for r in calls if r['response'].get('done_reason') in ('length','max_tokens')]
pending, settled = set(), set()
events = []
for p in (RAW/'val').glob('shard*/trace.jsonl'):
    for e in rows(p):
        events.append(e)
        if e['event_type'] == 'backend_request_started': pending.add(e['reservation_id'])
        if e['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown'): settled.add(e['reservation_id'])
assert not pending-settled
for r in new:
    assert r['response'].get('done') is True
    assert all(isinstance(r['response'].get(k),int) and r['response'][k]>=0 for k in ('prompt_eval_count','eval_count'))
audit = dict(status='passed' if not lengths else 'accounting_passed_output_completeness_warning',
    n_conditions=112, downloaded_files=len(manifest['files']), all_hashes_verified=True,
    source_locks_verified=True, parent_val_identical=True, original_parser_rechecked=True,
    invalid=sum(r['status']!='ok' for r in moc), budget=budget,
    logical_model_tokens=sum(r['logical_model_tokens'] for r in moc),
    startup_tokens=sum(tokens(r) for r in startup),
    logical_calls=len(calls), cached_calls=sum(r['cache_hit'] for r in calls),
    unsettled_reservations=0, cleanup=read(RAW/'cleanup.json'), output_cap_responses=lengths,
    trace_event_counts=dict(collections.Counter(e['event_type'] for e in events)))
(SUPPLEMENT/'completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

results = parent['results']+moc
summary = dict(scope='Completed val only; no tuning; three arms on identical56 questions',
               n_questions=56, n_conditions=len(results), arms={}, paired_vs_moc={},
               original_moc_completion_audit=audit,
               training=parent['summary']['training'],
               caveat='AQuA transfer abstains all nonroot messages; no-target-training costs are not comparable to task training without separating one-time training expense.')
for density in (.3,.7):
    groups = {}
    for arm in ('moc','frozen_transfer','adapted'):
        group = sorted((r for r in results if r['arm']==arm and r['density']==density),key=lambda r:r['question_id'])
        assert len(group)==56
        groups[arm]=group
        summary['arms'][f'{arm}|{density}'] = dict(n=56,correct=sum(r['correct'] for r in group),
            accuracy=sum(r['correct'] for r in group)/56,invalid=sum(r['status']!='ok' for r in group),
            logical_model_tokens=sum(r['logical_model_tokens'] for r in group),
            main_model_tokens=sum(r['main_model_tokens'] for r in group),
            merger_model_tokens=sum(r['merger_model_tokens'] for r in group),
            new_model_tokens=sum(r['new_model_tokens'] for r in group),
            merger_calls=sum(r['merger_calls'] for r in group))
    baseline=groups['moc']
    for arm in ('frozen_transfer','adapted'):
        group=groups[arm]
        assert [r['question_id'] for r in group]==[r['question_id'] for r in baseline]
        diffs=[r['correct']-b['correct'] for r,b in zip(group,baseline)]
        wins,losses=diffs.count(1),diffs.count(-1);n=wins+losses
        rng=random.Random(42);boot=sorted(sum(rng.choices(diffs,k=56))/56 for _ in range(20000))
        summary['paired_vs_moc'][f'{arm}|{density}']=dict(
            gain_questions=wins,loss_questions=losses,accuracy_difference=sum(diffs)/56,
            bootstrap95_percentile=[boot[499],boot[19499]],bootstrap_replicates=20000,seed=42,
            mcnemar_exact_two_sided=min(1,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1,
            logical_token_change=sum(r['logical_model_tokens'] for r in group)/sum(r['logical_model_tokens'] for r in baseline)-1)
(ROOT/'three_arm_val_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=True,indent=2))
