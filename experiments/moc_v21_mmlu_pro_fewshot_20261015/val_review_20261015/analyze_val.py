"""Read-only local validation analysis; never accesses author test targets."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import typing

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

manifest = read(ROOT/'download_manifest.json')
for name, digest in manifest['files'].items():
    assert hashlib.sha256((ROOT/'records'/name).read_bytes()).hexdigest() == digest, name
lock = read(RUN/'experiment_lock.json')
gold_path = RUN/'data/val_gold_LOCAL_ONLY.jsonl'
assert hashlib.sha256(gold_path.read_bytes()).hexdigest() == lock['local_gold_sha256']['val']
assert read(ROOT/'records/experiment_lock.json') == lock
policy = read(ROOT/'records/policy_lock.json')
assert policy['no_val_or_test_fitting'] and policy['frozen_before_val_and_test']
assert hashlib.sha256((ROOT/'records/group_utility_model.json').read_bytes()).hexdigest() == policy['model_sha256']

# Compile only the two original author parser methods, avoiding heavyweight imports.
source = RUN.parents[1]/'coding_base/MOC-official/datasets/data_process.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'MMLUProDataset')
methods = {}
for name in ('postprocess_answer', 'record_to_target_answer'):
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    namespace = {'re': re, 'Union': typing.Union, 'List': typing.List}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), namespace)
    methods[name] = namespace[name]
gold = {r['question_id']: methods['record_to_target_answer'](r) for r in rows(gold_path)}
results = [r for p in (ROOT/'records/val').glob('shard*/results.jsonl') for r in rows(p)]
expected = {(q, a, d) for q in gold for a in ('frozen_transfer', 'adapted') for d in (.3, .7)}
keys = [(r['question_id'], r['arm'], r['density']) for r in results]
assert len(keys) == len(set(keys)) and set(keys) == expected
for r in results:
    if r['status'] == 'ok':
        assert methods['postprocess_answer'](None, r['final_output']) == r['answer']
    r['correct'] = int(r['status'] == 'ok' and r['answer'] == gold[r['question_id']])

summary = {'scope': 'Completed MMLU-Pro val only; descriptive, no policy changes',
           'n_questions': len(gold), 'n_conditions': len(results),
           'all_downloaded_file_hashes_verified': True, 'original_parser_rechecked': True,
           'training': {k: read(ROOT/'records/training_report.json')[k]
                        for k in ('planned_questions', 'effective_questions', 'labels', 'excluded')},
           'arms': {}, 'paired': {}}
for density in (.3, .7):
    grouped = {}
    for arm in ('frozen_transfer', 'adapted'):
        group = sorted([r for r in results if r['density'] == density and r['arm'] == arm], key=lambda r: r['question_id'])
        grouped[arm] = group
        selections = [s for r in group for s in r.get('sampling', {}).get('selections', [])]
        summary['arms'][f'{arm}|{density}'] = dict(
            n=len(group), correct=sum(r['correct'] for r in group), invalid=sum(r['status'] != 'ok' for r in group),
            accuracy=sum(r['correct'] for r in group)/len(group),
            logical_model_tokens=sum(r['logical_model_tokens'] for r in group),
            new_model_tokens=sum(r['new_model_tokens'] for r in group),
            logical_calls=sum(r['logical_calls'] for r in group),
            selection_sizes=dict(collections.Counter(len(s['source_ids']) for s in selections)),
            stop_reasons=dict(collections.Counter(s['stop_reason'] for s in selections)),
            encoding_seconds=sum(r['sampling'].get('encoding_seconds', 0) for r in group),
            routing_seconds=sum(r['sampling'].get('routing_seconds', 0) for r in group),
            encoding_input_tokens=sum(r['sampling'].get('encoding_input_tokens', 0) for r in group))
    old, new = grouped['frozen_transfer'], grouped['adapted']
    assert [r['question_id'] for r in old] == [r['question_id'] for r in new]
    diffs = [b['correct']-a['correct'] for a,b in zip(old,new)]
    wins, losses = diffs.count(1), diffs.count(-1)
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(20000))
    n = wins+losses
    summary['paired'][str(density)] = dict(
        adapted_gain_questions=wins, adapted_loss_questions=losses,
        accuracy_difference=sum(diffs)/len(diffs),
        bootstrap95_percentile=[boot[499], boot[19499]], bootstrap_replicates=20000, seed=42,
        mcnemar_exact_two_sided=min(1, 2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1,
        logical_model_token_change=sum(r['logical_model_tokens'] for r in new)/sum(r['logical_model_tokens'] for r in old)-1)
(ROOT/'val_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(summary, ensure_ascii=True, indent=2))
