"""Val-only analysis. Evaluation answers stay local; no backend calls."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'remote_snapshot'
OLD = ROOT.parents[2] / 'experiments/moc_stateful_sampling_val_20261014'
V2 = ROOT.parent / 'moc_v2_evidence_bundles_train_20261014'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def comparison(module, left, right, gold):
    out = module.paired(left, right, gold)
    # The v1 helper's 5pp margin is not part of this v2.1 protocol.
    for name in ('prespecified_margin', 'bootstrap_noninferiority_screen'):
        out.pop(name, None)
    keys = set(left) & set(right)
    out.update(reference_correct=sum(left[q]['answer'] == gold[q] for q in keys),
               candidate_correct=sum(right[q]['answer'] == gold[q] for q in keys),
               reference_tokens=sum(left[q]['stats']['total_model_tokens'] for q in keys),
               candidate_tokens=sum(right[q]['stats']['total_model_tokens'] for q in keys),
               interval_scope='descriptive paired question bootstrap; exposed development val; no noninferiority claim')
    return out


def main():
    spec = importlib.util.spec_from_file_location('original_paired_metrics', OLD/'analyze_local.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gold = {r['question_id']: r['correct'] for r in rows(OLD/'data/val_evaluation_only_LOCAL_ONLY.jsonl')}
    original = [r for p in (OLD/'remote_snapshot/runs').glob('shard*/results.jsonl') for r in rows(p)]
    v2 = [r for p in (V2/'remote_snapshot/val').glob('shard*/results.jsonl') for r in rows(p)]
    current = [r for p in (SNAP/'val').glob('shard*/results.jsonl') for r in rows(p)]
    assert len(current) == 120 and len({(r['question_id'], r['density']) for r in current}) == 120
    old_limits = json.loads((OLD/'completion_audit.json').read_text())['output_limit_calls']
    metrics, comparisons, sensitivity = {}, {}, {}
    invalid = [r for r in current if r['status'] != 'ok']
    for density in (.3, .7):
        valid = {r['question_id']: r for r in current if r['density'] == density and r['status'] == 'ok'}
        selections = [s for r in valid.values() for s in r['sampling']['selections']]
        receivers = [s for s in selections if s['candidates']]
        scores = [c for s in receivers for layer in s['layers'] for c in layer.get('scores', [])]
        count_sizes = Counter(str(len(s['source_ids'])) for s in receivers)
        reads = Counter(str(m['hop']) for s in receivers for m in s['candidates'] if m['source'] in s['source_ids'])
        available = Counter(str(m['hop']) for s in receivers for m in s['candidates'])
        by_role = {}
        for s in receivers:
            records = [c for layer in s['layers'] for c in layer.get('scores', [])]
            role = records[0]['receiver_role'] if records else 'no_feasible_action'
            bucket = by_role.setdefault(role, Counter())
            bucket['receivers'] += 1
            bucket['empty'] += not bool(s['source_ids'])
            bucket['singleton'] += len(s['source_ids']) == 1
            bucket['pair'] += len(s['source_ids']) == 2
        metrics[str(density)] = dict(
            valid_questions=len(valid), correct=sum(r['answer'] == gold[q] for q, r in valid.items()),
            full_inference_model_tokens=sum(r['logical_model_tokens'] for r in valid.values()),
            new_validation_model_tokens=sum(r['new_model_tokens'] for r in valid.values()),
            logical_calls=sum(r['logical_calls'] for r in valid.values()),
            cache_hits=sum(r['cache_hits'] for r in valid.values()),
            ordinary_receivers=len(selections), nonroot_receivers=len(receivers),
            empty_nonroot_receivers=count_sizes['0'], selected_set_sizes=dict(count_sizes),
            read_messages_by_hop=dict(reads), available_messages_by_hop=dict(available),
            group_reads=count_sizes['2'], candidate_group_scores=len(scores),
            unsupported_candidate_groups=sum(not c['supported'] for c in scores),
            stop_reasons=dict(Counter(s['stop_reason'] for s in receivers)),
            communication_text_tokens=sum(s['communication_tokens'] for s in selections),
            encoding_seconds=sum(r['sampling']['encoding_seconds'] for r in valid.values()),
            routing_seconds=sum(r['sampling']['routing_seconds'] for r in valid.values()),
            sketch_bytes_generated=sum(r['sampling']['sketch_bytes_generated'] for r in valid.values()),
            selection_by_receiver_role={k: dict(v) for k, v in by_role.items()})
        transformed = {q: dict(r, stats={'total_model_tokens': r['logical_model_tokens']}) for q, r in valid.items()}
        baselines = {arm: {r['question_id']: r for r in original if r['arm'] == arm and r['density'] == density}
                     for arm in ('moc', 'moc_without_phi', 'vanilla_mas', 'adaptive_sampling')}
        baselines['v2'] = {r['question_id']: dict(r, stats={'total_model_tokens': r['logical_model_tokens']})
                           for r in v2 if r['density'] == density and r['status'] == 'ok'}
        for arm, baseline in baselines.items():
            assert set(baseline) == set(gold), (density, arm, len(baseline))
            comparisons[f'{density}_vs_{arm}'] = comparison(module, baseline, transformed, gold)
            excluded = {r['question_id'] for r in invalid if r['density'] == density}
            excluded |= {r['question_id'] for r in old_limits if r['density'] == density}
            sensitivity[f'{density}_vs_{arm}'] = dict(excluded_question_ids=sorted(excluded),
                paired=comparison(module, {q: r for q, r in baseline.items() if q not in excluded},
                    {q: r for q, r in transformed.items() if q not in excluded}, gold))
    raw = [r for p in (SNAP/'val').glob('shard*/calls_raw.jsonl') for r in rows(p)]
    budget = json.loads((SNAP/'budget.json').read_text())
    startup = rows(SNAP/'startup_calls_raw.jsonl')
    warm = sum(r['response']['prompt_eval_count'] + r['response']['eval_count'] for r in startup)
    inference = sum(r['logical_model_tokens'] for r in current)
    baseline_cost = sum(comparisons[f'{d}_vs_moc']['reference_tokens'] for d in (.3, .7))
    shared_train_cost = json.loads((V2/'analysis.json').read_text())['new_cost']['training_tokens_including_exclusions']
    summary = dict(metrics=metrics, comparisons=comparisons, integrity_sensitivity=sensitivity,
        invalid_val=invalid, inherited_baseline_output_limits=old_limits,
        cost=dict(full_logical_inference_tokens=inference,
            actual_new_val_tokens=sum(r['response']['prompt_eval_count'] + r['response']['eval_count'] for r in raw if not r['cache_hit']),
            warmup_tokens=warm, actual_new_tokens_including_warmup=budget['actual_tokens'],
            actual_new_calls_including_warmup=budget['new_calls'], cache_hits=sum(r['cache_hit'] for r in raw),
            unresolved_reserved_tokens=budget['reserved_tokens'], new_v21_training_tokens=0,
            inherited_v2_training_tokens=shared_train_cost, previous_v1_training_and_waste_tokens=657528,
            moc_inference_tokens=baseline_cost,
            inference_ratio_vs_moc=inference/baseline_cost,
            shared_training_plus_v21_inference_ratio_vs_moc=(inference+shared_train_cost)/baseline_cost,
            investment_scope='v2 training reused once, not paid again; v1 historical costs listed separately; cached calls count fully in logical inference'),
        interpretation='exposed diagnostic val60; two densities share the same question groups; frozen model; no baseline rerun or test',
        causal_limit='chosen actions/accuracy association does not establish causal usefulness; set-size cap and merger removal contribute to cost differences')
    (ROOT/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'metrics': metrics, 'cost': summary['cost'], 'moc_comparisons': {d: comparisons[f'{d}_vs_moc'] for d in (.3, .7)}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
