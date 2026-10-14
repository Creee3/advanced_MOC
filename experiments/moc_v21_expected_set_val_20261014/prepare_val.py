"""Prepare val-only exact history from preserved v2 artifacts, without gold.

No training calls, baseline calls, fitting, or outcome-based cache selection.
The caller freezes the checkpoint/config/source hashes after this preparation.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OLD = REPO/'experiments/moc_v2_evidence_bundles_train_20261014'
sys.path.insert(0, str(REPO/'coding_base'))
from moc_adaptive_sampling.counterfactual import check_response, request_key, ExactCallCache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    if (ROOT/'status.json').exists() or (ROOT/'budget.json').exists() or (ROOT/'val').exists():
        raise RuntimeError('do not prepare over an existing launched attempt')
    config = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    data = ROOT/'data'
    data.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative in ('data/val_inputs.jsonl', 'val_role_snapshot.json',
                     'data/historical_calls_LOCAL_ONLY.jsonl'):
        source, destination = OLD/relative, ROOT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        sources[relative] = dict(source=str(source.relative_to(REPO)), sha256=sha(source))
    inputs = rows(data/'val_inputs.jsonl')
    if len(inputs) != 60 or len({item['question_id'] for item in inputs}) != 60:
        raise RuntimeError('expected the original distinct val60 inputs')
    if any(set(item) != {'question_id', 'question', 'options'} for item in inputs):
        raise RuntimeError('val inputs must contain only frozen ID, question, and options (no gold)')
    old_history = OLD/'data/val_historical_calls_LOCAL_ONLY.jsonl'
    records = rows(old_history)
    sources[str(old_history.relative_to(REPO))] = dict(sha256=sha(old_history))
    rejected = []
    for stage in ('training', 'val'):
        for raw_path in sorted((OLD/'remote_snapshot'/stage).glob('shard*/calls_raw.jsonl')):
            raw_sha = sha(raw_path)
            sources[str(raw_path.relative_to(REPO))] = dict(sha256=raw_sha)
            trace = raw_path.with_name('trace.jsonl')
            tool_values = {}
            if trace.exists():
                sources[str(trace.relative_to(REPO))] = dict(sha256=sha(trace))
                for event in rows(trace):
                    if event.get('event_type') in ('new_tool_result', 'cached_tool_result'):
                        tool_values.setdefault(event['request_key'], set()).add(str(event['result']))
            for index, event in enumerate(rows(raw_path)):
                try:
                    check_response(event['response'])
                    key = request_key(event['request'], config['main_digest'])
                    if event['request']['model'] != config['main_model'] or key != event['key']:
                        raise ValueError('historical model or key mismatch')
                except (KeyError, ValueError) as error:
                    rejected.append(dict(source=str(raw_path.relative_to(REPO)), raw_line=index+1, reason=str(error)))
                    continue
                for value in tool_values.get(key, {None}):
                    record = dict(request=event['request'], response=event['response'], provenance={
                        'origin': 'frozen_v2_exact_call_no_gold', 'source': str(raw_path.relative_to(REPO)),
                        'source_sha256': raw_sha, 'raw_line': index+1,
                        'original_provenance': event.get('provenance')})
                    if value is not None:
                        record['tool_result'] = value
                    records.append(record)
    destination = data/'val_historical_calls_LOCAL_ONLY.jsonl'
    destination.write_text(''.join(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'
                                   for record in records), encoding='utf-8', newline='\n')
    # Inspect all records together; runtime repeats this same all-key conflict rule.
    cache = ExactCallCache(config['main_digest'], None)
    admission = cache.load_historical(rows(data/'historical_calls_LOCAL_ONLY.jsonl') + records)
    save(ROOT/'val_preparation.json', dict(
        schema='v21_val_only_exact_history', val_questions=len(inputs), validation_conditions=120,
        training_run=False, baseline_run=False, test_run=False, gold_read=False,
        sources=sources, valid_history_records=len(records), rejected_records=rejected,
        cache_admission=admission, cache_conflicts=cache.historical_conflicts,
        val_inputs_sha256=sha(data/'val_inputs.jsonl'), val_history_sha256=sha(destination),
        training_history_sha256=sha(data/'historical_calls_LOCAL_ONLY.jsonl')))
    print(json.dumps(dict(val_questions=60, validation_conditions=120, **admission)))


if __name__ == '__main__':
    main()
