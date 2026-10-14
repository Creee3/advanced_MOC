import json
from common import ROOT,CONFIG,rows,save
from moc_adaptive_sampling.counterfactual import ExactCallCache,check_response,request_key
from hashlib import sha256
def sha(path):return sha256(path.read_bytes()).hexdigest()

def restore_main_cache(backend, out):
    """Admit old and own raw jointly, excluding every conflicting key.

    Existing costs remain in the shared budget. Replayed cached calls still
    contribute full logical deployment tokens to the resulting condition.
    A truncated response remains raw and paid, but is not cache-admissible.
    """
    files = [ROOT/'data/historical_calls_LOCAL_ONLY.jsonl', ROOT/'data/val_historical_calls_LOCAL_ONLY.jsonl']
    historical = [record for path in files if path.exists() for record in rows(path)]
    prior_raw = rows(out/'calls_raw.jsonl') if (out/'calls_raw.jsonl').exists() else []
    prior_trace = rows(out/'trace.jsonl') if (out/'trace.jsonl').exists() else []
    backend.recorder.counter = max((event.get('event_index', 0) for event in prior_trace), default=0)
    tool_values = {}
    for event in prior_trace:
        if event.get('event_type') in ('new_tool_result', 'cached_tool_result'):
            tool_values.setdefault(event['request_key'], set()).add(str(event['result']))
    own_records, rejected = [], []
    source_sha = sha(out/'calls_raw.jsonl') if prior_raw else None
    for index, event in enumerate(prior_raw):
        try:
            check_response(event['response'])
            key = request_key(event['request'], CONFIG['main_digest'])
            if key != event['key']:
                raise ValueError('raw request-key integrity mismatch')
        except (KeyError, ValueError) as error:
            rejected.append(dict(raw_line=index+1, reason=str(error)))
            continue
        values = tool_values.get(key, {None})
        for value in values:
            record = dict(request=event['request'], response=event['response'], provenance={
                'origin': 'same_v21_attempt_resume', 'raw_sha256': source_sha,
                'raw_line': index+1, 'original_provenance': event.get('provenance')})
            if value is not None:
                record['tool_result'] = value
            own_records.append(record)
    # Rebuild in one pass: otherwise a later conflicting output could leave
    # an earlier convenient output available in the cache.
    backend.cache.entries.clear()
    backend.cache.historical_conflicts.clear()
    report = backend.cache.load_historical(historical + own_records)
    backend.cache.events = list(prior_raw)
    save(out/'resume_cache_admission.json', dict(
        report, conflicts=backend.cache.historical_conflicts,
        rejected_prior_raw=rejected, previous_raw_events=len(prior_raw),
        source_raw_sha256=source_sha))
