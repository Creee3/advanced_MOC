"""Read-only deadline diagnostics; no model calls or gold access."""
import json
from pathlib import Path
R=Path('/home/liuxin/ICML2027/experiments/moc_v21_svamp_train40_test_20261015')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
trace=[e for stage in ('training','test') for p in (R/stage).glob('shard*/trace.jsonl') for e in rows(p)]
starts={e['reservation_id']:e for e in trace if e['event_type']=='backend_request_started'}
settled={e['reservation_id']:e for e in trace if e['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown')}
raw=[e for stage in ('training','test') for p in (R/stage).glob('shard*/*calls_raw.jsonl') for e in rows(p)]
warm=rows(R/'startup_calls_raw.jsonl')
actual=[e for e in raw if not e['cache_hit']]+warm
print(json.dumps(dict(budget=json.loads((R/'budget.json').read_text()),
    starts=len(starts),settled=len(settled),unsettled=[e for k,e in starts.items() if k not in settled],
    raw_actual_tokens=sum(e['response']['prompt_eval_count']+e['response']['eval_count'] for e in actual),
    raw_actual_calls=len(actual),warmup_calls=len(warm),
    logs={p.name:p.read_text(errors='replace').splitlines()[-35:] for p in R.glob('test_moc_worker*.log')},
    cleanup=json.loads((R/'cleanup.json').read_text()),status=json.loads((R/'status.json').read_text())),ensure_ascii=True))
