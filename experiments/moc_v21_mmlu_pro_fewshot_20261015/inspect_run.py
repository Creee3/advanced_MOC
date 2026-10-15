"""Read actual processes and newline-committed work, never only stale PIDs."""
import json
from pathlib import Path
import time
from common import ROOT,rows
status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
pids=[status.get('supervisor_pid'),*status.get('worker_pids',[]),*[x['pid'] for x in status.get('services',[])]]
processes=[]
for pid in filter(None,pids):
    try:cmdline=Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,ProcessLookupError):cmdline=None
    processes.append(dict(pid=pid,cmdline=cmdline))
summary={}
for stage in ('training','val','test'):
    r=[x for p in (ROOT/stage).glob('shard*/'+('questions.jsonl' if stage=='training' else 'results.jsonl')) for x in rows(p)]
    raw=[p for p in (ROOT/stage).glob('shard*/calls_raw.jsonl')]
    summary[stage]=dict(committed=len(r),valid=sum(x['status']=='ok' for x in r),
        raw_age_seconds=[round(time.time()-p.stat().st_mtime,1) for p in raw])
print(json.dumps(dict(status=status,processes=processes,progress=summary,
    remaining_wall_seconds=round(json.loads((ROOT/'run_start.json').read_text())['deadline_unix']-time.time()) if (ROOT/'run_start.json').exists() else None),indent=2))
