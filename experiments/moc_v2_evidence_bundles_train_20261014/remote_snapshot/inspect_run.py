"""Read-only actual stage, commitments, raw timestamps, process and GPU check."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
out={k:status.get(k) for k in ('phase','supervisor_pid','worker_pids','completed_val_conditions','error','cleanup_complete','updated_utc','budget')}
pids=[status.get('supervisor_pid'),*status.get('worker_pids',[])]
pids += [s['pid'] for s in status.get('services',[])]
out['actual_processes']=[dict(pid=p,exists=Path('/proc',str(p)).exists()) for p in pids if p]
out['training']=[];out['val']=[]
for phase in ('training','val'):
    for directory in sorted((ROOT/phase).glob('shard*')):
        def count(name):
            path=directory/name
            return sum(bool(s.strip()) for s in path.read_text().splitlines()) if path.exists() else 0
        raw=directory/'calls_raw.jsonl'
        item={'shard':directory.name,'raw_age_seconds':round(time.time()-raw.stat().st_mtime,1) if raw.exists() else None,
              'committed_questions':count('questions.jsonl'),'excluded_questions':count('excluded_questions.jsonl'),
              'validation_results':count('results.jsonl')}
        out[phase].append(item)
out['gpu_now']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(out,indent=2))
