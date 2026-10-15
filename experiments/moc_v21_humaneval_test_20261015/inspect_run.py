"""Read-only actual stage, commitments, raw timestamps, process and GPU check."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {}
out={k:status.get(k) for k in ('phase','supervisor_pid','worker_pids','completed_test_conditions','error','cleanup_complete','updated_utc','budget')}
pids=[status.get('supervisor_pid'),*status.get('worker_pids',[])]
pids += [s['pid'] for s in status.get('services',[])]
out['actual_processes']=[]
for p in filter(None,pids):
    try:cmdline=Path('/proc',str(p),'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError):cmdline=None
    out['actual_processes'].append(dict(pid=p,exists=cmdline is not None,cmdline=cmdline))
out['training']=[];out['test']=[]
for phase in ('training','test'):
    for directory in sorted((ROOT/phase).glob('shard*')):
        def count(name):
            path=directory/name
            return sum(bool(s.strip()) for s in path.read_text().splitlines()) if path.exists() else 0
        raw=directory/'calls_raw.jsonl'
        item={'shard':directory.name,'raw_age_seconds':round(time.time()-raw.stat().st_mtime,1) if raw.exists() else None,
              'committed_questions':count('questions.jsonl'),'excluded_questions':count('excluded_questions.jsonl'),
              'validation_results':count('results.jsonl')}
        out[phase].append(item)
records=[]
for path in sorted((ROOT/'test').glob('shard*/results.jsonl')):
    content=path.read_text();lines=content.splitlines()
    if content and not content.endswith('\n'):lines=lines[:-1]
    records.extend(json.loads(s) for s in lines if s.strip())
keys={(r['question_id'],r['arm'],r['density']) for r in records}
assert len(keys)==len(records),'duplicate committed conditions'
counts={}
for qid,arm,density in keys:counts[qid]=counts.get(qid,0)+1
out.update(terminal_conditions=len(records),target=656,valid=sum(r['status']=='ok' for r in records),
    invalid=sum(r['status']=='invalid' for r in records),questions_started=len(counts),
    questions_all_four_terminal=sum(n==4 for n in counts.values()))
for name in ('budget','run_start','scoring_complete','cleanup'):
    path=ROOT/(name+'.json')
    if path.exists():out[name]=json.loads(path.read_text())
if 'run_start' in out:out['remaining_wall_seconds']=round(out['run_start']['deadline_unix']-time.time())
out['gpu_now']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(out,indent=2))
