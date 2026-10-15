"""At most two GPUs: parallel adaptive stages, then serial two-GPU MOC stages."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
RUNS=[BASE/'moc_v21_gsm8k_fewshot_20261015',BASE/'moc_v21_mmlu_pro_fewshot_20261015']
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def save(v):
    t=ROOT/'coordinator_status.tmp';t.write_text(json.dumps(dict(v,updated_utc=datetime.now(timezone.utc).isoformat()),indent=2));t.replace(ROOT/'coordinator_status.json')
def start(folder,mode):
    s=read(folder/'status.json')
    if s:
        if mode=='adaptive' and s.get('phase') in ('adaptive_complete','complete'):return
        if mode=='baseline' and s.get('phase')=='complete':return
        pid=s.get('supervisor_pid')
        if pid and Path('/proc',str(pid),'cmdline').exists():raise RuntimeError('existing live supervisor; do not duplicate')
        if s.get('phase')!='adaptive_complete' or mode!='baseline':raise RuntimeError('failure requires diagnostic recovery, not coordinator blind restart')
    cmd=[read(folder/'config.json')['python_executable'],'-u',str(folder/'supervise.py'),'--mode',mode]
    if s:cmd+=['--resume']
    with (folder/'supervisor.log').open('a') as log:
        p=subprocess.Popen(cmd,cwd=folder,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (folder/f'launcher_{mode}.json').write_text(json.dumps(dict(pid=p.pid,mode=mode,started_utc=datetime.now(timezone.utc).isoformat())))
def main():
    lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'STOP').exists():raise RuntimeError('coordinator STOP present')
    save(dict(phase='parallel_adaptive',pid=os.getpid(),runs=[str(p) for p in RUNS]))
    start(RUNS[0],'adaptive')
    # Wait until GPU1 is actually loaded, preventing independent selectors
    # from racing to select the same unclaimed idle GPU.
    limit=time.time()+300
    while not read(RUNS[0]/'status.json').get('services'):
        if read(RUNS[0]/'status.json').get('phase') in ('failed','stopped') or time.time()>limit:raise RuntimeError('first adaptive service not ready')
        time.sleep(5)
    start(RUNS[1],'adaptive')
    while True:
        states=[read(p/'status.json') for p in RUNS]
        if (ROOT/'STOP').exists():raise RuntimeError('STOP coordinator; do not launch further phases')
        if any(s.get('phase') in ('failed','stopped') for s in states):raise RuntimeError('dataset stopped; inspect concrete root cause before recovery')
        if all(s.get('phase') in ('adaptive_complete','complete') and s.get('cleanup_complete') for s in states):break
        time.sleep(10)
    for folder in RUNS:
        save(dict(phase='serial_moc_baseline',pid=os.getpid(),dataset=folder.name))
        start(folder,'baseline')
        while True:
            s=read(folder/'status.json')
            if (ROOT/'STOP').exists():raise RuntimeError('STOP; no further launches')
            if s.get('phase') in ('failed','stopped'):raise RuntimeError('MOC baseline failure; inspect before recovery')
            if s.get('phase')=='complete' and s.get('cleanup_complete'):break
            time.sleep(10)
    save(dict(phase='generation_complete',pid=os.getpid(),runs=[str(p) for p in RUNS]))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(dict(phase='needs_attention',error_type=type(e).__name__,error=str(e),pid=os.getpid()));raise
