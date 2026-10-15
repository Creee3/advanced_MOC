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
def process_active(pid,proc_root=Path('/proc')):
    if not pid:return False
    try:
        root=proc_root/str(pid)
        cmdline=(root/'cmdline').read_bytes()
        state=(root/'stat').read_text().rsplit(') ',1)[1].split()[0]
    except (FileNotFoundError,ProcessLookupError):
        return False
    return bool(cmdline) and state not in ('Z','X')

def start(folder,mode):
    if (ROOT/'STOP').exists() or (folder/'STOP').exists():raise RuntimeError('STOP; do not launch')
    s=read(folder/'status.json')
    if s:
        if mode=='adaptive' and s.get('phase') in ('adaptive_complete','complete'):return
        if mode=='baseline' and s.get('phase')=='complete':return
        if s.get('phase')!='adaptive_complete' or mode!='baseline':raise RuntimeError('failure requires diagnostic recovery, not coordinator blind restart')
        if not s.get('cleanup_complete') or read(folder/'budget.json').get('reserved_tokens',0):
            raise RuntimeError('previous phase cleanup/accounting incomplete; inspect before launch')
        # Final cleanup is published just before supervisor exit. Wait for exit
        # without treating an empty cmdline or zombie as a live supervisor.
        wait_until=time.monotonic()+30
        while process_active(s.get('supervisor_pid')):
            if (ROOT/'STOP').exists() or (folder/'STOP').exists():raise RuntimeError('STOP during phase transition')
            if time.monotonic()>=wait_until:raise RuntimeError('previous supervisor still live after bounded exit wait; do not duplicate')
            time.sleep(.5)
        if time.time()>=read(folder/'run_start.json')['deadline_unix']:
            raise RuntimeError('original dataset deadline exhausted; do not launch or extend')
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
