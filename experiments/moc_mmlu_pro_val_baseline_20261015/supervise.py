"""One immutable eight-hour dataset budget; adaptive/main-only or original-MOC phase."""
import argparse
import ctypes
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import gpu_runtime as gpu
from common import ROOT,CONFIG,save,rows

STATE={};STOPPING=False
def publish(**kw):
    STATE.update(kw);STATE['updated_utc']=datetime.now(timezone.utc).isoformat()
    if (ROOT/'budget.json').exists():STATE['budget']=json.loads((ROOT/'budget.json').read_text())
    STATE['progress']={phase:[json.loads(p.read_text()) for p in (ROOT/phase).glob('shard*/progress.json')] for phase in ('training','val','test')}
    save(ROOT/'status.json',STATE)
def stopped():return STOPPING or (ROOT/'STOP').exists()
def handle(*a):
    global STOPPING
    STOPPING=True
def eligible(count,deadline):
    previous={};start=time.monotonic()
    while time.monotonic()-start<180 and time.time()<deadline and not stopped():
        now=[g for g in gpu.gpu_snapshot() if g['uuid'] in CONFIG['allowed_gpu_uuids']];stable=[g for g in now if g['uuid'] in previous and g['utilization']<=15 and previous[g['uuid']]['utilization']<=15]
        main=sorted([g for g in stable if min(g['free_mib'],previous[g['uuid']]['free_mib'])>=21000],key=lambda g:(-g['free_mib'],g['index']))
        if main:
            if count==1:return [main[0]]
            merger=[g for g in stable if g['uuid']!=main[0]['uuid'] and min(g['free_mib'],previous[g['uuid']]['free_mib'])>=10000]
            if merger:return [main[0],sorted(merger,key=lambda g:(-g['free_mib'],g['index']))[0]]
        previous={g['uuid']:g for g in now};publish(phase='waiting_for_gpu',gpu_snapshot=now);time.sleep(10)
    raise RuntimeError('no eligible GPUs; do not use CPU or increase GPU count')
def worker_env():
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUTF8='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    return e
def stage(phase,arm,services,deadline):
    workers=[];started=time.time()
    for shard in range(1 if phase=='fit' else CONFIG['workers']):
        cmd=[CONFIG['python_executable'],'-u',str(ROOT/'runner.py'),'--phase',phase,'--shard',str(shard),
             '--arm',arm,'--main-host',services[0]['host']]
        if arm=='moc':cmd+=['--merger-host',services[1]['host']]
        workers.append(gpu.spawn(cmd,ROOT/f'{phase}_{arm}_worker{shard}.log',worker_env(),str(ROOT)))
    publish(phase=phase,arm=arm,worker_pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if stopped() or time.time()>=deadline:raise RuntimeError('STOP or original absolute deadline')
        if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('worker failure; preserve raw and costs')
        for shard,p in enumerate(workers):
            files=[ROOT/phase/f'shard{shard}'/n for n in ('calls_raw.jsonl','trace.jsonl','progress.json')]
            latest=max([started]+[x.stat().st_mtime for x in files if x.exists()])
            if p.poll() is None and time.time()-latest>600:raise RuntimeError('actual worker trace silence')
        for s in services:gpu.healthy(s)
        publish();time.sleep(10)
    if any(p.returncode for p in workers):raise RuntimeError('worker final nonzero exit')
    publish()
def verify_coverage(adaptive=False):
    results=[r for p in (ROOT/'val').glob('shard*/results.jsonl') for r in rows(p)]
    keys=[(r['question_id'],r['arm'],r['density']) for r in results]
    expected={(q['question_id'],'moc',d) for q in rows(ROOT/'data/val_inputs.jsonl') for d in (.3,.7)}
    assert len(keys)==len(set(keys)) and set(keys)==expected,(len(keys),len(expected))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['adaptive','baseline'],required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
    frozen=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert json.loads((ROOT/'preflight_report.json').read_text())['status']=='passed'
    assert subprocess.check_output(['git','-C',CONFIG['author_directory'],'rev-parse','HEAD'],text=True).strip()==CONFIG['author_commit']
    assert not subprocess.check_output(['git','-C',CONFIG['author_directory'],'status','--porcelain'],text=True).strip()
    if stopped():raise RuntimeError('STOP present')
    if (ROOT/'run_start.json').exists():
        if not args.resume:raise RuntimeError('existing run; no budget reset')
        old=json.loads((ROOT/'status.json').read_text());budget=json.loads((ROOT/'budget.json').read_text())
        if not old.get('cleanup_complete') or old['phase'] in ('complete','stopped') or budget['reserved_tokens']:
            raise RuntimeError('recovery needs inspected cleanup and reconciled costs')
        folder=ROOT/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');folder.mkdir(parents=True,exist_ok=False)
        for n in ('status.json','budget.json','cleanup.json','run_start.json'):
            if (ROOT/n).exists():shutil.copyfile(ROOT/n,folder/n)
        deadline=json.loads((ROOT/'run_start.json').read_text())['deadline_unix']
    else:
        assert args.mode=='baseline' and not args.resume
        deadline=time.time()+CONFIG['gpu_policy']['max_wall_seconds']
        save(ROOT/'run_start.json',dict(deadline_unix=deadline,started_utc=datetime.now(timezone.utc).isoformat()))
        save(ROOT/'budget.json',dict(new_calls=0,actual_tokens=0,reserved_tokens=0,unknown_cost_upper_bound_tokens=0))
    if time.time()>=deadline:raise RuntimeError('original deadline exhausted; cannot extend')
    publish(phase='preflight',mode=args.mode,supervisor_pid=os.getpid(),cleanup_complete=False)
    try:
        services=[]
        for i,device in enumerate(eligible(1 if args.mode=='adaptive' else 2,deadline)):
            services.append(gpu.start_service('main0' if i==0 else 'merger',device,CONFIG['main_ports'][0] if i==0 else CONFIG['merger_port']))
            publish(services=services)
        assert args.mode=='baseline'
        stage('val','moc',services,deadline)
        verify_coverage(False)
        publish(phase='complete')
    except BaseException as e:
        publish(phase='stopped' if stopped() else 'failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        gpu.capture_children()
        for p in reversed(gpu.PROCESSES):gpu.owned_stop(p)
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid,identity in gpu.CHILDREN.items():
                cur=gpu.process_identity(pid)
                if cur and cur['start']==identity['start'] and cur['state']!='Z':
                    try:os.kill(pid,sig)
                    except ProcessLookupError:pass
            time.sleep(1)
        while True:
            try:
                pid,_=os.waitpid(-1,os.WNOHANG)
                if not pid:break
            except ChildProcessError:break
        ports=[]
        for port in [*CONFIG['main_ports'],CONFIG['merger_port']]:
            with socket.socket() as s:
                if not s.connect_ex(('127.0.0.1',port)):ports.append(port)
        children=[pid for pid,entry in gpu.CHILDREN.items() if (cur:=gpu.process_identity(pid)) and cur['start']==entry['start']]
        owned=[p.pid for p in gpu.PROCESSES if p.poll() is None]
        clean=dict(remaining_ports=ports,owned_children_remaining=children,owned_processes_remaining=owned,
                   cleanup_complete=not ports and not children and not owned)
        save(ROOT/'cleanup.json',clean);publish(**clean)

if __name__=='__main__':main()
