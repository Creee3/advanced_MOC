"""Bounded train/calibration/val v2, reusing previously verified GPU helpers."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import gpu_runtime as gpu

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
STATE={}
STOPPING=False

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8'); temp.replace(path)

def publish(**updates):
    STATE.update(updates)
    STATE['updated_utc']=datetime.now(timezone.utc).isoformat()
    STATE['training_progress']=[json.loads(p.read_text()) for p in (ROOT/'training').glob('shard*/progress.json')]
    STATE['val_progress']=[json.loads(p.read_text()) for p in (ROOT/'val').glob('shard*/progress.json')]
    STATE['completed_val_conditions']=sum(x['processed_conditions'] for x in STATE['val_progress'])
    if (ROOT/'budget.json').exists():
        STATE['budget']=json.loads((ROOT/'budget.json').read_text())
    save(ROOT/'status.json',STATE)

def stopped():
    return STOPPING or (ROOT/'STOP').exists()

def handle(*args):
    global STOPPING
    STOPPING=True

def choose_two(deadline):
    previous=set(); start=time.monotonic()
    while time.monotonic()-start<CONFIG['gpu_policy']['max_wait_seconds'] and time.time()<deadline and not stopped():
        snapshots=gpu.gpu_snapshot()
        eligible=[s for s in snapshots if s['utilization']<=15 and s['free_mib']>=21000]
        stable=[s for s in eligible if s['uuid'] in previous]
        stable.sort(key=lambda s:(-s['free_mib'],s['index']))
        if len(stable)>=2:
            return stable[:2]
        previous={s['uuid'] for s in eligible}
        publish(phase='waiting_for_gpu',gpu_snapshot=snapshots)
        time.sleep(CONFIG['gpu_policy']['poll_seconds'])
    raise RuntimeError('GPU not available within bounded wait; no CPU fallback')

def env():
    value=os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTHONUTF8='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    return value

def stage(name,script,services,deadline):
    workers=[]
    for shard in range(CONFIG['workers']):
        cmd=[CONFIG['python_executable'],'-u',str(ROOT/script),'--shard',str(shard),'--main-host',services[shard]['host']]
        workers.append(gpu.spawn(cmd,ROOT/f'{name}_worker{shard}.log',env(),str(ROOT)))
    publish(phase=name,worker_pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if stopped() or time.time()>deadline:
            raise RuntimeError('STOP or bounded v2 wall deadline')
        for shard,p in enumerate(workers):
            if p.poll() not in (None,0):
                raise RuntimeError(f'{name} worker{shard} exited {p.returncode}; raw and logs preserved')
            trace=ROOT/name/f'shard{shard}'/'calls_raw.jsonl'
            if p.poll() is None and trace.exists() and time.time()-trace.stat().st_mtime>CONFIG['gpu_policy']['quiet_timeout_seconds']:
                raise RuntimeError('worker trace silence deadline')
        for service in services:
            gpu.healthy(service)
        publish()
        time.sleep(CONFIG['gpu_policy']['poll_seconds'])
    assert all(p.returncode==0 for p in workers)
    publish()

def main():
    signal.signal(signal.SIGTERM,handle); signal.signal(signal.SIGINT,handle)
    lock=(ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'status.json').exists() or (ROOT/'budget.json').exists():
        raise RuntimeError('existing run state; do not repeat completed work or reset budgets')
    if stopped():
        raise RuntimeError('STOP present before startup')
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    import ctypes
    assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
    frozen=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    deadline=time.time()+CONFIG['gpu_policy']['max_wall_seconds']
    save(ROOT/'run_start.json',dict(deadline_unix=deadline,started_utc=datetime.now(timezone.utc).isoformat()))
    save(ROOT/'budget.json',dict(new_calls=0,actual_tokens=0,reserved_tokens=0))
    publish(phase='preflight',supervisor_pid=os.getpid(),cleanup_complete=False)
    try:
        services=[]
        for i,g in enumerate(choose_two(deadline)):
            services.append(gpu.start_service(f'main{i}',g,CONFIG['main_ports'][i]))
            ledger=json.loads((ROOT/'budget.json').read_text())
            warm=services[-1]['warmup']
            ledger['actual_tokens']+=warm.get('prompt_eval_count',0)+warm.get('eval_count',0)
            ledger['new_calls']+=1
            save(ROOT/'budget.json',ledger)
            publish(services=services)
        stage('training','collect_v2.py',services,deadline)
        publish(phase='fit_and_calibration')
        fitted=gpu.spawn([CONFIG['python_executable'],'-u',str(ROOT/'fit_v2.py')],ROOT/'fit_v2.log',env(),str(ROOT))
        if fitted.wait(timeout=180)!=0:
            raise RuntimeError('fit/calibration failed; do not run val')
        assert (ROOT/'policy_lock.json').exists()
        publish(phase='policy_frozen',training_report=json.loads((ROOT/'training_report.json').read_text()))
        stage('val','run_val.py',services,deadline)
        if STATE['completed_val_conditions']!=CONFIG['validation_conditions']:
            raise RuntimeError('validation condition coverage mismatch')
        publish(phase='complete')
    except BaseException as error:
        publish(phase='stopped' if stopped() else 'failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        gpu.capture_children()
        for p in reversed(gpu.PROCESSES):
            gpu.owned_stop(p)
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid,identity in gpu.CHILDREN.items():
                current=gpu.process_identity(pid)
                if current and current['start']==identity['start'] and current['state']!='Z':
                    try:
                        os.kill(pid,sig)
                    except ProcessLookupError:
                        pass
            time.sleep(1)
        while True:
            try:
                child,_=os.waitpid(-1,os.WNOHANG)
                if child==0:
                    break
            except ChildProcessError:
                break
        remaining=[]
        for port in CONFIG['main_ports']:
            with socket.socket() as check:
                if check.connect_ex(('127.0.0.1',port))==0:
                    remaining.append(port)
        children=[pid for pid,e in gpu.CHILDREN.items() if (cur:=gpu.process_identity(pid)) and cur['start']==e['start']]
        owned=[p.pid for p in gpu.PROCESSES if p.poll() is None]
        cleanup=dict(remaining_ports=remaining,owned_children_remaining=children,owned_processes_remaining=owned,
                     cleanup_complete=not remaining and not children and not owned)
        save(ROOT/'cleanup.json',cleanup)
        publish(**cleanup)

if __name__=='__main__':
    main()
