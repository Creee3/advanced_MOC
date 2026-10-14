"""Bounded frozen val-only supervision; reuse verified GPU ownership helpers."""
import argparse
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
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def publish(**updates):
    STATE.update(updates)
    STATE['updated_utc']=datetime.now(timezone.utc).isoformat()
    progress=[json.loads(p.read_text()) for p in (ROOT/'test').glob('shard*/progress.json')]
    STATE['test_progress']=progress
    # Committed records, rather than a stale worker counter, determine coverage.
    keys=[]
    for path in (ROOT/'test').glob('shard*/results.jsonl'):
        text=path.read_text()
        lines=text.splitlines()
        # Only newline-terminated records are committed while a writer is alive.
        # Interior malformed records still fail; never discard completed data.
        if text and not text.endswith('\n'):
            lines=lines[:-1]
        keys += [(r['question_id'],r['arm'],r['density'])
                 for line in lines if line.strip() for r in [json.loads(line)]]
    if len(keys)!=len(set(keys)):
        raise RuntimeError('duplicate committed validation conditions')
    STATE['completed_test_conditions']=len(keys)
    if (ROOT/'budget.json').exists():
        STATE['budget']=json.loads((ROOT/'budget.json').read_text())
    save(ROOT/'status.json',STATE)


def stopped():
    return STOPPING or (ROOT/'STOP').exists()


def handle(*args):
    global STOPPING
    STOPPING=True


def choose_gpus(deadline):
    previous=set(); started=time.monotonic(); policy=CONFIG['gpu_policy']
    while time.monotonic()-started<policy['max_wait_seconds'] and time.time()<deadline and not stopped():
        snapshot=gpu.gpu_snapshot()
        stable=[g for g in snapshot if g['uuid'] in previous and g['utilization']<=15]
        mains=sorted([g for g in stable if g['free_mib']>=21000],key=lambda g:(-g['free_mib'],g['index']))
        if mains:
            mergers=sorted([g for g in stable if g['uuid']!=mains[0]['uuid'] and g['free_mib']>=10000],key=lambda g:(-g['free_mib'],g['index']))
            if mergers:return [mains[0],mergers[0]]
        previous={g['uuid'] for g in snapshot if g['utilization']<=15 and g['free_mib']>=10000}
        publish(phase='waiting_for_gpu',gpu_snapshot=snapshot)
        time.sleep(10)
    raise RuntimeError('no eligible distinct main/merger GPUs; no CPU fallback')


def worker_env():
    value=os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUTF8='1',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    return value


def run_val(services,deadline):
    workers=[]
    started=time.time()
    for shard in range(CONFIG['workers']):
        service=services[0]
        command=[CONFIG['python_executable'],'-u',str(ROOT/'run_test.py'),
                 '--shard',str(shard),'--main-host',service['host'],'--merger-host',services[-1]['host']]
        workers.append(gpu.spawn(command,ROOT/f'test_worker{shard}.log',worker_env(),str(ROOT)))
    publish(phase='test',worker_pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if stopped() or time.time()>deadline:
            raise RuntimeError('STOP or original cumulative wall deadline')
        for shard,process in enumerate(workers):
            if process.poll() not in (None,0):
                raise RuntimeError(f'val worker{shard} exited {process.returncode}; preserve raw/logs')
            traces=[ROOT/'test'/f'shard{shard}'/name for name in ('calls_raw.jsonl','merger_calls_raw.jsonl','trace.jsonl','progress.json')]
            latest=max([started]+[p.stat().st_mtime for p in traces if p.exists()])
            if process.poll() is None and time.time()-latest>CONFIG['gpu_policy']['quiet_timeout_seconds']:
                raise RuntimeError('worker actual trace silence deadline')
        for service in services:
            gpu.healthy(service)
        publish()
        time.sleep(CONFIG['gpu_policy']['poll_seconds'])
    if any(p.returncode!=0 for p in workers):
        raise RuntimeError('nonzero final worker exit')
    publish()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,handle)
    signal.signal(signal.SIGINT,handle)
    lock=(ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if stopped():
        raise RuntimeError('STOP present before startup')
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    import ctypes
    assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
    frozen=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**frozen['source_hashes'],**frozen['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert json.loads((ROOT/'preflight_report.json').read_text())['status']=='passed'
    author=CONFIG['author_directory']
    assert subprocess.check_output(['git','-C',author,'rev-parse','HEAD'],text=True).strip()==CONFIG['author_commit']
    assert not subprocess.check_output(['git','-C',author,'status','--porcelain'],text=True).strip()
    if args.resume:
        old=json.loads((ROOT/'status.json').read_text())
        if old.get('phase') in ('complete','stopped') or not old.get('cleanup_complete'):
            raise RuntimeError('recovery requires failed state and verified cleanup')
        budget=json.loads((ROOT/'budget.json').read_text())
        if budget['reserved_tokens'] or budget['new_calls']>=CONFIG['max_new_calls'] or budget['actual_tokens']>=CONFIG['max_new_model_tokens']:
            raise RuntimeError('unresolved/cumulative cost limit forbids recovery')
        deadline=json.loads((ROOT/'run_start.json').read_text())['deadline_unix']
        if time.time()>=deadline:
            raise RuntimeError('original wall limit already reached')
        history=ROOT/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        history.mkdir(parents=True,exist_ok=False)
        import shutil
        for name in ('status.json','cleanup.json','budget.json','run_start.json','launcher.json'):
            if (ROOT/name).exists():
                shutil.copyfile(ROOT/name,history/name)
        STATE.update(attempt_archive=str(history.relative_to(ROOT)))
    else:
        if (ROOT/'status.json').exists() or (ROOT/'budget.json').exists() or any((ROOT/'test').glob('shard*/results.jsonl')):
            raise RuntimeError('existing run state; do not reset budgets')
        deadline=time.time()+CONFIG['gpu_policy']['max_wall_seconds']
        save(ROOT/'run_start.json',dict(deadline_unix=deadline,started_utc=datetime.now(timezone.utc).isoformat()))
        save(ROOT/'budget.json',dict(new_calls=0,actual_tokens=0,reserved_tokens=0))
    publish(phase='preflight',supervisor_pid=os.getpid(),cleanup_complete=False,
            train_gate_passed=False,user_authorized_test=True)
    try:
        services=[]
        for index,device in enumerate(choose_gpus(deadline)):
            if time.time()>=deadline or stopped():
                raise RuntimeError('STOP/deadline before service')
            services.append(gpu.start_service('main0' if index==0 else 'merger',device,CONFIG['main_ports'][0] if index==0 else CONFIG['merger_port']))
            publish(services=services)
        run_val(services,deadline)
        if STATE['completed_test_conditions']!=CONFIG['validation_conditions']:
            raise RuntimeError('validation coverage mismatch')
        publish(phase='complete')
    except BaseException as error:
        publish(phase='stopped' if stopped() else 'failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        gpu.capture_children()
        for process in reversed(gpu.PROCESSES):
            gpu.owned_stop(process)
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
        for port in [*CONFIG['main_ports'],CONFIG['merger_port']]:
            with socket.socket() as check:
                if check.connect_ex(('127.0.0.1',port))==0:
                    remaining.append(port)
        children=[pid for pid,entry in gpu.CHILDREN.items()
                  if (cur:=gpu.process_identity(pid)) and cur['start']==entry['start']]
        owned=[p.pid for p in gpu.PROCESSES if p.poll() is None]
        cleanup=dict(remaining_ports=remaining,owned_children_remaining=children,
                     owned_processes_remaining=owned,cleanup_complete=not remaining and not children and not owned)
        save(ROOT/'cleanup.json',cleanup)
        publish(**cleanup)


if __name__=='__main__':
    main()
