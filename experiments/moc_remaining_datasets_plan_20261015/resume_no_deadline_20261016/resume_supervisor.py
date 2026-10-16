"""Resume only unfinished original-MOC test, with no experiment wall deadline."""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from resume_worker import validate_authorization
from preflight_archive import active, read, sha

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--main-gpu',required=True);p.add_argument('--merger-gpu',required=True);p.add_argument('--merger-port',type=int,required=True);p.add_argument('--review',default='pre_resume_review.json');a=p.parse_args()
    root=Path(a.root).resolve();validate_authorization(root)
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    review_path=(HERE/a.review).resolve();assert review_path.parent==HERE
    reviewed=read(review_path);assert reviewed['status']=='passed'
    old=read(root/'status.json');review=reviewed['runs'][root.name]
    assert old['cleanup_complete'] and old['phase'] in ('failed','adaptive_complete')
    assert not active(old.get('supervisor_pid')) and not (root/'STOP').exists() and not (HERE.parent/'STOP').exists()
    sys.path.insert(0,str(root))
    import common
    import gpu_runtime as gpu
    import supervise as original
    config=common.CONFIG
    assert subprocess.check_output(['git','-C',config['author_directory'],'rev-parse','HEAD'],text=True).strip()==config['author_commit']
    assert not subprocess.check_output(['git','-C',config['author_directory'],'status','--porcelain'],text=True).strip()
    assert read(root/'preflight_report.json')['status']=='passed'
    assert read(root/'budget.json')==review['budget']
    assert read(root/'run_start.json')==review['original_run_start']
    for name,m in review['terminal_files'].items():assert sha(root/name)==m['sha256']
    attempt=root/'attempts'/('human_no_deadline_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));attempt.mkdir(parents=True,exist_ok=False)
    for n in ('budget.json','status.json','cleanup.json','run_start.json','policy_lock.json'):shutil.copyfile(root/n,attempt/n)
    # Reconcile only the audited dead-worker reservation, retaining actual/calls.
    if review['unresolved']:
        assert root.name=='moc_v21_svamp_train40_test_20261015' and len(review['unresolved'])==1
        rid,event=next(iter(review['unresolved'].items()));assert rid=='main-1792111215015014674'
        trace=root/'test'/'shard1'/'trace.jsonl';records=common.rows(trace)
        assert any(e.get('reservation_id')==rid and e['event_type']=='backend_request_started' for e in records)
        assert not any(e.get('reservation_id')==rid and e['event_type']=='backend_request_reconciled_unknown' for e in records)
        with common.budget_lock():
            budget=read(root/'budget.json');assert budget==review['budget']
            budget['reserved_tokens']-=event['reserved_tokens']
            budget['unknown_cost_upper_bound_tokens']+=event['reserved_tokens']
            common.save(root/'budget.json',budget)
        common.append(trace,dict(event_index=max(e.get('event_index',0) for e in records)+1,
            event_type='backend_request_reconciled_unknown',time_unix=time.time(),reservation_id=rid,
            unknown_cost_upper_bound_tokens=event['reserved_tokens'],reserved_tokens=event['reserved_tokens'],
            question_id=event['question_id'],arm='moc',density=event['density'],phase='test',
            authorization_id='human_continue_no_deadline_20261016',reason='Original deadline killed request without usage; carry independent conservative unknown once'))
        common.save(attempt/'unknown_reconciliation.json',dict(before=review['budget'],after=budget,reservation_id=rid))
    assert read(root/'budget.json')['reserved_tokens']==0
    signal.signal(signal.SIGTERM,original.handle);signal.signal(signal.SIGINT,original.handle)
    original_stop=original.stopped
    original.stopped=lambda:original_stop() or (HERE.parent/'STOP').exists() or (HERE/'STOP').exists()
    original_spawn=gpu.spawn
    def spawn(cmd,log,env,cwd):
        if len(cmd)>2 and cmd[2]==str(root/'runner.py'):
            cmd=[*cmd[:2],str(HERE/'resume_worker.py'),'--root',str(root),*cmd[3:]]
        return original_spawn(cmd,log,env,cwd)
    gpu.spawn=spawn
    original.publish(phase='preflight',mode='baseline',arm='moc',supervisor_pid=os.getpid(),worker_pids=[],services=[],cleanup_complete=False,
        experiment_wall_deadline=None,human_authorization_id='human_continue_no_deadline_20261016',resume_attempt=str(attempt),
        original_deadline_unix=review['original_run_start']['deadline_unix'])
    services=[]
    try:
        first={g['uuid']:g for g in gpu.gpu_snapshot()};time.sleep(10)
        second={g['uuid']:g for g in gpu.gpu_snapshot()}
        original.publish(admission_checks=dict(first=first,second=second),assigned_gpus=[a.main_gpu,a.merger_gpu])
        assert a.main_gpu!=a.merger_gpu
        for uuid,minimum in ((a.main_gpu,21000),(a.merger_gpu,10000)):
            assert uuid in first and uuid in second and max(first[uuid]['utilization'],second[uuid]['utilization'])<=15
            assert min(first[uuid]['free_mib'],second[uuid]['free_mib'])>=minimum
        for kind,uuid,port in [('main0',a.main_gpu,config['main_ports'][0]),('merger',a.merger_gpu,a.merger_port)]:
            if original.stopped():raise RuntimeError('STOP before service startup')
            services.append(gpu.start_service(kind,second[uuid],port));original.publish(services=services)
        original.stage('test','moc',services,math.inf)
        original.verify_coverage(False)
        # Existing terminal bytes must remain an unchanged prefix after append.
        for name,m in review['terminal_files'].items():
            assert hashlib.sha256((root/name).read_bytes()[:m['bytes']]).hexdigest()==m['sha256']
        assert read(root/'run_start.json')==review['original_run_start']
        original.publish(phase='complete',error=None,error_type=None)
    except BaseException as error:
        original.publish(phase='stopped' if original.stopped() else 'failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        gpu.capture_children()
        for proc in reversed(gpu.PROCESSES):gpu.owned_stop(proc)
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
        for port in [config['main_ports'][0],a.merger_port]:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:ports.append(port)
        children=[pid for pid,identity in gpu.CHILDREN.items() if (cur:=gpu.process_identity(pid)) and cur['start']==identity['start'] and cur['state'] not in ('Z','X')]
        owned=[proc.pid for proc in gpu.PROCESSES if proc.poll() is None]
        clean=dict(remaining_ports=ports,owned_children_remaining=children,owned_processes_remaining=owned,cleanup_complete=not ports and not children and not owned)
        common.save(root/'cleanup.json',clean);original.publish(**clean)

if __name__=='__main__':main()
