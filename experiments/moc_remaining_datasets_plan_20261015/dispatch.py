"""Bounded tick: admit first launches only with idle GPUs; never restart failures."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
RUNS=[BASE/'moc_v21_mmlu_train40_test_20261015',BASE/'moc_v21_svamp_train40_test_20261015']

def read(p):return json.loads(p.read_text()) if p.exists() else {}
def active(pid):
    try:
        p=Path('/proc')/str(pid)
        return bool((p/'cmdline').read_bytes()) and (p/'stat').read_text().rsplit(') ',1)[1].split()[0] not in ('Z','X')
    except (OSError,ProcessLookupError):return False
def snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    return {u.strip():dict(uuid=u.strip(),util=int(v),free=int(f)) for u,v,f in (line.split(',') for line in rows.splitlines())}
def stable_devices(a,b,count):
    eligible=[v for k,v in b.items() if k in a and max(v['util'],a[k]['util'])<=15]
    main=[v for v in eligible if min(v['free'],a[v['uuid']]['free'])>=21000]
    if not main:return []
    first=sorted(main,key=lambda v:(-v['free'],v['uuid']))[0]
    if count==1:return [first]
    others=[v for v in eligible if v['uuid']!=first['uuid'] and min(v['free'],a[v['uuid']]['free'])>=10000]
    return [first,sorted(others,key=lambda v:(-v['free'],v['uuid']))[0]] if others else []
def audit(folder):
    c=read(folder/'config.json');lock=read(folder/'experiment_lock.json')
    assert c['train_questions']==40 and c['val_questions']==0 and c['validation_conditions']==0
    assert read(folder/'preflight_report.json')['status']=='passed'
    for name,h in {**lock['source_hashes'],**lock['data_hashes']}.items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h,name
    assert not (folder/'data/test_gold_LOCAL_ONLY.jsonl').exists(),'test gold must stay local'
    return c
def launch(folder,mode):
    if (ROOT/'STOP').exists() or (folder/'STOP').exists():raise RuntimeError('STOP')
    c=audit(folder);s=read(folder/'status.json');start=read(folder/'run_start.json')
    if active(s.get('supervisor_pid')):raise RuntimeError('existing live supervisor')
    if start:
        if mode!='baseline' or s.get('phase')!='adaptive_complete':raise RuntimeError('existing run requires diagnosis; no blind restart')
        if not s.get('cleanup_complete') or read(folder/'budget.json').get('reserved_tokens',0):raise RuntimeError('unclean transition/accounting')
        if time.time()>=start['deadline_unix']:raise RuntimeError('original deadline exhausted; no extension')
    elif mode!='adaptive':raise RuntimeError('baseline before training')
    for port in [*c['main_ports'],c['merger_port']] if mode=='baseline' else c['main_ports']:
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('port already occupied')
    cmd=[c['python_executable'],'-u',str(folder/'supervise.py'),'--mode',mode]+(['--resume'] if start else [])
    with (folder/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=folder,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    entry=dict(pid=child.pid,mode=mode,started_utc=datetime.now(timezone.utc).isoformat())
    (folder/f'launcher_{mode}.json').write_text(json.dumps(entry,indent=2))
    return dict(dataset=folder.name,**entry)
def tick():
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _tick()
def _tick():
    if (ROOT/'STOP').exists():return dict(phase='stopped',action='none')
    states=[read(p/'status.json') for p in RUNS]
    for folder,s in zip(RUNS,states):
        if (folder/'STOP').exists():return dict(phase='needs_attention',dataset=folder.name,error='STOP present')
        if s and s.get('phase') not in ('adaptive_complete','complete') and not active(s.get('supervisor_pid')):
            return dict(phase='needs_attention',dataset=folder.name,error=s.get('error') or 'inactive unfinished supervisor; diagnose before resume')
        if s.get('phase') in ('failed','stopped'):return dict(phase='needs_attention',dataset=folder.name,error=s.get('error'))
    live=[s for s in states if active(s.get('supervisor_pid'))]
    if any(s.get('mode')=='baseline' for s in live):return dict(phase='running_baseline',action='none')
    if len(live)>=2:return dict(phase='running_adaptive',action='none')
    unstarted=[p for p in RUNS if not (p/'run_start.json').exists()]
    if unstarted:
        candidate=unstarted[0];mode='adaptive';count=1
    else:
        if not all(s.get('phase') in ('adaptive_complete','complete') and s.get('cleanup_complete') for s in states):
            return dict(phase='running_adaptive',action='none')
        # Final cleanup publication precedes supervisor exit; next tick handles it.
        if live:return dict(phase='waiting_previous_exit',action='none')
        pending=[p for p,s in zip(RUNS,states) if s.get('phase')!='complete']
        if not pending:return dict(phase='generation_complete',action='audit_required')
        candidate=pending[0];mode='baseline';count=2
    start=read(candidate/'run_start.json')
    if start and time.time()>=start['deadline_unix']:return dict(phase='needs_attention',dataset=candidate.name,error='original deadline exhausted; do not launch')
    audit(candidate)
    a=snapshot();time.sleep(10);b=snapshot();devices=stable_devices(a,b,count)
    if not devices:return dict(phase='waiting_for_gpu',dataset=candidate.name,mode=mode,gpu_snapshot=b,action='none')
    return dict(phase='launched',launch=launch(candidate,mode),eligible_devices=devices)

if __name__=='__main__':
    try:result=tick()
    except BaseException as error:result=dict(phase='needs_attention',error_type=type(error).__name__,error=str(error))
    result['updated_utc']=datetime.now(timezone.utc).isoformat()
    temp=ROOT/'dispatch_status.tmp';temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(ROOT/'dispatch_status.json')
    print(json.dumps(result,ensure_ascii=False))
