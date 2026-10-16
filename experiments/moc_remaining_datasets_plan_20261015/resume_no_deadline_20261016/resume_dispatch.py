"""Bounded human-authorized recovery admission: up to three disjoint GPU pairs."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
from preflight_archive import active, read, sha
from resume_worker import validate_authorization

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
PYTHON='/home/liuxin/.conda/envs/moc/bin/python'
PORTS={'moc_v21_gsm8k_fewshot_20261015':11450,'moc_v21_mmlu_pro_fewshot_20261015':11456,'moc_v21_svamp_train40_test_20261015':11455}

def snapshot():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    return {uuid.strip():dict(index=int(index),uuid=uuid.strip(),util=int(util),free=int(free)) for index,uuid,util,free in (line.split(',') for line in out.splitlines())}

def pair_devices(first,second,count):
    stable=[v for uuid,v in second.items() if uuid in first and max(first[uuid]['util'],v['util'])<=15]
    mains=sorted([v for v in stable if min(v['free'],first[v['uuid']]['free'])>=21000],key=lambda v:(-v['free'],v['index']))
    for n in range(min(count,len(mains)),0,-1):
        chosen=mains[:n];used={g['uuid'] for g in chosen}
        mergers=sorted([v for v in stable if v['uuid'] not in used and min(v['free'],first[v['uuid']]['free'])>=10000],key=lambda v:(v['free']>=21000,-v['free'],v['index']))
        if len(mergers)>=n:return list(zip(chosen,mergers[:n]))
    return []

def tick():
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    assert not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists()
    reviewed=read(HERE/'pre_resume_review.json');assert reviewed['status']=='passed'
    assert read(HERE/'local_download_verified.json')['archive_sha256']==reviewed['archive_sha256']
    assert sha(HERE/'authorization.json')==reviewed['authorization_sha256']
    launches=read(HERE/'launches.json') if (HERE/'launches.json').exists() else {}
    pending=[]
    # Pro has the largest remaining baseline; GSM/SVAMP keep their old terminals.
    names=['moc_v21_mmlu_pro_fewshot_20261015','moc_v21_gsm8k_fewshot_20261015','moc_v21_svamp_train40_test_20261015']
    for name in names:
        root=BASE/name;validate_authorization(root)
        status=read(root/'status.json')
        if status['phase']=='complete' or active(status.get('supervisor_pid')):continue
        if name in launches:
            return dict(phase='needs_attention',dataset=name,error='Previously launched recovery exited; diagnose before any retry')
        assert not (root/'STOP').exists() and status['cleanup_complete']
        assert read(root/'budget.json')==reviewed['runs'][name]['budget']
        config=read(root/'config.json')
        for port in [config['main_ports'][0],PORTS[name]]:
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,(name,port)
        pending.append(name)
    if not pending:return dict(phase='running_or_complete',launches=launches)
    first=snapshot();time.sleep(10);second=snapshot();pairs=pair_devices(first,second,len(pending))
    if not pairs:return dict(phase='waiting_for_gpu',pending=pending,gpu_snapshot=second)
    started=[]
    for name,(main,merger) in zip(pending,pairs):
        root=BASE/name
        cmd=[PYTHON,'-u',str(HERE/'resume_supervisor.py'),'--root',str(root),'--main-gpu',main['uuid'],'--merger-gpu',merger['uuid'],'--merger-port',str(PORTS[name])]
        with (root/'supervisor.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        entry=dict(pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat(),main=main,merger=merger,merger_port=PORTS[name],cmd=cmd)
        launches[name]=entry;started.append(dict(dataset=name,**entry))
        temp=HERE/'launches.tmp';temp.write_text(json.dumps(launches,indent=2)+'\n');temp.replace(HERE/'launches.json')
    return dict(phase='launched',started=started,pending=pending[len(pairs):],gpu_snapshot=second)

if __name__=='__main__':
    with (HERE/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:result=tick()
        except BaseException as error:result=dict(phase='needs_attention',error_type=type(error).__name__,error=str(error))
        result['updated_utc']=datetime.now(timezone.utc).isoformat()
        target=HERE/'dispatch_status.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
        print(json.dumps(result))
