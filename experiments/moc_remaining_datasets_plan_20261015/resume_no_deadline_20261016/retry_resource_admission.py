"""One explicit diagnosed resource-admission retry; no retry loop."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
from resume_dispatch import snapshot,pair_devices,PYTHON
from preflight_archive import active,read
HERE=Path(__file__).resolve().parent
NAME='moc_v21_svamp_train40_test_20261015'
R=HERE.parent.parent/NAME
def main():
    with (HERE/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        review=read(HERE/'resource_retry_review_svamp.json')
        assert read(HERE/'resource_retry_download_verified.json')['archive_sha256']==review['archive_sha256']
        launches=read(HERE/'launches.json');old=launches[NAME]
        assert old['pid']==3007461 and 'resource_retry' not in old
        status=read(R/'status.json');assert status['cleanup_complete'] and not active(status['supervisor_pid'])
        assert not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists() and not (R/'STOP').exists()
        assert read(R/'budget.json')==review['runs'][NAME]['budget']
        a=snapshot();time.sleep(10);b=snapshot();pairs=pair_devices(a,b,1)
        if not pairs:
            print(json.dumps(dict(phase='waiting_for_gpu',action='none')));return
        main,merger=pairs[0]
        cmd=[PYTHON,'-u',str(HERE/'resume_supervisor.py'),'--root',str(R),'--main-gpu',main['uuid'],'--merger-gpu',merger['uuid'],'--merger-port','11455','--review','resource_retry_review_svamp.json']
        with (R/'supervisor.log').open('a') as log:
            proc=subprocess.Popen(cmd,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        entry=dict(pid=proc.pid,started_utc=datetime.now(timezone.utc).isoformat(),main=main,merger=merger,merger_port=11455,cmd=cmd,
            prior_admission_launch=old,resource_retry=1,review='resource_retry_review_svamp.json')
        launches[NAME]=entry;temp=HERE/'launches.tmp';temp.write_text(json.dumps(launches,indent=2)+'\n');temp.replace(HERE/'launches.json')
        (HERE/'resource_retry_launch.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(dict(phase='launched',**entry)))
if __name__=='__main__':main()
