"""Single owned post-generation CPU evaluation; preserves frozen scorer/code."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def save(name,value):
    path=ROOT/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text())
    status=json.loads((ROOT/'status.json').read_text())
    assert status['phase']=='complete' and status['cleanup_complete']
    if (ROOT/'scoring_complete.json').exists():
        print('scoring already complete');return
    if not args.worker:
        prior=ROOT/'scoring_launcher.json'
        if prior.exists():
            record=json.loads(prior.read_text());proc=Path('/proc')/str(record['pid'])
            if proc.exists() and b'launch_scoring.py' in (proc/'cmdline').read_bytes():
                print(json.dumps(dict(already_running=True,**record)));return
        with (ROOT/'scoring.log').open('a') as log:
            proc=subprocess.Popen([cfg['python_executable'],'-u',str(Path(__file__).resolve()),'--worker'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
        record=dict(pid=proc.pid,started_unix=time.time(),cpu_only=True,model_calls=0)
        save('scoring_launcher.json',record);print(json.dumps(record));return
    lock=(ROOT/'scoring.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=json.loads((ROOT/'run_start.json').read_text())['deadline_unix']
    if (ROOT/'STOP').exists() or time.time()>=deadline:raise RuntimeError('STOP/deadline before scoring')
    command=[cfg['python_executable'],'-u',str(ROOT/'score_completed.py'),'--targets',str(ROOT/'eval/test_targets_LOCAL_ONLY.jsonl')]
    proc=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    save('scoring_status.json',dict(phase='scoring',wrapper_pid=os.getpid(),child_pid=proc.pid,started_unix=time.time()))
    try:
        while proc.poll() is None:
            if (ROOT/'STOP').exists() or time.time()>=deadline:raise RuntimeError('STOP/deadline during scoring')
            time.sleep(1)
        if proc.returncode:raise RuntimeError('scorer exited '+str(proc.returncode))
        save('scoring_status.json',dict(phase='complete',wrapper_pid=os.getpid(),child_pid=proc.pid,
            completed_unix=time.time(),model_calls=0))
    except BaseException as error:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        save('scoring_status.json',dict(phase='failed',error=str(error),wrapper_pid=os.getpid(),child_pid=proc.pid))
        raise


if __name__=='__main__':main()
