"""One reviewed runtime-only recovery; preserve old state before every mutation.

No gold, sampling parameters, model weights or completed valid outputs are read
for selection. The request with unknown usage is counted separately at its full
reservation upper bound. This script will not repeat an already-applied repair.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parent
QID='3b61294603818b9c9476250d101c0d7e6078ca0611a71a400b354e18c2651f60'
PATCHES={'common.py','merger_backend.py','run_test.py','supervise.py','resume_cache.py'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(p)


def stats(events):
    usage=lambda e:e['response']['prompt_eval_count']+e['response']['eval_count']
    return dict(logical_calls=len(events),new_calls=sum(not e['cache_hit'] for e in events),
        cache_hits=sum(e['cache_hit'] for e in events),logical_model_tokens=sum(map(usage,events)),
        new_model_tokens=sum(usage(e) for e in events if not e['cache_hit']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--patch-directory',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    patch=args.patch_directory.resolve()
    manifest_path=patch/'runtime_recovery_manifest_20261015.json'
    assert sha(manifest_path)==args.manifest_sha256
    manifest=read(manifest_path)
    assert manifest['approval']['user_message']=='继续跑呗'
    assert manifest['approval']['remaining_seconds']==22125
    assert set(manifest['patches'])==PATCHES
    assert not (ROOT/'STOP').exists(),'explicit STOP must not be overridden'
    assert not (ROOT/'recovery_applied_20261015.json').exists(),'one-shot recovery already applied'
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    guard=(ROOT/'supervisor.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(ROOT/'status.json');budget=read(ROOT/'budget.json');start=read(ROOT/'run_start.json')
    assert state['phase']=='failed' and state['cleanup_complete']
    assert state['completed_test_conditions']==324
    for pid in [state['supervisor_pid'],*state['worker_pids'],*[s['pid'] for s in state['services']]]:
        assert not Path('/proc',str(pid)).exists(),f'old owned PID still exists: {pid}'
    for port in (11448,11450):
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
    assert budget==dict(new_calls=2666,actual_tokens=1774668,reserved_tokens=10192),budget
    assert start['deadline_unix']==1792021081.7877574
    elapsed=datetime.fromisoformat(state['updated_utc']).timestamp()-datetime.fromisoformat(start['started_utc']).timestamp()
    assert 6674<elapsed<6675 and elapsed+22125<=28800
    old_lock=read(ROOT/'experiment_lock.json');old_policy=read(ROOT/'policy_lock.json')
    assert sha(ROOT/'experiment_lock.json')==old_policy['experiment_lock_sha256']
    for name,digest in {**old_lock['source_hashes'],**old_lock['data_hashes']}.items():
        assert sha(ROOT/name)==digest,('original drift',name)
    for name,change in manifest['patches'].items():
        assert sha(ROOT/name)==change['before_sha256']
        assert sha(patch/name)==change['after_sha256']
    for name,digest in manifest['additional_files'].items():assert sha(patch/name)==digest
    original_results=[r for s in range(2) for r in rows(ROOT/'test'/f'shard{s}'/'results.jsonl')]
    assert len(original_results)==324 and all(r['status']=='ok' for r in original_results)
    assert len({(r['question_id'],r['arm'],r['density']) for r in original_results})==324
    assert not any((r['question_id'],r['arm'],r['density'])==(QID,'moc',.3) for r in original_results)
    # Prove the invalid condition and all of its costs from the original journal.
    out=ROOT/'test/shard0'
    selected=lambda name:[e for e in rows(out/name) if (e.get('question_id'),e.get('arm'),e.get('density'))==(QID,'moc',.3)]
    main_events=selected('calls_raw.jsonl');merger_events=selected('merger_calls_raw.jsonl')
    assert len(main_events)==8 and not merger_events
    traces=rows(out/'trace.jsonl')
    timeouts=[e for e in traces if e.get('question_id')==QID and e.get('timed_out')]
    assert len(timeouts)==1 and timeouts[0]['event_type']=='code_execution_finished'
    assert 'incomplete graph outputs' in (ROOT/'test_worker0.log').read_text()
    measured_calls=0;measured_tokens=0
    for s in range(2):
        for name in ('calls_raw.jsonl','merger_calls_raw.jsonl'):
            for e in rows(ROOT/'test'/f'shard{s}'/name):
                if not e['cache_hit']:
                    measured_calls+=1
                    measured_tokens+=e['response']['prompt_eval_count']+e['response']['eval_count']
    for e in rows(ROOT/'startup_calls_raw.jsonl'):
        assert 'response' in e
        measured_calls+=1
        measured_tokens+=e['response']['prompt_eval_count']+e['response']['eval_count']
    assert measured_calls==2665 and measured_tokens==budget['actual_tokens']
    # Preserve every current original file, not just terminal manager JSONs.
    history=ROOT/'attempts'/datetime.now(timezone.utc).strftime('approved_recovery_%Y%m%dT%H%M%S_%fZ')
    history.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and 'attempts' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts)
    archived={str(p.relative_to(ROOT)):sha(p) for p in files}
    with tarfile.open(history/'before_recovery.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    save(history/'manifest.json',dict(files=archived,archive_sha256=sha(history/'before_recovery.tar.gz')))
    save(ROOT/'recovery_applied_20261015.json',dict(stage='archive_complete',attempt=str(history.relative_to(ROOT))))
    for name in PATCHES|set(manifest['additional_files']):shutil.copyfile(patch/name,ROOT/name)
    shutil.copyfile(manifest_path,ROOT/manifest_path.name)
    # The original policy/model/data remain fixed; hashes are amended only for
    # the five named runtime files. Both original locks remain in the archive.
    lock=read(ROOT/'experiment_lock.json')
    for name,change in manifest['patches'].items():lock['source_hashes'][name]=change['after_sha256']
    lock['runtime_amendment_sha256']=args.manifest_sha256
    save(ROOT/'experiment_lock.json',lock)
    policy=read(ROOT/'policy_lock.json')
    for name,change in manifest['patches'].items():policy['source_hashes'][name]=change['after_sha256']
    policy['experiment_lock_sha256']=sha(ROOT/'experiment_lock.json')
    policy['runtime_amendment_sha256']=args.manifest_sha256
    save(ROOT/'policy_lock.json',policy)
    config=read(ROOT/'config.json')
    testenv=os.environ.copy()
    testenv.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1',OMP_NUM_THREADS='2')
    for name in ('recovery_checks_20261015.py','preflight.py'):
        report=subprocess.run([config['python_executable'],'-u',str(ROOT/name)],cwd=ROOT,
            env=testenv,capture_output=True,text=True,timeout=180)
        (history/(name+'.log')).write_text(report.stdout+report.stderr)
        assert report.returncode==0,report.stdout+report.stderr
    assert read(ROOT/'preflight_report.json')['status']=='passed'
    a,b=stats(main_events),stats(merger_events)
    invalid=dict(question_id=QID,density=.3,arm='moc',status='invalid',
        reason='code timeout; incomplete graph outputs; recovered from original raw without regeneration',
        **{k:a[k]+b[k] for k in a},main_calls=8,merger_calls=0,
        main_model_tokens=a['logical_model_tokens'],merger_model_tokens=0,
        recovery_evidence=dict(attempt=str(history.relative_to(ROOT)),
            code_timeout_event_index=timeouts[0]['event_index'],runtime_only_recovery=True))
    with (out/'results.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(invalid)+'\n')
    # Unknown usage stays distinct from measured actual usage and is permanently
    # charged conservatively by reserve() against the original total token cap.
    budget.update(reserved_tokens=0,unknown_cost_upper_bound_tokens=10192,unknown_call_count=1)
    save(ROOT/'budget.json',budget)
    reconciliation=dict(known_actual_tokens=1774668,recorded_calls=2665,ledger_calls=2666,
        unknown_call_count=1,unknown_usage_upper_bound_tokens=10192,
        unknown_call_context=dict(shard=1,question_id='1e1c06e3ce72d2d8a629a73e0c10145bda70e7642d4a2596293af130cfe84318',arm='moc',density=.7),
        exact_usage_unavailable=True,actual_tokens_not_inflated=True,original_budget=dict(new_calls=2666,actual_tokens=1774668,reserved_tokens=10192))
    save(ROOT/'cost_reconciliation_20261015.json',reconciliation)
    now=time.time()
    start.update(original_deadline_unix=start['deadline_unix'],deadline_unix=now+22125,
        prior_active_wall_seconds=elapsed,recovery_active_window_seconds=22125,
        user_approval=manifest['approval'],extension_started_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT/'run_start.json',start)
    review=read(ROOT/'recovery_review_20261015.json')
    review.update(status='approved_preflight_passed',deadline_extended=True,budget_reset=False,
        cloud_sources_modified=True,regression_checks='5 passed; 4 native Graph fake conditions passed',
        original_conditions_preserved=324,recovered_invalid_conditions=1,
        approval=manifest['approval'],runtime_manifest_sha256=args.manifest_sha256,
        attempt_archive=str(history.relative_to(ROOT)),new_deadline_unix=start['deadline_unix'])
    save(ROOT/'recovery_review_20261015.json',review)
    save(ROOT/'recovery_applied_20261015.json',dict(stage='verified_ready_to_launch',attempt=str(history.relative_to(ROOT)),deadline_unix=start['deadline_unix']))
    # Release our preparation lock; the supervisor takes it before GPU admission.
    fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
    launched=subprocess.check_output([config['python_executable'],str(ROOT/'launch.py'),'--resume'],cwd=ROOT,text=True)
    save(ROOT/'recovery_applied_20261015.json',dict(stage='launched',attempt=str(history.relative_to(ROOT)),launcher=json.loads(launched),deadline_unix=start['deadline_unix']))
    print(launched.strip())


if __name__=='__main__':main()
