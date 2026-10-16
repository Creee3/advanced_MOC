"""Archive a completed resumed baseline, all raw costs and trusted runtime source."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile
from preflight_archive import read,rows,sha,active
from resume_worker import validate_authorization

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
TARGETS={'moc_v21_gsm8k_fewshot_20261015':300,'moc_v21_mmlu_pro_fewshot_20261015':280,'moc_v21_svamp_train40_test_20261015':300}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=TARGETS);a=parser.parse_args()
    root=BASE/a.name;out=HERE/'completed_archives'/a.name
    if (out/'archive_receipt.json').exists():
        print(json.dumps(read(out/'archive_receipt.json')));return
    assert not out.exists(),'Incomplete prior archive attempt; diagnose before repeating'
    validate_authorization(root)
    status=read(root/'status.json');budget=read(root/'budget.json');config=read(root/'config.json');policy=read(root/'policy_lock.json')
    assert status['phase']=='complete' and status['cleanup_complete'] and read(root/'cleanup.json')['cleanup_complete']
    assert status['experiment_wall_deadline'] is None and not status.get('error')
    assert not any(active(pid) for pid in [status['supervisor_pid'],*status.get('worker_pids',[]),*[s['pid'] for s in status['services']]])
    for s in status['services']:
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',s['port']))!=0
    assert budget['reserved_tokens']==0
    assert sha(root/'group_utility_model.json')==policy['model_sha256']
    for n,h in policy['training_commits_sha256'].items():assert sha(root/n)==h
    assert not (root/'data/test_gold_LOCAL_ONLY.jsonl').exists()
    tests=[r for p in (root/'test').glob('shard*/results.jsonl') for r in rows(p)]
    keys=[(r['question_id'],r['arm'],r['density']) for r in tests]
    expected={(q['question_id'],arm,d) for q in rows(root/'data/test_inputs.jsonl') for arm in ('adapted','moc') for d in (.3,.7)}
    assert len(keys)==len(set(keys))==4*TARGETS[a.name] and set(keys)==expected
    raw=[r for phase in ('training','val','test') for p in (root/phase).glob('shard*/*calls_raw.jsonl') for r in rows(p)]
    warm=rows(root/'startup_calls_raw.jsonl');known=[r for r in raw if not r['cache_hit']]+warm
    tokens=lambda r:r['response']['prompt_eval_count']+r['response']['eval_count']
    assert sum(map(tokens,known))==budget['actual_tokens']
    traces=[r for phase in ('training','val','test') for p in (root/phase).glob('shard*/trace.jsonl') for r in rows(p)]
    started={r['reservation_id']:r for r in traces if r['event_type']=='backend_request_started'}
    settled={r['reservation_id']:r for r in traces if r['event_type']=='backend_request_settled'}
    unknown={r['reservation_id']:r for r in traces if r['event_type']=='backend_request_reconciled_unknown'}
    assert len(started)==sum(r['event_type']=='backend_request_started' for r in traces)
    assert not set(settled)&set(unknown) and not set(started)-set(settled)-set(unknown)
    assert sum(r['unknown_cost_upper_bound_tokens'] for r in unknown.values())==budget['unknown_cost_upper_bound_tokens']
    assert len(known)+len(unknown)==budget['new_calls']
    assert sum(r['actual_tokens'] for r in settled.values())+sum(map(tokens,warm))==budget['actual_tokens']
    assert budget['actual_tokens']+budget['unknown_cost_upper_bound_tokens']<=config['max_new_model_tokens'] and budget['new_calls']<=config['max_new_calls']
    previous=read(HERE/'pre_resume_review.json')['runs'][a.name]
    assert read(root/'run_start.json')==previous['original_run_start']
    for n,m in previous['terminal_files'].items():assert hashlib.sha256((root/n).read_bytes()[:m['bytes']]).hexdigest()==m['sha256']
    author=Path(config['author_directory'])
    assert subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'],text=True).strip()==config['author_commit']
    assert not subprocess.check_output(['git','-C',str(author),'status','--porcelain'],text=True).strip()
    files=[]
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        assert not p.is_symlink()
        if p.is_file():files.append((p,'run/'+p.relative_to(root).as_posix()))
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in ('.py','.json','.md','.txt') and p.name not in ('progress.md',):
            files.append((p,'runtime_management/'+p.name))
    tracked=subprocess.check_output(['git','-C',str(author),'ls-files'],text=True).splitlines()
    for name in tracked:
        p=author/name
        if p.is_file() and (p.suffix=='.py' or p.name in ('LICENSE','LICENSE.txt','README.md','requirements.txt')):
            assert not p.is_symlink();files.append((p,'author_source/'+name))
    manifest={n:dict(sha256=sha(p),bytes=p.stat().st_size) for p,n in files};assert len(manifest)==len(files)
    out.mkdir(parents=True,exist_ok=False);archive=out/'full_run.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p,n in files:tar.add(p,arcname=n,recursive=False)
    assert all(sha(p)==manifest[n]['sha256'] for p,n in files),'Source changed during archive'
    (out/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt=dict(status='generation_complete_archived',created_utc=datetime.now(timezone.utc).isoformat(),remote_directory=str(out),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,member_manifest_sha256=sha(out/'member_sha256.json'),members=len(manifest),
        questions=TARGETS[a.name],conditions=len(tests),invalid=sum(r['status']!='ok' for r in tests),budget=budget,
        source_data_policy_verified=True,raw_known_actual_exact=True,known_calls=len(known),unknown_calls=len(unknown),
        unresolved_reservations=0,original_terminal_prefixes_preserved=True,original_run_start_preserved=True,
        owned_cleanup_verified=True,author_commit=config['author_commit'],author_worktree_clean=True,
        all_runtime_management_source_included=True,author_python_source_included=True,no_gold_scoring=True)
    (out/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
