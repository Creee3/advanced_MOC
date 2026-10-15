"""Archive one completed, cleaned train40 run without changing experiment files."""
import hashlib
import json
from pathlib import Path
import socket
import tarfile
from datetime import datetime, timezone

BASE=Path('/home/liuxin/ICML2027/experiments')
NAME='moc_v21_mmlu_train40_test_20261015'
ROOT=BASE/NAME
OUT=BASE/'moc_remaining_datasets_plan_20261015'/'completed_archives'/NAME
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    if (OUT/'archive_receipt.json').exists():
        print(json.dumps(read(OUT/'archive_receipt.json')));return
    status=read(ROOT/'status.json');budget=read(ROOT/'budget.json')
    assert status['phase']=='complete' and status['cleanup_complete']
    assert read(ROOT/'cleanup.json')['cleanup_complete']
    assert budget['reserved_tokens']==budget['unknown_cost_upper_bound_tokens']==0
    for pid in [status['supervisor_pid'],*status.get('worker_pids',[]),*[s['pid'] for s in status['services']]]:
        p=Path('/proc')/str(pid)
        if p.exists():assert not (p/'cmdline').read_bytes() or (p/'stat').read_text().rsplit(') ',1)[1].split()[0] in ('Z','X')
    for service in status['services']:
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',service['port']))!=0
    lock=read(ROOT/'experiment_lock.json')
    assert all(sha(ROOT/n)==h for n,h in {**lock['source_hashes'],**lock['data_hashes']}.items())
    policy=read(ROOT/'policy_lock.json');assert sha(ROOT/'group_utility_model.json')==policy['model_sha256']
    tests=[x for p in (ROOT/'test').glob('shard*/results.jsonl') for x in rows(p)]
    keys=[(x['question_id'],x['arm'],x['density']) for x in tests]
    expected={(q['question_id'],a,d) for q in rows(ROOT/'data/test_inputs.jsonl') for a in ('adapted','moc') for d in (.3,.7)}
    assert len(keys)==len(set(keys))==1140 and set(keys)==expected
    raw=[x for stage in ('training','test') for p in (ROOT/stage).glob('shard*/*calls_raw.jsonl') for x in rows(p)]
    warm=rows(ROOT/'startup_calls_raw.jsonl');new=[x for x in raw if not x['cache_hit']]+warm
    tokens=lambda x:x['response']['prompt_eval_count']+x['response']['eval_count']
    assert sum(map(tokens,new))==budget['actual_tokens'] and len(new)==budget['new_calls']
    trace=[x for stage in ('training','test') for p in (ROOT/stage).glob('shard*/trace.jsonl') for x in rows(p)]
    started={x['reservation_id'] for x in trace if x['event_type']=='backend_request_started'}
    settled={x['reservation_id'] for x in trace if x['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown')}
    assert not started-settled
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        assert not p.is_symlink()
        if p.is_file():files.append(p)
    manifest={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    OUT.mkdir(parents=True,exist_ok=False)
    archive=OUT/'full_run.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    assert all(sha(ROOT/n)==v['sha256'] for n,v in manifest.items())
    (OUT/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    receipt=dict(status='generation_complete_archived',created_utc=datetime.now(timezone.utc).isoformat(),
        remote_directory=str(OUT),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=len(files),
        member_manifest_sha256=sha(OUT/'member_sha256.json'),budget=budget,conditions=len(tests),
        invalid=sum(x['status']!='ok' for x in tests),source_data_policy_verified=True,
        raw_cost_exact=True,unsettled_reservations=0,cleanup_verified=True,no_gold_scoring=True)
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
