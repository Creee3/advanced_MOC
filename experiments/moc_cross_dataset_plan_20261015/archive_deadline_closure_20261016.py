"""One-time expired-window preservation; read-only runs, no launch or gold scoring."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import tarfile
import time

BASE=Path('/home/liuxin/ICML2027/experiments')
ROOT=BASE/'moc_cross_dataset_plan_20261015'
NAMES=['moc_v21_gsm8k_fewshot_20261015','moc_v21_mmlu_pro_fewshot_20261015']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def live(pid):
    try:return bool((Path('/proc')/str(pid)/'cmdline').read_bytes())
    except OSError:return False
def collect():
    summaries={}
    for name in NAMES:
        root=BASE/name;status=read(root/'status.json');start=read(root/'run_start.json');budget=read(root/'budget.json')
        assert time.time()>start['deadline_unix']
        assert read(root/'cleanup.json')['cleanup_complete'] and status['cleanup_complete']
        pids=[status.get('supervisor_pid'),*status.get('worker_pids',[]),*[s['pid'] for s in status.get('services',[])]]
        assert not any(live(pid) for pid in pids if pid)
        assert budget['reserved_tokens']==0 and budget.get('unknown_cost_upper_bound_tokens',0)==0
        raw=[e for stage in ['training','val','test'] for p in (root/stage).glob('shard*/*calls_raw.jsonl') for e in rows(p)]
        actual=[e for e in raw if e['cache_hit'] is False]
        warm=rows(root/'startup_calls_raw.jsonl')
        known=sum(e['response']['prompt_eval_count']+e['response']['eval_count'] for e in actual+warm)
        assert known==budget['actual_tokens'],(name,known,budget)
        assert len(actual)+len(warm)==budget['new_calls']
        trace=[e for stage in ['training','val','test'] for p in (root/stage).glob('shard*/trace.jsonl') for e in rows(p)]
        starts={e['reservation_id']:e for e in trace if e['event_type']=='backend_request_started'}
        settled={e['reservation_id'] for e in trace if e['event_type'] in ['backend_request_settled','backend_request_reconciled_unknown']}
        assert not set(starts)-settled,(name,'unsettled trace requests')
        lock=read(root/'experiment_lock.json')
        assert all(sha(root/n)==h for n,h in {**lock['source_hashes'],**lock['data_hashes']}.items())
        tests=[e for p in (root/'test').glob('shard*/results.jsonl') for e in rows(p)]
        keys=[(e['question_id'],e['arm'],e['density']) for e in tests];assert len(keys)==len(set(keys))
        counts={}
        for e in tests:
            key=f"{e['arm']}|{e['density']}|{e['status']}";counts[key]=counts.get(key,0)+1
        logs={p.name:p.read_text(errors='replace').splitlines()[-20:] for p in root.glob('test_moc_worker*.log')}
        summaries[name]=dict(run_start=start,budget=budget,known_raw_tokens=known,new_raw_calls=len(actual),warmup_calls=len(warm),
            raw_cache_hits=sum(e['cache_hit'] for e in raw),trace_reservations=len(starts),unsettled_reservations=0,
            test_coverage=counts,duplicates=0,all_committed_valid=all(e['status']=='ok' for e in tests),
            cleanup_complete=True,locked_hashes_match=True,worker_error_tails=logs)
    for port in [11448,11449,11450]:
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,port
    assert not live(read(ROOT/'coordinator_status.json').get('pid'))
    return summaries
def main():
    output=ROOT/'expiry_closure_20261016.json'
    if output.exists():
        print(json.dumps(dict(status='already_archived',record=read(output))));return
    summaries=collect()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    attempt=ROOT/'attempts'/('deadline_closure_'+stamp);attempt.mkdir(parents=True,exist_ok=False)
    files=[]
    for name in NAMES:
        for p in sorted((BASE/name).rglob('*')):
            if '__pycache__' in p.parts or p.suffix=='.pyc':continue
            assert not p.is_symlink(),str(p)
            if p.is_file():files.append((p,name+'/'+p.relative_to(BASE/name).as_posix()))
    for p in sorted(ROOT.iterdir()):
        if p.is_file():files.append((p,'coordinator/'+p.name))
    manifest={n:dict(sha256=sha(p),bytes=p.stat().st_size) for p,n in files}
    archive=attempt/'expired_runs_fullraw.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p,n in files:tar.add(p,arcname=n,recursive=False)
    assert all(sha(p)==manifest[n]['sha256'] for p,n in files),'files changed during snapshot'
    (attempt/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='expired_incomplete_preserved',cause='original eight-hour deadline; worker BudgetExceeded then supervisor nonzero; no infrastructure recovery',
        created_utc=datetime.now(timezone.utc).isoformat(),attempt=str(attempt),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        member_manifest_sha256=sha(attempt/'member_sha256.json'),members=len(manifest),runs=summaries,
        no_budget_or_deadline_mutation=True,no_launch_or_restart=True,no_test_gold_scoring=True,new_cohort_unmodified=True,
        note='GSM originalMOC incomplete; Pro originalMOC never launched. Existing adapted results preserved. Resume requires fresh explicit human authorization, never infer an extension.')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (attempt/'closure_report.json').write_text(output.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],attempt=str(attempt),archive_sha256=report['archive_sha256'],archive_bytes=report['archive_bytes'],members=report['members'])))
if __name__=='__main__':main()
