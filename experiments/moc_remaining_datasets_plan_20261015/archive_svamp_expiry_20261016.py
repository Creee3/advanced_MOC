"""One-time read-only expired-run preservation with independent unknown-cost audit.

The run's original budget/status/trace are preserved byte for byte. A separate
reconciliation moves dead-worker unresolved reservations to conservative unknown
cost in the audit only; it never labels missing usage as actual or charges twice.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import tarfile
import time

BASE=Path('/home/liuxin/ICML2027/experiments')
NAME='moc_v21_svamp_train40_test_20261015'
RUN=BASE/NAME
PLAN=BASE/'moc_remaining_datasets_plan_20261015'
OUT=PLAN/'expired_archives'/NAME

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def live(pid):
    try:
        p=Path('/proc')/str(pid)
        return bool((p/'cmdline').read_bytes()) and (p/'stat').read_text().rsplit(') ',1)[1].split()[0] not in ('Z','X')
    except OSError:return False

def main():
    receipt=OUT/'closure_report.json'
    if receipt.exists():
        r=read(receipt);print(json.dumps({k:r[k] for k in ('status','remote_directory','archive_sha256','archive_bytes','members')}));return
    assert not OUT.exists(),'Incomplete prior archive attempt requires inspection'
    status=read(RUN/'status.json');start=read(RUN/'run_start.json');budget=read(RUN/'budget.json')
    assert start['deadline_unix']==1792111216.6382632 and time.time()>start['deadline_unix']
    assert status['phase']=='failed' and status['error']=='STOP or original absolute deadline'
    assert not (RUN/'STOP').exists()
    cleanup=read(RUN/'cleanup.json')
    assert status['cleanup_complete'] and cleanup['cleanup_complete']
    assert not cleanup['owned_processes_remaining'] and not cleanup['owned_children_remaining'] and not cleanup['remaining_ports']
    assert not any(live(p) for p in [status['supervisor_pid'],*status['worker_pids'],*[s['pid'] for s in status['services']]])
    for port in (11454,11455):
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
    lock=read(RUN/'experiment_lock.json');policy=read(RUN/'policy_lock.json')
    assert all(sha(RUN/n)==h for n,h in {**lock['source_hashes'],**lock['data_hashes']}.items())
    assert sha(RUN/'group_utility_model.json')==policy['model_sha256']=='728a99c19645a2ff230132ddd220ec243a7ea644bc11776e261ae193e4d4633e'
    raw_by_stage={stage:[e for p in (RUN/stage).glob('shard*/*calls_raw.jsonl') for e in rows(p)] for stage in ('training','test')}
    raw=sum(raw_by_stage.values(),[]);warm=rows(RUN/'startup_calls_raw.jsonl')
    actual=[e for e in raw if not e['cache_hit']]+warm
    tokens=lambda e:e['response']['prompt_eval_count']+e['response']['eval_count']
    assert sum(map(tokens,actual))==budget['actual_tokens']
    traces=[e for stage in ('training','test') for p in (RUN/stage).glob('shard*/trace.jsonl') for e in rows(p)]
    starts={e['reservation_id']:e for e in traces if e['event_type']=='backend_request_started'}
    settled={e['reservation_id']:e for e in traces if e['event_type']=='backend_request_settled'}
    previous_unknown={e['reservation_id']:e for e in traces if e['event_type']=='backend_request_reconciled_unknown'}
    assert not set(settled)&set(previous_unknown)
    unresolved={k:e for k,e in starts.items() if k not in settled and k not in previous_unknown}
    upper=sum(e['reserved_tokens'] for e in unresolved.values())
    assert upper==budget['reserved_tokens']==10192 and len(unresolved)==1
    assert len(actual)+len(unresolved)==budget['new_calls']==7444
    assert sum(e['actual_tokens'] for e in settled.values())+sum(map(tokens,warm))==budget['actual_tokens']
    reconciliation=[dict(event_type='backend_request_reconciled_unknown',reservation_id=k,
        reserved_tokens=e['reserved_tokens'],unknown_cost_upper_bound_tokens=e['reserved_tokens'],
        question_id=e['question_id'],arm=e['arm'],density=e['density'],
        reason='Supervisor enforced original deadline and killed owned worker/service; no response usage exists',
        source='independent closure audit; original trace and budget unchanged') for k,e in unresolved.items()]
    pending=set(starts)-set(settled)-set(previous_unknown)-{e['reservation_id'] for e in reconciliation}
    assert not pending
    effective=dict(actual_tokens=budget['actual_tokens'],new_calls=budget['new_calls'],pending_tokens=0,
        unknown_cost_upper_bound_tokens=budget.get('unknown_cost_upper_bound_tokens',0)+upper)
    effective['cap_accounted_tokens']=effective['actual_tokens']+effective['pending_tokens']+effective['unknown_cost_upper_bound_tokens']
    config=read(RUN/'config.json')
    assert effective['cap_accounted_tokens']<=config['max_new_model_tokens'] and effective['new_calls']<=config['max_new_calls']
    tests=[e for p in (RUN/'test').glob('shard*/results.jsonl') for e in rows(p)]
    keys=[(e['question_id'],e['arm'],e['density']) for e in tests]
    assert len(keys)==len(set(keys))==1028
    counts={}
    for e in tests:
        k=f"{e['arm']}|{e['density']}|{e['status']}";counts[k]=counts.get(k,0)+1
    assert counts=={'adapted|0.3|ok':300,'adapted|0.7|ok':300,'moc|0.3|ok':215,'moc|0.7|ok':213}
    files=[]
    for p in sorted(RUN.rglob('*')):
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        assert not p.is_symlink()
        if p.is_file():files.append((p,NAME+'/'+p.relative_to(RUN).as_posix()))
    for pattern in ('*.py','*.json','*.md','monitor_prompt_utf8.txt'):
        for p in sorted(PLAN.glob(pattern)):
            if not p.name.startswith('healthcheck_'):files.append((p,'coordinator/'+p.name))
    assert len({n for p,n in files})==len(files)
    manifest={n:dict(sha256=sha(p),bytes=p.stat().st_size) for p,n in files}
    OUT.mkdir(parents=True,exist_ok=False)
    archive=OUT/'expired_run_fullraw.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p,n in files:tar.add(p,arcname=n,recursive=False)
    assert all(sha(p)==manifest[n]['sha256'] for p,n in files),'Run changed during archive'
    (OUT/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='expired_incomplete_preserved',created_utc=datetime.now(timezone.utc).isoformat(),
        remote_directory=str(OUT),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        member_manifest_sha256=sha(OUT/'member_sha256.json'),members=len(files),run_start=start,
        original_budget=budget,effective_cost_audit=effective,independent_reconciliations=reconciliation,
        original_trace_started=len(starts),original_trace_settled=len(settled),audit_pending_count=len(pending),
        known_raw_calls=len(actual),known_raw_tokens=sum(map(tokens,actual)),warmup_calls=len(warm),warmup_tokens=sum(map(tokens,warm)),
        stage_costs={stage:dict(actual_tokens=sum(tokens(e) for e in rs if not e['cache_hit']),
            new_calls=sum(not e['cache_hit'] for e in rs),logical_tokens=sum(map(tokens,rs)),cache_hits=sum(e['cache_hit'] for e in rs)) for stage,rs in raw_by_stage.items()},
        test_coverage=counts,missing_original_moc=172,duplicates=0,invalid=0,
        cleanup=cleanup,locked_source_data_policy_verified=True,no_launch_or_restart=True,
        no_original_budget_status_trace_or_deadline_mutation=True,no_test_gold_scoring=True,
        cause='Original eight-hour deadline enforced by supervisor; worker0 also logged BudgetExceeded',
        note='Incomplete original-MOC comparison: no accuracy/comparison scoring. Preserve all adapted and partial baseline terminals; no automatic extension or recovery. Unknown is a conservative bound, not actual; raw reservation is replaced only in this independent audit, never added twice.')
    receipt.write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','remote_directory','archive_sha256','archive_bytes','members','effective_cost_audit')}))

if __name__=='__main__':main()
