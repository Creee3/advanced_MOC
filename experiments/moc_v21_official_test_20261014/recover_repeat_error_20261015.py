"""One-shot state-only recovery after the first merger repeat-limit failure.

All executable/model/input locks stay unchanged. Do not rerun the known failed
MOC condition or fabricate missing API token usage. Keep the existing deadline.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'attempts/preserved_repeat_error_20261015T040854_718704Z'
QID='67969c897d8d2ac6ef5dea3b67076e88065fb224d0ec0a997311d893ccb6da83'
EXPECTED_PENDING={'merger-1792036662713673806','merger-1792036678712704403','merger-1792036710356809412'}

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(p)
def stats(events):
    usage=lambda e:e['response']['prompt_eval_count']+e['response']['eval_count']
    return dict(logical_calls=len(events),new_calls=sum(not e['cache_hit'] for e in events),
        cache_hits=sum(e['cache_hit'] for e in events),logical_model_tokens=sum(map(usage,events)),
        new_model_tokens=sum(usage(e) for e in events if not e['cache_hit']))

def main():
    assert not (ROOT/'STOP').exists()
    marker=ROOT/'repeat_error_recovery_20261015.json'
    assert not marker.exists(),'one-shot recovery already applied'
    guard=(ROOT/'supervisor.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    s=read(ROOT/'status.json');budget=read(ROOT/'budget.json');start=read(ROOT/'run_start.json')
    assert s['phase']=='failed' and s['cleanup_complete'] and s['completed_test_conditions']==737
    assert s['updated_utc']=='2026-10-15T03:58:35.017996+00:00'
    assert budget==dict(new_calls=6113,actual_tokens=4148592,reserved_tokens=49152,
        unknown_cost_upper_bound_tokens=10192,unknown_call_count=1)
    assert start['deadline_unix']==1792050474.4160016 and time.time()<start['deadline_unix']
    for pid in [s['supervisor_pid'],*s['worker_pids'],*[x['pid'] for x in s['services']]]:
        assert not Path('/proc',str(pid)).exists()
    for port in (11448,11450):
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
    manifest=read(ARCHIVE/'manifest.json')
    assert sha(ARCHIVE/'before_recovery.tar.gz')==manifest['archive_sha256']=='fcf6e51d3a88b765aeb2f14434d16ba5e586031d46c9772024eeefcc05e2cd3a'
    for name,digest in manifest['files'].items():assert sha(ROOT/name)==digest,('post-archive drift',name)
    lock=read(ROOT/'experiment_lock.json');policy=read(ROOT/'policy_lock.json')
    assert sha(ROOT/'experiment_lock.json')==policy['experiment_lock_sha256']
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():assert sha(ROOT/name)==digest,name
    assert read(ROOT/'preflight_report.json')['status']=='passed'
    all_raw=[];pending={};traces_by_shard={}
    for folder in (ROOT/'test').glob('shard*'):
        traces=rows(folder/'trace.jsonl');traces_by_shard[folder.name]=traces
        settled={e['reservation_id'] for e in traces if e.get('event_type')=='backend_request_settled'}
        for e in traces:
            if e.get('event_type')=='backend_request_started' and e['reservation_id'] not in settled:
                pending[e['reservation_id']]=dict({k:v for k,v in e.items() if k!='request'},shard=folder.name,
                    request_sha256=hashlib.sha256(json.dumps(e['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest())
        for name in ('calls_raw.jsonl','merger_calls_raw.jsonl'):all_raw.extend(rows(folder/name))
    assert set(pending)==EXPECTED_PENDING and sum(e['reserved_tokens'] for e in pending.values())==49152
    known=[e for e in all_raw if not e['cache_hit']]+rows(ROOT/'startup_calls_raw.jsonl')
    assert len(known)==6109 and sum(e['response']['prompt_eval_count']+e['response']['eval_count'] for e in known)==4148592
    errors=[e for e in traces_by_shard['shard0'] if e.get('event_type')=='merger_backend_error'
        and (e.get('question_id'),e.get('arm'),e.get('density'))==(QID,'moc',.7)]
    assert len(errors)==2 and all('token repeat limit reached' in e['error'] for e in errors)
    out=ROOT/'test/shard0'
    committed=[r for f in (ROOT/'test').glob('shard*/results.jsonl') for r in rows(f)]
    assert len(committed)==737 and len({(r['question_id'],r['arm'],r['density']) for r in committed})==737
    assert not any((r['question_id'],r['arm'],r['density'])==(QID,'moc',.7) for r in committed)
    selected=lambda name:[e for e in rows(out/name) if (e.get('question_id'),e.get('arm'),e.get('density'))==(QID,'moc',.7)]
    main_events=selected('calls_raw.jsonl');merger_events=selected('merger_calls_raw.jsonl')
    assert len(main_events)==8
    a,b=stats(main_events),stats(merger_events)
    receipt=dict(stage='audited_before_state_changes',root_cause='merger_token_repeat_limit_500',recovery_attempt=1,
        checked_utc=datetime.now(timezone.utc).isoformat(),archive=str(ARCHIVE.relative_to(ROOT)),
        archive_sha256=manifest['archive_sha256'],original_conditions=737,old_budget=budget.copy(),
        deadline_unix=start['deadline_unix'],deadline_changed=False,source_hashes_changed=False,
        pending_requests=pending,known_actual_tokens=4148592,known_calls=6109,
        previous_unknown_cost_upper_bound=10192,new_unknown_cost_upper_bound=49152,
        total_unknown_cost_upper_bound=59344,total_unknown_call_count=4)
    save(marker,receipt)
    item=dict(question_id=QID,arm='moc',density=.7,status='invalid',
        reason='original merger failed: token repeat limit reached (500); fallback excluded; recovered from raw without regeneration',
        **{k:a[k]+b[k] for k in a},main_calls=a['logical_calls'],merger_calls=b['logical_calls'],
        main_model_tokens=a['logical_model_tokens'],merger_model_tokens=b['logical_model_tokens'],
        unknown_merger_calls=2,unknown_model_tokens_upper_bound=32768,
        recovery_evidence=dict(archive=str(ARCHIVE.relative_to(ROOT)),merger_error_event_indices=[e['event_index'] for e in errors]))
    with (out/'results.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(item)+'\n')
    for shard,traces in traces_by_shard.items():
        last=max(e['event_index'] for e in traces)
        with (ROOT/'test'/shard/'trace.jsonl').open('a',encoding='utf-8') as stream:
            for reservation,e in pending.items():
                if e['shard']!=shard:continue
                last+=1
                event=dict(event_type='backend_request_reconciled_unknown',event_index=last,time_unix=time.time(),
                    reservation_id=reservation,upper_bound_tokens=e['reserved_tokens'],actual_usage_unknown=True,
                    question_id=e['question_id'],arm=e['arm'],density=e['density'],
                    source_start_event_index=e['event_index'],audit=marker.name)
                stream.write(json.dumps(event)+'\n')
    budget.update(reserved_tokens=0,unknown_cost_upper_bound_tokens=59344,unknown_call_count=4)
    save(ROOT/'budget.json',budget)
    assert sha(ROOT/'run_start.json')==manifest['files']['run_start.json']
    assert budget['new_calls']<30000 and budget['actual_tokens']+59344<12000000
    receipt.update(stage='verified_ready_to_resume',new_budget=budget,recovered_invalid_conditions=1)
    save(marker,receipt)
    fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
    config=read(ROOT/'config.json')
    launched=subprocess.check_output([config['python_executable'],str(ROOT/'launch.py'),'--resume'],cwd=ROOT,text=True)
    receipt.update(stage='launched',launcher=json.loads(launched));save(marker,receipt)
    print(json.dumps({k:receipt[k] for k in ('stage','launcher','new_budget','recovered_invalid_conditions','deadline_unix')},ensure_ascii=False))

if __name__=='__main__':main()
