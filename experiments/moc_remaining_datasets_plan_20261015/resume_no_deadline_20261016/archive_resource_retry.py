"""Archive one diagnosed SVAMP GPU-admission rejection before a bounded retry."""
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import tarfile
from preflight_archive import read, rows, sha, active
from resume_worker import validate_authorization
HERE=Path(__file__).resolve().parent
NAME='moc_v21_svamp_train40_test_20261015'
R=HERE.parent.parent/NAME
def main():
    output=HERE/'resource_retry_review_svamp.json'
    assert not output.exists(),'Retry archive already exists; inspect, never repeat blindly'
    validate_authorization(R)
    s=read(R/'status.json');b=read(R/'budget.json')
    assert s['phase']=='failed' and s['error_type']=='AssertionError' and s['cleanup_complete']
    assert s['supervisor_pid']==3007461 and not active(s['supervisor_pid']) and not s['services']
    assert not (R/'STOP').exists()
    assert b==dict(new_calls=7444,actual_tokens=3908073,reserved_tokens=0,unknown_cost_upper_bound_tokens=10192)
    assert read(R/'cleanup.json')['cleanup_complete']
    for port in (11454,11455):
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
    trace=[e for phase in ('training','test') for p in (R/phase).glob('shard*/trace.jsonl') for e in rows(p)]
    starts={e['reservation_id'] for e in trace if e['event_type']=='backend_request_started'}
    settled={e['reservation_id'] for e in trace if e['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown')}
    assert not starts-settled
    reconciled=[e for e in trace if e['event_type']=='backend_request_reconciled_unknown']
    assert len(reconciled)==1 and reconciled[0]['unknown_cost_upper_bound_tokens']==10192
    previous=read(HERE/'pre_resume_review.json')['runs'][NAME]
    for n,m in previous['terminal_files'].items():assert sha(R/n)==m['sha256']
    attempt=HERE/'attempts'/('resource_admission_retry_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));attempt.mkdir(parents=True,exist_ok=False)
    files=[p for p in sorted(R.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    assert not any(p.is_symlink() for p in files)
    manifest={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    archive=attempt/'svamp_resource_retry_fullraw.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(R).as_posix(),recursive=False)
    assert all(sha(R/n)==m['sha256'] for n,m in manifest.items())
    (attempt/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    review=dict(previous,budget=b,unresolved={})
    report=dict(status='passed',attempt=str(attempt),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=sha(attempt/'member_sha256.json'),members=len(files),runs={NAME:review},
        cause='Second idle GPU admission check rejected utilization; failed before any service/model call',
        diagnosed_pid=3007461,root_cause_retry_number=1,unknown_already_carried_once=True)
    output.write_text(json.dumps(report,indent=2)+'\n');(attempt/'review.json').write_bytes(output.read_bytes())
    print(json.dumps({k:report[k] for k in ('status','attempt','archive_sha256','archive_bytes','members')}))
if __name__=='__main__':main()
