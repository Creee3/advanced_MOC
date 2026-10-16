"""Read-only baseline recovery review and complete pre-resume archive."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import tarfile
import time
from resume_worker import validate_authorization, patched_source

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
EXPECTED={'moc_v21_gsm8k_fewshot_20261015':(600,504,7736331,11386,0),
          'moc_v21_mmlu_pro_fewshot_20261015':(560,0,4127233,5134,0),
          'moc_v21_svamp_train40_test_20261015':(600,428,3908073,7444,10192)}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def active(pid):
    try:
        p=Path('/proc')/str(pid)
        return bool((p/'cmdline').read_bytes()) and (p/'stat').read_text().rsplit(') ',1)[1].split()[0] not in ('Z','X')
    except OSError:return False

def main():
    receipt=HERE/'pre_resume_review.json'
    if receipt.exists():
        r=read(receipt);print(json.dumps({k:r[k] for k in ('status','attempt','archive_sha256','archive_bytes','members')}));return
    assert subprocess.check_output(['systemd-detect-virt'],text=True).strip() in ('lxc','lxd')
    summaries={};files=[]
    for name,(adapted,moc,tokens,calls,pending) in EXPECTED.items():
        root=BASE/name;validate_authorization(root)
        assert not (root/'STOP').exists() and not (HERE.parent/'STOP').exists()
        for mod in ('common','merger_backend'):compile(patched_source(mod,(root/(mod+'.py')).read_text()),str(root/(mod+'.py')),'exec')
        s=read(root/'status.json');budget=read(root/'budget.json');config=read(root/'config.json');policy=read(root/'policy_lock.json')
        assert sha(root/'group_utility_model.json')==policy['model_sha256']
        assert s['phase'] in ('failed','adaptive_complete') and s['cleanup_complete'] and read(root/'cleanup.json')['cleanup_complete']
        assert not any(active(pid) for pid in [s.get('supervisor_pid'),*s.get('worker_pids',[]),*[x['pid'] for x in s.get('services',[])]])
        for port in [*config['main_ports'],config['merger_port']]:
            with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
        assert (budget['actual_tokens'],budget['new_calls'],budget['reserved_tokens'],budget.get('unknown_cost_upper_bound_tokens',0))==(tokens,calls,pending,0)
        assert tokens+pending<config['max_new_model_tokens'] and calls<config['max_new_calls']
        test=[r for p in (root/'test').glob('shard*/results.jsonl') for r in rows(p)]
        keys=[(r['question_id'],r['arm'],r['density']) for r in test]
        assert len(keys)==len(set(keys))==adapted+moc
        assert sum(r['arm']=='adapted' for r in test)==adapted and sum(r['arm']=='moc' for r in test)==moc
        raw=[e for phase in ('training','val','test') for p in (root/phase).glob('shard*/*calls_raw.jsonl') for e in rows(p)]
        warm=rows(root/'startup_calls_raw.jsonl');known=[e for e in raw if not e['cache_hit']]+warm
        assert sum(e['response']['prompt_eval_count']+e['response']['eval_count'] for e in known)==tokens
        trace=[e for phase in ('training','val','test') for p in (root/phase).glob('shard*/trace.jsonl') for e in rows(p)]
        starts={e['reservation_id']:e for e in trace if e['event_type']=='backend_request_started'}
        settled={e['reservation_id'] for e in trace if e['event_type'] in ('backend_request_settled','backend_request_reconciled_unknown')}
        unresolved={k:v for k,v in starts.items() if k not in settled}
        assert sum(e['reserved_tokens'] for e in unresolved.values())==pending
        assert len(known)+len(unresolved)==calls
        summaries[name]=dict(budget=budget,original_run_start=read(root/'run_start.json'),
            adapted=adapted,moc=moc,unresolved=unresolved,policy_sha256=policy['model_sha256'],
            terminal_files={p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in (root/'test').glob('shard*/results.jsonl')},
            cleanup_verified=True,source_data_policy_verified=True)
        for p in sorted(root.rglob('*')):
            if '__pycache__' in p.parts or p.suffix=='.pyc':continue
            assert not p.is_symlink()
            if p.is_file():files.append((p,name+'/'+p.relative_to(root).as_posix()))
    attempt=HERE/'attempts'/('pre_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'));attempt.mkdir(parents=True,exist_ok=False)
    manifest={n:dict(sha256=sha(p),bytes=p.stat().st_size) for p,n in files}
    archive=attempt/'all_three_pre_resume.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p,n in files:tar.add(p,arcname=n,recursive=False)
    assert all(sha(p)==manifest[n]['sha256'] for p,n in files)
    (attempt/'member_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    review=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),attempt=str(attempt),
        archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=len(manifest),
        manifest_sha256=sha(attempt/'member_sha256.json'),runs=summaries,
        authorization_sha256=sha(HERE/'authorization.json'),no_model_calls=True,no_run_mutation=True)
    receipt.write_text(json.dumps(review,ensure_ascii=True,indent=2)+'\n')
    (attempt/'pre_resume_review.json').write_bytes(receipt.read_bytes())
    print(json.dumps({k:review[k] for k in ('status','attempt','archive_sha256','archive_bytes','members')}))
if __name__=='__main__':main()
