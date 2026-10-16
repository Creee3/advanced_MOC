"""Read-only proof of real no-deadline baseline continuation, without gold."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from preflight_archive import read,rows,sha,active
from resume_worker import validate_authorization
HERE=Path(__file__).resolve().parent
def main():
    launches=read(HERE/'launches.json');initial=read(HERE/'pre_resume_review.json')
    report=dict(checked_utc=datetime.now(timezone.utc).isoformat(),runs={},all_six_gpu_verified=False)
    gpus=[]
    for name,launch in launches.items():
        root=HERE.parent.parent/name;validate_authorization(root);s=read(root/'status.json')
        assert s['phase']=='test' and s['arm']=='moc' and s['experiment_wall_deadline'] is None and not s['cleanup_complete']
        assert s['supervisor_pid']==launch['pid'] and active(s['supervisor_pid'])
        assert len(s['worker_pids'])==2 and all(active(pid) for pid in s['worker_pids'])
        assert not (root/'STOP').exists() and not s.get('error')
        previous=initial['runs'][name]
        assert read(root/'run_start.json')==previous['original_run_start']
        for file,m in previous['terminal_files'].items():assert hashlib.sha256((root/file).read_bytes()[:m['bytes']]).hexdigest()==m['sha256']
        assert sha(root/'group_utility_model.json')==previous['policy_sha256']
        services=[]
        for service in s['services']:
            assert active(service['pid']) and 'cuMemAlloc 0' in service['cuda_allocation_probe']
            assert service['warmup']['done'] and service['warmup']['eval_count']>0
            with urllib.request.urlopen(service['host']+'/api/ps',timeout=10) as response:loaded=json.load(response)['models']
            assert len(loaded)==1 and loaded[0]['digest']==service['model']['digest'] and loaded[0]['size_vram']>=loaded[0]['size']
            services.append(dict(kind=service['kind'],pid=service['pid'],port=service['port'],gpu=service['gpu'],loaded=loaded[0],cuda_probe_passed=True,warmup_passed=True))
            gpus.append(service['gpu']['uuid'])
        assert len(services)==2
        records=[r for p in (root/'test').glob('shard*/results.jsonl') for r in rows(p)]
        keys=[(r['question_id'],r['arm'],r['density']) for r in records];assert len(keys)==len(set(keys))
        moc=[r for r in records if r['arm']=='moc'];assert len(moc)>previous['moc']
        raw_age=[round(time.time()-p.stat().st_mtime,1) for p in (root/'test').glob('shard*/calls_raw.jsonl')]
        assert len(raw_age)==2 and max(raw_age)<120
        budget=read(root/'budget.json');config=read(root/'config.json')
        assert budget['actual_tokens']+budget['reserved_tokens']+budget['unknown_cost_upper_bound_tokens']<=config['max_new_model_tokens']
        assert budget['new_calls']<=config['max_new_calls']
        assert budget['unknown_cost_upper_bound_tokens']==(10192 if 'svamp' in name else 0)
        pids=[s['supervisor_pid'],*s['worker_pids'],*[v['pid'] for v in s['services']]]
        report['runs'][name]=dict(supervisor_pid=s['supervisor_pid'],worker_pids=s['worker_pids'],
            actual_cmdlines={str(pid):(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode() for pid in pids},
            services=services,experiment_wall_deadline=None,original_run_start_unchanged=True,
            original_terminal_prefixes_preserved=True,policy_source_data_verified=True,budget=budget,
            moc_committed=len(moc),moc_counts={str(d):sum(r['density']==d for r in moc) for d in (.3,.7)},
            adapted_committed=sum(r['arm']=='adapted' for r in records),duplicates=0,invalid=sum(r['status']!='ok' for r in records),raw_age_seconds=raw_age)
    assert len(gpus)==len(set(gpus))==6
    report['all_six_gpu_verified']=True;report['status']='running_verified'
    (HERE/'launch_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],gpus=6,runs={n:dict(pid=r['supervisor_pid'],moc=r['moc_committed'],budget=r['budget']) for n,r in report['runs'].items()})))
if __name__=='__main__':main()
