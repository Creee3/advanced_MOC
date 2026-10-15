"""Deploy only frozen allowlisted files; test gold never uploaded; no GPU launch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parent
PY='/home/liuxin/.conda/envs/moc/bin/python'
BASE='/home/liuxin/ICML2027/experiments'
NAMES=['moc_v21_mmlu_train40_test_20261015','moc_v21_svamp_train40_test_20261015']

def remote(code,timeout=60):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','DevContainer',PY,'-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
    print(r.stdout.strip());return r.stdout
def main():
    for name in NAMES:
        out=ROOT.parent/name;manifest=json.loads((out/'upload_manifest.json').read_text());dest=BASE+'/'+name
        assert not any('test_gold' in n or 'val_gold' in n for n in manifest)
        state=remote(f'''from pathlib import Path
import json,hashlib,shutil,datetime
p=Path({dest!r})
if p.exists():
    assert not (p/'run_start.json').exists(),'existing model run: never overwrite'
    report=json.loads((p/'preflight_report.json').read_text()) if (p/'preflight_report.json').exists() else {{}}
    if report.get('status')=='passed':
        assert hashlib.sha256((p/'experiment_lock.json').read_bytes()).hexdigest()=={hashlib.sha256((out/'experiment_lock.json').read_bytes()).hexdigest()!r}
        print('VERIFIED_PREFLIGHT_DONE')
    else:
        old=p.with_name(p.name+'_preflight_attempt_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'))
        shutil.copytree(p,old)
        print('ARCHIVED_NO_MODEL_PREFLIGHT',old)
else:p.mkdir()
''')
        if 'VERIFIED_PREFLIGHT_DONE' in state:
            for n in ('preflight_report.json','upload_receipt.json'):
                subprocess.run(['scp',f'DevContainer:{dest}/{n}',str(out/n)],check=True)
            continue
        subprocess.run(['scp',str(out/'upload_LOCAL_ONLY.tar.gz'),f'DevContainer:{dest}/input_bundle.tar.gz'],check=True)
        code=f'''import hashlib,json,tarfile,subprocess,os
from pathlib import Path
root=Path({dest!r});manifest=json.loads({json.dumps(manifest)!r})
archive=root/'input_bundle.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={hashlib.sha256((out/'upload_LOCAL_ONLY.tar.gz').read_bytes()).hexdigest()!r}
with tarfile.open(archive) as tar:
    assert {{x.name for x in tar.getmembers()}}==set(manifest)
    for x in tar.getmembers():
        assert x.isfile() and (root/x.name).resolve().is_relative_to(root.resolve())
    tar.extractall(root,filter='data')
for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
(root/'upload_receipt.json').write_text(json.dumps(dict(files=len(manifest),hashes_verified=True,test_gold_present=(root/'data/test_gold_LOCAL_ONLY.jsonl').exists()),indent=2))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
p=subprocess.run([{PY!r},str(root/'preflight.py')],cwd=root,env=env,capture_output=True,text=True,timeout=180)
(root/'preflight_stdout.log').write_text(p.stdout+'\\n'+p.stderr)
assert p.returncode==0,p.stdout[-2000:]+p.stderr[-4000:]
print(root.name,json.loads((root/'preflight_report.json').read_text()))
'''
        remote(code,timeout=200)
        for n in ('preflight_report.json','upload_receipt.json'):
            subprocess.run(['scp',f'DevContainer:{dest}/{n}',str(out/n)],check=True)
    tests=ROOT/'regression_tests';shutil.copytree(ROOT.parents[1]/'coding_base/moc_adaptive_sampling/tests',tests,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    files=['dispatch.py','test_dispatch.py','task_plan.md','findings.md','progress.md','prepare_train40.py','deploy.py']
    files += [str(p.relative_to(ROOT)).replace('\\','/') for p in tests.glob('*.py')]
    with tarfile.open(ROOT/'management_LOCAL_ONLY.tar.gz','w:gz') as tar:
        for n in files:tar.add(ROOT/n,arcname=n,recursive=False)
    dest=BASE+'/'+ROOT.name
    remote(f"from pathlib import Path\np=Path({dest!r})\nassert not p.exists()\np.mkdir()\n")
    subprocess.run(['scp',str(ROOT/'management_LOCAL_ONLY.tar.gz'),f'DevContainer:{dest}/management.tar.gz'],check=True)
    remote(f'''import tarfile,subprocess,os,json
from pathlib import Path
p=Path({dest!r})
with tarfile.open(p/'management.tar.gz') as t:t.extractall(p,filter='data')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH={BASE+'/'+NAMES[0]+'/coding_base'!r},OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
reports=[]
for args in [['-m','unittest','test_dispatch','-v'],['-m','unittest','discover','-s','regression_tests','-v']]:
    r=subprocess.run([{PY!r},*args],cwd=p,env=env,capture_output=True,text=True,timeout=120)
    reports.append(dict(args=args,returncode=r.returncode,output=r.stdout+'\\n'+r.stderr))
(p/'cpu_regression_report.json').write_text(json.dumps(reports,indent=2))
assert all(x['returncode']==0 for x in reports),reports
print('Admission and original method CPU regressions passed', [x['output'][-120:] for x in reports])
''',timeout=260)
    subprocess.run(['scp',f'DevContainer:{dest}/cpu_regression_report.json',str(ROOT/'cpu_regression_report.json')],check=True)
if __name__=='__main__':main()
