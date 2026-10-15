"""Resolve isolated test-fixture paths and rerun original CPU regressions."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
REMOTE='/home/liuxin/ICML2027/experiments/'+ROOT.name
source=ROOT.parents[2]/'research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014/upstream/ladies/pytorch_ladies.py'
assert source.exists(),source
code=f'''from pathlib import Path
import json,subprocess,os,hashlib
root=Path({REMOTE!r})
report=root/'cpu_regression_report.json'
if not (root/'cpu_regression_report_attempt1.json').exists():report.rename(root/'cpu_regression_report_attempt1.json')
p=root/'regression_tests/source_harness.py'
s=p.read_text().replace('AUTHOR = WORKSPACE / "coding_base/MOC-official"','AUTHOR = Path("/home/liuxin/ICML2027/experiments/moc_stateful_sampling_val_20261014/author_source")')
s=s.replace('ASSETS = WORKSPACE / "research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014"','ASSETS = Path(__file__).resolve().parent.parent / "regression_assets"')
p.write_text(s)
a=root/'regression_assets/upstream/ladies/pytorch_ladies.py';a.parent.mkdir(parents=True,exist_ok=True)
a.write_text({source.read_text(encoding='utf-8')!r})
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH='/home/liuxin/ICML2027/experiments/moc_v21_mmlu_train40_test_20261015/coding_base',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
reports=[]
for args in [['-m','unittest','test_dispatch','-v'],['-m','unittest','discover','-s','regression_tests','-v']]:
 r=subprocess.run(['/home/liuxin/.conda/envs/moc/bin/python',*args],cwd=root,env=env,capture_output=True,text=True,timeout=120)
 reports.append(dict(args=args,returncode=r.returncode,output=r.stdout+'\\n'+r.stderr))
(root/'cpu_regression_report.json').write_text(json.dumps(reports,indent=2))
assert all(r['returncode']==0 for r in reports),[r['output'][-3500:] for r in reports]
print([r['output'][-150:] for r in reports])
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','DevContainer','/home/liuxin/.conda/envs/moc/bin/python','-'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=240)
print(r.stdout);print(r.stderr);assert r.returncode==0
subprocess.run(['scp',f'DevContainer:{REMOTE}/cpu_regression_report.json',str(ROOT/'cpu_regression_report.json')],check=True)
