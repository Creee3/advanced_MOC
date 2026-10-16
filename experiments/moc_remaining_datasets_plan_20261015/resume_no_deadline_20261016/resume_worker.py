"""Explicit human-authorized wall-deadline override; frozen files stay unchanged.

Only two exact conditional clauses are removed at module load time. STOP,
reservation caps, all model requests, parsers and caching remain original.
This loads trusted experiment source, never generated code into the parent.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

HERE=Path(__file__).resolve().parent
PATCHES={
    'common':("if (ROOT/'STOP').exists() or time.time()>start['deadline_unix']:","if (ROOT/'STOP').exists():"),
    'merger_backend':("if (ROOT/'STOP').exists() or time.time() > json.loads((ROOT/'run_start.json').read_text())['deadline_unix']:","if (ROOT/'STOP').exists():"),
}

def patched_source(name,text):
    old,new=PATCHES[name]
    assert text.count(old)==1,(name,'unexpected frozen wall guard')
    return text.replace(old,new,1)

def validate_authorization(root):
    auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
    assert auth['experiment_wall_deadline'] is None and root.name in auth['runs']
    assert auth['resume_only_uncommitted_test_moc'] and auth['retain_cumulative_call_and_token_caps']
    lock=json.loads((root/'experiment_lock.json').read_text())
    for name,h in {**lock['source_hashes'],**lock['data_hashes']}.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    return auth

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args,rest=p.parse_known_args()
    root=Path(args.root).resolve();validate_authorization(root)
    assert '--phase' in rest and rest[rest.index('--phase')+1]=='test'
    assert '--arm' in rest and rest[rest.index('--arm')+1]=='moc'
    sys.path.insert(0,str(root))
    for name in ('common','merger_backend'):
        path=root/(name+'.py');source=patched_source(name,path.read_text(encoding='utf-8'))
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        exec(compile(source,str(path),'exec'),module.__dict__)
    sys.argv=[str(root/'runner.py'),*rest]
    runpy.run_path(str(root/'runner.py'),run_name='__main__')

if __name__=='__main__':main()
