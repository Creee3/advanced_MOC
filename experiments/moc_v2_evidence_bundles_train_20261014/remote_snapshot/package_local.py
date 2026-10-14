"""Freeze deployment artifacts; include train labels but no validation/test gold."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target=ROOT/'coding_base/moc_adaptive_sampling'
    shutil.copytree(REPO/'coding_base/moc_adaptive_sampling',target,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','tests','verify_local.py','audit_assets.py'))
    sources=list(ROOT.glob('*.py'))+list((ROOT/'coding_base').rglob('*.py'))
    fixed=[ROOT/'config.json',ROOT/'training_manifest.json',ROOT/'val_role_snapshot.json',ROOT/'protocol.md']
    data=[p for p in (ROOT/'data').glob('*.jsonl') if p.name!='dev_inputs.jsonl']
    allowed={'fixed_materials_LOCAL_ONLY.jsonl','train_gold_LOCAL_ONLY.jsonl','historical_calls_LOCAL_ONLY.jsonl',
             'val_inputs.jsonl','val_historical_calls_LOCAL_ONLY.jsonl'}
    assert {p.name for p in data}<=allowed
    lock={'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sources+fixed},
          'data_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in data},
          'gold_scope':'official train only','val_gold_included':False,'test_included':False,
          'v1_frozen_commit':'c0213541b6881b53d58ac5300a49df3ada456ac2'}
    (ROOT/'experiment_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    artifact=ROOT/'deployment.tar.gz'
    with tarfile.open(artifact,'w:gz') as archive:
        for p in sorted(set(sources+fixed+data+[ROOT/'experiment_lock.json'])):
            archive.add(p,arcname=p.relative_to(ROOT).as_posix())
    print(json.dumps({'files':len(sources+fixed+data)+1,'bytes':artifact.stat().st_size,'sha256':sha(artifact),
                      'train_materials':47,'val_conditions':120,'val_gold':False,'test':False}))

if __name__=='__main__':
    main()
