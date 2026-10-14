"""Freeze a val-only deployment, no training labels or evaluation gold."""
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
    if (ROOT/'status.json').exists() or (ROOT/'budget.json').exists():
        raise RuntimeError('do not repackage launched experiment')
    source=REPO/'coding_base/moc_adaptive_sampling'
    target=ROOT/'coding_base/moc_adaptive_sampling'
    shutil.copytree(source,target,dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','tests','verify_local.py','audit_assets.py'))
    sources=list(ROOT.glob('*.py'))+list((ROOT/'coding_base').rglob('*.py'))
    fixed=[ROOT/name for name in ('config.json','protocol.md','val_role_snapshot.json','expected_dags.json','group_utility_model.json')]
    data=list((ROOT/'data').glob('*.jsonl'))
    allowed={'val_inputs.jsonl','historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl'}
    assert {p.name for p in data}==allowed
    lock=dict(source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources+fixed},
        data_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in data},
        gold_scope='none',val_gold_included=False,test_included=False,training_labels_included=False,
        frozen_method_commit='bfc2204',train_gate_passed=False,user_authorized_diagnostic_val=True)
    (ROOT/'experiment_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    model=json.loads((ROOT/'group_utility_model.json').read_text())
    policy=dict(frozen_before_val=True,model_schema=model['schema'],model_sha256=sha(ROOT/'group_utility_model.json'),
        config_sha256=sha(ROOT/'config.json'),val_inputs_sha256=sha(ROOT/'data/val_inputs.jsonl'),
        source_hashes=lock['source_hashes'],experiment_lock_sha256=sha(ROOT/'experiment_lock.json'),
        train_gate_passed=False,user_authorized_diagnostic_val=True,selection_scope='lowest train OOF MSE corrected candidate, failed gate')
    (ROOT/'policy_lock.json').write_text(json.dumps(policy,indent=2)+'\n',encoding='utf-8')
    artifact=ROOT/'deployment.tar.gz'
    files=sorted(set(sources+fixed+data+[ROOT/'experiment_lock.json',ROOT/'policy_lock.json']))
    with tarfile.open(artifact,'w:gz') as archive:
        for path in files:
            archive.add(path,arcname=path.relative_to(ROOT).as_posix())
    report=dict(files=len(files),bytes=artifact.stat().st_size,sha256=sha(artifact),
        model_sha256=policy['model_sha256'],val_conditions=120,training=False,val_gold=False,test=False)
    (ROOT/'deployment_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
