"""Freeze originaltest inputs, unchanged v2.1 checkpoint and test-only harness."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (ROOT/'status.json').exists()
    shutil.copytree(REPO/'coding_base/moc_adaptive_sampling',ROOT/'coding_base/moc_adaptive_sampling',dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__','tests','verify_local.py','audit_assets.py'))
    source=list(ROOT.glob('*.py'))+list((ROOT/'coding_base').rglob('*.py'))
    fixed=[ROOT/n for n in ('config.json','protocol.md','val_role_snapshot.json','expected_dags.json','group_utility_model.json','data_preparation.json')]
    data=list((ROOT/'data').glob('*.jsonl'))
    assert {p.name for p in data}=={'test_inputs.jsonl','historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'}
    lock=dict(source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in source+fixed},
        data_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in data},
        test_gold_included=False,training_labels_included=False,test_included=True,
        train_gate_passed=False,user_authorized_test=True,official_test_historically_exposed=True)
    (ROOT/'experiment_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
    policy=dict(frozen_before_test=True,model_sha256=sha(ROOT/'group_utility_model.json'),
        config_sha256=sha(ROOT/'config.json'),test_inputs_sha256=sha(ROOT/'data/test_inputs.jsonl'),
        source_hashes=lock['source_hashes'],experiment_lock_sha256=sha(ROOT/'experiment_lock.json'),
        model_selection='unchanged completed v2.1 val checkpoint; no new fitting or tuning',historical_test_exposure_disclosed=True)
    (ROOT/'policy_lock.json').write_text(json.dumps(policy,indent=2)+'\n',encoding='utf8')
    assert policy['model_sha256']=='e2ce9cd7f69d3b4e6c2e6be7af1df0a852d44b972396989e5b02428a655fd210'
    files=sorted(set(source+fixed+data+[ROOT/'policy_lock.json',ROOT/'experiment_lock.json']))
    with tarfile.open(ROOT/'deployment.tar.gz','w:gz') as t:
        for p in files:t.add(p,arcname=p.relative_to(ROOT).as_posix())
    out=dict(files=len(files),sha256=sha(ROOT/'deployment.tar.gz'),bytes=(ROOT/'deployment.tar.gz').stat().st_size,
        conditions=1016,test_gold=False,model_sha256=policy['model_sha256'])
    (ROOT/'deployment_report.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))


if __name__=='__main__':main()
