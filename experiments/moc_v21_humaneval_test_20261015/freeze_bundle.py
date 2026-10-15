"""Freeze before HumanEval generation; no fit or test-based selection."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    if (ROOT/'status.json').exists(): raise RuntimeError('already launched: cannot refreeze')
    sources=[*ROOT.glob('*.py'),*ROOT.glob('coding_base/moc_adaptive_sampling/*.py'),ROOT/'config.json']
    data=[ROOT/'data/test_inputs.jsonl',ROOT/'group_utility_model.json',ROOT/'code_role_snapshot.json',ROOT/'dataset_provenance.json']
    old=ROOT.parent/'moc_v21_official_test_20261014/group_utility_model.json'
    assert sha(old)==sha(ROOT/'group_utility_model.json')
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(sources)}
    data_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(data)}
    save(ROOT/'experiment_lock.json',dict(source_hashes=source_hashes,data_hashes=data_hashes,
        scope='HumanEval164 x frozen v21 zero-shot transfer / original MOC x densities 0.3/0.7',
        no_humaneval_training=True,local_gold_sha256=sha(ROOT/'data/test_targets_LOCAL_ONLY.jsonl')))
    save(ROOT/'policy_lock.json',dict(frozen_before_test=True,source_hashes=source_hashes,
        model_sha256=sha(ROOT/'group_utility_model.json'),test_inputs_sha256=sha(ROOT/'data/test_inputs.jsonl'),
        checkpoint_unchanged=True,role_mapping=None,unsupported_roles='abstain exactly as frozen predictor',
        adapter='native CodeWriting / FinalWriteCode; internal tests and early return only on selected sources',
        final_gold_not_deployed=True))
    print(json.dumps(dict(source_files=len(sources),data_files=len(data),checkpoint_sha256=sha(ROOT/'group_utility_model.json'))))


if __name__=='__main__':main()
