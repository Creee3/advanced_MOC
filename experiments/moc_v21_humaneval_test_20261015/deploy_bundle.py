"""Prepare a SHA-verifiable upload archive without test targets or old raw."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def main():
    lock=json.loads((ROOT/'experiment_lock.json').read_text())
    names=set(lock['source_hashes'])|set(lock['data_hashes'])|{'experiment_lock.json','policy_lock.json'}
    names.update(p.name for p in ROOT.glob('*.md'))
    for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):
        names.add('data/'+name)
    artifact=ROOT/'upload_bundle_LOCAL_ONLY.tar.gz'
    with tarfile.open(artifact,'w:gz') as tar:
        for name in sorted(names):tar.add(ROOT/name,arcname=name,recursive=False)
    print(json.dumps(dict(archive=str(artifact),files=len(names),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())))


if __name__=='__main__':main()
