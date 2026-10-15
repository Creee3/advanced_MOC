"""Verify archive and all members before using raw data for local analysis."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1];SNAP=ROOT/'remote_snapshot'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest=json.loads((SNAP/'manifest.json').read_text())
    archive=SNAP/'final_raw_records.tar.gz'
    assert sha(archive)==manifest['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len(manifest['files'])
        assert {m.name for m in members}==set(manifest['files'])
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute()
            assert (SNAP/member.name).resolve().is_relative_to(SNAP.resolve())
        tar.extractall(SNAP,filter='data')
    for name,digest in manifest['files'].items():assert sha(SNAP/name)==digest,name
    print(json.dumps(dict(files_verified=len(manifest['files']),archive_sha256=manifest['archive_sha256'])))


if __name__=='__main__':main()
