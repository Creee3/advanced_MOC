"""Verify every downloaded archive member without evaluating test answers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

OUT=Path(__file__).resolve().parent.parent/'moc_v21_svamp_train40_test_20261015'/'expiry_closure_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((OUT/'closure_report.json').read_text())
manifest=json.loads((OUT/'member_sha256.json').read_text())
archive=OUT/'expired_run_fullraw.tar.gz'
assert sha(archive)==report['archive_sha256'] and archive.stat().st_size==report['archive_bytes']
assert sha(OUT/'member_sha256.json')==report['member_manifest_sha256']
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    assert len(members)==len(manifest)==report['members'] and len({m.name for m in members})==len(members)
    assert {m.name for m in members}==set(manifest)
    for m in members:
        p=PurePosixPath(m.name)
        assert m.isfile() and not p.is_absolute() and '..' not in p.parts
        data=tar.extractfile(m).read()
        assert len(data)==manifest[m.name]['bytes'] and hashlib.sha256(data).hexdigest()==manifest[m.name]['sha256']
    # Keep complete local raw/source copy, separate from the committed audit.
    for m in members:
        target=OUT/'records'/m.name
        target.parent.mkdir(parents=True,exist_ok=True)
        data=tar.extractfile(m).read()
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
result=dict(status='passed',verified_utc=datetime.now(timezone.utc).isoformat(),
    archive_sha256=sha(archive),manifest_sha256=sha(OUT/'member_sha256.json'),members_verified=len(manifest),
    original_run_files_unchanged=True,no_gold_access_or_scoring=True,effective_cost_audit=report['effective_cost_audit'])
(OUT/'download_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
