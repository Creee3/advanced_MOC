"""Local complete archive and per-member SHA verification before recovery."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
r=json.loads((HERE/'pre_resume_review.json').read_text())
m=json.loads((HERE/'pre_resume_member_sha256.json').read_text())
a=HERE/'all_three_pre_resume.tar.gz'
assert sha(a)==r['archive_sha256'] and a.stat().st_size==r['archive_bytes']
assert sha(HERE/'pre_resume_member_sha256.json')==r['manifest_sha256']
with tarfile.open(a) as tar:
    members=tar.getmembers();assert len(members)==len(m)==r['members']
    assert len({x.name for x in members})==len(members) and {x.name for x in members}==set(m)
    for entry in members:
        path=PurePosixPath(entry.name)
        assert entry.isfile() and not path.is_absolute() and '..' not in path.parts
        data=tar.extractfile(entry).read()
        assert hashlib.sha256(data).hexdigest()==m[entry.name]['sha256'] and len(data)==m[entry.name]['bytes']
out=dict(status='passed',archive_sha256=sha(a),members_verified=len(m),manifest_sha256=sha(HERE/'pre_resume_member_sha256.json'),no_gold_scoring=True)
(HERE/'local_download_verified.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
