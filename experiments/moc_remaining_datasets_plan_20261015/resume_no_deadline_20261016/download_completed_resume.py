"""Download immutable completed-run archive and verify every file locally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
NAMES=('moc_v21_gsm8k_fewshot_20261015','moc_v21_mmlu_pro_fewshot_20261015','moc_v21_svamp_train40_test_20261015')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=NAMES);a=parser.parse_args()
    out=HERE.parent.parent/a.name/'test_review_20261016';out.mkdir(parents=True,exist_ok=True)
    remote='/home/liuxin/ICML2027/experiments/moc_remaining_datasets_plan_20261015/resume_no_deadline_20261016/completed_archives/'+a.name
    for name in ('full_run.tar.gz','member_sha256.json','archive_receipt.json'):
        if not (out/name).exists():subprocess.run(['scp','DevContainer:'+remote+'/'+name,str(out/name)],check=True,timeout=120)
    report=json.loads((out/'archive_receipt.json').read_text());manifest=json.loads((out/'member_sha256.json').read_text())
    assert sha(out/'full_run.tar.gz')==report['archive_sha256'] and (out/'full_run.tar.gz').stat().st_size==report['archive_bytes']
    assert sha(out/'member_sha256.json')==report['member_manifest_sha256']
    with tarfile.open(out/'full_run.tar.gz') as tar:
        entries=tar.getmembers();assert len(entries)==len(manifest)==report['members'] and {x.name for x in entries}==set(manifest)
        for entry in entries:
            path=PurePosixPath(entry.name);assert entry.isfile() and not path.is_absolute() and '..' not in path.parts
            target=out/'records'/entry.name;meta=manifest[entry.name]
            if target.exists():
                assert target.stat().st_size==meta['bytes'] and sha(target)==meta['sha256'];continue
            target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();total=0
            with tar.extractfile(entry) as source,target.open('xb') as dest:
                for block in iter(lambda:source.read(1048576),b''):h.update(block);total+=len(block);dest.write(block)
            assert total==meta['bytes'] and h.hexdigest()==meta['sha256'],entry.name
    result=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),archive_sha256=report['archive_sha256'],
        members_verified=len(manifest),all_hashes_match=True,safe_regular_members_only=True,no_gold_scoring=True)
    (out/'download_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
