import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
destination=ROOT/"coding_base/moc_adaptive_sampling"
shutil.copytree(WORKSPACE/"coding_base/moc_adaptive_sampling",destination,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__","tests","verify_local.py","audit_assets.py"))
files=[p for p in ROOT.glob("*.py")]
files += [ROOT/x for x in ("config.json","frozen_topologies.json","protocol.md")]
files += [p for p in destination.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
files += [ROOT/"data"/x for x in ("train_inputs.jsonl","train_labels.jsonl","val_inputs.jsonl")]
assert all("LOCAL_ONLY" not in str(p) for p in files)
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
(ROOT/"deployment_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
with tarfile.open(ROOT/"deployment.tar.gz","w:gz") as archive:
    for name in hashes:
        archive.add(ROOT/name,arcname=name)
    archive.add(ROOT/"deployment_hashes.json",arcname="deployment_hashes.json")
print(json.dumps({"files":len(hashes),"archive_sha256":hashlib.sha256((ROOT/"deployment.tar.gz").read_bytes()).hexdigest(),
                  "archive_bytes":(ROOT/"deployment.tar.gz").stat().st_size,"val_gold_uploaded":False},indent=2))
