"""Archive complete raw evidence only after supervisor cleanup; no model calls."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
status=json.loads((ROOT/"status.json").read_text())
assert status["phase"]=="complete" and status["completed_conditions"]==540 and status["cleanup_complete"]
cleanup=json.loads((ROOT/"cleanup.json").read_text())
assert cleanup["ports_closed"] and not cleanup["owned_processes_remaining"] and not cleanup["owned_children_remaining"]
paths=[]
for folder in ("runs","training","attempts","coding_base"):
    paths += [p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
for pattern in ("*.py","*.json","*.jsonl","*.log","*.md"):
    paths += list(ROOT.glob(pattern))
paths += [ROOT/"data"/name for name in ("train_inputs.jsonl","train_labels.jsonl","val_inputs.jsonl")]
for folder in ("src","experiments"):
    paths += list((ROOT/"author_source"/folder).rglob("*.py"))
paths=sorted(set(p for p in paths if p.exists() and p.is_file()))
paths=[p for p in paths if p.name not in ("final_archive_manifest.json",)]
assert all("LOCAL_ONLY" not in p.name for p in paths)
manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(ROOT/"final_archive_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
archive=ROOT/"final_raw_records.tar.gz"
with tarfile.open(archive,"w:gz") as stream:
    for p in paths:
        stream.add(p,arcname=p.relative_to(ROOT).as_posix())
    stream.add(ROOT/"final_archive_manifest.json",arcname="final_archive_manifest.json")
print(json.dumps({"path":str(archive),"sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
                  "bytes":archive.stat().st_size,"files":len(paths)},indent=2))
