"""Write a bounded, read-only copy of this run's progress and completed records."""

import io
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
paths=[ROOT/name for name in ("status.json","launcher.json","runtime_ready.json","service_leases.jsonl",
       "gpu_smoke_gate.json","cleanup.json","supervisor.log","worker0.log","worker1.log",
       "training_worker0.log","training_worker1.log","fit_utility.log","training_report.json","policy_lock.json","utility_model.json",
       "training_recovery.json")]
paths += list((ROOT/"runs").glob("shard*/results.jsonl"))
paths += list((ROOT/"runs").glob("shard*/metadata.json"))
paths += list((ROOT/"runs").glob("shard*/progress.json"))
paths += list((ROOT/"runs").glob("shard*/stage_finished.json"))
paths += list((ROOT/"training").glob("shard*/questions.jsonl"))
paths += list((ROOT/"training").glob("shard*/labels.jsonl"))
paths += list((ROOT/"training").glob("shard*/progress.json"))
paths += list((ROOT/"training").glob("shard*/finished.json"))
paths += list((ROOT/"training").glob("shard*/excluded_questions.jsonl"))
with tarfile.open(ROOT/"live_snapshot.tar.gz","w:gz") as archive:
    for path in paths:
        if not path.exists():
            continue
        data=path.read_bytes()
        if path.suffix==".jsonl" and not data.endswith(b"\n"):
            data=data[:data.rfind(b"\n")+1]
        item=tarfile.TarInfo(path.relative_to(ROOT).as_posix())
        item.size=len(data)
        archive.addfile(item,io.BytesIO(data))
print(str(ROOT/"live_snapshot.tar.gz"))
