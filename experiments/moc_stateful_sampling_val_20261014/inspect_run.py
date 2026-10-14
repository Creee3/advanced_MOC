"""Read-only compact progress snapshot; no polling loop or process mutation."""

import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
path=ROOT/"status.json"
status=json.loads(path.read_text()) if path.exists() else {}
report={key:status.get(key) for key in ("phase","completed_conditions","total_conditions","workers",
         "training_progress","supervisor_pid","worker_pids","updated_utc","error_type","error","cleanup_complete")}
report["services"]=[{"kind":s["kind"],"gpu":s["gpu"]["index"],"pid":s["pid"],"port":s["port"],
                     "size":s["loaded"]["size"],"size_vram":s["loaded"]["size_vram"],
                     "warmup_output":s["warmup"]["response"]} for s in status.get("services",[])]
report["worker_last_events"]={}
for i in range(2):
    path=ROOT/f"worker{i}.log"
    if path.exists():
        lines=path.read_text(errors="replace").splitlines()
        report["worker_last_events"][str(i)]=[line for line in lines if line.startswith("COMPLETE")
            or any(token in line for token in ("Traceback","AssertionError","RuntimeError","TypeError"))][-5:]
trace_errors=[]
for path in (ROOT/"runs").glob("shard*/trace.jsonl"):
    for line in path.read_text().splitlines():
        item=json.loads(line)
        if item["event_type"] in ("condition_error","llm_error","receiver_error"):
            trace_errors.append(item)
report["trace_errors"]=trace_errors[-3:]
report["training_worker_last_events"]={}
for i in range(2):
    path=ROOT/f"training_worker{i}.log"
    if path.exists():
        lines=path.read_text(errors="replace").splitlines()
        report["training_worker_last_events"][str(i)]=[line for line in lines if line.startswith("TRAIN") or "Traceback" in line][-3:]
report["gpu_now"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader"],text=True).splitlines()
print(json.dumps(report,ensure_ascii=False,indent=2))
