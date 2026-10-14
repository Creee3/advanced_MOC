"""Launch one bounded GPU supervisor; never schedule recurring work."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
config=json.loads((ROOT/"config.json").read_text())
if (ROOT/"status.json").exists():
    raise RuntimeError("existing task status; do not duplicate or silently restart")
assert json.loads((ROOT/"runtime_ready.json").read_text())["status"]=="passed"
env=os.environ.copy()
env.update(PYTHONUTF8="1",NO_PROXY="127.0.0.1,localhost",no_proxy="127.0.0.1,localhost")
with (ROOT/"supervisor.log").open("a") as stream:
    process=subprocess.Popen([config["python_executable"],"-u",str(ROOT/"supervise.py")],
                             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                             cwd=str(ROOT),env=env,start_new_session=True)
report={"pid":process.pid,"started_utc":datetime.now(timezone.utc).isoformat(),
        "conditions":config["conditions_total"],"config_sha256":hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
        "maximum_wall_seconds":config["gpu_policy"]["max_wall_seconds"],
        "stop_file":str(ROOT/"STOP"),"val_authorized":True,"test_run":False,
        "training_questions":48,"val_questions":60}
(ROOT/"launcher.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
