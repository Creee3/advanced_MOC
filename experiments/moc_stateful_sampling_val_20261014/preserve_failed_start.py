"""Preserve a zero-experiment startup failure before an authorized retry."""
import json
from pathlib import Path
import shutil
import socket

ROOT=Path(__file__).resolve().parent
status=json.loads((ROOT/"status.json").read_text())
assert status["phase"] in ("stopped","failed") and status["cleanup_complete"]
assert status["completed_conditions"]==0 and not (ROOT/"training").exists() and not (ROOT/"runs").exists()
for port in (11448,11449,11450):
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1",port))!=0
out=ROOT/"attempts/startup_failure_01"
out.mkdir(parents=True,exist_ok=False)
for name in ("status.json","launcher.json","cleanup.json","supervisor.log","supervisor.lock","main0_ollama.log","STOP"):
    path=ROOT/name
    if path.exists():
        shutil.move(str(path),str(out/name))
print(str(out))
