"""Launch exactly one immutable v2 train/val batch, with no recurring job."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
config=json.loads((root/'config.json').read_text())
if (root/'status.json').exists() or (root/'budget.json').exists() or (root/'STOP').exists():
    raise RuntimeError('existing state or STOP; no silent duplicate/restart')
with (root/'supervisor.log').open('a') as stream:
    process=subprocess.Popen([config['python_executable'],'-u',str(root/'supervise.py')],cwd=str(root),
                             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                             env=os.environ.copy(),start_new_session=True)
report={'pid':process.pid,'started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'v2 original train47/calibration then only v2 val60x2','test':False}
(root/'launcher.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
