"""Read-only /proc, listener and GPU check for completed MOC work."""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess

root = Path('/home/liuxin/ICML2027/experiments/moc_stateful_sampling_val_20261014')
status = json.loads((root / 'status.json').read_text())
recorded = {status.get('supervisor_pid'), *status.get('worker_pids', [])}
recorded.update(s['pid'] for s in status.get('services', []))
records = []
for folder in Path('/proc').iterdir():
    if not folder.name.isdigit() or int(folder.name) == os.getpid():
        continue
    try:
        uid = folder.stat().st_uid
        cmd = (folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        cwd = os.readlink(folder / 'cwd')
        env = (folder / 'environ').read_bytes().split(b'\0') if uid == os.getuid() else []
        private_service = any(e in (b'OLLAMA_HOST=127.0.0.1:11448', b'OLLAMA_HOST=127.0.0.1:11449', b'OLLAMA_HOST=127.0.0.1:11450') for e in env)
        if uid == os.getuid() and ('ICML2027' in cmd or 'ICML2027' in cwd or private_service):
            stat = (folder / 'stat').read_text().rsplit(') ', 1)[1].split()
            records.append(dict(pid=int(folder.name), parent=int(stat[1]), state=stat[0], cwd=cwd, command=cmd, private_service=private_service))
    except (OSError, PermissionError):
        continue
ports = []
for port in (11448, 11449, 11450):
    with socket.socket() as conn:
        conn.settimeout(1)
        ports.append(dict(port=port, accepting=conn.connect_ex(('127.0.0.1', port)) == 0))
def query(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=15)
    return dict(exit_code=p.returncode, output=p.stdout.strip(), error=p.stderr.strip())
print(json.dumps(dict(
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    phase=status.get('phase'), completed_conditions=status.get('completed_conditions'),
    recorded_pid_existence=[dict(pid=p, exists=Path('/proc', str(p)).exists()) for p in sorted(p for p in recorded if p)],
    current_project_or_private_service_processes=records, private_ports=ports,
    gpu_processes=query(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']),
    gpu_now=query(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader']),
    listeners=query(['ss', '-ltnp']),
), ensure_ascii=False, indent=2))
