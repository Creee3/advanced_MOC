"""Read-only final ownership/process/port check after CPU scoring."""
from datetime import datetime,timezone
import json
from pathlib import Path
import socket

ROOT=Path(__file__).resolve().parents[1]


def main():
    status=json.loads((ROOT/'status.json').read_text())
    scoring=json.loads((ROOT/'scoring_status.json').read_text())
    pids=[status['supervisor_pid'],*status['worker_pids'],*[s['pid'] for s in status['services']],
          scoring['wrapper_pid'],scoring['child_pid']]
    processes={str(pid):Path('/proc',str(pid)).exists() for pid in pids}
    assert not any(processes.values())
    ports={}
    for port in (11448,11450):
        with socket.socket() as s:ports[str(port)]='open' if s.connect_ex(('127.0.0.1',port))==0 else 'closed'
    assert set(ports.values())=={'closed'}
    result=dict(checked_utc=datetime.now(timezone.utc).isoformat(),owned_pids_exist=processes,
                private_ports=ports,generation_status=status['phase'],scoring_status=scoring['phase'],all_owned_clean=True)
    (ROOT/'final_process_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
