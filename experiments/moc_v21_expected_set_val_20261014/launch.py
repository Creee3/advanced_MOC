"""Launch one frozen val-only batch; preserve failed-attempt state on recovery."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'config.json').read_text())
    if (ROOT/'STOP').exists():
        raise RuntimeError('STOP present; never override explicit stop')
    existing = (ROOT/'status.json').exists() or (ROOT/'budget.json').exists()
    if existing and not args.resume:
        raise RuntimeError('existing state; no silent duplicate or budget reset')
    if args.resume:
        status = json.loads((ROOT/'status.json').read_text())
        if status['phase'] in ('complete','stopped'):
            raise RuntimeError('completed/stopped task is not recoverable by this entry')
        if not status.get('cleanup_complete'):
            raise RuntimeError('must inspect and verify owned cleanup before recovery')
        budget = json.loads((ROOT/'budget.json').read_text())
        if budget['reserved_tokens']:
            raise RuntimeError('unresolved reservation must be audited before recovery')
        if not (ROOT/'run_start.json').exists():
            raise RuntimeError('missing original deadline; cannot reset wall limit')
    with (ROOT/'supervisor.log').open('a') as stream:
        command = [config['python_executable'],'-u',str(ROOT/'supervise.py')]
        if args.resume:
            command.append('--resume')
        process = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
    report = dict(pid=process.pid, started_utc=datetime.now(timezone.utc).isoformat(),
                  scope='v21 frozen compact checkpoint, original val60x2 only',
                  resume=args.resume, train=False, test=False)
    name = 'launcher_resume_'+str(process.pid)+'.json' if args.resume else 'launcher.json'
    (ROOT/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
