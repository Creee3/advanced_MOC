"""Read-only heartbeat snapshot for the three authorized runs; never scores gold."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

NAMES = ('moc_v21_gsm8k_fewshot_20261015', 'moc_v21_mmlu_pro_fewshot_20261015',
         'moc_mmlu_pro_val_baseline_20261015')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def collect_remote():
    base = Path('/home/liuxin/ICML2027/experiments')
    coordinator = read(base/'moc_cross_dataset_plan_20261015/coordinator_status.json')
    process = Path('/proc')/str(coordinator.get('pid'))/'cmdline'
    coordinator['actual_cmdline'] = process.read_bytes().replace(b'\0', b' ').decode() if process.exists() else None
    report = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), coordinator=coordinator, runs={})
    for name in NAMES:
        root = base/name
        for doc in ('task_plan.md', 'progress.md', 'protocol.md'):
            (root/doc).read_text(encoding='utf-8')
        config, lock = read(root/'config.json'), read(root/'experiment_lock.json')
        result = json.loads(subprocess.check_output(['/home/liuxin/.conda/envs/moc/bin/python', str(root/'inspect_run.py')], text=True))
        result['all_locked_hashes_match'] = all(hashlib.sha256((root/n).read_bytes()).hexdigest() == h
            for n,h in {**lock['source_hashes'], **lock['data_hashes']}.items())
        result['budget_live'] = read(root/'budget.json')
        result['caps'] = {k: config[k] for k in ('max_new_calls', 'max_new_model_tokens')}
        result['stop_present'] = (root/'STOP').exists()
        result['coverage'], result['loaded'] = {}, {}
        for stage in ('val', 'test'):
            records = [json.loads(line) for p in (root/stage).glob('shard*/results.jsonl')
                       for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
            keys = [(r['question_id'], r['arm'], r['density']) for r in records]
            counts = {}
            for r in records:
                key = '|'.join(map(str, (r['arm'], r['density'], r['status'])))
                counts[key] = counts.get(key, 0)+1
            result['coverage'][stage] = dict(counts=counts, duplicates=len(keys)-len(set(keys)),
                invalid_reasons=[r.get('reason') for r in records if r['status'] != 'ok'])
        for service in result['status'].get('services', []):
            try:
                with urllib.request.urlopen(service['host']+'/api/ps', timeout=10) as response:
                    data = json.load(response)
                result['loaded'][str(service['port'])] = [
                    {k: m.get(k) for k in ('name', 'digest', 'size', 'size_vram')} for m in data['models']]
            except Exception as error:
                result['loaded'][str(service['port'])] = dict(error=str(error))
        result['supervisor_log_tail'] = (root/'supervisor.log').read_text(encoding='utf-8', errors='replace').splitlines()[-10:]
        if (root/'cleanup.json').exists():
            result['cleanup'] = read(root/'cleanup.json')
        report['runs'][name] = result
    return report

if __name__ == '__main__':
    if '--capture' not in sys.argv:
        print(json.dumps(collect_remote()))
    else:
        response = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'DevContainer',
            '/home/liuxin/.conda/envs/moc/bin/python', '-'], input=Path(__file__).read_text(encoding='utf-8'),
            text=True, encoding='utf-8', capture_output=True, timeout=90)
        if response.returncode:
            raise RuntimeError(response.stderr)
        report = json.loads(response.stdout)
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        target = Path(__file__).parent/('healthcheck_'+stamp+'.json')
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('Snapshot:', target)
        print('Coordinator:', json.dumps(report['coordinator']))
        for name, r in report['runs'].items():
            print(json.dumps(dict(name=name, phase=r['status'].get('phase'), arm=r['status'].get('arm'),
                error=r['status'].get('error'), progress=r['progress'], coverage=r['coverage'],
                remaining=r['remaining_wall_seconds'], budget=r['budget_live'], stop=r['stop_present'],
                hashes_match=r['all_locked_hashes_match'], all_processes_alive=all(p['cmdline'] for p in r['processes']),
                loaded=r['loaded'], cleanup=r.get('cleanup'), log=r['supervisor_log_tail'])))
