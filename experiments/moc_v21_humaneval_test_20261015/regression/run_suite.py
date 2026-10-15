"""Run existing unit tests with original assets in the existing cloud runtime."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'coding_base'),str(ROOT/'regression/tests')]
import source_harness as harness
cfg=json.loads((ROOT/'config.json').read_text())
harness.WORKSPACE=Path('/home/liuxin/ICML2027')
harness.AUTHOR=Path(cfg['author_directory'])
harness.ASSETS=ROOT/'regression'
suite=unittest.defaultTestLoader.discover(str(ROOT/'regression/tests'))
result=unittest.TextTestRunner(verbosity=1).run(suite)
files=[*sorted((ROOT/'regression/tests').glob('*.py')),ROOT/'regression/upstream/ladies/pytorch_ladies.py']
report=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
    errors=len(result.errors),failures=len(result.failures),real_model_calls=0,
    assets={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(ROOT/'regression_report.json').write_text(json.dumps(report,indent=2)+'\n')
sys.exit(not result.wasSuccessful())
