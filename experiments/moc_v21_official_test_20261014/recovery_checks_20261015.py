"""Regression for the actual shared-ledger failure; no backend/model calls."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'run_test.py').read_text(encoding='utf8'))
guard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='assert_own_cost_settled')
space={}
exec(compile(ast.Module(body=[guard],type_ignores=[]),'actual_run_test_guard','exec'),space)


class SharedReservationRegression(unittest.TestCase):
    def test_other_worker_inflight_does_not_block_recording_invalid(self):
        # Reproduce the observed situation: worker0 owns no outstanding call;
        # worker1 owns a normal10192 reservation in the same shared ledger.
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'budget.json';p.write_text(json.dumps(dict(reserved_tokens=10192)))
            main=SimpleNamespace(pending_requests={})
            merger=SimpleNamespace(pending_requests={})
            space['assert_own_cost_settled'](main,merger)
            record=dict(status='invalid',reason='code timeout; incomplete graph outputs')
            (Path(folder)/'results.jsonl').write_text(json.dumps(record)+'\n')
            self.assertEqual(json.loads(p.read_text())['reserved_tokens'],10192)
            self.assertEqual(json.loads((Path(folder)/'results.jsonl').read_text())['status'],'invalid')

    def test_own_unknown_main_request_must_stop(self):
        with self.assertRaisesRegex(RuntimeError,'this worker has unresolved'):
            space['assert_own_cost_settled'](SimpleNamespace(pending_requests={'main-1':10192}),SimpleNamespace(pending_requests={}))

    def test_own_unknown_merger_request_must_stop(self):
        with self.assertRaisesRegex(RuntimeError,'this worker has unresolved'):
            space['assert_own_cost_settled'](SimpleNamespace(pending_requests={}),SimpleNamespace(pending_requests={'merger-1':16384}))


if __name__=='__main__':unittest.main(verbosity=2)
