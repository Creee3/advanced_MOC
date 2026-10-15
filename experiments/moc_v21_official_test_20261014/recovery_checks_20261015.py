"""Regression for the actual shared-ledger failure; no backend/model calls."""
import ast
from contextlib import nullcontext
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
    def test_timeout_tool_result_cannot_be_reused_as_success(self):
        tree=ast.parse((ROOT/'resume_cache.py').read_text(encoding='utf8'))
        helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='timed_out_tool_keys')
        scope={}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),'actual_timeout_filter','exec'),scope)
        trace=[dict(event_type='code_execution_finished',timed_out=True,question_id='q'),
               dict(event_type='new_tool_result',request_key='failed',question_id='q'),
               dict(event_type='code_execution_finished',timed_out=False,question_id='q'),
               dict(event_type='new_tool_result',request_key='valid',question_id='q')]
        self.assertEqual(scope['timed_out_tool_keys'](trace),{'failed'})

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

    def test_unknown_cost_upper_bound_consumes_budget_without_faking_actual(self):
        tree=ast.parse((ROOT/'common.py').read_text(encoding='utf8'))
        reserve=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reserve')
        class BudgetExceeded(RuntimeError):pass
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            ledger=dict(new_calls=2,actual_tokens=80,reserved_tokens=0,unknown_cost_upper_bound_tokens=15)
            (root/'budget.json').write_text(json.dumps(ledger))
            scope=dict(ROOT=root,CONFIG=dict(max_new_calls=10,max_new_model_tokens=100),
                json=json,BudgetExceeded=BudgetExceeded,budget_lock=nullcontext,
                save=lambda p,v:p.write_text(json.dumps(v)))
            exec(compile(ast.Module(body=[reserve],type_ignores=[]),'actual_reserve','exec'),scope)
            with self.assertRaises(BudgetExceeded):scope['reserve'](6)
            self.assertEqual(json.loads((root/'budget.json').read_text()),ledger)
            scope['reserve'](5)
            charged=json.loads((root/'budget.json').read_text())
            self.assertEqual((charged['actual_tokens'],charged['unknown_cost_upper_bound_tokens'],charged['reserved_tokens']),(80,15,5))


if __name__=='__main__':unittest.main(verbosity=2)
