"""CPU-only checks for the authorized management override; no model imports."""
import ast
import importlib.util
from pathlib import Path
import unittest
from resume_worker import PATCHES, patched_source

HERE=Path(__file__).resolve().parent
BASE=HERE.parent.parent
NAMES=('moc_v21_gsm8k_fewshot_20261015','moc_v21_mmlu_pro_fewshot_20261015','moc_v21_svamp_train40_test_20261015')

class RuntimeOverrideTests(unittest.TestCase):
    def pair(self,first,second,count):
        tree=ast.parse((HERE/'resume_dispatch.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pair_devices')
        scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'pair','exec'),scope)
        return scope['pair_devices'](first,second,count)

    def test_three_disjoint_pairs_use_partial_cards_as_mergers(self):
        devices={str(i):dict(uuid=str(i),index=i,util=0,free=24081 if i<3 else 20747) for i in range(6)}
        pairs=self.pair(devices,devices,3)
        self.assertEqual(len(pairs),3)
        self.assertEqual(len({g['uuid'] for pair in pairs for g in pair}),6)
        self.assertTrue(all(main['free']>=21000 and merger['free']>=10000 for main,merger in pairs))

    def test_both_gpu_snapshots_must_qualify(self):
        first={'a':dict(uuid='a',index=0,util=60,free=24081),'b':dict(uuid='b',index=1,util=0,free=20000)}
        second={k:dict(v,util=0) for k,v in first.items()}
        self.assertEqual(self.pair(first,second,3),[])

    def test_fewer_idle_cards_limits_concurrency(self):
        devices={str(i):dict(uuid=str(i),index=i,util=0,free=24081) for i in range(3)}
        self.assertEqual(len(self.pair(devices,devices,3)),1)

    def test_only_wall_guard_changes_across_six_modules(self):
        for name in NAMES:
            for module in PATCHES:
                with self.subTest(run=name,module=module):
                    source=(BASE/name/(module+'.py')).read_text(encoding='utf-8')
                    tree=ast.parse(source);changed=0
                    for node in ast.walk(tree):
                        if isinstance(node,ast.AsyncFunctionDef) and node.name=='generate':
                            for child in ast.walk(node):
                                if isinstance(child,ast.If) and isinstance(child.test,ast.BoolOp) and isinstance(child.test.op,ast.Or) and 'deadline_unix' in ast.unparse(child.test):
                                    child.test=child.test.values[0];changed+=1
                    self.assertEqual(changed,1)
                    self.assertEqual(ast.dump(tree),ast.dump(ast.parse(patched_source(module,source))))

    def test_stop_survives_but_expired_clock_no_longer_stops(self):
        class Root:
            stopped=False
            def __truediv__(self,p):return self
            def exists(self):return self.stopped
        for module in PATCHES:
            source=(BASE/NAMES[0]/(module+'.py')).read_text()
            tree=ast.parse(patched_source(module,source))
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.AsyncFunctionDef) and n.name=='generate')
            guard=next(n for n in fn.body if isinstance(n,ast.If))
            code=compile(ast.Expression(guard.test),'guard','eval');root=Root()
            self.assertFalse(eval(code,{'ROOT':root}))
            root.stopped=True;self.assertTrue(eval(code,{'ROOT':root}))

    def test_cumulative_budget_function_remains_identical(self):
        for name in NAMES:
            source=(BASE/name/'common.py').read_text()
            def reserve(s):return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='reserve'))
            self.assertEqual(reserve(source),reserve(patched_source('common',source)))
            self.assertIn("ledger.get('unknown_cost_upper_bound_tokens',0)",source)

    def test_rejects_unexpected_frozen_source(self):
        with self.assertRaises(AssertionError):patched_source('common','pass')

    def test_resume_worker_restricts_to_original_moc_test(self):
        source=(HERE/'resume_worker.py').read_text()
        self.assertIn("rest[rest.index('--phase')+1]=='test'",source)
        self.assertIn("rest[rest.index('--arm')+1]=='moc'",source)
        for name in NAMES:
            runner=(BASE/name/'runner.py').read_text()
            self.assertIn('if key in done:continue',runner)
            self.assertIn('restore_main_cache(b,out)',runner)

if __name__=='__main__':unittest.main()
