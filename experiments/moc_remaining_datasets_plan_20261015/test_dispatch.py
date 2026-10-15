"""CPU-only admission tests: idle memory, cohort cap, STOP and no blind resume."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dispatch as d

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runs=[self.root/'mmlu',self.root/'svamp']
        for p in self.runs:p.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(d,'ROOT',self.root).start();patch.object(d,'RUNS',self.runs).start()
        patch.object(d.subprocess,'check_output',return_value='lxc\n').start()
        patch.object(d,'audit',return_value={}).start()
        self.launch=patch.object(d,'launch',return_value={'pid':123}).start()
        patch.object(d.time,'sleep').start()
        patch.object(d,'active',side_effect=lambda p:bool(p)).start()
    def state(self,i,s):
        (self.runs[i]/'status.json').write_text(json.dumps(s))
        (self.runs[i]/'run_start.json').write_text(json.dumps({'deadline_unix':d.time.time()+1000}))
    def test_memory_not_util_alone(self):
        a={'a':dict(uuid='a',util=0,free=3200)}
        self.assertEqual(d.stable_devices(a,a,1),[])
    def test_twice_stable(self):
        a={'a':dict(uuid='a',util=100,free=24000)};b={'a':dict(uuid='a',util=0,free=24000)}
        self.assertEqual(d.stable_devices(a,b,1),[])
    def test_pair_distinct(self):
        a={'a':dict(uuid='a',util=0,free=24000)}
        self.assertEqual(d.stable_devices(a,a,2),[])
        a['b']=dict(uuid='b',util=0,free=11000)
        self.assertEqual(len(d.stable_devices(a,a,2)),2)
    def test_stop(self):
        (self.root/'STOP').touch();self.assertEqual(d.tick()['phase'],'stopped');self.launch.assert_not_called()
    def test_two_live_no_third(self):
        for i in range(2):self.state(i,dict(phase='training',mode='adaptive',supervisor_pid=i+1))
        self.assertEqual(d.tick()['phase'],'running_adaptive');self.launch.assert_not_called()
    def test_failure_no_resume(self):
        self.state(0,dict(phase='failed',supervisor_pid=None,error='backend error'))
        self.assertEqual(d.tick()['phase'],'needs_attention');self.launch.assert_not_called()
    def test_no_gpu_no_window(self):
        patch.object(d,'snapshot',return_value={'a':dict(uuid='a',util=0,free=3200)}).start()
        self.assertEqual(d.tick()['phase'],'waiting_for_gpu');self.launch.assert_not_called()
        self.assertFalse((self.runs[0]/'run_start.json').exists())
    def test_deadline_blocks_baseline(self):
        for i in range(2):self.state(i,dict(phase='adaptive_complete',cleanup_complete=True,supervisor_pid=None))
        (self.runs[0]/'run_start.json').write_text(json.dumps({'deadline_unix':1}))
        self.assertEqual(d.tick()['phase'],'needs_attention');self.launch.assert_not_called()

if __name__=='__main__':unittest.main()
