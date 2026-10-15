"""CPU-only transition checks; subprocess launches are mocked."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

spec=importlib.util.spec_from_file_location('coordinator_test_target',Path(__file__).with_name('coordinator.py'))
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

class Transitions(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.folder=self.root/'run';self.folder.mkdir()
        self.root_patch=patch.object(c,'ROOT',self.root);self.root_patch.start()
        self.put('status.json',dict(phase='adaptive_complete',cleanup_complete=True,supervisor_pid=123))
        self.put('budget.json',dict(reserved_tokens=0,new_calls=100,actual_tokens=1234))
        self.put('run_start.json',dict(deadline_unix=time.time()+600))
        self.put('config.json',dict(python_executable='/fake/python'))
    def tearDown(self):
        self.root_patch.stop();self.tmp.cleanup()
    def put(self,name,value):
        (self.folder/name).write_text(json.dumps(value))
    def test_proc_zombie_empty_missing_and_live(self):
        proc=self.root/'proc';(proc/'8').mkdir(parents=True)
        for state,cmd,expected in [('S',b'python\0supervise.py',True),('Z',b'python',False),('S',b'',False)]:
            (proc/'8/cmdline').write_bytes(cmd);(proc/'8/stat').write_text('8 (python worker) '+state+' 1 2 3')
            self.assertEqual(c.process_active(8,proc),expected)
        self.assertFalse(c.process_active(9,proc))
    def test_wait_exit_then_baseline_resume_only(self):
        with patch.object(c,'process_active',side_effect=[True,False]),patch.object(c.time,'sleep'),patch.object(c.subprocess,'Popen',return_value=Mock(pid=999)) as launch:
            c.start(self.folder,'baseline')
            self.assertEqual(launch.call_args.args[0][-3:],['--mode','baseline','--resume'])
            self.assertEqual(json.loads((self.folder/'budget.json').read_text())['actual_tokens'],1234)
    def test_live_supervisor_is_not_duplicated(self):
        with patch.object(c,'process_active',return_value=True),patch.object(c.time,'monotonic',side_effect=[0,31]),patch.object(c.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'still live'):c.start(self.folder,'baseline')
            launch.assert_not_called()
    def test_pending_or_cleanup_failure_blocks(self):
        for clean,pending in [(False,0),(True,1)]:
            self.put('status.json',dict(phase='adaptive_complete',cleanup_complete=clean,supervisor_pid=123))
            self.put('budget.json',dict(reserved_tokens=pending))
            with patch.object(c.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'cleanup/accounting'):c.start(self.folder,'baseline')
                launch.assert_not_called()
    def test_original_deadline_blocks(self):
        self.put('run_start.json',dict(deadline_unix=time.time()-1))
        with patch.object(c,'process_active',return_value=False),patch.object(c.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'deadline'):c.start(self.folder,'baseline')
            launch.assert_not_called()
    def test_stop_blocks(self):
        (self.folder/'STOP').write_text('user stop')
        with patch.object(c.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'STOP'):c.start(self.folder,'baseline')
            launch.assert_not_called()
    def test_completed_adaptive_is_not_repeated(self):
        with patch.object(c.subprocess,'Popen') as launch:
            c.start(self.folder,'adaptive');launch.assert_not_called()

if __name__=='__main__':unittest.main()
