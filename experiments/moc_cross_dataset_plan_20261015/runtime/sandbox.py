"""Bounded isolated execution bridge; never execute generated code in workers."""
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parent


def run(code, test='', timeout=10, expression=None, separate_locals=False):
    payload=json.dumps(dict(code=code,test=test,timeout=timeout,expression=expression,separate_locals=separate_locals))
    # stdout/stderr to bounded files (child RLIMIT_FSIZE); dedicated result pipe.
    read_fd,write_fd=os.pipe()
    with tempfile.TemporaryFile() as output:
        process=subprocess.Popen(['/usr/bin/python3','-I',str(ROOT/'sandbox_child.py')],
            stdin=subprocess.PIPE,stdout=output,stderr=output,text=True,
            env={'PATH':'/usr/bin','HOME':'/nonexistent','PYTHONDONTWRITEBYTECODE':'1','MOC_RESULT_FD':str(write_fd)},
            cwd='/tmp',pass_fds=(write_fd,),start_new_session=True)
        os.close(write_fd)
        try:
            process.communicate(payload,timeout=timeout+3)
            data=os.read(read_fd,65536)
            output.seek(0); log=output.read(6000).decode(errors='replace')
            if process.returncode or not data:
                kind='ResourceLimit' if process.returncode in (-9,-24,-25) else 'SandboxFailure'
                return dict(passed=False,error_type=kind,error=log[-2000:],returncode=process.returncode)
            return json.loads(data)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.communicate()
            return dict(passed=False,error_type='TimeoutError',error=f'exceeded {timeout} seconds')
        finally: os.close(read_fd)


def execute(func, tests, timeout=10):
    success,failed,state=[],[],[]
    for test in tests:
        result=run(func,test,timeout)
        if result.get('error_type')=='SandboxFailure':
            raise RuntimeError('code executor infrastructure failed: '+str(result))
        state.append(result['passed'])
        if result['passed']:
            success.append(test)
        else:
            try:
                from src.tools.coding.python_executor import get_call_str
                expression=get_call_str(test)
                value=run(func,timeout=timeout,expression=expression)
                text=('TIMEOUT' if value.get('error_type') in ('TimeoutError','ResourceLimit')
                      else value.get('value',value.get('error','TIMEOUT')))
            except Exception as error:
                text=str(error)
            failed.append(f'{test} # output: {text}')
    return all(state), 'Tests passed:\n'+'\n'.join(success)+'\n\nTests failed:'+'\n'.join(failed), tuple(state)
