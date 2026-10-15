"""Linux Landlock + seccomp restricted Python executor, invoked by sandbox.py.

No namespaces are assumed inside the existing allocated LXC. Fail closed when
either kernel filesystem restriction or the syscall filter is unavailable.
"""
import ast
import ctypes
import errno
import json
import math
import os
import resource
import sys
import traceback
from typing import *


def restrict():
    libc = ctypes.CDLL(None, use_errno=True)
    # Landlock ABI discovery and path-beneath rules, Linux x86_64 syscall IDs.
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock unavailable')
    handled = (1 << 13)-1
    if abi >= 2: handled |= 1 << 13  # REFER
    if abi >= 3: handled |= 1 << 14  # TRUNCATE
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]
    rules = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0: raise OSError(ctypes.get_errno(), 'Landlock create')
    try:
        for path in sorted({os.path.realpath(p) for p in ('/usr', '/lib', '/lib64') if os.path.exists(p)}):
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            rule = Beneath((1 << 2) | (1 << 3), parent)  # read file / directory only
            try:
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
                    raise OSError(ctypes.get_errno(), 'Landlock rule')
            finally: os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0): raise OSError('no_new_privs')
        if libc.syscall(446, fd, 0): raise OSError(ctypes.get_errno(), 'Landlock restrict')
    finally: os.close(fd)
    sec = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW, then deny dangerous facilities
    if not ctx: raise RuntimeError('seccomp init')
    try:
        names = ('socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg '
                 'recvfrom recvmsg recvmmsg fork vfork clone clone3 execve execveat ptrace '
                 'process_vm_readv process_vm_writev kill tkill tgkill mount umount2 unshare '
                 'setns bpf keyctl add_key request_key perf_event_open open_by_handle_at '
                 'io_uring_setup io_uring_enter io_uring_register').split()
        for name in names:
            number=sec.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and sec.seccomp_rule_add(ctx,0x50000 | errno.EPERM,number,0):
                raise RuntimeError('seccomp rule '+name)
        if sec.seccomp_load(ctx): raise RuntimeError('seccomp load')
    finally: sec.seccomp_release(ctx)


def main():
    request=json.load(sys.stdin)
    result_fd=int(os.environ['MOC_RESULT_FD'])
    seconds=max(1,int(request['timeout']))
    resource.setrlimit(resource.RLIMIT_CPU,(seconds,seconds+1))
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.environ.clear()
    restrict()
    # Use a separate result channel so incidental model stdout is not parsed
    # as the runner result. As with HumanEval, this is not an anti-cheating API.
    result={'passed':False}
    try:
        namespace={'__name__':'__main__'}
        exec(request['code'],{},namespace) if request.get('separate_locals') else exec('from typing import *\n'+request['code'],namespace)
        if request.get('expression') is not None:
            result={'passed':True,'value':str(eval(request['expression'],namespace))[:10000]}
        else:
            exec(request.get('test',''),namespace)
            result={'passed':True}
    except BaseException as error:
        result={'passed':False,'error_type':type(error).__name__,'error':str(error)[:2000]}
    os.write(result_fd,json.dumps(result).encode())


if __name__=='__main__': main()
