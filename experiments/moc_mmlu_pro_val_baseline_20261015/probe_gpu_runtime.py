"""Bounded, model-free CUDA diagnostics; no drivers or host changes."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if len(sys.argv)>1:
    print(json.dumps({"phase":"started","visible":os.environ.get("CUDA_VISIBLE_DEVICES")}),flush=True)
    cuda=ctypes.CDLL("libcuda.so.1")
    print("cuInit",cuda.cuInit(0),flush=True)
    device=ctypes.c_int()
    print("cuDeviceGet",cuda.cuDeviceGet(ctypes.byref(device),0),device.value,flush=True)
    context=ctypes.c_void_p()
    print("cuCtxCreate",cuda.cuCtxCreate_v2(ctypes.byref(context),0,device),flush=True)
    buffer=ctypes.c_uint64()
    print("cuMemAlloc",cuda.cuMemAlloc_v2(ctypes.byref(buffer),ctypes.c_size_t(1024)),flush=True)
    print("cuMemFree",cuda.cuMemFree_v2(buffer),flush=True)
    print("cuCtxDestroy",cuda.cuCtxDestroy_v2(context),flush=True)
    sys.exit(0)

uuids=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True).splitlines()
report=[]
for line in uuids:
    index,uuid=line.split(",",1)
    if int(index) not in (0,3,7):
        continue
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=uuid.strip()
    proc=subprocess.Popen([sys.executable,str(__file__),"cuda"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,env=env,start_new_session=True)
    try:
        output,_=proc.communicate(timeout=15)
        result={"gpu":int(index),"returncode":proc.returncode,"output":output}
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        output,_=proc.communicate(timeout=5)
        result={"gpu":int(index),"timeout":True,"output":output}
    report.append(result)
    print(json.dumps(result),flush=True)
(ROOT/"cuda_probe.json").write_text(json.dumps(report,indent=2)+"\n")
