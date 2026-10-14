"""One bounded, container-only GPU run; stop only process groups created here."""

import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/"config.json").read_text())
POLICY = CONFIG["gpu_policy"]
STATE = {}
PROCESSES = []
CHILDREN = {}
STOPPING = False
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(path,value):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def publish(**kwargs):
    STATE.update(kwargs)
    STATE["updated_utc"] = datetime.now(timezone.utc).isoformat()
    progress = []
    for shard in range(CONFIG["workers"]):
        path = ROOT/"runs"/f"shard{shard}"/"progress.json"
        if path.exists():
            item = json.loads(path.read_text())
            item["shard"] = shard
            progress.append(item)
    STATE["workers"] = progress
    STATE["completed_conditions"] = sum(p["completed_conditions"] for p in progress)
    STATE["total_conditions"] = CONFIG["conditions_total"]
    save(ROOT/"status.json",STATE)


def stop_requested():
    return STOPPING or (ROOT/"STOP").exists()


def handle(*args):
    global STOPPING
    STOPPING = True


def api(host,path,value=None,timeout=15):
    data = json.dumps(value).encode() if value is not None else None
    req = urllib.request.Request(host+path,data=data,headers={"Content-Type":"application/json"})
    with OPENER.open(req,timeout=timeout) as response:
        return json.load(response)


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,utilization.gpu,memory.free",
                                      "--format=csv,noheader,nounits"],text=True,timeout=15)
    return [{"index":int(i),"uuid":uuid.strip(),"utilization":int(u),"free_mib":int(f)}
            for i,uuid,u,f in csv.reader(output.splitlines())]


def choose_gpus():
    start = time.monotonic()
    previous = set()
    while not stop_requested() and time.monotonic()-start < POLICY["max_wait_seconds"]:
        snapshot = gpu_snapshot()
        main = [g for g in snapshot if g["utilization"] <= POLICY["max_utilization_percent"]
                and g["free_mib"] >= POLICY["main_min_free_mib"] and g["uuid"] in previous]
        main.sort(key=lambda g:(-g["free_mib"],g["index"]))
        if len(main) >= 2:
            used = {g["uuid"] for g in main[:2]}
            merger = [g for g in snapshot if g["uuid"] not in used and g["uuid"] in previous
                      and g["utilization"] <= POLICY["max_utilization_percent"]
                      and g["free_mib"] >= POLICY["merger_min_free_mib"]]
            merger.sort(key=lambda g:(-g["free_mib"],g["index"]))
            if merger:
                return main[:2],merger[0]
        previous = {g["uuid"] for g in snapshot if g["utilization"] <= POLICY["max_utilization_percent"]}
        publish(phase="waiting_for_gpu",gpu_snapshot=snapshot)
        time.sleep(POLICY["poll_seconds"])
    raise RuntimeError("GPU selection timed out or user stopped; no CPU fallback")


def spawn(command,log,env,cwd):
    with log.open("a") as stream:
        proc = subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,
                                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    PROCESSES.append(proc)
    return proc


def owned_stop(proc):
    if proc.poll() is not None:
        return
    assert os.getpgid(proc.pid) == proc.pid
    os.killpg(proc.pid,signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=10)


def process_identity(pid):
    try:
        folder = Path("/proc")/str(pid)
        if folder.stat().st_uid != os.getuid():
            return None
        fields = (folder/"stat").read_text().rsplit(") ",1)[1].split()
        return {"pid":pid,"state":fields[0],"parent":int(fields[1]),
                "group":int(fields[2]),"start":fields[19]}
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return None


def capture_children():
    pids = subprocess.check_output(["ps","-u",str(os.getuid()),"-o","pid="],text=True).split()
    entries = [entry for pid in pids if (entry := process_identity(int(pid))) is not None]
    owners = {p.pid for p in PROCESSES}
    seen = owners.copy()
    while True:
        new = {e["pid"] for e in entries if e["parent"] in seen or e["group"] in owners}
        if new <= seen:
            break
        seen.update(new)
    for e in entries:
        if e["pid"] in seen - owners:
            CHILDREN[e["pid"]] = e


def start_service(kind,gpu,port):
    current = next(g for g in gpu_snapshot() if g["uuid"] == gpu["uuid"])
    minimum = POLICY["merger_min_free_mib"] if kind == "merger" else POLICY["main_min_free_mib"]
    assert current["free_mib"] >= minimum and current["utilization"] <= POLICY["max_utilization_percent"]
    with socket.socket() as check:
        check.bind(("127.0.0.1",port))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"],OLLAMA_HOST=f"127.0.0.1:{port}",
               OLLAMA_MODELS=CONFIG["model_directory"],OLLAMA_NUM_PARALLEL="1",
               OLLAMA_MAX_LOADED_MODELS="1",OLLAMA_CONTEXT_LENGTH=str(CONFIG["num_ctx"]),
               OLLAMA_KEEP_ALIVE="5m",OLLAMA_NO_CLOUD="1",OLLAMA_NOPRUNE="1",NO_PROXY="127.0.0.1,localhost",
               no_proxy="127.0.0.1,localhost")
    # Explicitly select the existing v13 backend used by successful v0.
    # No model, driver, compiler or runtime installation is changed.
    library_root="/home/liuxin/.local/opt/ollama/0.40.2/lib/ollama"
    env.update(GGML_BACKEND_PATH=library_root+"/cuda_v13/libggml-cuda.so",
               OLLAMA_LLM_LIBRARY="cuda_v13",LD_LIBRARY_PATH=library_root+":"+library_root+"/cuda_v13")
    probe=subprocess.run([CONFIG["python_executable"],str(ROOT/"probe_gpu_runtime.py"),"cuda"],
                         env=env,capture_output=True,text=True,timeout=20)
    assert probe.returncode==0 and "cuMemAlloc 0" in probe.stdout,probe.stdout+probe.stderr
    proc = spawn([CONFIG["ollama_executable"],"serve"],ROOT/f"{kind}_ollama.log",env,CONFIG["author_directory"])
    host = f"http://127.0.0.1:{port}"
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Ollama service exited")
        try:
            version = api(host,"/api/version")
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("Ollama readiness deadline")
    family = "merger" if kind == "merger" else "main"
    model = CONFIG[family+"_model"]
    tags = api(host,"/api/tags")
    entry = next(m for m in tags["models"] if m["name"] == model)
    assert entry["digest"] == CONFIG[family+"_digest"]
    assert entry["details"]["quantization_level"] == "Q4_0"
    warm = api(host,"/api/generate",{"model":model,"prompt":"Reply with OK.","stream":False,
               "options":{"num_predict":8,"num_ctx":CONFIG["num_ctx"],"num_gpu":999,"temperature":0,"seed":42}},timeout=180)
    assert warm.get("done") and warm.get("eval_count",0) > 0
    loaded = next(m for m in api(host,"/api/ps")["models"] if m["name"] == model)
    assert loaded["size_vram"] >= loaded["size"], {"error":"CPU offload", "loaded":loaded}
    record = {"kind":kind,"gpu":gpu,"pid":proc.pid,"port":port,"host":host,"version":version,
              "model":entry,"loaded":loaded,"warmup":warm,"time_utc":datetime.now(timezone.utc).isoformat()}
    with (ROOT/"service_leases.jsonl").open("a") as stream:
        record.update(cuda_allocation_probe=probe.stdout,explicit_existing_cuda_library=env["GGML_BACKEND_PATH"])
        stream.write(json.dumps(record)+"\n")
    return record


def healthy(service):
    capture_children()
    assert next(p for p in PROCESSES if p.pid == service["pid"]).poll() is None
    models = api(service["host"],"/api/ps")["models"]
    # Ollama may release after keep-alive; reload on next call still constrained to GPU UUID.
    for m in models:
        assert m["name"] == service["model"]["name"] and m["size_vram"] >= m["size"]
    text = (ROOT/f"{service['kind']}_ollama.log").read_text(errors="replace")
    bad = [line for line in text.splitlines() if any(word in line.lower() for word in
           ("truncating input prompt","out of memory","failed to load model"))]
    assert not bad,bad[-3:]


def run_training(services,deadline):
    workers=[]
    for shard in range(CONFIG["workers"]):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUTF8="1")
        command=[CONFIG["python_executable"],"-u",str(ROOT/"collect_training.py"),"--shard",str(shard),
                 "--main-host",services[shard]["host"],"--merger-host",services[2]["host"]]
        workers.append(spawn(command,ROOT/f"training_worker{shard}.log",env,CONFIG["author_directory"]))
    publish(phase="training_labels",worker_pids=[p.pid for p in workers])
    while any(p.poll() is None for p in workers):
        if stop_requested() or time.monotonic()>deadline:
            raise RuntimeError("STOP or training wall deadline")
        for shard,p in enumerate(workers):
            if p.poll() not in (None,0):
                raise RuntimeError(f"training worker{shard} exited {p.returncode}")
        progress=[]
        for shard in range(CONFIG["workers"]):
            folder=ROOT/"training"/f"shard{shard}"
            path=folder/"progress.json"
            if path.exists():
                progress.append({"shard":shard,**json.loads(path.read_text())})
            trace=folder/"trace.jsonl"
            if trace.exists() and workers[shard].poll() is None and time.time()-trace.stat().st_mtime>POLICY["quiet_timeout_seconds"]:
                raise RuntimeError("training worker silence deadline")
        for service in services:
            healthy(service)
        publish(training_progress=progress)
        time.sleep(POLICY["poll_seconds"])
    fit=spawn([CONFIG["python_executable"],str(ROOT/"fit_utility.py")],ROOT/"fit_utility.log",env,CONFIG["author_directory"])
    if fit.wait(timeout=60)!=0:
        raise RuntimeError("utility fitting failed")
    assert (ROOT/"policy_lock.json").exists()
    publish(phase="policy_frozen",training_report=json.loads((ROOT/"training_report.json").read_text()),
            policy_lock=json.loads((ROOT/"policy_lock.json").read_text()))


def run_stage(services,limit,deadline):
    workers = []
    for shard in range(2):
        command = [CONFIG["python_executable"],"-u",str(ROOT/"run_experiment.py"),"--shard",str(shard),
                   "--main-host",services[shard]["host"],"--merger-host",services[2]["host"]]
        if limit:
            command += ["--limit-questions",str(limit)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="",OLLAMA_HOST=services[shard]["host"],HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUTF8="1")
        proc = spawn(command,ROOT/f"worker{shard}.log",env,CONFIG["author_directory"])
        workers.append(proc)
    publish(phase="val_smoke_running" if limit else "val_running",worker_pids=[p.pid for p in workers])
    start = time.monotonic()
    while any(p.poll() is None for p in workers):
        if stop_requested() or time.monotonic() > deadline:
            raise RuntimeError("User STOP or bounded six-hour deadline")
        for i,p in enumerate(workers):
            if p.poll() not in (None,0):
                raise RuntimeError(f"worker{i} exited {p.returncode}; see worker{i}.log")
            if p.poll() is None:
                trace = ROOT/"runs"/f"shard{i}"/"trace.jsonl"
                quiet = time.time()-trace.stat().st_mtime if trace.exists() else time.monotonic()-start
                if quiet > POLICY["quiet_timeout_seconds"]:
                    raise RuntimeError(f"worker{i} silent for {quiet:.0f}s")
        for service in services:
            healthy(service)
        publish()
        time.sleep(POLICY["poll_seconds"])
    for p in workers:
        assert p.returncode == 0
    publish()


def main():
    signal.signal(signal.SIGINT,handle)
    signal.signal(signal.SIGTERM,handle)
    lock = (ROOT/"supervisor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Adopt this supervisor's orphan descendants so exited runners can be reaped.
    import ctypes
    assert ctypes.CDLL(None).prctl(36,1,0,0,0) == 0
    assert subprocess.check_output(["systemd-detect-virt"],text=True).strip() in ("lxc","lxd")
    manifest = json.loads((ROOT/"deployment_hashes.json").read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    assert json.loads((ROOT/"runtime_ready.json").read_text())["status"] == "passed"
    status = subprocess.check_output(["git","-C",CONFIG["author_directory"],"status","--porcelain=v1"],text=True).strip()
    assert not status,status
    deadline = time.monotonic()+POLICY["max_wall_seconds"]
    publish(phase="preflight",supervisor_pid=os.getpid(),cleanup_complete=False)
    try:
        mains,merger = choose_gpus()
        services = []
        for index,gpu in enumerate(mains):
            services.append(start_service(f"main{index}",gpu,CONFIG["main_ports"][index]))
            publish(services=services)
        services.append(start_service("merger",merger,CONFIG["merger_port"]))
        publish(services=services)
        run_training(services,deadline)
        run_stage(services,1,deadline)
        assert STATE["completed_conditions"] == 18
        save(ROOT/"gpu_smoke_gate.json",{"status":"passed","completed_conditions":18,
                                          "quality_scope":"val conditions reused; no tuning or duplicate replays"})
        run_stage(services,0,deadline)
        assert STATE["completed_conditions"] == CONFIG["conditions_total"]
        publish(phase="complete")
    except BaseException as exc:
        publish(phase="stopped" if stop_requested() else "failed",error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        capture_children()
        for p in reversed(PROCESSES):
            owned_stop(p)
        # Any descendant that created a new process group is matched by PID and
        # kernel start time before signalling; never touch another task's process.
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for pid,identity in CHILDREN.items():
                current = process_identity(pid)
                if current and current["start"] == identity["start"] and current["state"] != "Z":
                    try:
                        os.kill(pid,sig)
                    except ProcessLookupError:
                        pass
            time.sleep(1)
        while True:
            try:
                child,_ = os.waitpid(-1,os.WNOHANG)
                if child == 0:
                    break
            except ChildProcessError:
                break
        ports = CONFIG["main_ports"]+[CONFIG["merger_port"]]
        remaining = []
        for port in ports:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1",port)) == 0:
                    remaining.append(port)
        children_remaining = [pid for pid,e in CHILDREN.items()
                              if (current := process_identity(pid)) and current["start"] == e["start"]]
        publish(cleanup_complete=not remaining and not children_remaining,remaining_private_ports=remaining,
                owned_children_remaining=children_remaining,
                owned_processes_remaining=[p.pid for p in PROCESSES if p.poll() is None])
        save(ROOT/"cleanup.json",{"ports_closed":not remaining,"remaining_ports":remaining,
                                  "owned_processes_remaining":[p.pid for p in PROCESSES if p.poll() is None],
                                  "owned_children_remaining":children_remaining,
                                  "gpu_snapshot_after":gpu_snapshot()})


if __name__ == "__main__":
    main()
