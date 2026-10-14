"""Keep completed train questions; archive and exclude incomplete/limited records."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket

from training_integrity import committed_labels

ROOT=Path(__file__).resolve().parent

def rows(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")

def write_rows(path,values):
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in values),encoding="utf-8")

def main():
    status=json.loads((ROOT/"status.json").read_text())
    assert status["phase"]=="failed" and status["cleanup_complete"]
    assert status["completed_conditions"]==0 and not (ROOT/"runs").exists() and not (ROOT/"policy_lock.json").exists()
    for port in (11448,11449,11450):
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1",port))!=0
    config=json.loads((ROOT/"config.json").read_text())
    out=ROOT/"attempts/training_failure_02"
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/"training",out/"training")
    for p in ROOT.glob("*.log"):
        shutil.copyfile(p,out/p.name)
    shutil.copyfile(ROOT/"config.json",out/"config.json")
    previous_start=json.loads((ROOT/"launcher.json").read_text())["started_utc"]
    active_seconds=(datetime.fromisoformat(status["updated_utc"])-datetime.fromisoformat(previous_start)).total_seconds()
    # Preserve the original overall GPU wall ceiling across this continuation.
    config["gpu_policy"]["max_wall_seconds"]=max(60,config["gpu_policy"]["max_wall_seconds"]-int(active_seconds+1))
    report={"previous_failure":"fixed train output limit", "valid_completed_questions":0,"valid_labels":0,
            "invalid_questions":[],"abandoned_calls":{"main_calls":0,"prompt_tokens":0,"completion_tokens":0},
            "prior_gpu_wall_seconds":active_seconds,"remaining_gpu_wall_seconds":config["gpu_policy"]["max_wall_seconds"],
            "algorithm_parameters_unchanged":True,"no_val_inference_yet":True,"shards":[]}
    for shard in range(2):
        folder=ROOT/"training"/f"shard{shard}"
        questions=rows(folder/"questions.jsonl")
        valid_ids={q["question_id"] for q in questions}
        labels=rows(folder/"labels.jsonl")
        committed=committed_labels(labels,questions)
        events=rows(folder/"trace.jsonl")
        failed={e["question_id"] for e in events if e["event_type"]=="llm_call" and e["response"].get("done_reason")=="length"}
        assert not failed&valid_ids
        exclusions=[]
        abandoned=[]
        for qid in sorted(failed):
            calls=[e for e in events if e["event_type"]=="llm_call" and e.get("question_id")==qid]
            examples=[r for r in labels if r["question_id"]==qid]
            index=examples[0]["question_index"] if examples else next(i for i,r in enumerate(rows(ROOT/"data/train_inputs.jsonl")) if r["question_id"]==qid)
            excluded={"question_id":qid,"question_index":index,"status":"invalid_output_limit",
                      "reason":"recovered failed question; all paired labels excluded",
                      "main_calls":len(calls),"prompt_tokens":sum(e["response"]["prompt_eval_count"] for e in calls),
                      "completion_tokens":sum(e["response"]["eval_count"] for e in calls)}
            exclusions.append(excluded)
        pending={e.get("question_id") for e in events if e["event_type"]=="llm_call"}-valid_ids-failed
        for qid in pending:
            abandoned += [e for e in events if e["event_type"]=="llm_call" and e.get("question_id")==qid]
        for key,response_key in (("prompt_tokens","prompt_eval_count"),("completion_tokens","eval_count")):
            report["abandoned_calls"][key]+=sum(e["response"][response_key] for e in abandoned)
        report["abandoned_calls"]["main_calls"]+=len(abandoned)
        report["valid_completed_questions"]+=len(valid_ids)
        report["valid_labels"]+=len(committed)
        report["invalid_questions"]+=exclusions
        report["shards"].append({"shard":shard,"valid_questions":len(valid_ids),"valid_labels":len(committed),
                                 "removed_uncommitted_labels":len(labels)-len(committed),"pending_rerun_question_ids":sorted(pending)})
        write_rows(folder/"labels.jsonl",committed)
        write_rows(folder/"excluded_questions.jsonl",exclusions)
        write_rows(folder/"materials.jsonl",[r for r in rows(folder/"materials.jsonl") if r["question_id"] in valid_ids])
        # Preserve previous calls in the archive; the active trace contains
        # only committed questions, avoiding duplicate request IDs on resume.
        write_rows(folder/"trace.jsonl",[e for e in events if e.get("question_id") in valid_ids])
        write(folder/"progress.json",{"completed_questions":len(valid_ids),"excluded_questions":len(failed),
             "processed_questions":len(valid_ids|failed),"total_questions":24})
    write(ROOT/"config.json",config)
    write(ROOT/"training_recovery.json",report)
    for name in ("status.json","launcher.json","cleanup.json","supervisor.log","supervisor.lock","service_leases.jsonl",
                 "main0_ollama.log","main1_ollama.log","merger_ollama.log","training_worker0.log","training_worker1.log"):
        p=ROOT/name
        if p.exists():
            if (out/name).exists():
                p.unlink()  # exact single owned file, already archived above
            else:
                shutil.move(str(p),str(out/name))
    manifest=json.loads((ROOT/"deployment_hashes.json").read_text())
    manifest["config.json"]=hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest()
    write(ROOT/"deployment_hashes.json",manifest)
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
