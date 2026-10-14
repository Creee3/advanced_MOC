"""Raw-call/condition/paired-label/policy-lock/resource consistency checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SNAP=ROOT/"remote_snapshot"

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def main():
    config=json.loads((ROOT/"config.json").read_text())
    lock=json.loads((SNAP/"policy_lock.json").read_text())
    assert lock["frozen_before_validation"] and not lock["val_gold_on_server"]
    assert hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest()==lock["config_sha256"]
    assert hashlib.sha256((ROOT/"data/val_inputs.jsonl").read_bytes()).hexdigest()==lock["val_inputs_sha256"]
    assert hashlib.sha256((SNAP/"utility_model.json").read_bytes()).hexdigest()==lock["utility_model_sha256"]
    for name,digest in lock["execution_source_sha256"].items():
        assert hashlib.sha256((SNAP/name).read_bytes()).hexdigest()==digest,name
    expected={x["question_id"] for x in rows(ROOT/"data/val_inputs.jsonl")}
    gold={x["question_id"]:x["correct"] for x in rows(ROOT/"data/train_labels.jsonl")}
    totals=Counter()
    results={}
    roles=[]
    length_calls=[]
    for shard in range(2):
        folder=SNAP/"runs"/f"shard{shard}"
        saved=rows(folder/"results.jsonl")
        assert len(saved)==270
        for r in saved:
            key=(r["question_id"],r["arm"],r["density"])
            assert key not in results and r["status"]=="ok"
            results[key]=r
        roles.append(json.loads((folder/"metadata.json").read_text())["roles"])
        current=None
        calls=[]
        finished=[]
        for event in rows(folder/"trace.jsonl"):
            kind=event["event_type"]
            assert kind not in ("llm_error","receiver_error","condition_error")
            if kind=="condition_started":
                assert current is None
                current=(event["question_id"],event["arm"],event["density"])
                calls=[]
            elif kind=="question_started":
                assert set(event["raw_input"])=={"task"}
                assert [n["role"] for n in event["nodes"]]==roles[-1]
            elif kind=="llm_call":
                assert current is not None
                options=event["request"]["options"]
                assert options["seed"]==42 and options["num_ctx"]==8192
                assert options["num_gpu"]==999
                if event["request"]["model"]==config["main_model"]:
                    assert options["temperature"]==0 and options["num_predict"]==2000
                else:
                    assert event["request"]["model"]==config["merger_model"] and options["temperature"]==.1
                response=event["response"]
                assert response.get("done")
                if response.get("done_reason")=="length":
                    length_calls.append({"question_id":current[0],"arm":current[1],"density":current[2],
                                         "event_index":event["event_index"],"model":event["request"]["model"],
                                         "eval_count":response["eval_count"]})
                assert response.get("prompt_eval_count",0)>0 and response.get("eval_count",0)>0
                calls.append(event)
            elif kind=="condition_finished":
                key=(event["question_id"],event["arm"],event["density"])
                assert current==key
                stats=results[key]["stats"]
                assert stats==event["stats"]
                actual=sum(c["response"].get(k,0) for c in calls for k in ("prompt_eval_count","eval_count"))
                assert actual==stats["total_model_tokens"]
                main=sum(c["request"]["model"]==config["main_model"] for c in calls)
                merger=len(calls)-main
                assert main==stats["main_calls"]==(1 if key[1]=="single_llm" else 8)
                assert merger==stats["merger_calls"]
                assert (merger==5*stats["merges"]) if key[1]=="moc" else (merger==0)
                totals.update(main_calls=main,merger_calls=merger,total_model_tokens=actual)
                finished.append(key)
                current=None
        assert current is None and len(finished)==270
        training=SNAP/"training"/f"shard{shard}"
        training_questions=rows(training/"questions.jsonl")
        excluded=rows(training/"excluded_questions.jsonl") if (training/"excluded_questions.jsonl").exists() else []
        assert len(training_questions)+len(excluded)==24
        valid_train_ids={r["question_id"] for r in training_questions}
        invalid_train_ids={r["question_id"] for r in excluded}
        assert not valid_train_ids&invalid_train_ids
        pairs=rows(training/"labels.jsonl")
        assert hashlib.sha256((training/"labels.jsonl").read_bytes()).hexdigest()==lock["train_labels_sha256"][f"training/shard{shard}/labels.jsonl"]
        for pair in pairs:
            q=pair["question_id"]
            assert q in gold and q in valid_train_ids and q not in expected
            assert pair["candidate_source"] not in pair["selected_sources"]
            assert pair["label"]==int(pair["augmented"]["answer"]==gold[q])-int(pair["baseline"]["answer"]==gold[q])
        totals.update(training_pairs=len(pairs),valid_training_questions=len(training_questions),excluded_training_questions=len(excluded))
    assert roles[0]==roles[1] and len(results)==540
    for q in expected:
        assert (q,"single_llm",None) in results
        for d in (.3,.7):
            for arm in ("vanilla_mas","moc","moc_without_phi","adaptive_sampling"):
                assert (q,arm,d) in results
    for (qid,arm,d),result in results.items():
        if arm=="adaptive_sampling":
            ledger=result["sampling"]
            assert ledger["valid"] and ledger["provider_usage_observed"]
            for selection in ledger["selections"]:
                candidates={m["source"] for m in selection["candidates"]}
                assert set(selection["source_ids"])<=candidates and selection["communication_tokens"]<=1024
                assert len(set(selection["source_ids"]))==len(selection["source_ids"])
    status=json.loads((SNAP/"status.json").read_text())
    cleanup=json.loads((SNAP/"cleanup.json").read_text())
    assert status["phase"]=="complete" and status["completed_conditions"]==540 and status["cleanup_complete"]
    assert cleanup["ports_closed"] and not cleanup["owned_processes_remaining"] and not cleanup["owned_children_remaining"]
    leases=rows(SNAP/"service_leases.jsonl")
    assert len(leases)==3 and all(s["loaded"]["size_vram"]>=s["loaded"]["size"] for s in leases)
    affected={(e["question_id"],e["arm"],e["density"]) for e in length_calls}
    report={"status":"passed_with_output_limit_exception" if length_calls else "passed",
            "record_and_cost_consistency":"passed","output_completeness":"failed" if length_calls else "passed",
            "all_conditions_valid":not bool(length_calls),"output_limit_calls":length_calls,
            "untruncated_conditions":540-len(affected),"val_conditions":540,"val_questions":60,"planned_training_questions":48,
            "valid_training_questions":totals["valid_training_questions"],"excluded_training_questions":totals["excluded_training_questions"],
            "call_totals":dict(totals),"roles":roles[0],"raw_usage_matches_saved":True,
            "paired_train_labels_recomputed":True,"policy_frozen_before_val":True,
            "all_services_full_vram":True,"owned_cleanup_confirmed":True,
            "scope":"record consistency audit; not an independent replication or performance guarantee"}
    (ROOT/"completion_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
