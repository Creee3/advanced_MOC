"""Locked val evaluation; gold stays local and is never a policy feature."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
SNAP=ROOT/"remote_snapshot"

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def paired(left,right,gold):
    keys=sorted(set(left)&set(right))
    if not keys:
        return {"pairs":0}
    delta=np.asarray([int(right[q]["answer"]==gold[q])-int(left[q]["answer"]==gold[q]) for q in keys])
    lt=np.asarray([left[q]["stats"]["total_model_tokens"] for q in keys],dtype=float)
    rt=np.asarray([right[q]["stats"]["total_model_tokens"] for q in keys],dtype=float)
    improved,degraded=int((delta>0).sum()),int((delta<0).sum())
    n=improved+degraded
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(improved,degraded)+1))/2**n) if n else 1.
    rng=np.random.default_rng(42)
    draw=rng.integers(0,len(keys),size=(20000,len(keys)))
    boot=delta[draw].mean(axis=1)
    lower=float(np.quantile(boot,.05))
    cost_boot=rt[draw].sum(axis=1)/lt[draw].sum(axis=1)
    return {"pairs":len(keys),"accuracy_difference":float(delta.mean()),"improved":improved,"degraded":degraded,
            "two_sided_sign_p":p,"paired_accuracy_bootstrap95":np.quantile(boot,[.025,.975]).tolist(),
            "one_sided_accuracy_lower95":lower,"prespecified_margin":.05,"bootstrap_noninferiority_screen":lower>-.05,
            "total_token_ratio":float(rt.sum()/lt.sum()),"token_ratio_bootstrap95":np.quantile(cost_boot,[.025,.975]).tolist(),
            "observed_operational_target_met":bool(delta.sum()>=0 and rt.sum()<lt.sum()),
            "interpretation":"observed screening result; bootstrap/significance alone does not establish general quality guarantee"}

def main():
    gold={r["question_id"]:r["correct"] for r in rows(ROOT/"data/val_evaluation_only_LOCAL_ONLY.jsonl")}
    results=[r for p in (SNAP/"runs").glob("shard*/results.jsonl") for r in rows(p)]
    groups={}
    for r in results:
        assert r["status"]=="ok" and r["question_id"] in gold
        name=r["arm"]+(f"@{r['density']}" if r["density"] is not None else "")
        assert r["question_id"] not in groups.setdefault(name,{})
        groups[name][r["question_id"]]=r
    metrics={}
    for name,items in groups.items():
        stats=[r["stats"] for r in items.values()]
        metric={"n":len(items),"correct":sum(r["answer"]==gold[q] for q,r in items.items())}
        for key in ("total_model_tokens","main_prompt_tokens","main_completion_tokens","merger_prompt_tokens",
                    "merger_completion_tokens","ordinary_communication_text_tokens","final_communication_text_tokens","merger_calls","code_timeouts","main_output_limit_hits"):
            metric[key]=sum(x[key] for x in stats)
        ledger=[r["sampling"] for r in items.values() if r["sampling"]]
        if ledger:
            selections=[s for x in ledger for s in x["selections"]]
            nonroot=[s for s in selections if s["candidates"]]
            available=Counter(str(m["hop"]) for s in selections for m in s["candidates"])
            selected=Counter(str(m["hop"]) for s in selections for m in s["candidates"] if m["source"] in s["source_ids"])
            actions=Counter(x["action"] for s in selections for x in s["layers"])
            metric.update(candidate_messages_by_hop=dict(available),read_messages_by_hop=dict(selected),
                          nonroot_receivers=len(nonroot),empty_nonroot=sum(not s["source_ids"] for s in nonroot),
                          all_read_nonroot=sum(len(s["source_ids"])==len(s["candidates"]) for s in nonroot),
                          stopped_with_budget_remaining=sum(s["stop_reason"]=="predicted_net_gain_nonpositive" and s["communication_tokens"]<1024 for s in nonroot),
                          stop_reasons=dict(Counter(s["stop_reason"] for s in selections)),actions=dict(actions),
                          routing_seconds=sum(x["routing_seconds"] for x in ledger),encoding_seconds=sum(x["encoding_seconds"] for x in ledger),
                          sketch_bytes_generated=sum(x["sketch_bytes_generated"] for x in ledger))
        metrics[name]=metric
    comparisons={}
    for d in (.3,.7):
        right=groups.get("adaptive_sampling@"+str(d),{})
        for arm in ("moc","moc_without_phi","vanilla_mas"):
            left=groups.get(arm+"@"+str(d),{})
            comparisons[f"{arm}@{d} -> v1"]=paired(left,right,gold)
    train=[r for p in (SNAP/"training").glob("shard*/questions.jsonl") for r in rows(p)]
    train_tokens=sum(x["prompt_tokens"]+x["completion_tokens"] for x in train)
    excluded=[r for p in (SNAP/"training").glob("shard*/excluded_questions.jsonl") for r in rows(p)]
    invalid_tokens=sum(x["prompt_tokens"]+x["completion_tokens"] for x in excluded)
    recovery=json.loads((SNAP/"training_recovery.json").read_text()) if (SNAP/"training_recovery.json").exists() else {}
    abandoned=recovery.get("abandoned_calls",{})
    abandoned_tokens=abandoned.get("prompt_tokens",0)+abandoned.get("completion_tokens",0)
    report={"scope":"locked val60; two densities share questions and are not independent samples",
            "completed_conditions":len(results),"expected_conditions":540,"metrics":metrics,"paired":comparisons,
            "one_time_training": {"valid_questions":len(train),"excluded_questions":len(excluded),
                "main_calls":sum(x["main_calls"] for x in train+excluded)+abandoned.get("main_calls",0),
                "valid_training_tokens":train_tokens,"excluded_training_tokens":invalid_tokens,
                "abandoned_call_tokens":abandoned_tokens,"total_model_tokens":train_tokens+invalid_tokens+abandoned_tokens},
            "training_report":json.loads((SNAP/"training_report.json").read_text()) if (SNAP/"training_report.json").exists() else None,
            "inference_cost_scope":"all backend prompt/completion tokens, including ordinary/final/merger; training reported separately"}
    limited=[]
    for path in (SNAP/"runs").glob("shard*/trace.jsonl"):
        current=None
        for event in rows(path):
            if event["event_type"]=="condition_started":
                current=(event["question_id"],event["arm"],event["density"])
            elif event["event_type"]=="llm_call" and event["response"].get("done_reason")=="length":
                limited.append({"question_id":current[0],"arm":current[1],"density":current[2],
                                "event_index":event["event_index"],"eval_count":event["response"]["eval_count"]})
    complete_case={}
    for d in (.3,.7):
        excluded_ids={x["question_id"] for x in limited if x["density"]==d}
        if not excluded_ids:
            continue
        available={qid:r for qid,r in groups["adaptive_sampling@"+str(d)].items() if qid not in excluded_ids}
        comparison={}
        for arm in ("moc","moc_without_phi","vanilla_mas"):
            reference={qid:r for qid,r in groups[arm+"@"+str(d)].items() if qid not in excluded_ids}
            item=paired(reference,available,gold)
            item.update(reference_correct=sum(r["answer"]==gold[q] for q,r in reference.items()),
                        candidate_correct=sum(r["answer"]==gold[q] for q,r in available.items()),
                        reference_tokens=sum(r["stats"]["total_model_tokens"] for r in reference.values()),
                        candidate_tokens=sum(r["stats"]["total_model_tokens"] for r in available.values()))
            comparison[arm]=item
        complete_case[str(d)]={"excluded_question_ids":sorted(excluded_ids),"comparisons":comparison,
                              "scope":"secondary completeness analysis; remove flagged question from every arm at this density, no outcome-based filtering"}
    report["output_limit_calls"]=limited
    report["all_conditions_untruncated"]=not limited
    report["primary_interpretation"]="all60 observed capped-system outputs, with flagged condition; strict original output-completeness gate did not fully pass" if limited else "all conditions satisfy completeness gate"
    report["complete_case_sensitivity"]=complete_case
    report["training_amortization"]={}
    training_cost=train_tokens+invalid_tokens+abandoned_tokens
    for d in (.3,.7):
        candidate=metrics["adaptive_sampling@"+str(d)]["total_model_tokens"]
        reference=metrics["moc@"+str(d)]["total_model_tokens"]
        savings=reference-candidate
        report["training_amortization"][str(d)]={"inference_saved_tokens":savings,"one_time_training_tokens":training_cost,
             "training_plus_this_candidate_batch_tokens":candidate+training_cost,
             "this_batch_ratio_including_all_training":(candidate+training_cost)/reference,
             "illustrative_break_even_questions":training_cost/(savings/60) if savings>0 else None,
             "scope":"arithmetic using observed savings; not a guarantee of future savings, and training is a shared one-time cost"}
    (ROOT/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"completed_conditions":len(results),"metrics":metrics,"paired":comparisons,"training_tokens":train_tokens},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
