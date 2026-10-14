"""Training-only validity and atomic label admission; never inspect val outcomes."""

def inspect_calls(events, model):
    calls=[e for e in events if e["event_type"]=="llm_call"]
    assert calls and all(e["request"]["model"]==model for e in calls)
    assert all(e["response"].get("done") and e["response"].get("eval_count",0)>0 for e in calls)
    usage={"main_calls":len(calls),
           "prompt_tokens":sum(e["response"]["prompt_eval_count"] for e in calls),
           "completion_tokens":sum(e["response"]["eval_count"] for e in calls)}
    limited=[e for e in calls if e["response"].get("done_reason")=="length"]
    return calls,limited,usage


def committed_labels(labels, questions):
    committed={q["question_id"] for q in questions if q["status"]=="ok"}
    return [r for r in labels if r["question_id"] in committed]
