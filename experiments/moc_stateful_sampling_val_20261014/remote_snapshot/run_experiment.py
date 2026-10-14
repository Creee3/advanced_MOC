"""Actual author Graph.arun runs with original-paper baselines and one candidate."""

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

from trace_hooks import Observer, Recorder, QUESTION
from runtime_controls import stable_ids, backend_routes, effective_options, prompt_and_tool_observers

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
sys.path.insert(0, CONFIG["extension_directory"])
from moc_adaptive_sampling import SamplingConfig, LayerDependentSampler, MOCSamplingAdapter
from moc_adaptive_sampling.stateful import StatefulConfig, StatefulSampler, MarginalUtilityModel
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.paper_baselines import PaperRecipe, load_author_api, make_graph, run_single_llm


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)


def seed_all(torch, np):
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])


def task(row):
    return "Question: " + row["question"] + "\n" + "".join(option+"\n" for option in row["options"])


def call_stats(events, arm):
    calls = [e for e in events if e["event_type"] == "llm_call"]
    errors = [e for e in events if e["event_type"] in ("llm_error", "receiver_error")]
    assert not errors, errors
    main = [e for e in calls if e["request"]["model"] == CONFIG["main_model"]]
    merger = [e for e in calls if e["request"]["model"] == CONFIG["merger_model"]]
    assert len(main) == (1 if arm == "single_llm" else 8), len(main)
    merges = [e for e in events if e["event_type"] == "merge_finished"]
    if arm == "moc":
        assert len(merger) == 5 * len(merges)
        assert all(e["observer_consistent"] for e in merges)
    else:
        assert not merger and not merges
    for e in calls:
        response = e["response"]
        assert response.get("done") is True
        assert response.get("prompt_eval_count", 0) > 0 and response.get("eval_count", 0) > 0
        assert e["request"]["options"]["seed"] == CONFIG["seed"]
    consumed = [e for e in events if e["event_type"] == "communication_consumed"]
    assert len(consumed) == (0 if arm == "single_llm" else 8)
    if arm == "adaptive_sampling":
        assert not any(e["raw_fallback"] for e in consumed if not e["is_final"])
        assert all(e["communication_text_tokens"] <= CONFIG["candidate"]["communication_tokens"] for e in consumed if not e["is_final"])
    total = lambda group, key: sum(e["response"].get(key, 0) for e in group)
    executions = [e for e in events if e["event_type"] == "code_execution_finished"]
    return {"main_calls":len(main), "merger_calls":len(merger), "merges":len(merges),
            "main_prompt_tokens":total(main,"prompt_eval_count"), "main_completion_tokens":total(main,"eval_count"),
            "merger_prompt_tokens":total(merger,"prompt_eval_count"), "merger_completion_tokens":total(merger,"eval_count"),
            "total_model_tokens":sum(total(calls,k) for k in ("prompt_eval_count","eval_count")),
            "ordinary_communication_text_tokens":sum(e["communication_text_tokens"] for e in consumed if not e["is_final"]),
            "final_communication_text_tokens":sum(e["communication_text_tokens"] for e in consumed if e["is_final"]),
            "main_output_limit_hits":sum(e["response"]["eval_count"] >= CONFIG["main_num_predict"] for e in main),
            "model_seconds":sum(e["elapsed_seconds"] for e in calls),
            "code_timeouts":sum(e["timed_out"] for e in executions),
            "code_seconds":sum(e["seconds"] for e in executions)}


async def run(args):
    api = load_author_api(CONFIG["author_directory"])
    import numpy as np
    import torch
    import ollama
    from src.llm.profile_embedding import get_sentence_embedding
    gm = importlib.import_module("src.graph.graph")
    nm = importlib.import_module("src.graph.node")
    math_module = importlib.import_module("src.agents.math_solver")
    final = importlib.import_module("src.agents.final_decision").FinalRefer
    counter = LocalTokenizerCounter(CONFIG["tokenizer_path"], CONFIG["tokenizer_sha256"])
    # Warm MiniLM on CPU in existing environment; GPU memory reserved for actual LLMs.
    assert np.asarray(get_sentence_embedding("routing encoder readiness")).shape == (384,)
    recipe = PaperRecipe(llm_name=CONFIG["main_model"], edge_density=CONFIG["densities"][0],
                         use_cot=CONFIG["use_cot"], random_dag_seed=CONFIG["seed"],
                         ism_epsilon=CONFIG["ism_epsilon"])
    lock = json.loads((ROOT/"policy_lock.json").read_text())
    assert lock["frozen_before_validation"]
    assert hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest() == lock["config_sha256"]
    assert hashlib.sha256((ROOT/"data/val_inputs.jsonl").read_bytes()).hexdigest() == lock["val_inputs_sha256"]
    for name,digest in lock["execution_source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    model = MarginalUtilityModel.load(CONFIG["utility_checkpoint"],lock["utility_model_sha256"])
    sampler = StatefulSampler(StatefulConfig(**CONFIG["candidate"]), counter, model)
    input_rows = rows(ROOT / "data/val_inputs.jsonl")
    assert len(input_rows) == 60
    assert all(set(r) == {"question_id","question","options"} for r in input_rows)
    input_rows = [r for i,r in enumerate(input_rows) if i % CONFIG["workers"] == args.shard]
    if args.limit_questions:
        input_rows = input_rows[:args.limit_questions]
    out = ROOT / "runs" / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(out)
    if (out/"trace.jsonl").exists():
        recorder.counter = max((e["event_index"] for e in rows(out/"trace.jsonl")), default=0)
    previous = rows(out/"results.jsonl") if (out/"results.jsonl").exists() else []
    assert all(r["status"] == "ok" for r in previous)
    completed = {(r["question_id"],r["arm"],r["density"]) for r in previous}
    expected_topologies = json.loads((ROOT/"frozen_topologies.json").read_text())
    question_map = {task(r):r["question_id"] for r in input_rows}

    with ExitStack() as stack:
        stack.enter_context(backend_routes(ollama,args.main_host,args.merger_host))
        stack.enter_context(Observer(recorder,gm,nm,ollama,question_map))
        stack.enter_context(effective_options(ollama,CONFIG,counter))
        stack.enter_context(prompt_and_tool_observers(math_module,final,recorder,counter,CONFIG["code_timeout_seconds"]))
        seed_all(torch,np)
        with stable_ids():
            role_source = make_graph(api,recipe,"vanilla_mas")
        roles = [n.role for n in role_source.nodes.values()]
        save(out/"metadata.json", {"shard":args.shard,"config":CONFIG,"roles":roles,
                                  "source_commit":CONFIG["author_commit"],"gold_free_inputs":True,
                                  "main_host":args.main_host,"merger_host":args.merger_host})
        for index,row in enumerate(input_rows):
            # Rotate graph arm order without changing any model parameter.
            graph_arms = ["vanilla_mas","moc","moc_without_phi","adaptive_sampling"]
            rotation = int(row["question_id"][:2],16) % len(graph_arms)
            graph_arms = graph_arms[rotation:] + graph_arms[:rotation]
            conditions = [("single_llm", None)] + [(a,d) for d in CONFIG["densities"] for a in graph_arms]
            for arm,density in conditions:
                key = (row["question_id"],arm,density)
                if key in completed:
                    continue
                if (ROOT/"STOP").exists():
                    raise RuntimeError("user STOP requested")
                start = time.monotonic()
                event_start = len(recorder.records)
                seed_all(torch,np)
                token = QUESTION.set(row["question_id"])
                recorder.write("condition_started",question_id=row["question_id"],arm=arm,density=density)
                adapter = None
                try:
                    if arm == "single_llm":
                        result = await run_single_llm(api,recipe,{"task":task(row)})
                        text, extracted = result["llm_response"], result["extracted_answer"]
                    else:
                        current = replace(recipe,edge_density=density)
                        topology = api.get_kwargs("Random",7,density,CONFIG["seed"])
                        assert topology == expected_topologies[str(density)]
                        with stable_ids():
                            graph = make_graph(api,current,arm,roles=roles)
                        if arm == "adaptive_sampling":
                            adapter = MOCSamplingAdapter(graph,sampler,get_sentence_embedding,counter_name=counter.name)
                        answers,_ = await graph.arun({"task":task(row)},num_rounds=1,max_tries=1,max_time=300)
                        assert len(answers) == 1
                        text = answers[0]
                        prompt = api.PromptSetRegistry.get("aqua")
                        extracted = prompt.postprocess_answer(prompt,text)
                    stats = call_stats(recorder.records[event_start:],arm)
                    if adapter is not None:
                        adapter.ledger["provider_usage_observed"] = True
                        adapter.ledger["backend_usage"] = stats.copy()
                        recorder.write("sampling_ledger",question_id=row["question_id"],arm=arm,density=density,ledger=adapter.ledger)
                    item = {"question_id":row["question_id"],"arm":arm,"density":density,"status":"ok",
                            "answer":extracted,"final_output":text,"seconds":time.monotonic()-start,
                            "stats":stats,"sampling":adapter.ledger if adapter else None}
                    with (out/"results.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(item,ensure_ascii=False)+"\n")
                    completed.add(key)
                    recorder.write("condition_finished",question_id=row["question_id"],arm=arm,density=density,stats=stats)
                    save(out/"progress.json",{"completed_conditions":len(completed),"last_question":row["question_id"],
                                             "last_arm":arm,"last_density":density,"target_conditions":270})
                    print(f"COMPLETE shard={args.shard} conditions={len(completed)}/270 arm={arm} density={density}",flush=True)
                except BaseException as exc:
                    recorder.write("condition_error",question_id=row["question_id"],arm=arm,density=density,
                                   error_type=type(exc).__name__,error=str(exc))
                    raise
                finally:
                    QUESTION.reset(token)
                    if adapter is not None:
                        adapter.detach()
    save(out/"stage_finished.json",{"status":"complete","limit_questions":args.limit_questions,
                                    "completed_conditions":len(completed),"source_commit":CONFIG["author_commit"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--main-host",required=True)
    parser.add_argument("--merger-host",required=True)
    parser.add_argument("--limit-questions",type=int,default=0)
    asyncio.run(run(parser.parse_args()))
