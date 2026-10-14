"""Fixed-material, fixed-role paired receiver interventions on official train.

Training gold is read only after inference for local correctness differences.
All original prompts/tool behavior/decoding are shared within each pair.
"""
import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time

from trace_hooks import Observer, Recorder, QUESTION
from runtime_controls import stable_ids, backend_routes, effective_options, prompt_and_tool_observers
from training_integrity import inspect_calls

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
sys.path.insert(0, CONFIG["extension_directory"])
from moc_adaptive_sampling.sampler import Message, serialize_messages, unit_vector
from moc_adaptive_sampling.stateful import utility_features, structural_priors, count_context
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.paper_baselines import PaperRecipe, load_author_api, make_graph

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temp.replace(path)

def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")

def task(row):
    return "Question: " + row["question"] + "\n" + "".join(o+"\n" for o in row["options"])

async def run(args):
    api = load_author_api(CONFIG["author_directory"])
    import numpy as np
    import torch
    import ollama
    from src.llm.profile_embedding import get_sentence_embedding
    gm, nm = importlib.import_module("src.graph.graph"), importlib.import_module("src.graph.node")
    math_module = importlib.import_module("src.agents.math_solver")
    final = importlib.import_module("src.agents.final_decision").FinalRefer
    counter = LocalTokenizerCounter(CONFIG["tokenizer_path"], CONFIG["tokenizer_sha256"])
    budget = CONFIG["candidate"]["communication_tokens"]
    inputs = rows(ROOT / "data/train_inputs.jsonl")
    assert len(inputs) == CONFIG["training_questions"]
    gold = {x["question_id"]: x["correct"] for x in rows(ROOT / "data/train_labels.jsonl")}
    ordered_ids = [x["question_id"] for x in inputs]
    inputs = [r for i, r in enumerate(inputs) if i % CONFIG["workers"] == args.shard]
    out = ROOT / "training" / f"shard{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(out)
    if (out / "trace.jsonl").exists():
        recorder.counter = max(e["event_index"] for e in rows(out / "trace.jsonl"))
    completed = {x["question_id"] for x in rows(out / "questions.jsonl")} if (out / "questions.jsonl").exists() else set()
    excluded = {x["question_id"] for x in rows(out / "excluded_questions.jsonl")} if (out / "excluded_questions.jsonl").exists() else set()
    if (out / "labels.jsonl").exists() and not completed:
        raise RuntimeError("partial training labels without complete questions; use a new isolated run")
    recipe = PaperRecipe(llm_name=CONFIG["main_model"], edge_density=.3, use_cot=CONFIG["use_cot"],
                         random_dag_seed=CONFIG["seed"], ism_epsilon=CONFIG["ism_epsilon"])
    question_map = {task(r):r["question_id"] for r in inputs}
    with ExitStack() as stack:
        stack.enter_context(backend_routes(ollama, args.main_host, args.merger_host))
        stack.enter_context(Observer(recorder, gm, nm, ollama, question_map))
        stack.enter_context(effective_options(ollama, CONFIG, counter))
        stack.enter_context(prompt_and_tool_observers(math_module, final, recorder, counter, CONFIG["code_timeout_seconds"]))
        with stable_ids():
            role_graph = make_graph(api, recipe, "vanilla_mas")
        roles = [n.role for n in role_graph.nodes.values()]
        for row in inputs:
            qid = row["question_id"]
            if qid in completed or qid in excluded:
                continue
            if (ROOT / "STOP").exists():
                raise RuntimeError("STOP requested")
            index = ordered_ids.index(qid)
            density = CONFIG["densities"][index % len(CONFIG["densities"])]
            random.seed(CONFIG["seed"])
            np.random.seed(CONFIG["seed"])
            torch.manual_seed(CONFIG["seed"])
            question = {"task":task(row)}
            token = QUESTION.set(qid)
            event_start, started = len(recorder.records), time.monotonic()
            pending_labels=[]

            def exclude_limited(limited,usage):
                item={"question_id":qid,"question_index":index,"status":"invalid_output_limit",
                      "reason":"at least one train generation stopped at the fixed output cap; all pairs for this question excluded",
                      "output_limit_call_events":[e["event_index"] for e in limited],
                      "seconds":time.monotonic()-started,**usage}
                append(out/"excluded_questions.jsonl",item)
                recorder.write("training_question_excluded",**item)
                excluded.add(qid)
                save(out/"progress.json",{"completed_questions":len(completed),"excluded_questions":len(excluded),
                     "processed_questions":len(completed|excluded),"total_questions":len(inputs),"last_question":qid})
                print(f"TRAIN_EXCLUDED shard={args.shard} question={qid} reason=output_limit",flush=True)

            try:
                with stable_ids():
                    graph = make_graph(api, replace(recipe, edge_density=density), "vanilla_mas", roles=roles)
                await graph.arun(question, num_rounds=1, max_tries=1, max_time=300)
                _,limited,usage=inspect_calls(recorder.records[event_start:],CONFIG["main_model"])
                if limited:
                    exclude_limited(limited,usage)
                    continue
                predecessors = {i: tuple(p.id for p in n.spatial_predecessors) for i,n in graph.nodes.items()}
                sketches = {i:unit_vector(get_sentence_embedding(str(n.outputs[-1]))) for i,n in graph.nodes.items()}
                materials = {i:{"role":n.role,"text":str(n.outputs[-1]),"sketch":sketches[i].tolist()} for i,n in graph.nodes.items()}
                append(out / "materials.jsonl", {"question_id":qid,"density":density,"task":question["task"],
                                               "predecessors":predecessors,"materials":materials})
                pairs = 0
                for receiver_index in CONFIG["training_receivers"]:
                    node = list(graph.nodes.values())[receiver_index]
                    levels = graph.get_neighbors_by_hops(node.id, 4)
                    messages = [Message(n.id,n.role,hop,order,str(n.outputs[-1]),sketches[n.id])
                                for order,(hop,n) in enumerate((h,n) for h in sorted(levels, reverse=True) for n in levels[h])]
                    if not messages:
                        continue
                    priors,_ = structural_priors(node.id, messages, predecessors)
                    base = unit_vector(get_sentence_embedding(f"Task:\n{question['task']}\nReceiver role:\n{node.role}"))
                    rng = np.random.default_rng(int.from_bytes(hashlib.sha256((qid+str(receiver_index)).encode()).digest()[:8], "little"))
                    states = [[]]
                    if len(messages) > 1:
                        amount = min(len(messages)-1, 1 + index % 3)
                        drawn = rng.choice(len(messages), size=amount, replace=False)
                        context = [messages[int(i)] for i in drawn]
                        while context and count_context(counter, context) > budget:
                            context.pop()
                        if context:
                            states.append(context)
                    cache = {}
                    async def evaluate(selection):
                        key = tuple(sorted(m.source for m in selection))
                        if key not in cache:
                            text = await node._async_execute(question, {}, {}, serialize_messages(selection))
                            parsed = node.prompt_set.postprocess_answer(node.prompt_set, text)
                            cache[key] = {"answer":parsed, "output":text,
                                          "input_sha256":hashlib.sha256(serialize_messages(selection).encode()).hexdigest()}
                        return cache[key]
                    for state_index, selected in enumerate(states):
                        candidates = [m for m in messages if m.source not in {a.source for a in selected}
                                      and count_context(counter,[*selected,m]) <= budget]
                        if not candidates:
                            continue
                        # Fixed, label-blind subset: at most two candidates per state.
                        perm = rng.permutation(len(candidates))[:2]
                        baseline = await evaluate(selected)
                        for position in perm:
                            candidate = candidates[int(position)]
                            augmented = await evaluate([*selected,candidate])
                            features, info = utility_features(base, selected, candidate, messages, priors,
                                                               node.role, counter, budget)
                            label = int(augmented["answer"] == gold[qid]) - int(baseline["answer"] == gold[qid])
                            item = {"question_id":qid, "question_index":index, "receiver_id":node.id,
                                    "receiver_index":receiver_index,"receiver_role":node.role,"density":density,
                                    "state_index":state_index,"selected_sources":[m.source for m in selected],
                                    "candidate_source":candidate.source,"candidate_hop":candidate.hop,
                                    "features":features.tolist(),"feature_metadata":info,"label":label,
                                    "baseline":baseline,"augmented":augmented,
                                    "label_scope":"fixed-material local receiver answer, not downstream graph utility"}
                            pending_labels.append(item)
                            pairs += 1
                calls,limited,usage=inspect_calls(recorder.records[event_start:],CONFIG["main_model"])
                if limited:
                    exclude_limited(limited,usage)
                    continue
                # Commit only after all calls for this fixed-material question
                # pass validity. A failed/abandoned question never trains the model.
                for item in pending_labels:
                    append(out/"labels.jsonl",item)
                    recorder.write("controlled_pair",**item)
                item = {"question_id":qid,"question_index":index,"paired_labels":pairs,"main_calls":len(calls),
                        "prompt_tokens":sum(e["response"]["prompt_eval_count"] for e in calls),
                        "completion_tokens":sum(e["response"]["eval_count"] for e in calls),
                        "seconds":time.monotonic()-started,"status":"ok"}
                append(out / "questions.jsonl",item)
                completed.add(qid)
                save(out / "progress.json",{"completed_questions":len(completed),"excluded_questions":len(excluded),
                     "processed_questions":len(completed|excluded),"total_questions":len(inputs),"last_question":qid})
                print(f"TRAIN shard={args.shard} questions={len(completed)}/{len(inputs)} labels={pairs}",flush=True)
            finally:
                QUESTION.reset(token)
    assert len(completed|excluded)==len(inputs)
    save(out / "finished.json",{"status":"complete","completed_questions":len(completed),
         "excluded_questions":len(excluded),"processed_questions":len(completed|excluded)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,choices=(0,1),required=True)
    parser.add_argument("--main-host",required=True)
    parser.add_argument("--merger-host",required=True)
    asyncio.run(run(parser.parse_args()))
