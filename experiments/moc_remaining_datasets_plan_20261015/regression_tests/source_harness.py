"""Execute unchanged author AST bodies with offline dependency/service doubles.

This is a structural integration harness, not GPU/accuracy reproduction.
It does not substitute a handwritten execution loop or handwritten prompts.
"""

import ast
import asyncio
import itertools
import math
from pathlib import Path
import random
import shortuuid
import typing
from abc import ABC, abstractmethod
import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[3]
AUTHOR = WORKSPACE / "coding_base/MOC-official"
ASSETS = WORKSPACE / "research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014"


def load_symbols(path, names, namespace, *, methods=None, include_assignments=False):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    selected = []
    for statement in tree.body:
        if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in names:
            if isinstance(statement, ast.ClassDef):
                statement.decorator_list = []  # registries are environmental, not the tested body.
                if methods is not None:
                    statement.body = [m for m in statement.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in methods]
            selected.append(statement)
        elif include_assignments and isinstance(statement, (ast.Assign, ast.AnnAssign)):
            selected.append(statement)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class FakeLLM:
    def __init__(self, response="source reasoning\nThe answer is A"):
        self.response = response
        self.calls = []

    async def agen(self, messages, *args, **kwargs):
        self.calls.append(messages)
        await asyncio.sleep(0)
        return self.response


def author_namespace():
    namespace = {name: getattr(typing, name) for name in ("Any", "List", "Dict", "Optional", "Tuple", "Union")}
    namespace.update(ABC=ABC, abstractmethod=abstractmethod, asyncio=asyncio,
                     shortuuid=shortuuid, np=np, torch=torch, itertools=itertools,
                     math=math, random=random, PromptSet=object,
                     execute_code_get_return=lambda source: 42)
    load_symbols(AUTHOR / "src/prompt/aqua_prompt_set.py", {"AQUAPromptSet"}, namespace, include_assignments=True)
    prompt = namespace["AQUAPromptSet"]()

    class PromptRegistry:
        @staticmethod
        def get(domain):
            return prompt

    class LLMRegistry:
        @staticmethod
        def get(name):
            return FakeLLM()

    namespace.update(PromptSetRegistry=PromptRegistry, LLMRegistry=LLMRegistry)
    load_symbols(AUTHOR / "src/graph/node.py", {"Node"}, namespace)
    load_symbols(AUTHOR / "src/agents/math_solver.py", {"MathSolver"}, namespace)
    load_symbols(AUTHOR / "src/agents/final_decision.py", {"FinalRefer"}, namespace)
    methods = {"arun", "get_neighbors_by_hops", "update_memory", "connect_decision_node",
               "get_neighbor_summary_with_ism", "iterative_semantic_merging_with_clustering"}
    load_symbols(AUTHOR / "src/graph/graph.py", {"Graph", "min_max_norm"}, namespace, methods=methods)
    load_symbols(AUTHOR / "experiments/run_experiment.py", {"get_kwargs"}, namespace)
    load_symbols(AUTHOR / "experiments/run_direct.py", {"process_single_question"}, namespace)
    # Only matrix cosine/service backend doubles; sampler and author prompt/loop
    # bodies remain the actual parsed pinned sources.
    def cosine(a):
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        a = a / np.maximum(norm, 1e-12)
        return a @ a.T
    namespace["cosine_similarity"] = cosine
    return namespace


def fake_encode(text):
    # Explicit offline feature double, never claimed as MiniLM semantic output.
    return np.array([1.0, 0.2 if "Programming" in text else 0.1, 0.3, 0.05], dtype=np.float32)


def graph_fixture(count=5, *, dense=False, prefix="n", namespace=None):
    namespace = namespace or author_namespace()
    Graph, MathSolver, FinalRefer = (namespace[name] for name in ("Graph", "MathSolver", "FinalRefer"))
    graph = Graph()
    graph.optimized_spatial = graph.optimized_temporal = False
    graph.use_neighbor_summary = False
    graph.neighbor_hops, graph.ism_r, graph.ism_epsilon, graph.ism_kppa = 2, 1.0, 0.1, 45
    nodes = [MathSolver(id=f"{prefix}{i}", domain="aqua", llm_name="mock") for i in range(count)]
    for i, node in enumerate(nodes):
        if node.role == "Programming Expert":
            node.llm = FakeLLM("```python\nanswer = 42\n```")
        else:
            node.llm = FakeLLM(f"unique output for {node.id}\nThe answer is A")
        for j in range(i):
            if dense or j == i - 1:
                node.add_predecessor(nodes[j])
    graph.nodes = {n.id: n for n in nodes}
    graph.decision_node = FinalRefer(id=f"{prefix}final", domain="aqua", llm_name="mock")
    graph.construct_new_features = lambda task: torch.tensor([[float(i + 1), float(i % 2)] for i in range(count)])
    graph.gcn = lambda features, adjacency: features
    graph.mlp = lambda features: features
    graph.role_adj_matrix = None
    graph.construct_spatial_connection = lambda: 0
    graph.construct_temporal_connection = lambda round: 0
    graph.encode_text = fake_encode
    async def merge(messages, kppa):
        return "source reasoning\nThe answer is A"
    graph.merge_multiple_messages = merge
    return graph, namespace
