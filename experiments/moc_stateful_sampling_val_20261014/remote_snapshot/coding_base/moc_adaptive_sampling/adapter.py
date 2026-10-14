"""Per-graph adapter; leave the pinned author's files and baseline instances intact."""

from dataclasses import asdict
from types import MethodType
import hashlib
import time
import numpy as np

from .sampler import Message, LayerDependentSampler, unit_vector


class _MeteredLLM:
    def __init__(self, llm, adapter, source, stage):
        self._llm, self._adapter, self._source, self._stage = llm, adapter, source, stage

    def __getattr__(self, name):
        return getattr(self._llm, name)

    async def agen(self, messages, *args, **kwargs):
        adapter = self._adapter
        record = {"source": self._source, "stage": self._stage,
                  "prompt_text_tokens": sum(adapter.sampler.count_tokens(m["content"]) for m in messages)}
        adapter.ledger["llm_calls"].append(record)
        start = time.perf_counter()
        try:
            response = await self._llm.agen(messages, *args, **kwargs)
            record["completion_text_tokens"] = adapter.sampler.count_tokens(str(response))
            return response
        except BaseException as exc:
            record["error_type"] = type(exc).__name__
            raise
        finally:
            record["seconds"] = time.perf_counter() - start


class MOCSamplingAdapter:
    """Install on one author Graph; call its original arun, Node, and MathSolver.

    v0 supports one-round MathSolver graphs and FinalRefer. CodeWriting and
    Wiki/tool-dependent AnalyzeAgent require their own audited adapters; refuse
    them here rather than silently change early-return or tool semantics.
    """

    def __init__(self, graph, sampler: LayerDependentSampler, encode, *,
                 feature_space="MOC:all-MiniLM-L6-v2", counter_name="injected-text-counter"):
        if getattr(graph, "_moc_sampling_adapter", None) is not None:
            raise ValueError("this graph already has a sampling adapter")
        if any(getattr(node, "agent_name", None) != "MathSolver" for node in graph.nodes.values()):
            raise NotImplementedError("v0 supports MathSolver; CodeWriting/AnalyzeAgent tool paths are not adapted")
        if getattr(graph.decision_node, "agent_name", None) != "FinalRefer":
            raise NotImplementedError("v0 preserves the author's FinalRefer aggregation")
        if getattr(graph, "optimized_spatial", False) or getattr(graph, "optimized_temporal", False):
            raise NotImplementedError("v0 uses the author's fixed DAG topology")
        self.graph, self.sampler, self.encode = graph, sampler, encode
        self.feature_space, self.counter_name = feature_space, counter_name
        self._saved = []
        self._running = False
        self._artifacts = {}
        self._selections = {}
        self._failures = []
        self.ledger = {}
        self._install()

    def _replace(self, obj, attr, value):
        self._saved.append((obj, attr, attr in obj.__dict__, obj.__dict__.get(attr)))
        setattr(obj, attr, value)

    def _vector(self, text, category):
        start = time.perf_counter()
        vector = unit_vector(self.encode(text)).astype(np.float32)
        self.ledger["encoding_seconds"] += time.perf_counter() - start
        self.ledger["encoding_input_tokens"] += self.sampler.count_tokens(text)
        self.ledger["encoding_calls"][category] += 1
        self.ledger["sketch_bytes_generated"] += vector.nbytes
        return vector

    async def _select(self, node_id, query):
        if not self._running or self._failures:
            raise RuntimeError("sampling must run inside a healthy, one-round Graph.arun")
        graph = self.graph
        node = graph.nodes[node_id]
        levels = graph.get_neighbors_by_hops(node_id, self.sampler.config.max_hops)
        messages = []
        for hop in sorted(levels, reverse=True):
            for ancestor in levels[hop]:
                artifact = self._artifacts.get(ancestor.id)
                if artifact is None:
                    continue  # stale outputs, unfinished nodes, and future nodes are inaccessible.
                text, sketch = artifact
                messages.append(Message(ancestor.id, ancestor.role, hop, len(messages), text, sketch))
        base = self._vector(f"Task:\n{query}\nReceiver role:\n{node.role}", "receiver")
        local = [self._artifacts[p.id][1] for p in node.spatial_predecessors if p.id in self._artifacts]
        state = (base if getattr(self.sampler, "uses_selected_state", False) else
                 unit_vector(base + self.sampler.config.local_state_weight * np.mean(local, axis=0)) if local else base)
        predecessors = {i: tuple(p.id for p in n.spatial_predecessors if p.id in graph.nodes)
                        for i, n in graph.nodes.items()}
        stable_index = list(graph.nodes).index(node_id)
        start = time.perf_counter()
        selection_kwargs = {"request_key": f"{self.ledger['task_sha256']}:{stable_index}"}
        if getattr(self.sampler, "uses_selected_state", False):
            selection_kwargs["receiver_role"] = node.role
        result = self.sampler.select(node_id, state, messages, predecessors, **selection_kwargs)
        self.ledger["routing_seconds"] += time.perf_counter() - start
        # Logical bytes made available to this router, not physical network measurement.
        self.ledger["routing_sketch_bytes_read"] += sum(m.sketch.nbytes for m in messages) + base.nbytes
        self._selections[node_id] = result
        entry = asdict(result)
        entry.pop("context")
        entry.update(receiver=node_id, receiver_index=stable_index,
                     candidates=[{"source": m.source, "hop": m.hop,
                                  "author_order": m.author_order,
                                  "body_tokens": self.sampler.count_tokens(m.text)} for m in messages])
        self.ledger["selections"].append(entry)
        return result.context

    def _install(self):
        graph = self.graph
        adapter = self
        original_arun = graph.arun

        async def summary(_graph, node_id, query):
            try:
                return await adapter._select(node_id, query)
            except BaseException as exc:
                adapter._failures.append(f"selection:{node_id}:{type(exc).__name__}:{exc}")
                raise

        self._replace(graph, "get_neighbor_summary_with_ism", MethodType(summary, graph))
        self._replace(graph, "_moc_sampling_adapter", self)
        for node in graph.nodes.values():
            original_inner = node._async_execute
            original_outer = node.async_execute

            def make_inner(original, node_id):
                async def inner(_node, input, spatial_info, temporal_info, neighbor_summary="", **kwargs):
                    result = adapter._selections.get(node_id)
                    if adapter._failures or result is None or result.context != neighbor_summary:
                        raise RuntimeError("selection failed; raw-message fallback is forbidden")
                    # An empty selection remains empty in the original MathSolver.
                    return await original(input, {}, {}, result.context, **kwargs)
                return inner

            def make_outer(original, node_id):
                async def outer(_node, input, **kwargs):
                    if not adapter._running or kwargs.get("graph") is not graph:
                        raise RuntimeError("adapted nodes require their owning Graph.arun")
                    adapter._selections.pop(node_id, None)
                    try:
                        result = await original(input, **kwargs)
                        if not _node.outputs:
                            raise RuntimeError("node completed without an output")
                        text = str(_node.outputs[-1])
                        # Runs after Programming Expert appends its execution result.
                        adapter._artifacts[node_id] = (text, adapter._vector(text, "sender"))
                        return result
                    except BaseException as exc:
                        adapter._failures.append(f"node:{node_id}:{type(exc).__name__}:{exc}")
                        raise
                return outer

            self._replace(node, "_async_execute", MethodType(make_inner(original_inner, node.id), node))
            self._replace(node, "async_execute", MethodType(make_outer(original_outer, node.id), node))
            self._replace(node, "llm", _MeteredLLM(node.llm, self, node.id, "ordinary"))

        final = graph.decision_node
        original_final = final._async_execute

        async def final_inner(_node, *args, **kwargs):
            if adapter._failures or len(adapter._artifacts) != len(graph.nodes):
                raise RuntimeError("ordinary-node failure; final aggregation will not consume partial results")
            # The author still receives ALL ordinary outputs at FinalRefer.
            return await original_final(*args, **kwargs)

        self._replace(final, "_async_execute", MethodType(final_inner, final))
        self._replace(final, "llm", _MeteredLLM(final.llm, self, final.id, "final"))

        async def arun(_graph, input, num_rounds=1, max_tries=3, max_time=600):
            if num_rounds != 1:
                raise NotImplementedError("temporal/multi-round communication is outside v0")
            if adapter._running:
                raise RuntimeError("concurrent requests must use separate Graph instances")
            if not isinstance(input, dict) or not isinstance(input.get("task"), str):
                raise ValueError("expected the author's task dictionary")
            adapter._running = True
            adapter._artifacts, adapter._selections, adapter._failures = {}, {}, []
            adapter.ledger = {
                "task_sha256": hashlib.sha256(input["task"].encode("utf-8")).hexdigest(),
                "config": asdict(adapter.sampler.config), "feature_space": adapter.feature_space,
                "counter": adapter.counter_name,
                "token_scope": "text content only; chat-template overhead excluded; backend usage required for total cost claims",
                "provider_usage_observed": False, "merger_llm_calls": 0,
                "encoding_calls": {"sender": 0, "receiver": 0}, "encoding_seconds": 0.0,
                "encoding_input_tokens": 0, "sketch_bytes_generated": 0,
                "routing_sketch_bytes_read": 0, "routing_seconds": 0.0,
                "selections": [], "llm_calls": [], "valid": False,
            }
            for node in [*graph.nodes.values(), graph.decision_node]:
                node.outputs = []
            try:
                result = await original_arun(input, num_rounds=1, max_tries=max_tries, max_time=max_time)
                if adapter._failures:
                    raise RuntimeError("; ".join(adapter._failures))
                adapter.ledger["valid"] = True
                return result
            finally:
                adapter.ledger["failures"] = adapter._failures.copy()
                adapter._running = False

        self._replace(graph, "arun", MethodType(arun, graph))

    def detach(self):
        if self._running:
            raise RuntimeError("cannot detach during a request")
        for obj, attr, existed, old in reversed(self._saved):
            if existed:
                setattr(obj, attr, old)
            else:
                delattr(obj, attr)
        self._saved.clear()
