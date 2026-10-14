import asyncio
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from moc_adaptive_sampling import SamplingConfig, LayerDependentSampler, MOCSamplingAdapter
from moc_adaptive_sampling.sampler import Message, math_communication, serialize_messages
from moc_adaptive_sampling.kernels import ladies_probabilities, asgcn_probabilities
from moc_adaptive_sampling.features import LocalTokenizerCounter, pool_kv_sketch, pool_token_states
from moc_adaptive_sampling.paper_baselines import AuthorAPI, PaperRecipe, make_graph, run_single_llm
from source_harness import WORKSPACE, AUTHOR, ASSETS, load_symbols, graph_fixture, author_namespace, fake_encode, FakeLLM


def token_counter():
    return LocalTokenizerCounter(WORKSPACE / "experiments/moc_index_pull_20261013/tokenizer/tokenizer.json",
        "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922")


class KernelTests(unittest.TestCase):
    def test_ladies_proposal_against_executed_official_sampler(self):
        class SparseDouble:
            def __init__(self, values):
                self.values = np.asarray(values)
            def __getitem__(self, index):
                return SparseDouble(self.values[index])
            def multiply(self, other):
                return SparseDouble(self.values * (other.values if isinstance(other, SparseDouble) else other))
            def __array__(self, dtype=None, copy=None):
                return self.values.astype(dtype) if dtype is not None else self.values
            def sum(self, axis=None, out=None, **kwargs):
                return np.asmatrix(self.values.sum(axis=axis))
        class RecordingRandom:
            def seed(self, seed):
                self.rng = np.random.RandomState(seed)
            def choice(self, size, count, p, replace):
                self.p, self.replace = np.asarray(p), replace
                return self.rng.choice(size, count, p=p, replace=replace)
        class NumpyProxy:
            random = RecordingRandom()
            def __getattr__(self, attr):
                return getattr(np, attr)
        proxy = NumpyProxy()
        ns = {"np": proxy, "sparse_mx_to_torch_sparse_tensor": lambda x: x, "row_normalize": lambda x: x}
        load_symbols(ASSETS / "upstream/ladies/pytorch_ladies.py", {"ladies_sampler"}, ns)
        adjacency = np.array([[.5, .5, 0.], [0., .25, .75], [0., 0., 1.]])
        ns["ladies_sampler"](7, np.array([0, 1]), [2], 3, SparseDouble(adjacency), 1)
        np.testing.assert_allclose(ladies_probabilities(adjacency[[0, 1]]), proxy.random.p)
        self.assertFalse(proxy.random.replace)

    def test_asgcn_proposal_column_convention_and_numerics(self):
        parents = np.eye(2)
        candidates = np.array([[1, 2], [2, 0], [0, 1]])
        support = np.array([[.5, .5, 0], [0, .25, .75]])
        weights = np.array([[.2, .3], [.4, -.1]])
        mass = np.array([1.1, np.sqrt(.3125) * .8 * 1.4, .75 * .8 * 1.4])
        np.testing.assert_allclose(asgcn_probabilities(support, parents, candidates, weights, 2), mass / mass.sum())
        np.testing.assert_array_equal(ladies_probabilities(np.zeros((1, 3))), np.zeros(3))

    def test_hidden_and_kv_require_explicit_scope(self):
        states = np.array([[1., 0], [0, 2], [7, 7]])
        np.testing.assert_allclose(pool_token_states(states, [True, True, False]), [1/np.sqrt(5), 2/np.sqrt(5)])
        with self.assertRaises(ValueError):
            pool_kv_sketch(states.reshape(3, 1, 2), states.reshape(3, 1, 2), [True]*3, positions_aligned=False)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = token_counter()

    def test_real_token_budget_covers_envelope_and_labels(self):
        message = Message("source", "Inspector", 1, 0, "A", np.ones(4))
        body_cost = self.counter(message.text)
        required = self.counter(math_communication(serialize_messages([message])))
        self.assertGreater(required, body_cost)
        for budget, expected in [(body_cost, []), (required - 1, []), (required, ["source"])]:
            sampler = LayerDependentSampler(SamplingConfig(max_hops=1, frontier_fanouts=(1,), communication_tokens=budget), self.counter)
            result = sampler.select("target", np.ones(4), [message], {"target": ["source"]})
            self.assertEqual(result.source_ids, expected)
            self.assertLessEqual(result.communication_tokens, budget)

    def test_bodies_can_be_skipped_without_blocking_remote_sketches(self):
        messages = [Message("near", "Inspector", 1, 1, "long " * 500, np.ones(4)),
                    Message("remote", "Math Solver", 2, 0, "answer A", np.ones(4))]
        sampler = LayerDependentSampler(SamplingConfig(max_hops=2, frontier_fanouts=(1,1), communication_tokens=100), self.counter)
        result = sampler.select("receiver", np.ones(4), messages, {"receiver": ["near"], "near": ["remote"]})
        self.assertEqual(result.source_ids, ["remote"])
        self.assertEqual(result.layers[0]["routing_frontier"], ["near"])
        self.assertEqual(result.layers[0]["read"], [])

    def test_receiver_conditions_probability_and_novelty_stops(self):
        messages = [Message("x", "Inspector", 1, 0, "x", np.array([1.,0.])),
                    Message("y", "Inspector", 1, 1, "y", np.array([0.,1.]))]
        config = SamplingConfig(max_hops=1, frontier_fanouts=(1,))
        sampler = LayerDependentSampler(config, self.counter)
        p = sampler.select("r", np.array([1.,0.]), messages, {"r":["x","y"]}, "one").layers[0]["proposal"]
        p2 = sampler.select("r", np.array([0.,1.]), messages, {"r":["x","y"]}, "one").layers[0]["proposal"]
        self.assertGreater(p[0], p[1])
        self.assertGreater(p2[1], p2[0])
        chain = [Message("a", "Inspector", 1, 1, "a", np.array([1.,0.])),
                 Message("b", "Inspector", 2, 0, "b", np.array([1.,0.]))]
        stop_sampler = LayerDependentSampler(SamplingConfig(max_hops=2, frontier_fanouts=(1,1), redundancy_weight=1), self.counter)
        result = stop_sampler.select("r", np.array([1.,0.]), chain, {"r":["a"], "a":["b"]})
        self.assertEqual(result.source_ids, ["a"])
        self.assertEqual(result.stop_reason, "no_remaining_proxy_gain_within_budget")

    def test_real_token_cap_over_many_ragged_graphs(self):
        for seed in range(20):
            messages = [Message(str(i), "Inspector", i//3+1, 11-i,
                                "reason " * (i*13+1), fake_encode(str(i))) for i in range(12)]
            preds = {"r": [str(i) for i in range(3)]}
            for i in range(9):
                preds[str(i)] = [str(j) for j in range((i//3+1)*3, (i//3+2)*3)]
            budget = 80 + seed*21
            sampler = LayerDependentSampler(SamplingConfig(communication_tokens=budget, seed=seed), self.counter)
            result = sampler.select("r", fake_encode("task"), messages, preds)
            self.assertEqual(result.communication_tokens, self.counter(math_communication(result.context)))
            self.assertLessEqual(result.communication_tokens, budget)
            self.assertEqual(len(result.source_ids), len(set(result.source_ids)))
            self.assertEqual(result.source_ids, sorted(result.source_ids, key=lambda i: 11-int(i)))


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = token_counter()

    async def test_actual_author_arun_reaches_four_hops_and_final_unchanged(self):
        graph, _ = graph_fixture()
        sampler = LayerDependentSampler(SamplingConfig(), self.counter)
        adapter = MOCSamplingAdapter(graph, sampler, fake_encode, counter_name=self.counter.name, feature_space="OFFLINE_TEST_DOUBLE")
        with contextlib.redirect_stdout(io.StringIO()):
            await graph.arun({"task": "test task"})
        self.assertTrue(adapter.ledger["valid"])
        selection = adapter.ledger["selections"][-1]
        self.assertEqual([c["hop"] for c in selection["candidates"]], [4,3,2,1])
        self.assertEqual(selection["source_ids"], ["n0","n1","n2","n3"])
        final_prompt = graph.decision_node.llm._llm.calls[-1][1]["content"]
        for node in graph.nodes.values():
            self.assertIn(node.outputs[-1], final_prompt)
        programming = next(n for n in graph.nodes.values() if n.role == "Programming Expert")
        self.assertIn("the answer is 42", adapter._artifacts[programming.id][0])
        self.assertEqual(adapter.ledger["merger_llm_calls"], 0)
        self.assertEqual(adapter.ledger["encoding_calls"], {"sender":5,"receiver":5})
        self.assertEqual(len(adapter.ledger["llm_calls"]), 6)
        adapter.detach()
        self.assertNotIn("_moc_sampling_adapter", graph.__dict__)

    async def test_empty_selection_and_stale_outputs_do_not_leak(self):
        graph, _ = graph_fixture()
        for node in graph.nodes.values():
            node.outputs = ["STALE_PRIVATE_MESSAGE"]
        adapter = MOCSamplingAdapter(graph, LayerDependentSampler(SamplingConfig(communication_tokens=0), self.counter), fake_encode)
        with contextlib.redirect_stdout(io.StringIO()):
            await graph.arun({"task":"test task"})
        for node in graph.nodes.values():
            prompt = node.llm._llm.calls[-1][1]["content"]
            self.assertNotIn("Agent n", prompt)
            self.assertNotIn("STALE_PRIVATE_MESSAGE", prompt)
        self.assertTrue(all(s["source_ids"] == [] for s in adapter.ledger["selections"]))

    async def test_selector_failure_fails_closed_despite_author_catch(self):
        graph, _ = graph_fixture()
        def broken(text):
            raise ValueError("broken sketch")
        adapter = MOCSamplingAdapter(graph, LayerDependentSampler(SamplingConfig(), self.counter), broken)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            await graph.arun({"task":"test"}, max_tries=1)
        self.assertFalse(adapter.ledger["valid"])
        self.assertEqual(adapter.ledger["llm_calls"], [])

    async def test_instance_isolation_and_baseline_remains_vanilla(self):
        candidate, _ = graph_fixture(prefix="c")
        baseline, _ = graph_fixture(prefix="b")
        original_method = baseline.get_neighbor_summary_with_ism.__func__
        MOCSamplingAdapter(candidate, LayerDependentSampler(SamplingConfig(communication_tokens=0), self.counter), fake_encode)
        with contextlib.redirect_stdout(io.StringIO()):
            await asyncio.gather(candidate.arun({"task":"test"}), baseline.arun({"task":"test"}, num_rounds=1))
        self.assertIs(baseline.get_neighbor_summary_with_ism.__func__, original_method)
        self.assertIn("Agent b0", baseline.nodes["b1"].llm.calls[-1][1]["content"])
        self.assertNotIn("Agent c0", candidate.nodes["c1"].llm._llm.calls[-1][1]["content"])

    async def test_temporal_and_unadapted_code_paths_refused(self):
        graph, _ = graph_fixture()
        MOCSamplingAdapter(graph, LayerDependentSampler(SamplingConfig(), self.counter), fake_encode)
        with self.assertRaises(NotImplementedError):
            await graph.arun({"task":"test"}, num_rounds=2)
        code, _ = graph_fixture()
        next(iter(code.nodes.values())).agent_name = "CodeWriting"
        with self.assertRaises(NotImplementedError):
            MOCSamplingAdapter(code, LayerDependentSampler(SamplingConfig(), self.counter), fake_encode)

    async def test_author_bfs_uses_unique_shortest_hops(self):
        graph, _ = graph_fixture()
        graph.nodes["n4"].add_predecessor(graph.nodes["n0"])
        levels = graph.get_neighbors_by_hops("n4", 4)
        flattened = [(hop, n.id) for hop in sorted(levels) for n in levels[hop]]
        self.assertEqual(flattened, [(1,"n3"),(1,"n0"),(2,"n2"),(3,"n1")])
        self.assertEqual(len(flattened), len({node_id for hop, node_id in flattened}))


class BaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_llm_uses_original_direct_function_and_one_call(self):
        namespace = author_namespace()
        llm = FakeLLM("The answer is A")
        api = AuthorAPI(None, namespace["get_kwargs"], namespace["process_single_question"],
                        SimpleNamespace(get=lambda name: llm), namespace["PromptSetRegistry"])
        result = await run_single_llm(api, PaperRecipe(), {"task":"test","answer":"A"})
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][0]["content"], namespace["AQUAPromptSet"].get_direct_constraint(False))
        self.assertTrue(result["is_correct"])

    async def test_author_topology_and_baseline_settings_delegated(self):
        namespace = author_namespace()
        seen = []
        def constructor(**kwargs):
            seen.append(kwargs)
            return SimpleNamespace(**kwargs)
        api = AuthorAPI(constructor, namespace["get_kwargs"], None, None, None)
        recipe = PaperRecipe()
        vanilla = make_graph(api, recipe, "vanilla_mas")
        moc = make_graph(api, recipe, "moc")
        self.assertFalse(vanilla.use_neighbor_summary)
        self.assertTrue(moc.use_neighbor_summary)
        self.assertEqual(moc.neighbor_hops, 2)
        self.assertEqual(moc.ism_epsilon, .1)
        self.assertEqual(vanilla.fixed_spatial_masks, moc.fixed_spatial_masks)
        self.assertEqual(vanilla.fixed_spatial_masks, namespace["get_kwargs"]("Random", 7, .5, 42)["fixed_spatial_masks"])
        without_phi = make_graph(api, recipe, "moc_without_phi")
        rendered = await without_phi.iterative_semantic_merging_with_clustering([(3,"a","Inspector","old")])
        self.assertEqual(rendered, "Agent a,role is Inspector,output is:\nold")

    async def test_moc_still_executes_original_iterative_merging(self):
        graph, _ = graph_fixture(count=7, dense=True)
        graph.use_neighbor_summary = True
        merges = []
        original = graph.merge_multiple_messages
        async def counted_merge(messages, kppa):
            merges.append((messages, kppa))
            return await original(messages, kppa)
        graph.merge_multiple_messages = counted_merge
        with contextlib.redirect_stdout(io.StringIO()):
            await graph.arun({"task":"test"}, num_rounds=1)
        self.assertGreater(len(merges), 0)
        self.assertTrue(all(kppa == 45 for messages, kppa in merges))
        self.assertFalse(hasattr(graph, "_moc_sampling_adapter"))


if __name__ == "__main__":
    unittest.main()
