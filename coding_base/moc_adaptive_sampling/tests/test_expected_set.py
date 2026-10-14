import asyncio
import contextlib
import io
import unittest
import numpy as np

from moc_adaptive_sampling.sampler import Message
from moc_adaptive_sampling.expected_set import ExpectedSetConfig, ExpectedSetSampler


def messages():
    return [Message('far', 'Math Solver', 4, 0, 'longer remote evidence', np.array([1.,0.])),
            Message('near', 'Mathematical Analyst', 1, 1, 'short fact', np.array([0.,1.]))]


class Model:
    def __init__(self, gain, supported=lambda info:True, spread=0.):
        self.gain, self.supported, self.spread = gain, supported, spread
        self.infos = []

    def predict_expected(self, x, info):
        self.infos.append(info)
        mean = self.gain(info)
        return dict(help=max(0.,mean), harm=max(0.,-mean), mean=mean,
                    supported=self.supported(info), bootstrap_spread=self.spread)


class ExpectedSetTests(unittest.TestCase):
    def select(self, model, budget=1024):
        return ExpectedSetSampler(ExpectedSetConfig(communication_tokens=budget), len, model).select(
            'r', [1.,1.], messages(), {'r':['near'], 'near':['far']}, receiver_role='Inspector')

    def test_total_gain_can_choose_pair_even_if_singleton_gain_per_token_higher(self):
        model = Model(lambda info:.65 if info['group_size']==2 else .6)
        result = self.select(model)
        self.assertEqual(result.source_ids, ['far','near'])
        self.assertEqual(sum(e['action']=='read' for e in result.layers), 1)
        self.assertTrue(all(i['selected_count']==0 for i in model.infos))
        pair = next(r for r in result.layers[0]['scores'] if r['group_size']==2)
        single = next(r for r in result.layers[0]['scores'] if r['group_size']==1)
        self.assertLess(pair['net_mean']/pair['projected_tokens'], single['net_mean']/single['projected_tokens'])
        self.assertLessEqual(result.communication_tokens, 1024)

    def test_high_bootstrap_spread_does_not_force_reads(self):
        result = self.select(Model(lambda info:-.1, spread=999.))
        self.assertEqual(result.source_ids, [])
        self.assertEqual(result.context, '')
        self.assertEqual(result.stop_reason, 'expected_net_utility_nonpositive')
        self.assertFalse(result.layers[0]['uncertainty_used_for_decision'])

    def test_unsupported_high_mean_cannot_authorize_optimistic_read(self):
        model = Model(lambda info:.8 if info['group_size']==2 else -.1,
                      supported=lambda info:info['group_size']==1)
        result = self.select(model)
        self.assertEqual(result.source_ids, [])
        self.assertTrue(any(r['abstained'] for r in result.layers[0]['scores']))
        unsupported = self.select(Model(lambda info:.8, supported=lambda info:False))
        self.assertEqual(unsupported.stop_reason, 'no_supported_expected_action')

    def test_cost_can_make_positive_gain_less_attractive_than_empty(self):
        result = self.select(Model(lambda info:.0001))
        self.assertEqual(result.source_ids, [])
        self.assertTrue(all(r['mean']>0 and r['net_mean']<0 for r in result.layers[0]['scores']))

    def test_zero_budget_and_unsupported_larger_groups(self):
        result = self.select(Model(lambda info:.9), budget=0)
        self.assertEqual(result.communication_tokens, 0)
        self.assertEqual(result.stop_reason, 'no_group_fits_budget')
        with self.assertRaises(ValueError):
            ExpectedSetConfig(max_group_size=3)

    def test_exact_budget_and_deterministic_author_order(self):
        model = Model(lambda info:.8 if info['group_size']==2 else -.5)
        initial = self.select(model)
        capped = self.select(model, budget=initial.communication_tokens)
        self.assertEqual(capped.source_ids, initial.source_ids)
        smaller = self.select(model, budget=initial.communication_tokens-1)
        self.assertEqual(smaller.source_ids, [])
        self.assertLess(capped.context.index('Agent far'), capped.context.index('Agent near'))

    def test_real_author_arun_and_final_still_use_all_node_outputs(self):
        from source_harness import graph_fixture, fake_encode
        from moc_adaptive_sampling.adapter import MOCSamplingAdapter
        for mean in (-.1, .8):
            graph,_ = graph_fixture()
            sampler = ExpectedSetSampler(ExpectedSetConfig(communication_tokens=4000), len, Model(lambda info:mean))
            adapter = MOCSamplingAdapter(graph, sampler, fake_encode)
            with contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(graph.arun({'task':'one-shot integration'}, num_rounds=1, max_tries=1))
            self.assertTrue(adapter.ledger['valid'])
            self.assertEqual(len(adapter.ledger['llm_calls']), len(graph.nodes)+1)
            self.assertEqual(len(graph.decision_node.spatial_predecessors), len(graph.nodes))
            if mean<0:
                self.assertFalse(any(e['source_ids'] for e in adapter.ledger['selections']))
            else:
                self.assertTrue(any(e['source_ids'] for e in adapter.ledger['selections']))
            adapter.detach()


if __name__=='__main__':
    unittest.main()
