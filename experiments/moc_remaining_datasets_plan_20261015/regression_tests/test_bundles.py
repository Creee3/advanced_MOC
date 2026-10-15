import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from moc_adaptive_sampling.sampler import Message
from moc_adaptive_sampling.bundles import (BundleConfig, BundleSampler, EvidenceGroup,
    GROUP_FEATURE_NAMES, evidence_groups, group_features)
from moc_adaptive_sampling.counterfactual import (ExactCallCache, request_key,
    replay_intervention, topological_order, factorial_effects)
from moc_adaptive_sampling.uncertainty import GroupUtilityModel


def inventory():
    return [Message('far','Math Solver',4,0,'remote fact',np.array([1.,0.])),
            Message('bridge','Mathematical Analyst',3,1,'supporting premise',np.array([0.,1.])),
            Message('near','Inspector',1,2,'near statement',np.array([1.,1.]))]


class GroupTests(unittest.TestCase):
    def test_original_graph_loop_and_final_aggregation_accept_group_selector(self):
        from source_harness import graph_fixture,fake_encode
        from moc_adaptive_sampling.adapter import MOCSamplingAdapter
        graph,_=graph_fixture()
        class Model:
            def predict_interval(self,x,info):
                return dict(help=.7,harm=0.,mean=.7,lower=.4,upper=.9,supported=True)
        adapter=MOCSamplingAdapter(graph,BundleSampler(BundleConfig(communication_tokens=4000),len,Model()),fake_encode)
        with contextlib.redirect_stdout(io.StringIO()):
            result=asyncio.run(graph.arun({'task':'group integration check'},num_rounds=1,max_tries=1))
        self.assertTrue(adapter.ledger['valid'])
        self.assertEqual(len(adapter.ledger['llm_calls']),len(graph.nodes)+1)
        self.assertTrue(any(s['source_ids'] for s in adapter.ledger['selections']))
        self.assertEqual(len(graph.decision_node.spatial_predecessors),len(graph.nodes))
        adapter.detach()

    def test_paths_keep_singletons_and_do_not_require_all_intermediates(self):
        groups=evidence_groups(inventory(), {'bridge':['far'],'near':['bridge']},2)
        sources=[g.sources for g in groups]
        self.assertIn(('far',),sources)
        self.assertIn(('far','bridge'),sources)
        self.assertNotIn(('far','near'),sources)
        self.assertTrue(all(len(g.sources)<=2 for g in groups))

    def test_group_budget_counts_union_and_recomputes_selected_state(self):
        m=inventory(); count=len
        group=EvidenceGroup(('far','bridge'),('far','bridge'))
        x,empty=group_features([1.,1.],[],group,m,{}, {'bridge':['far']},'Inspector',count,1000)
        y,one=group_features([1.,1.],[m[0]],group,m,{}, {'bridge':['far']},'Inspector',count,1000)
        self.assertEqual(empty['group_size'],2)
        self.assertEqual(one['added_sources'],['bridge'])
        self.assertEqual(empty['projected_tokens'],one['projected_tokens'])
        self.assertFalse(np.array_equal(x,y))

    def test_selects_complementary_group_and_preserves_original_order(self):
        class Model:
            def predict_interval(self,x,info):
                mean=.8 if info['group_size']==2 else -.2
                return dict(help=max(mean,0),harm=max(-mean,0),mean=mean,lower=mean,upper=mean,supported=True)
        m=inventory()[:2]
        result=BundleSampler(BundleConfig(communication_tokens=1000),len,Model()).select(
            'r',[1.,1.],m,{'r':['bridge'],'bridge':['far']},receiver_role='Inspector')
        self.assertEqual(result.source_ids,['far','bridge'])
        self.assertIn('Agent far',result.context)
        self.assertEqual(sum(e['action']=='read' for e in result.layers),1)
        self.assertLessEqual(result.communication_tokens,1000)

    def test_unsupported_prediction_cannot_certify_stop(self):
        class Model:
            def predict_interval(self,x,info):
                return dict(help=0.,harm=.5,mean=-.5,lower=-.6,upper=-.4,supported=False)
        result=BundleSampler(BundleConfig(communication_tokens=1000),len,Model()).select(
            'r',[1.,1.],inventory()[:1],{'r':['far']},receiver_role='Inspector')
        self.assertEqual(result.source_ids,['far'])
        self.assertEqual(next(e for e in result.layers if e['action']=='read')['selection_mode'],'uncertain_optimistic')

    def test_supported_negative_bounds_allow_empty_context(self):
        class Model:
            def predict_interval(self,x,info):
                return dict(help=0.,harm=.5,mean=-.5,lower=-.7,upper=-.2,supported=True)
        result=BundleSampler(BundleConfig(communication_tokens=1000),len,Model()).select(
            'r',[1.,0.],inventory()[:1],{'r':['far']})
        self.assertEqual(result.context,'')
        self.assertEqual(result.stop_reason,'surrogate_upper_bound_nonpositive')


class ReplayTests(unittest.TestCase):
    def test_shuffled_ids_follow_actual_dag_and_only_recompute_descendants(self):
        material={'predecessors':{'tail':['middle'],'untouched':[],'middle':['root'],'root':[]},
                  'materials':{i:{'text':i+' old'} for i in ('tail','untouched','middle','root')}}
        called=[]
        async def execute(node,outputs,sources):
            called.append((node,sources))
            return node+' new'
        async def final(outputs):
            self.assertEqual(outputs['untouched'],'untouched old')
            return outputs['tail']
        result=asyncio.run(replay_intervention(material,'middle',[],execute,final))
        self.assertEqual(called,[('middle',[]),('tail',None)])
        self.assertEqual(result['final_output'],'tail new')
        self.assertEqual(material['materials']['tail']['text'],'tail old')
        with self.assertRaises(ValueError):
            asyncio.run(replay_intervention(material,'middle',['tail'],execute,final))
        with self.assertRaises(ValueError):
            topological_order({'a':['b'],'b':['a']})

    def test_cache_reuses_exact_calls_and_accounts_virtual_vs_new(self):
        request={'model':'m','messages':[{'role':'user','content':'same'}], 'stream':False,'options':{'seed':42}}
        response={'done':True,'done_reason':'stop','eval_count':2,'prompt_eval_count':5,'message':{'content':'A'}}
        calls=[]
        async def generate(req):
            calls.append(req)
            return response
        cache=ExactCallCache('digest',generate)
        cache.add(dict(request,args=[]),response,{'origin':'historical'})
        asyncio.run(cache.call(request))
        self.assertEqual(calls,[])
        changed=dict(request,options={'seed':43})
        asyncio.run(cache.call(changed))
        self.assertEqual(cache.totals()['new_model_tokens'],7)
        self.assertEqual(cache.totals()['logical_model_tokens'],14)
        self.assertNotEqual(request_key(request,'digest'),request_key(request,'other-digest'))
        with self.assertRaises(ValueError):
            cache.add(request,dict(response,message={'content':'B'}),{})

    def test_truncated_call_is_logged_and_costed_but_not_cached(self):
        request={'model':'m','messages':[], 'stream':False,'options':{}}
        async def generate(req):
            return {'done':True,'done_reason':'length','eval_count':2000,'prompt_eval_count':5,'message':{'content':'partial'}}
        with tempfile.TemporaryDirectory() as path:
            cache=ExactCallCache('digest',generate,Path(path)/'raw.jsonl')
            with self.assertRaises(ValueError):
                asyncio.run(cache.call(request))
            self.assertEqual(cache.totals()['new_model_tokens'],2005)
            self.assertFalse(cache.entries)
            self.assertEqual(len((Path(path)/'raw.jsonl').read_text().splitlines()),1)

    def test_ambiguous_history_is_excluded_without_outcome_selection(self):
        request={'model':'m','messages':[], 'stream':False,'options':{}}
        response={'done':True,'done_reason':'stop','eval_count':2,'prompt_eval_count':5,'message':{'content':'A'}}
        cache=ExactCallCache('digest',None)
        report=cache.load_historical([dict(request=request,response=response,provenance={'event':1}),
                                      dict(request=request,response=dict(response,message={'content':'B'}),provenance={'event':2})])
        self.assertEqual(report['conflicting_requests'],1)
        self.assertEqual(report['reusable_requests'],0)
        self.assertFalse(cache.entries)

    def test_factorial_contrasts_distinguish_joint_only_help_and_harm(self):
        self.assertTrue(factorial_effects(dict(empty=0,a=0,b=0,ab=1))['bundle_only_success'])
        self.assertTrue(factorial_effects(dict(empty=1,a=1,b=1,ab=0))['bundle_only_harm'])
        self.assertFalse(factorial_effects(dict(empty=0,a=1,b=0,ab=1))['bundle_only_success'])


class UncertaintyTests(unittest.TestCase):
    def test_group_bootstrap_calibration_separates_questions_and_reports_small_sample(self):
        width=len(GROUP_FEATURE_NAMES)
        x=np.zeros((12,width)); x[:,0]=np.tile([-1,0,1],4)
        labels=np.tile([-1,0,1],4)
        ids=[f'q{i//2}' for i in range(12)]
        info=dict(receiver_role='Inspector',max_hop=1,group_size=1,selected_count=0)
        model=GroupUtilityModel.fit(x,labels,ids,[info]*12,bootstrap_replicates=8)
        with self.assertRaises(ValueError):
            model.calibrate(x[:2],labels[:2],ids[:2],[info]*2)
        model.calibrate(x[:4],labels[:4],['c0','c1','c2','c3'],[info]*4,alpha=.1)
        pred=model.predict_interval(x[0],info)
        self.assertIsNone(model.radius)
        self.assertFalse(pred['supported'])
        self.assertEqual((pred['lower'],pred['upper']),(-1.,1.))
        model.calibrate(x[:10],labels[:10],[f'c{i}' for i in range(10)],[info]*10,alpha=.1)
        self.assertTrue(model.predict_interval(x[0],info)['supported'])
        self.assertFalse(model.predict_interval(x[0],dict(info,receiver_role='Programming Expert'))['supported'])
        with tempfile.TemporaryDirectory() as path:
            checkpoint=Path(path)/'model.json'; model.write(checkpoint)
            restored=GroupUtilityModel.load(checkpoint)
            self.assertAlmostEqual(restored.predict_interval(x[0],info)['mean'],model.predict_interval(x[0],info)['mean'])
            with self.assertRaises(ValueError):
                GroupUtilityModel.load(checkpoint,expected_sha256='wrong')
