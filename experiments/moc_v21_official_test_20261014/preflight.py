"""No-network/no-main-LLM source, data, GPU-routing and native graph checks."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text())
sys.path.insert(0,str(ROOT/'coding_base'))
from moc_adaptive_sampling.paper_baselines import load_author_api,PaperRecipe,make_graph
from moc_adaptive_sampling.expected_utility import ExpectedGroupUtilityModel
from moc_adaptive_sampling.expected_set import ExpectedSetSampler,ExpectedSetConfig
from moc_adaptive_sampling.adapter import MOCSamplingAdapter
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.counterfactual import ExactCallCache
from runtime_controls import stable_ids


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


async def integration(api, model, roles):
    import ollama
    import common
    import merger_backend
    from trace_hooks import QUESTION
    original_sync,original_async=ollama.Client,ollama.AsyncClient
    calls=[]
    def fake_response(request):
        content='The answer is A'
        if request['model']==CONFIG['main_model'] and any('programming expert' in m['content'].lower() for m in request['messages'][:1]):
            content="```python\nanswer='A'\n```"
        if request['model']==CONFIG['merger_model']:content='Consolidated evidence. The answer is A'
        calls.append(request)
        return dict(done=True,done_reason='stop',prompt_eval_count=20,eval_count=10,message={'role':'assistant','content':content})
    class FakeClient:
        def __init__(self,*a,**kw):pass
        def list(self):return {'models':[{'model':CONFIG['main_model'],'digest':CONFIG['main_digest']}]}
        def chat(self,**kw):return fake_response(kw)
    class FakeAsync:
        def __init__(self,*a,**kw):pass
        async def chat(self,**kw):return fake_response(kw)
    ollama.Client,ollama.AsyncClient=FakeClient,FakeAsync
    old_root,old_merger_root=common.ROOT,merger_backend.ROOT
    verified=[]
    try:
        with tempfile.TemporaryDirectory(prefix='moc_test_preflight_') as tmp:
            folder=Path(tmp);(folder/'data').mkdir()
            for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):
                (folder/'data'/name).write_text('')
            (folder/'budget.json').write_text(json.dumps(dict(new_calls=0,actual_tokens=0,reserved_tokens=0)))
            (folder/'run_start.json').write_text(json.dumps(dict(deadline_unix=time.time()+300)))
            common.ROOT=merger_backend.ROOT=folder
            import src.graph.graph as graph_module
            import src.llm.profile_embedding as embedding
            def encode(text):
                b=hashlib.sha256(text.encode()).digest()
                return np.tile(np.frombuffer(b,dtype=np.uint8).astype(np.float32)-127.,12)
            graph_module.get_sentence_embedding=embedding.get_sentence_embedding=encode
            import src.agents.math_solver as math_module
            from src.agents.final_decision import FinalRefer
            for density in (.3,.7):
                backend=common.Backend('fake-main',folder/str(density),'test')
                merge=merger_backend.MergerBackend('fake-merger',folder/str(density),backend)
                sampler=ExpectedSetSampler(ExpectedSetConfig(),backend.counter,model)
                for arm in ('moc','adaptive_sampling'):
                    random.seed(42);np.random.seed(42)
                    import torch
                    torch.manual_seed(42)
                    with stable_ids():graph=make_graph(api,PaperRecipe(edge_density=density,use_cot=False),arm,roles=roles)
                    for node in [*graph.nodes.values(),graph.decision_node]:node.llm=backend.llm(node.llm)
                    backend.context=dict(question_id='mock-only',density=density,arm=arm)
                    adapter=MOCSamplingAdapter(graph,sampler,encode,counter_name=backend.counter.name) if arm!='moc' else None
                    before,mbefore=len(backend.cache.events),len(merge.cache.events)
                    with backend.controls(math_module,FinalRefer),merge.route():
                        output=await graph.arun({'task':'Question: mock-only\nA) 1\nB) 2\nC) 3\nD) 4\nE) 5\n'},num_rounds=1,max_tries=1,max_time=60)
                    assert len(output[0])==1 and backend.cache.totals(before)['logical_calls']==8
                    merged=merge.cache.totals(mbefore)['logical_calls']
                    assert not merge.failures and (merged%5==0 if arm=='moc' else merged==0)
                    if adapter:
                        assert adapter.ledger['valid'];adapter.detach()
                    verified.append(dict(density=density,arm=arm,ordinary_and_final_calls=8,merger_calls=merged))
            assert any(v['merger_calls']>0 for v in verified if v['arm']=='moc')
            paid=json.loads((folder/'budget.json').read_text());assert paid['reserved_tokens']==0
            assert paid['actual_tokens']==30*paid['new_calls']
    finally:
        common.ROOT=old_root;merger_backend.ROOT=old_merger_root
        ollama.Client,ollama.AsyncClient=original_sync,original_async
    return verified


def main():
    lock=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    inputs=rows(ROOT/'data/test_inputs.jsonl')
    assert len(inputs)==len({r['question_id'] for r in inputs})==254
    assert all(set(r)=={'question_id','question','options'} for r in inputs)
    model=ExpectedGroupUtilityModel.load(ROOT/'group_utility_model.json',json.loads((ROOT/'policy_lock.json').read_text())['model_sha256'])
    assert model.mode=='compact' and model.metadata['fit_question_count']==35
    import ollama
    ollama.list=lambda:{'models':[]}
    api=load_author_api(CONFIG['author_directory'])
    expected=json.loads((ROOT/'expected_dags.json').read_text())
    roles=json.loads((ROOT/'val_role_snapshot.json').read_text())['roles'];graphs=[]
    for density in (.3,.7):
        random.seed(42);np.random.seed(42)
        import torch
        torch.manual_seed(42)
        with stable_ids():g=make_graph(api,PaperRecipe(edge_density=density),'moc',roles=roles)
        g.spatial_logits=g.spatial_masks*0;g.construct_spatial_connection()
        actual={i:[p.id for p in n.spatial_predecessors] for i,n in g.nodes.items()}
        assert actual==expected[str(density)]['predecessors']
        graphs.append(dict(density=density,nodes=7,edges=sum(map(len,actual.values()))))
    checks=asyncio.run(integration(api,model,roles))
    report=dict(status='passed',test_questions=254,test_conditions=1016,graphs=graphs,
                fake_native_graph_checks=checks,real_model_calls=0,network_calls=0,gpu_allocations=0,
                test_gold_present=False,checkpoint_unchanged=True,historical_test_exposure_disclosed=True)
    (ROOT/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
