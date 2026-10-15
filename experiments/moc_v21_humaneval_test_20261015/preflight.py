"""Synthetic native CodeWriting checks; no benchmark execution or real LLM calls."""
import asyncio
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
from runtime_controls import stable_ids


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sandbox_checks():
    from sandbox import run
    good=run('def plus(a,b): return a+b','assert plus(1,2)==3')
    assert good['passed'],good
    bad=run('def plus(a,b): return a-b','assert plus(1,2)==3')
    assert not bad['passed'] and bad['error_type']=='AssertionError',bad
    denied=[]
    for name,code in [('home_read',"open('/home/liuxin/.ssh/id_rsa').read()"),
                      ('network',"import socket; socket.socket()"),
                      ('subprocess',"import os; os.fork()"),
                      ('write',"open('/tmp/moc_sandbox_write_probe','w').write('x')")]:
        result=run(code)
        assert not result['passed'],(name,result)
        denied.append(dict(facility=name,error_type=result['error_type']))
    timed=run('while True: pass',timeout=1)
    assert not timed['passed']
    return dict(positive_test=True,negative_test=True,restrictions=denied,timeout_bounded=True)


async def integration(api,model,roles):
    import ollama,common,merger_backend
    from trace_hooks import QUESTION
    old_sync,old_async=ollama.Client,ollama.AsyncClient
    success=[True];calls=[]
    def response(request):
        content='```python\ndef plus(a,b): return a+b\n```' if success[0] else '```python\ndef plus(a,b): return a-b\n```'
        if request['model']==CONFIG['merger_model']:content='A synthetic summary of plus implementation.'
        assert 'HIDDEN_GOLD_MUST_NEVER_APPEAR' not in json.dumps(request)
        calls.append(request)
        return dict(done=True,done_reason='stop',prompt_eval_count=20,eval_count=10,message={'role':'assistant','content':content})
    class Client:
        def __init__(self,*a,**kw):pass
        def list(self):return {'models':[{'model':CONFIG['main_model'],'digest':CONFIG['main_digest']}]}
        def chat(self,**kw):return response(kw)
    class AsyncClient:
        def __init__(self,*a,**kw):pass
        async def chat(self,**kw):return response(kw)
    ollama.Client,ollama.AsyncClient=Client,AsyncClient
    original_root,original_mroot=common.ROOT,merger_backend.ROOT
    checks=[]
    try:
        with tempfile.TemporaryDirectory(prefix='moc_humaneval_fake_') as tmp:
            folder=Path(tmp);(folder/'data').mkdir()
            for name in ('historical_calls_LOCAL_ONLY.jsonl','val_historical_calls_LOCAL_ONLY.jsonl','merger_historical_calls_LOCAL_ONLY.jsonl'):
                (folder/'data'/name).write_text('')
            (folder/'budget.json').write_text(json.dumps(dict(new_calls=0,actual_tokens=0,reserved_tokens=0)))
            (folder/'run_start.json').write_text(json.dumps(dict(deadline_unix=time.time()+600)))
            common.ROOT=merger_backend.ROOT=folder
            import src.graph.graph as graph_module
            import src.llm.profile_embedding as embedding
            def encode(text):
                b=hashlib.sha256(text.encode()).digest()
                return np.tile(np.frombuffer(b,dtype=np.uint8).astype(np.float32)-127.,12)
            graph_module.get_sentence_embedding=embedding.get_sentence_embedding=encode
            import src.agents.code_writing as code_module
            from src.agents.final_decision import FinalWriteCode
            for passing in (True,False):
                success[0]=passing
                for density in (.3,.7):
                    # Separate cache so contradictory fake responses never share history.
                    backend=common.Backend('fake-main',folder/f'{passing}-{density}','test')
                    merge=merger_backend.MergerBackend('fake-merger',folder/f'{passing}-{density}',backend)
                    sampler=ExpectedSetSampler(ExpectedSetConfig(),backend.counter,model)
                    for arm in ('moc','adaptive_sampling'):
                        random.seed(42);np.random.seed(42)
                        import torch
                        torch.manual_seed(42)
                        with stable_ids():graph=make_graph(api,PaperRecipe(domain='humaneval',edge_density=density),arm,roles=roles)
                        for node in [*graph.nodes.values(),graph.decision_node]:node.llm=backend.llm(node.llm)
                        backend.context=dict(question_id='synthetic-only',density=density,arm=arm)
                        adapter=MOCSamplingAdapter(graph,sampler,encode,counter_name=backend.counter.name,code_writing=True) if arm!='moc' else None
                        before,mbefore=len(backend.cache.events),len(merge.cache.events)
                        with backend.controls(code_module,FinalWriteCode),merge.route():
                            output=await graph.arun({'task':'def plus(a,b):\n    """Add.\n    >>> plus(1, 2)\n    3\n    """\n'},num_rounds=1,max_tries=1,max_time=120)
                        count=backend.cache.totals(before)['logical_calls']
                        assert len(output[0])==1 and 1<=count<=8
                        if not passing:assert count==8
                        merged=merge.cache.totals(mbefore)['logical_calls']
                        assert not merge.failures and (merged%5==0 if arm=='moc' else merged==0)
                        if adapter:
                            assert adapter.ledger['valid']
                            for record in adapter.ledger['selections']:
                                if roles[record['receiver_index']]!='Programming Expert':assert not record['source_ids']
                            adapter.detach()
                        checks.append(dict(public_example_passes=passing,density=density,arm=arm,
                            main_calls=count,early_returns=8-count,merger_calls=merged))
            assert any(r['early_returns'] for r in checks if r['public_example_passes'] and r['arm']=='moc')
            assert any(r['merger_calls'] for r in checks if r['arm']=='moc')
            # Synthetic forced selection tests adapter mechanics independently
            # of the frozen predictor (whose real support is never relaxed).
            from moc_adaptive_sampling.sampler import Selection,serialize_messages
            class SelectFirst:
                uses_selected_state=True
                config=ExpectedSetConfig()
                count_tokens=backend.counter
                def select(self,receiver,state,messages,predecessors,**kwargs):
                    chosen=messages[:1]
                    text=serialize_messages(chosen)
                    return Selection([m.source for m in chosen],text,self.count_tokens(text),[],
                                     'synthetic_adapter_probe')
            success[0]=True
            random.seed(42);np.random.seed(42);torch.manual_seed(42)
            with stable_ids():graph=make_graph(api,PaperRecipe(domain='humaneval',edge_density=.7),'adaptive_sampling',roles=roles)
            for node in [*graph.nodes.values(),graph.decision_node]:node.llm=backend.llm(node.llm)
            before=len(backend.cache.events);trace_before=len(backend.recorder.records)
            adapter=MOCSamplingAdapter(graph,SelectFirst(),encode,code_writing=True)
            with backend.controls(code_module,FinalWriteCode):
                await graph.arun({'task':'def plus(a,b):\n    """\n    >>> plus(1, 2)\n    3\n    """\n'},num_rounds=1,max_tries=1,max_time=120)
            assert backend.cache.totals(before)['logical_calls']==2
            consumed={r['receiver_id']:r for r in backend.recorder.records[trace_before:]
                      if r['event_type']=='communication_consumed' and not r['is_final']}
            for record in adapter.ledger['selections']:
                assert consumed[record['receiver']]['spatial_sources']==record['source_ids']
            assert adapter.ledger['valid'];adapter.detach()
            checks.append(dict(scope='synthetic selected-source-only early return',passed=True,main_calls=2))
            paid=json.loads((folder/'budget.json').read_text())
            assert paid['reserved_tokens']==0 and paid['actual_tokens']==30*paid['new_calls']
    finally:
        common.ROOT=original_root;merger_backend.ROOT=original_mroot
        ollama.Client,ollama.AsyncClient=old_sync,old_async
    return checks


def main():
    lock=json.loads((ROOT/'experiment_lock.json').read_text())
    for name,digest in {**lock['source_hashes'],**lock['data_hashes']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    data=rows(ROOT/'data/test_inputs.jsonl')
    assert len(data)==len({r['question_id'] for r in data})==164
    assert all(set(r)=={'question_id','prompt','entry_point'} for r in data)
    assert not (ROOT/'data/test_targets_LOCAL_ONLY.jsonl').exists(),'do not deploy gold before generation completes'
    isolation=sandbox_checks()
    model=ExpectedGroupUtilityModel.load(ROOT/'group_utility_model.json',json.loads((ROOT/'policy_lock.json').read_text())['model_sha256'])
    import ollama
    ollama.list=lambda:{'models':[]}
    api=load_author_api(CONFIG['author_directory'])
    roles=json.loads((ROOT/'code_role_snapshot.json').read_text())['roles']
    checks=asyncio.run(integration(api,model,roles))
    report=dict(status='passed',test_questions=164,test_conditions=656,
        fake_native_graph_checks=checks,isolation_checks=isolation,real_model_calls=0,gpu_allocations=0,
        gold_present=False,checkpoint_unchanged=True,covered_receiver_roles=sorted({k[0] for k in model.coverage}),
        unseen_code_roles=sorted(set(roles)-{k[0] for k in model.coverage}))
    (ROOT/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
