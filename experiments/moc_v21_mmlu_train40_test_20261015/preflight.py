"""CPU-only native Graph integration and restricted math executor checks."""
import asyncio
import json
from pathlib import Path
import sys
import time
import numpy as np
from common import ROOT,CONFIG,save,load_author_api
from domain_runtime import parse,target,observers
from runtime_controls import stable_ids
from moc_adaptive_sampling.paper_baselines import PaperRecipe,make_graph
from moc_adaptive_sampling.expected_set import ExpectedSetConfig,ExpectedSetSampler
from moc_adaptive_sampling.adapter import MOCSamplingAdapter
import sandbox

class Predict:
    def predict_expected(self,x,info):return dict(help=.5,harm=0.,mean=.5,supported=True)
class Recorder:
    def __init__(self):self.records=[]
    def write(self,kind,**kw):self.records.append(dict(event_type=kind,**kw))
class Dummy:
    async def agen(self,messages,*a,**kw):return 'The answer is 3' if CONFIG['domain']=='svamp' else 'D'
    def _convert_messages(self,m):return m
class Backend:
    recorder=Recorder();counter=staticmethod(lambda s:len(s.split()));context={}

async def main():
    api=load_author_api(CONFIG['author_directory'])
    original_get=api.LLMRegistry.__dict__['get'];api.LLMRegistry.get=classmethod(lambda *a,**k:Dummy())
    import ollama,importlib
    old_chat=ollama.AsyncClient.chat
    async def merger(client,*a,**kw):return dict(done=True,message={'content':'synthetic compact summary'},prompt_eval_count=10,eval_count=5)
    ollama.AsyncClient.chat=merger
    module=importlib.import_module('src.agents.math_solver' if CONFIG['domain']=='svamp' else 'src.agents.analyze_agent')
    final=importlib.import_module('src.agents.final_decision').FinalRefer
    checks=[];b=Backend()
    try:
        for d in (.3,.7):
            for arm in ('vanilla_mas','adaptive_sampling','moc'):
                with stable_ids():g=make_graph(api,PaperRecipe(domain=CONFIG['domain'],edge_density=d,use_cot=False),arm,roles=CONFIG['roles'])
                calls=[]
                for node in [*g.nodes.values(),g.decision_node]:
                    role=getattr(node,'role','final')
                    class LLM(Dummy):
                        async def agen(self,messages,*a,role=role,**kw):
                            calls.append(messages)
                            if role=='Programming Expert':return '```python\nanswer=1+2\n```'
                            return await super().agen(messages,*a,**kw)
                    node.llm=LLM()
                adapter=None
                if arm=='adaptive_sampling':
                    sampler=ExpectedSetSampler(ExpectedSetConfig(communication_tokens=1024,max_group_size=2,cost_penalty=.02,local_state_weight=0),b.counter,Predict())
                    adapter=MOCSamplingAdapter(g,sampler,lambda t:np.asarray([1.,2.,3.]),analyze_agent=CONFIG['domain']=='mmlu')
                with observers(module,final,b):result=await g.arun({'task':'Synthetic preflight problem; choose or compute 3.'},num_rounds=1,max_tries=1,max_time=60)
                assert len(calls)==8 and len(result[0])==1 and all(n.outputs for n in g.nodes.values())
                if adapter:
                    assert adapter.ledger['valid'] and len(adapter.ledger['selections'])==7
                    for e in [x for x in b.recorder.records if x['event_type']=='communication_consumed'][-8:-1]:
                        assert e['spatial_sources']==[] and e['temporal_sources']==[],e
                    adapter.detach()
                checks.append(dict(density=d,arm=arm,main_calls=len(calls)))
        for code,expected in [('answer=3','3'),('def f(): return 7\nanswer=f()','7'),('pass','None'),('import math\ndef f(): return math.pi\nanswer=f()',None)]:
            r=sandbox.run(code,timeout=5,expression="locals().get('answer')",separate_locals=True)
            if expected is None:assert not r['passed'] and r['error_type']=='NameError',r
            else:assert r.get('value')==expected,r
        denied=[]
        for code in ["open('/home/liuxin/.ssh/id_rsa').read()","import socket;socket.socket()","import os;os.fork()","open('/tmp/moc_probe','w').write('x')"]:
            r=sandbox.run(code,timeout=2);assert not r['passed'];denied.append(r['error_type'])
        assert not sandbox.run('while True: pass',timeout=1)['passed']
        assert parse('The answer is 3')=='3' if CONFIG['domain']=='svamp' else parse('D')=='D'
        save(ROOT/'preflight_report.json',dict(status='passed',native_graph_checks=checks,
            math_separate_namespace_checks=4,isolation_denials=denied,timeout=True,model_calls=0,time_unix=time.time()))
    finally:
        api.LLMRegistry.get=original_get;ollama.AsyncClient.chat=old_chat

if __name__=='__main__':asyncio.run(main())
