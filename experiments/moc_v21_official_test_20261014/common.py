"""Shared immutable requests, exact cache, resource budget and author replay."""
import asyncio
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
sys.path.insert(0,CONFIG['extension_directory'])
from moc_adaptive_sampling.counterfactual import ExactCallCache, request_key
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.paper_baselines import PaperRecipe, make_graph, load_author_api
from moc_adaptive_sampling.sampler import serialize_messages
from trace_hooks import Recorder, QUESTION, RECEIVER, plain
from runtime_controls import stable_ids, prompt_and_tool_observers

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)

def append(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

class BudgetExceeded(RuntimeError):
    pass

class InvalidUnit(ValueError):
    pass

@contextmanager
def budget_lock():
    import fcntl
    with (ROOT/'budget.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)

def reserve(tokens):
    with budget_lock():
        ledger=json.loads((ROOT/'budget.json').read_text())
        if ledger['new_calls']>=CONFIG['max_new_calls'] or ledger['actual_tokens']+ledger.get('unknown_cost_upper_bound_tokens',0)+ledger['reserved_tokens']+tokens>CONFIG['max_new_model_tokens']:
            raise BudgetExceeded('new-call/token budget reached; no sample expansion')
        ledger['new_calls']+=1; ledger['reserved_tokens']+=tokens
        save(ROOT/'budget.json',ledger)

def charge(reservation,response):
    with budget_lock():
        ledger=json.loads((ROOT/'budget.json').read_text())
        actual=response.get('prompt_eval_count',0)+response.get('eval_count',0)
        ledger['reserved_tokens']-=reservation
        ledger['actual_tokens']+=actual
        save(ROOT/'budget.json',ledger)


class Backend:
    def __init__(self, host, directory, stage):
        import ollama
        self.client=ollama.Client(host=host,timeout=300)
        # Constructors only query our private service; never a public/default port.
        ollama.list=self.client.list
        self.counter=LocalTokenizerCounter(CONFIG['tokenizer_path'],CONFIG['tokenizer_sha256'])
        self.recorder=Recorder(directory)
        self.stage, self.last_key=stage,None
        self.context={}
        self.pending_requests={}
        self.cache=ExactCallCache(CONFIG['main_digest'],self.generate,Path(directory)/'calls_raw.jsonl')
        files=[ROOT/'data/historical_calls_LOCAL_ONLY.jsonl']
        if stage in ('val','test') and (ROOT/'data/val_historical_calls_LOCAL_ONLY.jsonl').exists():
            files.append(ROOT/'data/val_historical_calls_LOCAL_ONLY.jsonl')
        cache_records=[r for p in files for r in rows(p)]
        report=self.cache.load_historical(cache_records)
        save(Path(directory)/'cache_admission.json',dict(report,conflicts=self.cache.historical_conflicts))
        current=next(m for m in plain(self.client.list())['models'] if m.get('model')==CONFIG['main_model'] or m.get('name')==CONFIG['main_model'])
        assert current['digest']==CONFIG['main_digest']

    def request(self,messages):
        return dict(model=CONFIG['main_model'],messages=messages,stream=False,
                    options=dict(num_predict=CONFIG['main_num_predict'],temperature=CONFIG['main_temperature'],
                                 seed=CONFIG['seed'],num_ctx=CONFIG['num_ctx'],num_gpu=999))

    async def generate(self, request):
        start=json.loads((ROOT/'run_start.json').read_text())
        if (ROOT/'STOP').exists() or time.time()>start['deadline_unix']:
            raise BudgetExceeded('STOP or GPU wall deadline')
        text_tokens=sum(self.counter(m['content']) for m in request['messages'])
        if text_tokens>CONFIG['num_ctx']-64:
            raise InvalidUnit('prompt would exceed safe context envelope')
        reservation=CONFIG['num_ctx']+CONFIG['main_num_predict']
        reserve(reservation)
        reservation_id='main-'+str(time.time_ns())
        self.pending_requests[reservation_id]=reservation
        self.recorder.write('backend_request_started',stage=self.stage,request=request,**self.context,
                            reservation_id=reservation_id,reserved_tokens=reservation)
        try:
            response=plain(await asyncio.to_thread(self.client.chat,**request))
        except BaseException as error:
            self.recorder.write('backend_error',stage=self.stage,request=request,**self.context,
                                error_type=type(error).__name__,error=str(error),reserved_cost_unknown=reservation)
            raise
        if not all(isinstance(response.get(k),int) and response[k]>=0 for k in ('prompt_eval_count','eval_count')):
            self.recorder.write('backend_usage_missing',stage=self.stage,request=request,**self.context,
                                reserved_cost_unknown=reservation)
            raise RuntimeError('main usage missing; keep unresolved reservation')
        charge(reservation,response)
        self.pending_requests.pop(reservation_id)
        self.recorder.write('backend_request_settled',stage=self.stage,**self.context,
                            reservation_id=reservation_id,actual_tokens=response['prompt_eval_count']+response['eval_count'])
        return response

    def llm(self,original):
        owner=self
        class Metered:
            async def agen(self,messages,*args,**kwargs):
                converted=original._convert_messages(messages)
                request=owner.request(converted)
                owner.last_key=request_key(request,CONFIG['main_digest'])
                response=await owner.cache.call(request,stage=owner.stage,**owner.context)
                return response['message']['content']
        return Metered()

    @contextmanager
    def controls(self, math_module, final_class):
        with prompt_and_tool_observers(math_module,final_class,self.recorder,self.counter,CONFIG['code_timeout_seconds']):
            observed=math_module.execute_code_get_return
            def execute(code):
                entry=self.cache.entries.get(self.last_key,{})
                if 'tool_result' in entry:
                    self.recorder.write('cached_tool_result',request_key=self.last_key,result=entry['tool_result'],**self.context)
                    return entry['tool_result']
                result=observed(code)
                # f-string suffix only uses its string representation.
                self.recorder.write('new_tool_result',request_key=self.last_key,result=str(result),**self.context)
                if any(e['event_type']=='code_execution_finished' and e.get('timed_out') for e in self.recorder.records[-2:]):
                    raise InvalidUnit('code timeout; unit excluded')
                entry['tool_result']=str(result)
                return result
            math_module.execute_code_get_return=execute
            try:
                yield
            finally:
                math_module.execute_code_get_return=observed


class AuthorReplay:
    def __init__(self,api,backend,material):
        self.backend,self.material=backend,material
        recipe=PaperRecipe(edge_density=material['density'],use_cot=CONFIG['use_cot'])
        roles=[m['role'] for m in material['materials'].values()]
        with stable_ids():
            self.graph=make_graph(api,recipe,'vanilla_mas',roles=roles)
        if self.graph.optimized_spatial or not all(float(x) in (0.,1.) for x in self.graph.spatial_masks):
            raise RuntimeError('replay only supports the original fixed binary DAG')
        # arun normally computes these from the GNN. Fixed binary masks never
        # inspect the logits, so hydration avoids an unused feature forward.
        self.graph.spatial_logits=self.graph.spatial_masks*0
        self.graph.construct_spatial_connection()
        self.nodes=self.graph.nodes
        actual={i:[p.id for p in node.spatial_predecessors] for i,node in self.nodes.items()}
        if actual!=material['predecessors']:
            raise RuntimeError('actual author DAG differs from frozen materials')
        if list(self.nodes)!=list(material['materials']):
            raise RuntimeError('author node IDs/order differ')
        for node in [*self.nodes.values(),self.graph.decision_node]:
            node.llm=backend.llm(node.llm)
        self.messages=[]

    async def execute(self,node_id,outputs,sources):
        node=self.nodes[node_id]
        self.backend.context.update(receiver_id=node_id,receiver_role=node.role)
        token=RECEIVER.set({'receiver_id':node_id,'role':node.role,'receiver_event_id':None})
        try:
            if sources is None:
                spatial={p:{'role':self.nodes[p].role,'output':outputs[p]} for p in self.material['predecessors'][node_id]}
                return await node._async_execute({'task':self.material['task']},spatial,{},'')
            summary=serialize_messages([m for m in self.messages if m.source in sources])
            return await node._async_execute({'task':self.material['task']},{},{},summary)
        finally:
            RECEIVER.reset(token)

    async def finalize(self,outputs):
        node=self.graph.decision_node
        self.backend.context.update(receiver_id=node.id,receiver_role='FinalRefer')
        spatial={i:{'role':self.nodes[i].role,'output':outputs[i]} for i in self.material['baseline_usage']['final_order']}
        return await node._async_execute({'task':self.material['task']},spatial,{},'')

    def parse(self,text):
        prompt=self.graph.decision_node.prompt_set
        return prompt.postprocess_answer(prompt,text)

    def full_logical_tokens(self,affected,call_start):
        frozen=sum(value for source,value in self.material['baseline_usage']['by_node'].items()
                   if source not in affected and source!=self.material['baseline_usage']['decision_id'])
        return frozen+self.backend.cache.totals(call_start)['logical_model_tokens']


def imports(host,directory,stage):
    backend=Backend(host,directory,stage)
    api=load_author_api(CONFIG['author_directory'])
    math_module=importlib.import_module('src.agents.math_solver')
    final_class=importlib.import_module('src.agents.final_decision').FinalRefer
    return api,backend,math_module,final_class
