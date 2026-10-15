"""Native prompts/parsers, isolated original separate-globals/locals math execution."""
from contextlib import contextmanager
import importlib
import time
from common import CONFIG,QUESTION,RECEIVER,InvalidUnit
import sandbox

def dataset():
    from datasets.data_process import GSM8KDataset,MMLUProDataset
    return GSM8KDataset if CONFIG['domain']=='gsm8k' else MMLUProDataset
def parse(text):
    cls=dataset()
    return cls.postprocess_answer(cls.__new__(cls),text)
def target(value):
    return dataset().record_to_target_answer({'answer':value})

@contextmanager
def observers(module,final_class,backend):
    ordinary=getattr(module,'MathSolver' if CONFIG['domain']=='gsm8k' else 'AnalyzeAgent')
    original=ordinary._process_inputs;last=final_class._process_inputs
    def log(node,spatial,temporal,summary,prompts,final=False):
        backend.recorder.write('communication_consumed',question_id=QUESTION.get(),receiver_id=node.id,
            receiver_role=getattr(node,'role',None),is_final=final,spatial_sources=list(spatial),
            temporal_sources=list(temporal),neighbor_summary=summary,system_prompt=prompts[0],user_prompt=prompts[1],
            full_user_text_tokens=backend.counter(prompts[1]),summary_used=bool(summary),
            raw_fallback=not bool(summary) and bool(spatial or temporal))
    if CONFIG['domain']=='gsm8k':
        def inputs(node,raw,spatial,temporal,neighbor_summary='',**kwargs):
            prompts=original(node,raw,spatial,temporal,neighbor_summary,**kwargs)
            log(node,spatial,temporal,neighbor_summary,prompts);return prompts
    else:
        async def inputs(node,raw,spatial,temporal,neighbor_summary='',**kwargs):
            prompts=await original(node,raw,spatial,temporal,neighbor_summary,**kwargs)
            log(node,spatial,temporal,neighbor_summary,prompts);return prompts
    def final(node,raw,spatial,temporal,neighbor_summary='',**kwargs):
        prompts=last(node,raw,spatial,temporal,neighbor_summary,**kwargs)
        log(node,spatial,temporal,neighbor_summary,prompts,True);return prompts
    saved_code=getattr(module,'execute_code_get_return',None)
    def execute(code):
        start=time.monotonic()
        result=sandbox.run(code,timeout=5,expression="locals().get('answer')",separate_locals=True)
        if result.get('error_type')=='SandboxFailure':raise RuntimeError('isolated math executor infrastructure failed: '+str(result))
        timed=result.get('error_type') in ('TimeoutError','ResourceLimit')
        value=result.get('value') if result['passed'] else 'Error occurred: '+result.get('error','unknown error')
        backend.recorder.write('code_execution_finished',question_id=QUESTION.get(),code=code,
            timed_out=timed,seconds=time.monotonic()-start,result=value,isolated_result=result)
        if timed:raise InvalidUnit('code timeout; condition invalid')
        return value
    ordinary._process_inputs=inputs;final_class._process_inputs=final
    if saved_code is not None:module.execute_code_get_return=execute
    try:yield
    finally:
        ordinary._process_inputs=original;final_class._process_inputs=last
        if saved_code is not None:module.execute_code_get_return=saved_code
