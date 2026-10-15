"""Observe native CodeWriting prompts and replace only unsafe in-process exec."""
from contextlib import contextmanager
import time
from trace_hooks import QUESTION
import sandbox


@contextmanager
def code_observers(code_module, final_class, recorder, counter):
    native_code=code_module.CodeWriting._process_inputs
    native_final=final_class._process_inputs
    executor=code_module.PyExecutor
    native_execute=executor.execute

    def execute(self,func,tests,timeout=5,verbose=True):
        start=time.monotonic()
        result=sandbox.execute(func,tests,timeout)
        recorder.write('internal_test_execution',question_id=QUESTION.get(),
            source='public_prompt_doctests_only',code=func,tests=tests,
            timeout=timeout,seconds=time.monotonic()-start,passed=result[0],state=result[2],feedback=result[1])
        return result

    def wrap(native,is_final):
        def inputs(node, raw, spatial, temporal, neighbor_summary='', **kwargs):
            prompts=native(node,raw,spatial,temporal,neighbor_summary,**kwargs)
            early=prompts[0]=='is_solved'
            recorder.write('communication_consumed',question_id=QUESTION.get(),receiver_id=node.id,
                receiver_role=getattr(node,'role',None),is_final=is_final,early_return=early,
                spatial_sources=list(spatial),temporal_sources=list(temporal),summary_used=bool(neighbor_summary),
                neighbor_summary=neighbor_summary,system_prompt=prompts[0],user_prompt=prompts[1],
                full_user_text_tokens=counter(prompts[1]),
                raw_fallback=not bool(neighbor_summary) and bool(spatial or temporal))
            return prompts
        return inputs
    executor.execute=execute
    code_module.CodeWriting._process_inputs=wrap(native_code,False)
    final_class._process_inputs=wrap(native_final,True)
    try: yield
    finally:
        executor.execute=native_execute
        code_module.CodeWriting._process_inputs=native_code
        final_class._process_inputs=native_final
