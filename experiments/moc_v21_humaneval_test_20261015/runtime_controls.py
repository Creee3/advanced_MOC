"""Shared deployment/measurement controls; no new baseline prompts or topology."""

from contextlib import contextmanager
import itertools
import signal
import time
from trace_hooks import QUESTION, RECEIVER


@contextmanager
def stable_ids():
    import shortuuid
    original = shortuuid.ShortUUID.random
    counter = itertools.count()
    def fixed(instance, length=None):
        assert length == 4
        return f"{next(counter):04x}"
    shortuuid.ShortUUID.random = fixed
    try:
        yield
    finally:
        shortuuid.ShortUUID.random = original


@contextmanager
def backend_routes(ollama, main_host, merger_host):
    original_sync, original_async, original_list = ollama.chat, ollama.AsyncClient.chat, ollama.list
    main = ollama.Client(host=main_host, timeout=300)
    merger = ollama.AsyncClient(host=merger_host, timeout=300)
    def sync(*args, **kwargs):
        if kwargs.get("model") != "gemma2:27b":
            raise ValueError("unexpected synchronous model")
        return main.chat(*args, **kwargs)
    async def async_chat(client, *args, **kwargs):
        if kwargs.get("model") != "gemma2:9b":
            raise ValueError("unexpected asynchronous model")
        return await original_async(merger, *args, **kwargs)
    ollama.chat, ollama.AsyncClient.chat, ollama.list = sync, async_chat, main.list
    try:
        yield
    finally:
        ollama.chat, ollama.AsyncClient.chat, ollama.list = original_sync, original_async, original_list


@contextmanager
def effective_options(ollama, config, count_tokens):
    original_sync, original_async = ollama.chat, ollama.AsyncClient.chat
    def prepare(kwargs):
        copied = dict(kwargs)
        model = copied["model"]
        options = dict(copied.get("options") or {})
        # Same explicit all-GPU placement as warmup; keep this option uniform
        # for every arm so the server cannot reload into an automatic CPU path.
        options.update(seed=config["seed"], num_ctx=config["num_ctx"], num_gpu=999)
        if model == config["main_model"]:
            options.update(temperature=config["main_temperature"], num_predict=config["main_num_predict"])
        elif model == config["merger_model"]:
            assert options["temperature"] == config["merger_temperature"]
        else:
            raise ValueError("unknown model; model downloads are not allowed")
        # Leave room for template framing; actual backend truncation still checked by supervisor.
        text_tokens = sum(count_tokens(m["content"]) for m in copied["messages"])
        if text_tokens > config["num_ctx"] - 64:
            raise ValueError(f"prompt would exceed safe context envelope: {text_tokens}")
        copied["options"] = options
        return copied
    def sync(*args, **kwargs):
        return original_sync(*args, **prepare(kwargs))
    async def async_chat(client, *args, **kwargs):
        return await original_async(client, *args, **prepare(kwargs))
    ollama.chat, ollama.AsyncClient.chat = sync, async_chat
    try:
        yield
    finally:
        ollama.chat, ollama.AsyncClient.chat = original_sync, original_async


@contextmanager
def prompt_and_tool_observers(math_module, final_class, recorder, count_tokens, code_timeout):
    MathSolver = math_module.MathSolver
    original_math, original_final = MathSolver._process_inputs, final_class._process_inputs
    original_code = math_module.execute_code_get_return

    def context(node=None):
        current = RECEIVER.get() or {}
        return {"question_id":QUESTION.get(), "receiver_id":node.id if node is not None else current.get("receiver_id"),
                "receiver_event_id":current.get("receiver_event_id")}

    def math_inputs(node, inputs, spatial, temporal, neighbor_summary="", **kwargs):
        prompts = original_math(node, inputs, spatial, temporal, neighbor_summary, **kwargs)
        task_prompt = node.prompt_set.get_answer_prompt(question=inputs["task"], role=node.role)
        assert prompts[1].startswith(task_prompt)
        payload = prompts[1][len(task_prompt):]
        recorder.write("communication_consumed", **context(node), is_final=False,
                       communication_payload=payload, communication_text_tokens=count_tokens(payload),
                       communication_utf8_bytes=len(payload.encode("utf-8")),
                       summary_used=bool(neighbor_summary), raw_fallback=not bool(neighbor_summary) and bool(spatial or temporal))
        return prompts

    def final_inputs(node, inputs, spatial, temporal, neighbor_summary="", **kwargs):
        prompts = original_final(node, inputs, spatial, temporal, neighbor_summary, **kwargs)
        payload = "".join(id + ": " + info["output"] + "\n\n" for id, info in spatial.items())
        assert not payload or prompts[1].endswith(payload)
        recorder.write("communication_consumed", **context(node), is_final=True,
                       communication_payload=payload, communication_text_tokens=count_tokens(payload),
                       communication_utf8_bytes=len(payload.encode("utf-8")), raw_fallback=False)
        return prompts

    class Deadline(BaseException):
        pass
    def alarm(*args):
        raise Deadline()
    def execute(code):
        start = time.monotonic()
        previous = signal.signal(signal.SIGALRM, alarm)
        timed_out = False
        try:
            signal.setitimer(signal.ITIMER_REAL, code_timeout)
            try:
                result = original_code(code)
            except Deadline:
                timed_out = True
                result = f"Error occurred: code execution exceeded {code_timeout} seconds"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        recorder.write("code_execution_finished", **context(), timed_out=timed_out,
                       seconds=time.monotonic()-start, result=result)
        return result
    MathSolver._process_inputs, final_class._process_inputs = math_inputs, final_inputs
    math_module.execute_code_get_return = execute
    try:
        yield
    finally:
        MathSolver._process_inputs, final_class._process_inputs = original_math, original_final
        math_module.execute_code_get_return = original_code
