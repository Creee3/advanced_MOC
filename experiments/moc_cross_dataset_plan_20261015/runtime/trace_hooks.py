"""Observe author calls without changing inputs, outputs, scores or call count."""
import contextvars
import json
from pathlib import Path
import threading
import time

QUESTION = contextvars.ContextVar('moc_trace_question', default=None)
RECEIVER = contextvars.ContextVar('moc_trace_receiver', default=None)
MERGE = contextvars.ContextVar('moc_trace_merge', default=None)


def plain(value):
    if hasattr(value, 'model_dump'):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.counter = 0
        self.records = []

    def write(self, kind, **payload):
        with self.lock:
            self.counter += 1
            item = {'event_index': self.counter, 'event_type': kind,
                    'time_unix': time.time(), **plain(payload)}
            self.records.append(item)
            with (self.directory / 'trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            return item


class Observer:
    def __init__(self, recorder, graph_module, node_module, ollama_module, question_map):
        self.recorder = recorder
        self.graph_module = graph_module
        self.node_module = node_module
        self.ollama_module = ollama_module
        self.question_map = question_map
        self.restores = []

    def replace(self, obj, name, wrapper):
        original = getattr(obj, name)
        self.restores.append((obj, name, original))
        setattr(obj, name, wrapper(original))

    def context(self):
        receiver = RECEIVER.get()
        merge = MERGE.get()
        return {'question_id': QUESTION.get(),
                'receiver_id': receiver['receiver_id'] if receiver else None,
                'receiver_event_id': receiver['receiver_event_id'] if receiver else None,
                'receiver_role': receiver.get('role') if receiver else None,
                'merge_id': merge['merge_id'] if merge else None}

    def install(self):
        owner = self

        def wrap_arun(original):
            async def observed(graph, inputs, *args, **kwargs):
                qid = owner.question_map[inputs['task']]
                token = QUESTION.set(qid)
                owner.recorder.write('question_started', question_id=qid,
                    raw_input=inputs, nodes=[{'id': n.id, 'role': n.role,
                    'agent_name': n.agent_name} for n in graph.nodes.values()],
                    decision_id=graph.decision_node.id)
                try:
                    result = await original(graph, inputs, *args, **kwargs)
                    owner.recorder.write('question_finished', question_id=qid,
                        final_answers=result[0],
                        nodes=[{'id': n.id, 'role': n.role, 'outputs': n.outputs,
                                'predecessors': [p.id for p in n.spatial_predecessors]}
                               for n in graph.nodes.values()],
                        final_predecessors=[n.id for n in graph.decision_node.spatial_predecessors])
                    return result
                finally:
                    QUESTION.reset(token)
            return observed

        def wrap_node(original):
            async def observed(node, inputs, *args, **kwargs):
                start = owner.recorder.write('receiver_started', question_id=QUESTION.get(),
                    receiver_id=node.id, role=node.role, agent_name=node.agent_name,
                    raw_input=inputs, spatial_info=node.get_spatial_info(),
                    temporal_info=node.get_temporal_info())
                token = RECEIVER.set({'receiver_id': node.id, 'role': node.role,
                                     'receiver_event_id': start['event_index']})
                try:
                    result = await original(node, inputs, *args, **kwargs)
                    owner.recorder.write('receiver_finished', **owner.context(),
                        actual_role=node.role, outputs=node.outputs)
                    return result
                except Exception as error:
                    owner.recorder.write('receiver_error', **owner.context(),
                        error_type=type(error).__name__, error=str(error))
                    raise
                finally:
                    RECEIVER.reset(token)
            return observed

        def wrap_summary(original):
            async def observed(graph, node_id, query):
                by_hop = graph.get_neighbors_by_hops(node_id, graph.neighbor_hops)
                owner.recorder.write('summary_started', **owner.context(), query=query,
                    neighbors=[{'hop': hop, 'id': n.id, 'role': n.role,
                                'output': str(n.outputs[-1]) if n.outputs else ''}
                               for hop in sorted(by_hop, reverse=True) for n in by_hop[hop]])
                result = await original(graph, node_id, query)
                owner.recorder.write('summary_finished', **owner.context(),
                                     neighbor_summary=result)
                return result
            return observed

        def wrap_merge(original):
            async def observed(graph, messages, kppa):
                start = owner.recorder.write('merge_started', **owner.context(),
                                             messages=messages, kppa=kppa)
                active = {'merge_id': start['event_index'], 'calls': [],
                          'cosines': [], 'native_cosines': [], 'embeddings': []}
                token = MERGE.set(active)
                try:
                    result = await original(graph, messages, kppa)
                    outputs = [call['response']['message']['content'].strip()
                               for call in active['calls'] if 'response' in call]
                    scores = []
                    for i in range(0, len(active['native_cosines']), 2):
                        total = 0
                        for value in active['native_cosines'][i:i+2]:
                            # Match the author's += and NumPy scalar precision.
                            total += value
                        scores.append(float(total))
                    complete = len(outputs) == 5 and len(active['cosines']) == 10
                    selected = max(range(5), key=lambda i: scores[i]) if complete else None
                    consistent = complete and result == outputs[selected]
                    owner.recorder.write('merge_finished', **owner.context(),
                        messages=messages, kppa=kppa, candidates=outputs,
                        cosine_components=active['cosines'], scores=scores,
                        embeddings=active['embeddings'], selected_index=selected,
                        returned_summary=result, observer_consistent=consistent,
                        candidate_call_events=[c['event_index'] for c in active['calls']])
                    return result
                finally:
                    MERGE.reset(token)
            return observed

        def wrap_cosine(original):
            def observed(*args, **kwargs):
                result = original(*args, **kwargs)
                active = MERGE.get()
                if active is not None:
                    active['native_cosines'].append(result[0][0])
                    active['cosines'].append(float(result[0][0]))
                return result
            return observed

        def wrap_embedding(original):
            def observed(text, *args, **kwargs):
                result = original(text, *args, **kwargs)
                active = MERGE.get()
                if active is not None:
                    active['embeddings'].append({'text': text, 'embedding': plain(result)})
                return result
            return observed

        def finish_call(request, response, started):
            record = owner.recorder.write('llm_call', **owner.context(),
                request=request, response=response,
                elapsed_seconds=time.monotonic()-started,
                request_json_bytes_estimate=len(json.dumps(plain(request), ensure_ascii=False).encode('utf-8')))
            active = MERGE.get()
            if active is not None:
                active['calls'].append(record)

        def wrap_sync(original):
            def observed(*args, **kwargs):
                started = time.monotonic()
                try:
                    result = original(*args, **kwargs)
                    finish_call({'args': args, **kwargs}, result, started)
                    return result
                except Exception as error:
                    owner.recorder.write('llm_error', **owner.context(),
                        request={'args': args, **kwargs}, error=str(error))
                    raise
            return observed

        def wrap_async(original):
            async def observed(client, *args, **kwargs):
                started = time.monotonic()
                try:
                    result = await original(client, *args, **kwargs)
                    finish_call({'args': args, **kwargs}, result, started)
                    return result
                except Exception as error:
                    owner.recorder.write('llm_error', **owner.context(),
                        request={'args': args, **kwargs}, error=str(error))
                    raise
            return observed

        self.replace(self.graph_module.Graph, 'arun', wrap_arun)
        self.replace(self.node_module.Node, 'async_execute', wrap_node)
        self.replace(self.graph_module.Graph, 'get_neighbor_summary_with_ism', wrap_summary)
        self.replace(self.graph_module.Graph, 'merge_multiple_messages', wrap_merge)
        self.replace(self.graph_module, 'cosine_similarity', wrap_cosine)
        self.replace(self.graph_module, 'get_sentence_embedding', wrap_embedding)
        self.replace(self.ollama_module, 'chat', wrap_sync)
        self.replace(self.ollama_module.AsyncClient, 'chat', wrap_async)
        return self

    def uninstall(self):
        for obj, name, original in reversed(self.restores):
            setattr(obj, name, original)
        self.restores.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
