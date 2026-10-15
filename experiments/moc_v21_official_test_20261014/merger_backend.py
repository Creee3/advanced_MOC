"""Meter original five-candidate MOC calls on the private9B service."""
import json
import time
from contextlib import contextmanager
from pathlib import Path

from common import CONFIG, ROOT, BudgetExceeded, InvalidUnit, reserve, charge, rows, save
from trace_hooks import plain
from moc_adaptive_sampling.counterfactual import ExactCallCache, check_response


class MergerBackend:
    def __init__(self, host, directory, context_backend):
        import ollama
        self.original_chat = ollama.AsyncClient.chat
        self.client = ollama.AsyncClient(host=host, timeout=300)
        self.owner = context_backend
        self.failures = []
        self.pending_requests = {}
        self.cache = ExactCallCache(CONFIG['merger_digest'], self.generate, Path(directory)/'merger_calls_raw.jsonl')
        history = ROOT/'data/merger_historical_calls_LOCAL_ONLY.jsonl'
        previous = self.cache.journal
        prior = rows(previous) if previous.exists() else []
        records, rejected = rows(history), []
        for index, event in enumerate(prior):
            try:
                check_response(event['response'])
                records.append(dict(request=event['request'], response=event['response'], provenance=event['provenance']))
            except (KeyError, ValueError) as error:
                rejected.append(dict(raw_line=index+1, reason=str(error)))
        report = self.cache.load_historical(records)
        self.cache.events = prior
        save(Path(directory)/'merger_cache_admission.json', dict(report, conflicts=self.cache.historical_conflicts,
            previous_raw_events=len(prior), rejected_prior_raw=rejected))

    async def generate(self, request):
        if (ROOT/'STOP').exists() or time.time() > json.loads((ROOT/'run_start.json').read_text())['deadline_unix']:
            raise BudgetExceeded('STOP or cumulative wall deadline')
        if sum(self.owner.counter(m['content']) for m in request['messages']) > CONFIG['num_ctx']-64:
            raise InvalidUnit('prompt would exceed safe context envelope')
        reservation = 2*CONFIG['num_ctx']
        reserve(reservation)
        reservation_id = 'merger-'+str(time.time_ns())
        self.pending_requests[reservation_id] = reservation
        self.owner.recorder.write('backend_request_started', stage='test', request=request, **self.owner.context,
                                  reservation_id=reservation_id, reserved_tokens=reservation)
        try:
            response = plain(await self.original_chat(self.client, **request))
        except BaseException as error:
            self.owner.recorder.write('merger_backend_error', request=request, **self.owner.context,
                error_type=type(error).__name__, error=str(error), reserved_cost_unknown=reservation)
            raise
        if not all(isinstance(response.get(k), int) and response[k]>=0 for k in ('prompt_eval_count','eval_count')):
            raise RuntimeError('merger usage missing; keep unresolved reservation')
        charge(reservation, response)
        self.pending_requests.pop(reservation_id)
        self.owner.recorder.write('backend_request_settled',stage='test',**self.owner.context,
            reservation_id=reservation_id,actual_tokens=response['prompt_eval_count']+response['eval_count'])
        return response

    @contextmanager
    def route(self):
        import ollama
        original = ollama.AsyncClient.chat
        owner = self
        async def routed(client, *args, **kwargs):
            try:
                if args or kwargs.get('model') != CONFIG['merger_model']:
                    raise RuntimeError('unexpected author merger request')
                request = dict(kwargs)
                options = dict(request.get('options') or {})
                if options != {'temperature': CONFIG['merger_temperature']}:
                    raise RuntimeError('author merger options differ from frozen default')
                options.update(seed=42, num_ctx=8192, num_gpu=999)
                request['options'] = options
                response = await owner.cache.call(request, stage='test', **owner.owner.context)
                check_response(response)
                return response
            except BaseException as error:
                owner.failures.append(dict(error_type=type(error).__name__, error=str(error)))
                raise
        ollama.AsyncClient.chat = routed
        try:
            yield
        finally:
            ollama.AsyncClient.chat = original
