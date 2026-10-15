"""Fixed-material DAG interventions with exact-request caching and cost ledgers."""
import hashlib
import json
from pathlib import Path
from .sampler import Message


def messages_for(material, receiver, max_hops=4):
    """Match the author's visited BFS and descending-hop serialization order."""
    seen,frontier={receiver},[receiver]
    levels={}
    for hop in range(1,max_hops+1):
        following=[]
        for node in frontier:
            for parent in material['predecessors'][node]:
                if parent not in seen:
                    seen.add(parent); following.append(parent)
        if not following:
            break
        levels[hop]=following; frontier=following
    messages=[]
    for hop in sorted(levels,reverse=True):
        for source in levels[hop]:
            m=material['materials'][source]
            messages.append(Message(source,m['role'],hop,len(messages),m['text'],m['sketch']))
    return messages


def canonical_request(request):
    value = dict(request)
    args = value.pop('args', [])
    if args:
        raise ValueError('positional backend arguments not supported')
    allowed = {'model', 'messages', 'stream', 'options'}
    if set(value) != allowed or value['stream'] is not False:
        raise ValueError('only explicit nonstream chat requests are cacheable')
    return value


def request_key(request, model_digest):
    data = {'digest':model_digest, 'request':canonical_request(request)}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def check_response(response):
    if not response.get('done') or response.get('eval_count', 0) <= 0:
        raise ValueError('incomplete backend response')
    if response.get('done_reason') == 'length':
        raise ValueError('output cap reached; response is not admissible')
    if not isinstance(response.get('message', {}).get('content'), str):
        raise ValueError('missing response body')
    for key in ('prompt_eval_count', 'eval_count'):
        if not isinstance(response.get(key), int) or response[key] < 0:
            raise ValueError('missing actual usage')


class ExactCallCache:
    """Cache content never depends on answer correctness; never overwrites raw.

    Historical records are explicitly linked by hash. Every lookup counts
    virtual inference usage; only misses incur new experimental token expense.
    """
    def __init__(self, model_digest, generate, journal=None):
        self.model_digest, self.generate = model_digest, generate
        self.journal = Path(journal) if journal else None
        self.entries, self.events = {}, []
        self.historical_conflicts = {}

    def load_historical(self, records):
        grouped = {}
        for record in records:
            check_response(record['response'])
            grouped.setdefault(request_key(record['request'],self.model_digest),[]).append(record)
        for key, group in grouped.items():
            outputs={json.dumps({'body':r['response']['message']['content'],
                                 'tool_result':r.get('tool_result')},sort_keys=True,ensure_ascii=False) for r in group}
            if len(outputs)>1:
                self.historical_conflicts[key]=[r['provenance'] for r in group]
                continue  # no convenient historical output is selected.
            first=group[0]
            self.add(first['request'],first['response'],first['provenance'])
            if 'tool_result' in first:
                self.entries[key]['tool_result']=first['tool_result']
        return {'historical_requests':len(grouped), 'reusable_requests':len(self.entries),
                'conflicting_requests':len(self.historical_conflicts)}

    def add(self, request, response, provenance):
        check_response(response)
        key = request_key(request, self.model_digest)
        candidate = {'request':canonical_request(request), 'response':response, 'provenance':provenance}
        if key in self.entries and self.entries[key]['response']['message']['content'] != response['message']['content']:
            raise ValueError('same request has conflicting historical outputs')
        self.entries.setdefault(key, candidate)
        return key

    async def call(self, request, **context):
        key = request_key(request, self.model_digest)
        hit = key in self.entries
        if not hit:
            response = await self.generate(canonical_request(request))
            # Persist before validity checks: truncated/failed calls still cost.
            event = dict(context, key=key, cache_hit=False, request=canonical_request(request),
                         response=response, provenance={'origin':'new_v2_call'})
            self.record(event)
            self.add(request, response, event['provenance'])
        else:
            saved = self.entries[key]
            response = saved['response']
            self.record(dict(context, key=key, cache_hit=True, request=saved['request'],
                             response=response, provenance=saved['provenance']))
        return response

    def record(self, event):
        self.events.append(event)
        if self.journal:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')

    def totals(self, start=0):
        events = self.events[start:]
        def usage(items):
            return sum(x.get('response', {}).get('prompt_eval_count', 0)+
                       x.get('response', {}).get('eval_count', 0) for x in items)
        return {'logical_calls':len(events), 'new_calls':sum(not x['cache_hit'] for x in events),
                'cache_hits':sum(x['cache_hit'] for x in events),
                'logical_model_tokens':usage(events),
                'new_model_tokens':usage([x for x in events if not x['cache_hit']])}


def topological_order(predecessors):
    nodes = list(predecessors)
    if any(p not in predecessors for ps in predecessors.values() for p in ps):
        raise ValueError('DAG has an unknown predecessor')
    degrees = {n:len(set(predecessors[n])) for n in nodes}
    queue = [n for n in nodes if degrees[n]==0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in nodes:
            if node in predecessors[child]:
                degrees[child]-=1
                if degrees[child]==0:
                    queue.append(child)
    if len(order)!=len(nodes):
        raise ValueError('replay requires an acyclic graph')
    return order


async def replay_intervention(material, receiver_id, sources, execute, finalize):
    """Recompute receiver and descendants, preserve all other upstream outputs.

    execute(node_id, outputs, sources_or_None) delegates to original agent code.
    None means original Vanilla spatial inputs, [] means truly empty context.
    finalize delegates to original all-node FinalRefer in author node order.
    """
    preds, frozen = material['predecessors'], material['materials']
    order = topological_order(preds)
    if receiver_id not in frozen or set(frozen)!=set(preds):
        raise ValueError('incomplete fixed material')
    ancestor_ids, pending = set(), list(preds[receiver_id])
    while pending:
        current = pending.pop()
        if current not in ancestor_ids:
            ancestor_ids.add(current)
            pending.extend(preds[current])
    if not set(sources) <= ancestor_ids:
        raise ValueError('intervention reads future or nonancestor messages')
    outputs = {i:m['text'] for i,m in frozen.items()}
    affected = set()
    for node in order:
        if node==receiver_id or any(p in affected for p in preds[node]):
            outputs[node] = await execute(node, outputs, list(sources) if node==receiver_id else None)
            affected.add(node)
    result = await finalize(outputs)
    return {'receiver_output':outputs[receiver_id], 'final_output':result,
            'affected_nodes':[n for n in order if n in affected], 'outputs':outputs}


def factorial_effects(outcomes):
    """2x2 contrasts; binary final outcomes do not prove general synergy."""
    if set(outcomes)!= {'empty','a','b','ab'} or any(v not in (0,1) for v in outcomes.values()):
        raise ValueError('need four binary outcomes')
    y0,ya,yb,yab = (outcomes[k] for k in ('empty','a','b','ab'))
    return {'a_gain':ya-y0, 'b_gain':yb-y0, 'bundle_gain':yab-y0,
            'interaction':yab-ya-yb+y0,
            'bundle_only_success':bool(y0==ya==yb==0 and yab==1),
            'bundle_only_harm':bool(y0==ya==yb==1 and yab==0)}
