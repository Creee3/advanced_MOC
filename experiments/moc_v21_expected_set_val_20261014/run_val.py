"""Frozen v2.1 only on the existing diagnostic val60; no baseline calls.

Completed (question_id, arm, density) records, including invalid terminal
conditions, are never regenerated. Partial work preserves its raw usage and
can reuse only mutually consistent exact requests on a later bounded attempt.
"""
import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from common import ROOT, CONFIG, rows, save, append, imports, BudgetExceeded, InvalidUnit, QUESTION
from runtime_controls import stable_ids
from moc_adaptive_sampling.counterfactual import ExactCallCache, check_response, request_key
from moc_adaptive_sampling.expected_set import ExpectedSetConfig, ExpectedSetSampler
from moc_adaptive_sampling.expected_utility import ExpectedGroupUtilityModel, SCHEMA
from moc_adaptive_sampling.adapter import MOCSamplingAdapter
from moc_adaptive_sampling.paper_baselines import PaperRecipe, make_graph

ARM = 'v21_expected_set'
DENSITIES = (.3, .7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_records(out, assigned):
    """Reject duplicate/out-of-shard records rather than silently overriding."""
    allowed = {(r['question_id'], ARM, density) for r in assigned for density in DENSITIES}
    records = rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done = {}
    for item in records:
        key = item['question_id'], item['arm'], item['density']
        if key not in allowed or key in done or item.get('status') not in ('ok', 'invalid'):
            raise RuntimeError('duplicate, nonterminal, or out-of-shard saved val condition')
        if item['status'] == 'ok' and item.get('logical_calls') != 8:
            raise RuntimeError('saved successful condition has incomplete author graph')
        done[key] = item
    return done


def restore_exact_cache(backend, out):
    """Admit old and own raw jointly, excluding every conflicting key.

    Existing costs remain in the shared budget. Replayed cached calls still
    contribute full logical deployment tokens to the resulting condition.
    A truncated response remains raw and paid, but is not cache-admissible.
    """
    files = [ROOT/'data/historical_calls_LOCAL_ONLY.jsonl', ROOT/'data/val_historical_calls_LOCAL_ONLY.jsonl']
    historical = [record for path in files if path.exists() for record in rows(path)]
    prior_raw = rows(out/'calls_raw.jsonl') if (out/'calls_raw.jsonl').exists() else []
    prior_trace = rows(out/'trace.jsonl') if (out/'trace.jsonl').exists() else []
    backend.recorder.counter = max((event.get('event_index', 0) for event in prior_trace), default=0)
    tool_values = {}
    for event in prior_trace:
        if event.get('event_type') in ('new_tool_result', 'cached_tool_result'):
            tool_values.setdefault(event['request_key'], set()).add(str(event['result']))
    own_records, rejected = [], []
    source_sha = sha(out/'calls_raw.jsonl') if prior_raw else None
    for index, event in enumerate(prior_raw):
        try:
            check_response(event['response'])
            key = request_key(event['request'], CONFIG['main_digest'])
            if key != event['key']:
                raise ValueError('raw request-key integrity mismatch')
        except (KeyError, ValueError) as error:
            rejected.append(dict(raw_line=index+1, reason=str(error)))
            continue
        values = tool_values.get(key, {None})
        for value in values:
            record = dict(request=event['request'], response=event['response'], provenance={
                'origin': 'same_v21_attempt_resume', 'raw_sha256': source_sha,
                'raw_line': index+1, 'original_provenance': event.get('provenance')})
            if value is not None:
                record['tool_result'] = value
            own_records.append(record)
    # Rebuild in one pass: otherwise a later conflicting output could leave
    # an earlier convenient output available in the cache.
    backend.cache.entries.clear()
    backend.cache.historical_conflicts.clear()
    report = backend.cache.load_historical(historical + own_records)
    backend.cache.events = list(prior_raw)
    save(out/'resume_cache_admission.json', dict(
        report, conflicts=backend.cache.historical_conflicts,
        rejected_prior_raw=rejected, previous_raw_events=len(prior_raw),
        source_raw_sha256=source_sha))


def check_lock():
    lock = json.loads((ROOT/'policy_lock.json').read_text(encoding='utf-8'))
    if not lock.get('frozen_before_val') or lock.get('model_schema') != SCHEMA:
        raise RuntimeError('v2.1 model policy was not frozen before diagnostic val')
    for path, expected in ((ROOT/'config.json', lock['config_sha256']),
                           (ROOT/'data/val_inputs.jsonl', lock['val_inputs_sha256'])):
        if sha(path) != expected:
            raise RuntimeError('frozen val policy/data hash mismatch: '+str(path))
    for relative, expected in lock.get('source_hashes', {}).items():
        if sha(ROOT/relative) != expected:
            raise RuntimeError('frozen source hash mismatch: '+relative)
    model = ExpectedGroupUtilityModel.load(ROOT/'group_utility_model.json', lock['model_sha256'])
    if model.mode != 'compact' or model.metadata.get('fit_question_count') != 35:
        raise RuntimeError('only the existing centered_compact train35 checkpoint is authorized')
    return model


async def run(args):
    model = check_lock()
    out = ROOT/'val'/f'shard{args.shard}'
    inputs = rows(ROOT/'data/val_inputs.jsonl')
    if len(inputs) != 60 or len({r['question_id'] for r in inputs}) != 60:
        raise RuntimeError('frozen val60 input coverage mismatch')
    assigned = [row for index, row in enumerate(inputs) if index % CONFIG['workers'] == args.shard]
    done = completed_records(out, assigned)
    completed = sum(item['status'] == 'ok' for item in done.values())
    invalid = len(done) - completed
    if len(done) == len(assigned)*len(DENSITIES):
        saved_costs = ExactCallCache(CONFIG['main_digest'], None)
        saved_costs.events = rows(out/'calls_raw.jsonl') if (out/'calls_raw.jsonl').exists() else []
        save(out/'progress.json', dict(completed_conditions=completed, invalid_conditions=invalid,
                                     processed_conditions=len(done), target_conditions=len(done),
                                     **saved_costs.totals()))
        save(out/'finished.json', dict(status='complete', completed_conditions=completed,
                                     invalid_conditions=invalid, already_complete_on_entry=True))
        return
    api, backend, math_module, final_class = imports(args.main_host, out, 'val')
    restore_exact_cache(backend, out)
    from src.llm.profile_embedding import get_sentence_embedding
    roles = json.loads((ROOT/'val_role_snapshot.json').read_text(encoding='utf-8'))['roles']
    params = CONFIG['model']
    sampler = ExpectedSetSampler(ExpectedSetConfig(
        communication_tokens=CONFIG['communication_tokens'], max_group_size=params['max_group_size'],
        cost_penalty=params['cost_penalty'], local_state_weight=0.), backend.counter, model)
    with backend.controls(math_module, final_class):
        for row in assigned:
            qid = row['question_id']
            token = QUESTION.set(qid)
            try:
                task = 'Question: '+row['question']+'\n'+''.join(option+'\n' for option in row['options'])
                for density in DENSITIES:
                    key = qid, ARM, density
                    if key in done:
                        continue
                    backend.context = {'question_id': qid, 'density': density, 'arm': ARM}
                    start = len(backend.cache.events)
                    adapter = None
                    try:
                        random.seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])
                        import torch
                        torch.manual_seed(CONFIG['seed'])
                        with stable_ids():
                            graph = make_graph(api, PaperRecipe(edge_density=density, use_cot=CONFIG['use_cot']),
                                               'adaptive_sampling', roles=roles)
                        for node in [*graph.nodes.values(), graph.decision_node]:
                            node.llm = backend.llm(node.llm)
                        adapter = MOCSamplingAdapter(graph, sampler, get_sentence_embedding,
                                                     counter_name=backend.counter.name)
                        # Preserve the author's complete Graph.arun and all-output FinalRefer.
                        result = await graph.arun({'task': task}, num_rounds=1, max_tries=1, max_time=300)
                        if len(result[0]) != 1 or any(not node.outputs for node in graph.nodes.values()):
                            raise InvalidUnit('incomplete graph outputs')
                        prompt = graph.decision_node.prompt_set
                        answer = prompt.postprocess_answer(prompt, result[0][0])
                        stats = backend.cache.totals(start)
                        if stats['logical_calls'] != 8:
                            raise InvalidUnit('expected seven ordinary calls plus FinalRefer')
                        item = dict(question_id=qid, density=density, arm=ARM, status='ok', answer=answer,
                                    final_output=result[0][0], **stats, sampling=adapter.ledger)
                        completed += 1
                    except BudgetExceeded:
                        raise
                    except Exception as error:
                        reasons = [str(error), *(adapter.ledger.get('failures', []) if adapter else [])]
                        if any('BudgetExceeded' in reason for reason in reasons):
                            raise BudgetExceeded('call/token/STOP limit reached inside author graph')
                        integrity = ('output cap reached', 'code timeout', 'prompt would exceed safe context envelope',
                                     'incomplete backend response', 'incomplete graph outputs',
                                     'expected seven ordinary calls')
                        if not isinstance(error, InvalidUnit) and not any(
                            phrase in reason for phrase in integrity for reason in reasons):
                            raise
                        item = dict(question_id=qid, density=density, arm=ARM, status='invalid',
                                    reason=str(error), **backend.cache.totals(start))
                        invalid += 1
                    finally:
                        if adapter:
                            adapter.detach()
                    append(out/'results.jsonl', item)
                    done[key] = item
                    save(out/'progress.json', dict(
                        completed_conditions=completed, invalid_conditions=invalid,
                        processed_conditions=len(done), target_conditions=len(assigned)*len(DENSITIES),
                        last_question=qid, **backend.cache.totals()))
                    print(f'VAL v2.1 processed={len(done)}/{len(assigned)*len(DENSITIES)} '
                          f'valid={completed} density={density}', flush=True)
            finally:
                QUESTION.reset(token)
    save(out/'finished.json', dict(status='complete', completed_conditions=completed, invalid_conditions=invalid))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--main-host', required=True)
    asyncio.run(run(parser.parse_args()))
