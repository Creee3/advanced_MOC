"""Only frozen v2.1 and original MOC on HumanEval164 public prompts; gold is not deployed to generation."""
import argparse
import asyncio
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
from common import ROOT, CONFIG, rows, save, append, imports, BudgetExceeded, InvalidUnit, QUESTION
from merger_backend import MergerBackend
from runtime_controls import stable_ids
from moc_adaptive_sampling.expected_set import ExpectedSetConfig, ExpectedSetSampler
from moc_adaptive_sampling.expected_utility import ExpectedGroupUtilityModel
from moc_adaptive_sampling.adapter import MOCSamplingAdapter
from moc_adaptive_sampling.paper_baselines import PaperRecipe, make_graph
from moc_adaptive_sampling.counterfactual import check_response

ARMS = ('v21_expected_set', 'moc')
DENSITIES = (.3, .7)


def assert_own_cost_settled(backend, merger):
    """A different worker's live reservation cannot invalidate this condition.

    The shared ledger is still enforced by reserve/charge. Only unresolved
    requests actually owned by this worker require halting its error handler.
    """
    pending = {**backend.pending_requests, **merger.pending_requests}
    if pending:
        raise RuntimeError('this worker has unresolved backend cost: '+str(pending))


def totals(main, merger, mstart, gstart):
    a, b = main.cache.totals(mstart), merger.cache.totals(gstart)
    result = {key: a[key]+b[key] for key in a}
    result.update(main_calls=a['logical_calls'], merger_calls=b['logical_calls'],
                  main_model_tokens=a['logical_model_tokens'], merger_model_tokens=b['logical_model_tokens'])
    return result


def completed_records(out, assigned):
    allowed = {(q['question_id'], arm, density) for _, q in assigned for arm in ARMS for density in DENSITIES}
    records = rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    done = {}
    for row in records:
        key = row['question_id'], row['arm'], row['density']
        if key not in allowed or key in done or row['status'] not in ('ok', 'invalid'):
            raise RuntimeError('duplicate/nonterminal/out-of-shard committed test condition')
        if row['status'] == 'ok' and not 1 <= row['main_calls'] <= 8:
            raise RuntimeError('incomplete saved author graph')
        done[key] = row
    return done


async def run(args):
    lock = json.loads((ROOT/'policy_lock.json').read_text())
    if not lock['frozen_before_test']:
        raise RuntimeError('test policy not frozen')
    for name, digest in lock['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    model = ExpectedGroupUtilityModel.load(ROOT/'group_utility_model.json', lock['model_sha256'])
    inputs = rows(ROOT/'data/test_inputs.jsonl')
    assert len(inputs) == len({r['question_id'] for r in inputs}) == 164
    assert hashlib.sha256((ROOT/'data/test_inputs.jsonl').read_bytes()).hexdigest() == lock['test_inputs_sha256']
    assigned = [(index, row) for index, row in enumerate(inputs) if index % CONFIG['workers'] == args.shard]
    out = ROOT/'test'/f'shard{args.shard}'
    done = completed_records(out, assigned)
    out.mkdir(parents=True, exist_ok=True)
    api, backend, math_module, final_class = imports(args.main_host, out, 'test')
    # Joint conflict rejection for main request and tool-result histories on resume.
    from resume_cache import restore_main_cache
    restore_main_cache(backend, out)
    merger = MergerBackend(args.merger_host, out, backend)
    from src.llm.profile_embedding import get_sentence_embedding
    roles = json.loads((ROOT/'code_role_snapshot.json').read_text())['roles']
    sampler = ExpectedSetSampler(ExpectedSetConfig(communication_tokens=1024, max_group_size=2,
        cost_penalty=.02, local_state_weight=0.), backend.counter, model)
    with backend.controls(math_module, final_class), merger.route():
        for index, row in assigned:
            qid = row['question_id']; token = QUESTION.set(qid)
            try:
                task = row['prompt']
                for dindex, density in enumerate(DENSITIES):
                    arm_order = ARMS if (index+dindex)%2 == 0 else tuple(reversed(ARMS))
                    for arm in arm_order:
                        key = qid, arm, density
                        if key in done:
                            continue
                        backend.context = dict(question_id=qid, arm=arm, density=density)
                        start, gstart = len(backend.cache.events), len(merger.cache.events)
                        failure_start = len(merger.failures)
                        adapter = None
                        try:
                            random.seed(42); np.random.seed(42)
                            import torch
                            torch.manual_seed(42)
                            with stable_ids():
                                graph = make_graph(api, PaperRecipe(domain="humaneval", edge_density=density, use_cot=False),
                                    'moc' if arm == 'moc' else 'adaptive_sampling', roles=roles)
                            for node in [*graph.nodes.values(), graph.decision_node]:
                                node.llm = backend.llm(node.llm)
                            if arm != 'moc':
                                adapter = MOCSamplingAdapter(graph, sampler, get_sentence_embedding, counter_name=backend.counter.name, code_writing=True)
                            output = await graph.arun({'task': task}, num_rounds=1, max_tries=1, max_time=300)
                            if len(merger.failures) > failure_start:
                                if any(e['error_type'] == 'BudgetExceeded' for e in merger.failures[failure_start:]):
                                    raise BudgetExceeded('cost/STOP limit inside original merger')
                                raise InvalidUnit('original merger failed; fallback is not valid MOC: '+str(merger.failures[failure_start:]))
                            if len(output[0]) != 1 or any(not node.outputs for node in graph.nodes.values()):
                                raise InvalidUnit('incomplete graph outputs')
                            stats = totals(backend, merger, start, gstart)
                            if not 1 <= stats['main_calls'] <= 8 or (arm != 'moc' and stats['merger_calls']):
                                raise InvalidUnit('expected up to seven ordinary calls plus FinalWriteCode')
                            if arm == 'moc' and stats['merger_calls'] % 5:
                                raise InvalidUnit('incomplete original five-candidate merger')
                            from datasets.data_process import HumanEvalDataset
                            answer = HumanEvalDataset.postprocess_answer(None, output[0][0])
                            item = dict(question_id=qid, density=density, arm=arm, status='ok',
                                answer=answer, final_output=output[0][0], early_return_count=8-stats['main_calls'], **stats,
                                sampling=adapter.ledger if adapter else None)
                        except BudgetExceeded:
                            raise
                        except Exception as error:
                            assert_own_cost_settled(backend, merger)
                            reasons = [str(error), *(adapter.ledger.get('failures', []) if adapter else [])]
                            if any('BudgetExceeded' in reason for reason in reasons):
                                raise BudgetExceeded('limit inside author graph') from error
                            phrases = ('output cap reached', 'code timeout', 'prompt would exceed safe context envelope',
                                       'incomplete backend response', 'incomplete graph outputs', 'expected seven ordinary calls')
                            if not isinstance(error, InvalidUnit) and not any(p in r for p in phrases for r in reasons):
                                raise
                            item = dict(question_id=qid, density=density, arm=arm, status='invalid',
                                reason=str(error), **totals(backend, merger, start, gstart))
                        finally:
                            if adapter:
                                adapter.detach()
                        append(out/'results.jsonl', item); done[key] = item
                        costs = totals(backend, merger, 0, 0)
                        save(out/'progress.json', dict(processed_conditions=len(done),
                            completed_conditions=sum(r['status'] == 'ok' for r in done.values()),
                            invalid_conditions=sum(r['status'] == 'invalid' for r in done.values()),
                            target_conditions=len(assigned)*4, last_question=qid, **costs))
                        print(f'TEST processed={len(done)}/{len(assigned)*4} arm={arm} density={density} status={item["status"]}', flush=True)
            finally:
                QUESTION.reset(token)
    save(out/'finished.json', dict(status='complete', processed_conditions=len(done),
        valid_conditions=sum(r['status'] == 'ok' for r in done.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--main-host', required=True)
    parser.add_argument('--merger-host', required=True)
    asyncio.run(run(parser.parse_args()))
