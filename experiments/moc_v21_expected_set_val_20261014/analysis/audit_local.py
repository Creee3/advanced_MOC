"""Audit immutable raw records, real call charges, coverage and frozen decisions."""
from collections import Counter
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT/'remote_snapshot'
sys.path.insert(0, str(SNAP/'coding_base'))
from moc_adaptive_sampling.counterfactual import ExactCallCache, check_response, request_key


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SNAP/'final_manifest.json').read_text())
    by_file = {f['path']: f for f in manifest['files']}
    assert sha(SNAP/manifest['archive']) == manifest['archive_sha256']
    for name, item in by_file.items():
        assert sha(SNAP/name) == item['sha256'] and (SNAP/name).stat().st_size == item['bytes'], name
    frozen = json.loads((SNAP/'experiment_lock.json').read_text())
    policy = json.loads((SNAP/'policy_lock.json').read_text())
    for name, digest in {**frozen['source_hashes'], **frozen['data_hashes']}.items():
        assert sha(SNAP/name) == digest, name
        assert sha(ROOT/name) == digest, 'local differs: '+name
    assert sha(SNAP/'experiment_lock.json') == policy['experiment_lock_sha256']
    assert sha(SNAP/'group_utility_model.json') == policy['model_sha256']
    assert policy['frozen_before_val'] and not frozen['val_gold_included'] and not frozen['test_included']
    assert not frozen['training_labels_included']
    config = json.loads((SNAP/'config.json').read_text())
    inputs = rows(SNAP/'data/val_inputs.jsonl')
    expected_dags = json.loads((SNAP/'expected_dags.json').read_text())
    qids = [r['question_id'] for r in inputs]
    assert len(qids) == len(set(qids)) == 60
    author_parser = ROOT.parents[2]/'coding_base/MOC-official/src/prompt/aqua_prompt_set.py'
    parser_proof = json.loads((ROOT/'author_parser_remote_check.json').read_text())
    assert parser_proof['author_commit'] == config['author_commit'] and parser_proof['git_status'] == ''
    assert hashlib.sha256(author_parser.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == parser_proof['remote_parser_sha256']
    tree = ast.parse(author_parser.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AQUAPromptSet')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'postprocess_answer')
    function.decorator_list = []
    for arg in function.args.args:
        arg.annotation = None
    function.returns = None
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(author_parser), 'exec'), namespace)
    parse_answer = namespace['postprocess_answer']
    historical = [r for name in ('historical_calls_LOCAL_ONLY.jsonl', 'val_historical_calls_LOCAL_ONLY.jsonl')
                  for r in rows(SNAP/'data'/name)]
    all_results, all_calls, all_events = [], [], []
    cache_checks = 0
    for shard in (0, 1):
        folder = SNAP/'val'/f'shard{shard}'
        results, raw, events = rows(folder/'results.jsonl'), rows(folder/'calls_raw.jsonl'), rows(folder/'trace.jsonl')
        allowed = {(q, .3) for q in qids[shard::2]} | {(q, .7) for q in qids[shard::2]}
        assert len(results) == 60 and {(r['question_id'], r['density']) for r in results} == allowed
        assert len({(r['question_id'], r['density']) for r in results}) == 60
        cache = ExactCallCache(config['main_digest'], None)
        admission = cache.load_historical(historical)
        assert admission == {k: json.loads((folder/'cache_admission.json').read_text())[k] for k in admission}
        for call in raw:
            assert call['stage'] == 'val' and call['arm'] == 'v21_expected_set'
            assert (call['question_id'], call['density']) in allowed
            assert call['request']['model'] == config['main_model']
            assert call['request']['options'] == {'num_predict': 2000, 'temperature': 0, 'seed': 42, 'num_ctx': 8192, 'num_gpu': 999}
            check_response(call['response'])
            assert request_key(call['request'], config['main_digest']) == call['key']
            if call['cache_hit']:
                assert call['key'] in cache.entries
                saved = cache.entries[call['key']]['response']
                assert saved['message']['content'] == call['response']['message']['content']
                assert all(saved[k] == call['response'][k] for k in ('prompt_eval_count', 'eval_count'))
                cache_checks += 1
            else:
                assert call['key'] not in cache.entries
                cache.add(call['request'], call['response'], call['provenance'])
        for r in results:
            assert r['status'] == 'ok' and r['arm'] == 'v21_expected_set' and r['sampling']['valid']
            assert not r['sampling']['failures'] and r['sampling']['merger_llm_calls'] == 0
            matching = [c for c in raw if (c['question_id'], c['density']) == (r['question_id'], r['density'])]
            assert len(matching) == r['logical_calls'] == 8
            assert r['final_output'] == matching[-1]['response']['message']['content']
            assert r['answer'] == parse_answer(None, r['final_output'])
            assert sum(c['response']['prompt_eval_count']+c['response']['eval_count'] for c in matching) == r['logical_model_tokens']
            assert sum(c['response']['prompt_eval_count']+c['response']['eval_count'] for c in matching if not c['cache_hit']) == r['new_model_tokens']
            assert sum(c['cache_hit'] for c in matching) == r['cache_hits']
            assert sum(not c['cache_hit'] for c in matching) == r['new_calls']
            selected = r['sampling']['selections']
            assert len(selected) == 7
            for s in selected:
                assert 0 <= s['communication_tokens'] <= 1024 and len(s['source_ids']) <= 2
                candidates = {m['source']: m for m in s['candidates']}
                assert set(s['source_ids']) <= set(candidates)
                assert s['source_ids'] == [m['source'] for m in s['candidates'] if m['source'] in s['source_ids']]
                for layer in s['layers']:
                    assert not layer['selected_before'] and not layer['uncertainty_used_for_decision']
                    scores = layer['scores']
                    for c in scores:
                        assert c['selected_count'] == 0 and 1 <= c['max_hop'] <= 4
                        assert c['net_mean'] == c['mean'] - .02*c['projected_tokens']/1024
                    feasible = [c for c in scores if c['supported']]
                    best = min(feasible, key=lambda c: (-c['net_mean'], c['projected_tokens'], len(c['sources']), tuple(c['sources']))) if feasible else None
                    if s['source_ids']:
                        assert best and best['net_mean'] > 0 and best['sources'] == s['source_ids']
                        assert best['projected_tokens'] == s['communication_tokens']
                        if len(s['source_ids']) == 2:
                            u, v = best['path']
                            assert best['is_connected_path'] and u in expected_dags[str(r['density'])]['predecessors'][v]
                    else:
                        assert best is None or best['net_mean'] <= 0
        all_results += results
        all_calls += raw
        all_events += events
    first_val = min(e['time_unix'] for e in all_events)
    assert all(not e['timed_out'] for e in all_events if e['event_type'] == 'code_execution_finished')
    communication = [e for e in all_events if e['event_type'] == 'communication_consumed']
    assert len(communication) == 960
    assert Counter(e['question_id'] for e in communication) == Counter({qid: 16 for qid in qids})
    assert Counter(e['question_id'] for e in communication if e['is_final']) == Counter({qid: 2 for qid in qids})
    for name in ('group_utility_model.json', 'policy_lock.json', 'experiment_lock.json', *frozen['source_hashes']):
        assert by_file[name]['mtime_ns']/1e9 < first_val, name
    startup = rows(SNAP/'startup_calls_raw.jsonl')
    assert len(startup) == 1 and startup[0]['response']['done']
    tokens = lambda c: c['response']['prompt_eval_count'] + c['response']['eval_count']
    budget = json.loads((SNAP/'budget.json').read_text())
    new = [c for c in all_calls if not c['cache_hit']]
    assert budget['reserved_tokens'] == 0
    assert budget['actual_tokens'] == sum(map(tokens, new)) + sum(map(tokens, startup))
    assert budget['new_calls'] == len(new)+len(startup)
    assert budget['actual_tokens'] <= config['max_new_model_tokens'] and budget['new_calls'] <= config['max_new_calls']
    status, cleanup = (json.loads((SNAP/name).read_text()) for name in ('status.json', 'cleanup.json'))
    assert status['phase'] == 'complete' and status['cleanup_complete'] and cleanup['cleanup_complete']
    assert not any(cleanup[k] for k in ('remaining_ports', 'owned_children_remaining', 'owned_processes_remaining'))
    assert all(v == 'closed' for v in manifest['live_ports'].values())
    for service in status['services']:
        assert service['loaded']['size_vram'] >= service['loaded']['size']
        assert service['model']['digest'] == config['main_digest']
    started = json.loads((SNAP/'run_start.json').read_text())
    t0 = datetime.datetime.fromisoformat(started['started_utc']).timestamp()
    t1 = datetime.datetime.fromisoformat(status['updated_utc']).timestamp()
    assert t1-t0 <= 7200
    report = dict(status='passed', record_and_cost_consistency='passed', snapshot_files_verified=len(by_file),
        frozen_source_and_data_hashes_verified=len(frozen['source_hashes'])+len(frozen['data_hashes']),
        validation_conditions=len(all_results), valid_val_conditions=len(all_results), logical_calls=len(all_calls),
        historical_and_same_run_cache_hits_verified=cache_checks, output_limit_calls=[],
        all_new_tokens_including_warmup=budget['actual_tokens'], all_new_calls_including_warmup=budget['new_calls'],
        unresolved_reserved_tokens=0, owned_cleanup_complete=True, live_private_ports_closed=True,
        checkpoint_and_policy_timestamps_before_first_val=True, wall_seconds=t1-t0,
        observed_decisions_match_frozen_argmax=True, scope='consistency audit, not independent reproduction or evidence of policy usefulness')
    report.update(final_answers_reparsed_with_author_method=120, final_outputs_match_raw=120,
                  author_answer_parser_sha256=sha(author_parser), communication_events=960,
                  author_parser_lf_sha256=parser_proof['remote_parser_sha256'],
                  code_timeouts=0, selected_pairs_follow_author_dag=True)
    (ROOT/'completion_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
