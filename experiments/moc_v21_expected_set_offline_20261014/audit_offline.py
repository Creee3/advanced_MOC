"""Read-only independent arithmetic/integrity check of the fit-only OOF output."""
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row['question_id'], row['episode_id'], row['right']


def hierarchy_mean(values):
    per_question = defaultdict(lambda: defaultdict(list))
    for qid, eid, value in values:
        per_question[qid][eid].append(float(value))
    return sum(sum(sum(actions)/len(actions) for actions in episodes.values())/len(episodes)
               for episodes in per_question.values())/len(per_question)


def close(a, b):
    if abs(a-b) > 1e-12:
        raise ValueError(('arithmetic mismatch', a, b))


def main():
    report = json.loads((BASE/'oof_report.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (BASE/'oof_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    derived = [json.loads(line) for line in (BASE/'derived_fit_features.jsonl').read_text(encoding='utf-8').splitlines()]
    derived_map = {key(row): row for row in derived}
    assert len(records) == len(derived_map) == 176
    checked_hashes = []
    for item in report['sources'] + report['inputs'] + report['derived_data']:
        actual = sha(ROOT/item['path'])
        assert actual == item['sha256'], item['path']
        checked_hashes.append({'path': item['path'], 'sha256': actual})
    originals, episodes = {}, {}
    for item in report['inputs']:
        for line in (ROOT/item['path']).read_text(encoding='utf-8').splitlines():
            question = json.loads(line)
            if question['question_index'] >= 36:
                continue
            assert question['question_index'] != 32 and question['status'] == 'ok'
            for row in question['labels']:
                if row['left'] == 'empty':
                    assert row['fit_or_calibration'] == 'fit'
                    originals[key(row)] = row
            for episode in question['episodes']:
                episodes[question['question_id'], episode['episode_id']] = episode
    assert set(originals) == set(derived_map)
    repair_count = 0
    for action, row in derived_map.items():
        original = originals[action]
        allowed = copy.deepcopy(original)
        if original['feature_info']['group_size'] == 1:
            allowed['feature_info']['path'] = original['feature_info']['added_sources']
            allowed['feature_info']['is_connected_path'] = False
            # 32 member means +32 maxima +the seventh group extra.
            allowed['features'][70] = 0.
        assert allowed == row
        repair_count += original != row
    assert repair_count == report['schema_repair']['changed_initial_state_rows'] == 106
    qids = {r['question_id'] for r in derived}
    fold_of = {}
    for fold in report['fold_assignments']:
        assert len(fold['test_question_ids']) == 7
        for qid in fold['test_question_ids']:
            assert qid not in fold_of
            fold_of[qid] = fold['fold']
    assert set(fold_of) == qids and len(qids) == 35
    for fold in report['fold_assignments']:
        train_values = [(r['question_id'], r['episode_id'], r['downstream_delta']) for r in derived
                        if fold_of[r['question_id']] != fold['fold']]
        close(hierarchy_mean(train_values), fold['train_weighted_prior'])
    close(hierarchy_mean([(r['question_id'], r['episode_id'], r['delta']**2) for r in records]),
          report['null_references']['zero']['mean'])
    close(hierarchy_mean([(r['question_id'], r['episode_id'], (r['fold_train_prior']-r['delta'])**2)
                          for r in records]), report['null_references']['fold_train_mean']['mean'])
    models = {}
    for name, result in report['models'].items():
        close(hierarchy_mean([(r['question_id'], r['episode_id'], (r['predictions'][name]['mean']-r['delta'])**2)
                              for r in records]), result['question_macro_mse'])
        outcomes = result['restricted_replay_policy']['episodes']
        for outcome in outcomes:
            episode_key = outcome['question_id'], outcome['episode_id']
            competing = [r for r in records if (r['question_id'], r['episode_id']) == episode_key
                         and r['predictions'][name]['supported']]
            chosen = min(competing, key=lambda r: (-r['predictions'][name]['mean']+.02*r['communication_tokens']/1024,
                r['communication_tokens'], derived_map[key(r)]['feature_info']['group_size'],
                tuple(derived_map[key(r)]['feature_info']['added_sources']))) if competing else None
            best_positive = chosen and chosen['predictions'][name]['mean']-.02*chosen['communication_tokens']/1024 > 0
            assert outcome['variant'] == (chosen['right'] if best_positive else 'empty')
            assert outcome['downstream_delta'] == (chosen['delta'] if best_positive else 0)
            assert outcome['communication_tokens'] == (chosen['communication_tokens'] if best_positive else 0)
            close(outcome['observed_net_utility'], outcome['downstream_delta']-.02*outcome['communication_tokens']/1024)
            variants = episodes[episode_key]['variants']
            assert outcome['chosen_full_logical_model_tokens'] == variants[outcome['variant']]['full_logical_model_tokens']
            assert outcome['empty_full_logical_model_tokens'] == variants['empty']['full_logical_model_tokens']
        close(hierarchy_mean([(r['question_id'], r['episode_id'], r['observed_net_utility']) for r in outcomes]),
              result['restricted_replay_policy']['observed_net_utility']['mean'])
        checkpoint = ROOT/result['checkpoint']
        assert sha(checkpoint) == result['checkpoint_sha256']
        checkpoint_payload = json.loads(checkpoint.read_text(encoding='utf-8'))
        assert set(checkpoint_payload['metadata']['fit_questions']) == qids
        models[name] = {'mse_arithmetic': True, 'action_and_cost_arithmetic': True,
                        'checkpoint_sha256': sha(checkpoint),
                        'positive_net_cost_tie_episodes': result['restricted_replay_policy']['tie_audit']['positive_max_net_and_cost_tie_episodes']}
    assert report['selected_for_possible_calibration_diagnostic'] is None
    assert report['training_gate_passed'] is False
    audit = {'schema': 'moc_v21_offline_audit_v1', 'passed': True,
             'oof_report_sha256': sha(BASE/'oof_report.json'),
             'oof_predictions_sha256': sha(BASE/'oof_predictions.jsonl'),
             'source_data_hashes_checked': checked_hashes,
             'fit_questions': len(qids), 'initial_state_rows': len(records), 'schema_repaired_rows': repair_count,
             'folds_question_disjoint_and_complete': True, 'schema_rewards_costs_other_fields_unchanged': True,
             'null_prior_and_model_hierarchical_arithmetic_verified': True, 'models': models,
             'new_llm_calls': 0, 'new_model_tokens': 0,
             'calibration_outcomes_inspected': False, 'val_or_test_used': False,
             'scope': 'artifact integrity and arithmetic only; no statistical or deployment success claim'}
    (BASE/'offline_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'schema_repaired_rows': repair_count,
                      'models': models}, indent=2))


if __name__ == '__main__':
    main()
