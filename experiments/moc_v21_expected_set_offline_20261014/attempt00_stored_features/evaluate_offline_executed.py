"""Frozen five-fold question OOF checks on existing initial-state fit labels.

No LLM calls, val data, calibration labels, or baseline reruns. Episode actions
are restricted to the already observed empty/A/B/AB outcomes. Their final cost
is a fixed-material replay description, not a deployed Graph.arun estimate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'coding_base'))
from moc_adaptive_sampling.uncertainty import GroupUtilityModel, coverage_key
from moc_adaptive_sampling.expected_utility import (
    ExpectedGroupUtilityModel, hierarchical_sample_weights,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fit(input_paths):
    rows, episodes, questions = [], {}, []
    for path in input_paths:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            question = json.loads(line)
            # Original index32 was excluded; original >=36 are calibration.
            # Skip calibration before looking at its labels or outcomes.
            if question['question_index'] >= 36:
                continue
            if question['status'] != 'ok' or question['question_index'] == 32:
                raise ValueError('fit input contains invalid/excluded question')
            qid = question['question_id']
            questions.append(qid)
            by_id = {e['episode_id']: e for e in question['episodes']}
            for label in question['labels']:
                if label['fit_or_calibration'] != 'fit':
                    raise ValueError('original fit boundary mismatch')
                if label['left'] != 'empty':
                    continue
                info = label['feature_info']
                if info['selected_count'] != 0:
                    raise ValueError('initial-state label has nonempty state')
                if info['incremental_tokens'] != info['projected_tokens']:
                    raise ValueError('initial-state projected/incremental cost mismatch')
                if not 0 < info['projected_tokens'] <= 1024:
                    raise ValueError('stored action violates frozen budget')
                key = (qid, label['episode_id'])
                episodes[key] = by_id[label['episode_id']]
                rows.append(label)
    if len(questions) != 35 or len(set(questions)) != 35:
        raise ValueError('expected exactly original 35 fit questions')
    rows.sort(key=lambda r: (r['question_id'], r['episode_id'], r['right']))
    for key, episode in episodes.items():
        actual = {r['right'] for r in rows if (r['question_id'], r['episode_id']) == key}
        available = set(episode['variants']) - {'empty'}
        if actual != available or 'empty' not in episode['variants']:
            raise ValueError('incomplete episode action outcomes')
    return rows, episodes, sorted(questions)


def macro_question_values(values, qids, eids):
    result = {}
    values = np.asarray(values, dtype=float)
    for qid in sorted(set(qids)):
        episodes = []
        for eid in sorted({e for q, e in zip(qids, eids) if q == qid}):
            mask = np.asarray([q == qid and e == eid for q, e in zip(qids, eids)])
            episodes.append(float(values[mask].mean()))
        result[qid] = float(np.mean(episodes))
    return result


def descriptive_ci(values, draws=2000, seed=42):
    x = np.asarray(list(values.values()) if isinstance(values, dict) else values, dtype=float)
    if not len(x):
        return {'mean': None, 'low': None, 'high': None}
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    low, high = np.quantile(sampled, [.025, .975])
    return {'mean': float(x.mean()), 'low': float(low), 'high': float(high),
            'resampling_unit': 'fit question', 'draws': draws,
            'scope': 'descriptive OOF fixed-material surrogate; model selection and fold reuse not corrected'}


def legacy_predictions(model, x, infos):
    predictions = []
    for features, info in zip(x, infos):
        heads, gain, spread = model._predict(features)
        predictions.append({'help': float(heads[0]), 'harm': float(heads[1]),
                            'mean': gain, 'bootstrap_spread': spread,
                            'supported': coverage_key(info) in model.coverage})
    return predictions


def fit_model(name, x, y, qids, eids, infos):
    if name == 'frozen_v2_mean':
        return GroupUtilityModel.fit(x, y, qids, infos, regularization=10.,
                                     bootstrap_replicates=32, seed=42)
    return ExpectedGroupUtilityModel.fit(x, y, qids, infos, episode_ids=eids,
        mode='full' if name == 'centered_full' else 'compact',
        regularization=10., bootstrap_replicates=32, seed=42)


def predict_model(name, model, x, infos):
    if name == 'frozen_v2_mean':
        return legacy_predictions(model, x, infos)
    return [model.predict_expected(features, info) for features, info in zip(x, infos)]


def policy_summary(rows, episodes, predictions):
    outcomes = []
    for key in sorted(episodes):
        candidates = [i for i, r in enumerate(rows)
                      if (r['question_id'], r['episode_id']) == key]
        scores = []
        for i in candidates:
            row, prediction = rows[i], predictions[i]
            cost = row['feature_info']['projected_tokens']
            net = float(prediction['mean']) - .02 * cost / 1024
            if prediction['supported']:
                scores.append((net, -cost, row['right'], i))
        chosen = max(scores) if scores else None
        if chosen is not None and chosen[0] > 0:
            i = chosen[3]
            row = rows[i]
            variant = row['right']
            delta = int(row['downstream_delta'])
            cost = row['feature_info']['projected_tokens']
            size = row['feature_info']['group_size']
            predicted = float(predictions[i]['mean'])
        else:
            variant, delta, cost, size, predicted = 'empty', 0, 0, 0, 0.
        episode = episodes[key]
        chosen_full = episode['variants'][variant].get('full_logical_model_tokens')
        empty_full = episode['variants']['empty'].get('full_logical_model_tokens')
        oracle = max([0.] + [float(rows[i]['downstream_delta']) - .02 *
                    rows[i]['feature_info']['projected_tokens']/1024 for i in candidates])
        actual_net = delta - .02 * cost / 1024
        outcomes.append({'question_id': key[0], 'episode_id': key[1], 'variant': variant,
            'group_size': size, 'downstream_delta': delta, 'observed_net_utility': actual_net,
            'predicted_gain': predicted, 'communication_tokens': cost,
            'restricted_oracle_net': oracle, 'restricted_regret': oracle - actual_net,
            'chosen_full_logical_model_tokens': chosen_full,
            'empty_full_logical_model_tokens': empty_full,
            'full_logical_model_token_change': None if chosen_full is None or empty_full is None
                                                else chosen_full - empty_full})
    qids = [r['question_id'] for r in outcomes]
    eids = [r['episode_id'] for r in outcomes]
    stats = {}
    for field in ('downstream_delta', 'observed_net_utility', 'communication_tokens',
                  'restricted_regret', 'full_logical_model_token_change'):
        if all(r[field] is not None for r in outcomes):
            per_q = macro_question_values([r[field] for r in outcomes], qids, eids)
            stats[field] = {**descriptive_ci(per_q), 'per_question': per_q,
                            'episode_sum': float(sum(r[field] for r in outcomes))}
    stats['actions'] = {'empty': sum(r['group_size'] == 0 for r in outcomes),
                       'singleton': sum(r['group_size'] == 1 for r in outcomes),
                       'bundle': sum(r['group_size'] > 1 for r in outcomes)}
    if all(r['chosen_full_logical_model_tokens'] is not None for r in outcomes):
        stats['full_logical_model_tokens'] = {
            'chosen_sum_over_replayed_episodes': sum(r['chosen_full_logical_model_tokens'] for r in outcomes),
            'empty_sum_over_replayed_episodes': sum(r['empty_full_logical_model_tokens'] for r in outcomes),
            'scope': 'counterfactual fixed upstream, Vanilla downstream; episodes reuse one question graph; not deployed full-graph cost'}
    stats['episodes'] = outcomes
    return stats


def run(output_dir):
    source = ROOT / 'experiments' / 'moc_v2_evidence_bundles_train_20261014' / 'remote_snapshot'
    input_paths = [source / 'training' / f'shard{i}' / 'questions.jsonl' for i in (0, 1)]
    rows, episodes, unique = load_fit(input_paths)
    x = np.asarray([r['features'] for r in rows], dtype=float)
    y = np.asarray([r['downstream_delta'] for r in rows], dtype=float)
    qids = np.asarray([r['question_id'] for r in rows])
    eids = np.asarray([r['episode_id'] for r in rows])
    infos = [r['feature_info'] for r in rows]
    shuffled = np.asarray(unique)[np.random.default_rng(42).permutation(len(unique))]
    fold_groups = [list(fold) for fold in np.array_split(shuffled, 5)]
    predictions = {name: [None] * len(rows) for name in
                   ('frozen_v2_mean', 'centered_full', 'centered_compact')}
    prior_pred = np.zeros(len(rows))
    folds = []
    for fold_id, test_qids in enumerate(fold_groups):
        test = np.flatnonzero(np.isin(qids, test_qids))
        train = np.flatnonzero(~np.isin(qids, test_qids))
        weights = hierarchical_sample_weights(qids[train], eids[train])
        prior = float(np.average(y[train], weights=weights))
        prior_pred[test] = prior
        folds.append({'fold': fold_id, 'test_question_ids': test_qids,
                      'train_question_count': len(set(qids[train])),
                      'train_rows': len(train), 'test_rows': len(test),
                      'train_weighted_prior': prior})
        for name in predictions:
            model = fit_model(name, x[train], y[train], qids[train], eids[train], [infos[i] for i in train])
            fold_predictions = predict_model(name, model, x[test], [infos[i] for i in test])
            for index, prediction in zip(test, fold_predictions):
                predictions[name][int(index)] = prediction
    null_values = {'zero': np.zeros(len(rows)), 'fold_train_mean': prior_pred}
    nulls = {}
    for name, prediction in null_values.items():
        errors = (prediction-y)**2
        per_q = macro_question_values(errors, qids, eids)
        nulls[name] = {**descriptive_ci(per_q), 'row_mse': float(errors.mean()),
                       'per_question': per_q}
    models = {}
    checkpoint_dir = output_dir / 'offline_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, values in predictions.items():
        gain = np.asarray([p['mean'] for p in values])
        errors = (gain-y)**2
        per_q = macro_question_values(errors, qids, eids)
        mse = float(np.mean(list(per_q.values())))
        heads = np.asarray([[p['help'], p['harm']] for p in values])
        target = np.stack((y>0, y<0), axis=1).astype(float)
        briers = [float(np.mean(list(macro_question_values((heads[:, k]-target[:, k])**2,
                                                        qids, eids).values()))) for k in (0, 1)]
        policy = policy_summary(rows, episodes, values)
        net_mean = policy['observed_net_utility']['mean']
        passed = mse < nulls['zero']['mean'] and mse < nulls['fold_train_mean']['mean'] and net_mean > 0
        checkpoint = checkpoint_dir / f'{name}.json'
        fit_model(name, x, y, qids, eids, infos).write(checkpoint)
        differences = {}
        for null_name in nulls:
            differences[null_name] = descriptive_ci({qid: per_q[qid] - nulls[null_name]['per_question'][qid]
                                                    for qid in unique})
        models[name] = {'question_macro_mse': mse, 'row_mse': float(errors.mean()),
            'mse_interval': descriptive_ci(per_q), 'per_question_mse': per_q,
            'mse_difference_from_null': differences, 'help_brier': briers[0], 'harm_brier': briers[1],
            'oof_supported_rows': sum(bool(p['supported']) for p in values),
            'oof_total_rows': len(rows), 'restricted_replay_policy': policy,
            'passes_prespecified_training_gate': bool(passed),
            'checkpoint': str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
            'checkpoint_sha256': sha256(checkpoint)}
    eligible = [name for name, value in models.items() if value['passes_prespecified_training_gate']]
    selected = min(eligible, key=lambda name: (models[name]['question_macro_mse'], name)) if eligible else None
    own_sources = [Path(__file__), Path(__file__).with_name('protocol.md'),
                   ROOT/'coding_base'/'moc_adaptive_sampling'/'expected_utility.py',
                   ROOT/'coding_base'/'moc_adaptive_sampling'/'uncertainty.py',
                   ROOT/'coding_base'/'moc_adaptive_sampling'/'bundles.py',
                   ROOT/'coding_base'/'moc_adaptive_sampling'/'stateful.py']
    report = {'schema': 'moc_v21_fit_question_oof_v1', 'protocol_commit': '219aa66',
        'inputs': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256(p)} for p in input_paths],
        'sources': [{'path': str(p.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256(p)} for p in own_sources],
        'data': {'fit_questions': 35, 'episodes': len(episodes), 'initial_state_labels': len(rows),
                 'labels_negative': int((y<0).sum()), 'labels_zero': int((y==0).sum()), 'labels_positive': int((y>0).sum()),
                 'boundary': 'original index<36 excluding invalid index32; no cal/val labels used'},
        'configuration': {'folds': 5, 'seed': 42, 'regularization': 10., 'bootstrap_replicates': 32,
                          'cost_penalty': .02, 'communication_budget': 1024,
                          'decision': 'supported initial-state candidate with greatest total predicted net; empty if net<=0'},
        'fold_assignments': folds, 'null_references': nulls, 'models': models,
        'selected_for_possible_calibration_diagnostic': selected, 'training_gate_passed': bool(eligible),
        'new_llm_calls': 0, 'new_model_tokens': 0,
        'limitations': ['OOF uses existing train outcomes after v2 method failure, not independent confirmation.',
            'Only actually replayed A/B/AB actions compete; unobserved candidate groups are not evaluated.',
            'Binary downstream differences are single stochastic runs; repeated same input once varied historically.',
            'One-time choice here does not establish full-graph selection accuracy, runtime, or token savings.',
            'Per-question bootstrap intervals are descriptive and do not adjust candidate selection or fold dependence.']}
    (output_dir/'oof_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    records = []
    for i, row in enumerate(rows):
        records.append({'question_id': row['question_id'], 'episode_id': row['episode_id'],
                        'right': row['right'], 'delta': row['downstream_delta'],
                        'communication_tokens': row['feature_info']['projected_tokens'],
                        'predictions': {name: values[i] for name, values in predictions.items()},
                        'fold_train_prior': float(prior_pred[i])})
    (output_dir/'oof_predictions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')
    print(json.dumps({'training_gate_passed': report['training_gate_passed'], 'selected': selected,
        'null_mse': {name: value['mean'] for name, value in nulls.items()},
        'models': {name: {'mse': value['question_macro_mse'],
                         'policy_net': value['restricted_replay_policy']['observed_net_utility']['mean'],
                         'actions': value['restricted_replay_policy']['actions'],
                         'passes': value['passes_prespecified_training_gate']} for name, value in models.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.output_dir)
