"""Read-only-to-raw v2 analysis; validation gold stays in the original local workspace."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parents[2]/'experiments/moc_stateful_sampling_val_20261014'
SNAP=ROOT/'remote_snapshot'

def rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    lock=json.loads((SNAP/'policy_lock.json').read_text())
    assert lock['frozen_before_val']
    assert hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest()==lock['config_sha256']
    assert hashlib.sha256((SNAP/'group_utility_model.json').read_bytes()).hexdigest()==lock['model_sha256']
    source=OLD/'analyze_local.py'
    spec=importlib.util.spec_from_file_location('v1_paired_metric',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    gold={r['question_id']:r['correct'] for r in rows(OLD/'data/val_evaluation_only_LOCAL_ONLY.jsonl')}
    original=[r for p in (OLD/'remote_snapshot/runs').glob('shard*/results.jsonl') for r in rows(p)]
    current=[r for p in (SNAP/'val').glob('shard*/results.jsonl') for r in rows(p)]
    assert len(current)==120 and len({(r['question_id'],r['density']) for r in current})==120
    metrics,comparisons,sensitivity={},{},{}
    invalid=[r for r in current if r['status']!='ok']
    old_limits=json.loads((OLD/'completion_audit.json').read_text())['output_limit_calls']
    for density in (.3,.7):
        valid={r['question_id']:r for r in current if r['density']==density and r['status']=='ok'}
        selections=[s for r in valid.values() for s in r['sampling']['selections']]
        receivers=[s for s in selections if s['candidates']]
        reads=[x for s in selections for x in s['layers'] if x['action']=='read']
        action_modes=Counter(r['selection_mode'] for r in reads)
        stop_reasons=Counter(s['stop_reason'] for s in receivers)
        metrics[str(density)]={'valid_questions':len(valid),'correct':sum(r['answer']==gold[q] for q,r in valid.items()),
            'full_inference_model_tokens':sum(r['logical_model_tokens'] for r in valid.values()),
            'new_validation_model_tokens':sum(r['new_model_tokens'] for r in valid.values()),
            'cache_hits':sum(r['cache_hits'] for r in valid.values()),
            'nonroot_receivers':len(receivers),'empty_nonroot_receivers':sum(not s['source_ids'] for s in receivers),
            'read_action_modes':dict(action_modes),'stop_reasons':dict(stop_reasons),
            'group_reads':sum(len(x['read'][0]['added_sources'])>1 for x in reads),
            'read_hops':dict(Counter(str(m['hop']) for s in receivers for m in s['candidates'] if m['source'] in s['source_ids']))}
        transformed={q:dict(r,stats={'total_model_tokens':r['logical_model_tokens']}) for q,r in valid.items()}
        for arm in ('moc','moc_without_phi','vanilla_mas','adaptive_sampling'):
            baseline={r['question_id']:r for r in original if r['arm']==arm and r['density']==density}
            comparisons[f'{density}_vs_{arm}']=module.paired(baseline,transformed,gold)
            excluded={r['question_id'] for r in invalid if r['density']==density}
            excluded|={r['question_id'] for r in old_limits if r['density']==density}
            sensitivity[f'{density}_vs_{arm}']={'excluded_question_ids':sorted(excluded),
                'paired':module.paired({q:r for q,r in baseline.items() if q not in excluded},
                                       {q:r for q,r in transformed.items() if q not in excluded},gold)}
    committed=[q for p in (SNAP/'training').glob('shard*/questions.jsonl') for q in rows(p)]
    labels=[r for q in committed for r in q['labels']]
    grouped_disagreement={r['question_id'] for r in labels if r['local_delta']!=r['downstream_delta']}
    joint_only={q['question_id'] for q in committed for e in q['episodes'] if e['effects'].get('final',{}).get('bundle_only_success')}
    joint_harm={q['question_id'] for q in committed for e in q['episodes'] if e['effects'].get('final',{}).get('bundle_only_harm')}
    raw=[r for phase in ('training','val') for p in (SNAP/phase).glob('shard*/calls_raw.jsonl') for r in rows(p)]
    new_train=sum(r['response']['prompt_eval_count']+r['response']['eval_count'] for r in raw if r['stage']=='training' and not r['cache_hit'])
    new_val=sum(r['response']['prompt_eval_count']+r['response']['eval_count'] for r in raw if r['stage']=='val' and not r['cache_hit'])
    budget=json.loads((SNAP/'budget.json').read_text())
    calibration=json.loads((SNAP/'training_report.json').read_text())
    config=json.loads((ROOT/'config.json').read_text())
    radius=calibration['interval_radius']
    summary={'metrics':metrics,'comparisons':comparisons,'integrity_sensitivity':sensitivity,'invalid_val':invalid,
             'training':{'valid_questions':len(committed),'labels':len(labels),'disagreement_question_groups':len(grouped_disagreement),
                         'joint_only_help_question_groups':len(joint_only),'joint_only_harm_question_groups':len(joint_harm),
                         'downstream_counts':dict(Counter(str(r['downstream_delta']) for r in labels)),
                         'local_counts':dict(Counter(str(r['local_delta']) for r in labels))},
             'new_cost':{'training_tokens_including_exclusions':new_train,'validation_tokens_including_invalid':new_val,
                         'ledger_all_new_tokens_including_warmup':budget['actual_tokens'],'unresolved_reserved_tokens':budget['reserved_tokens'],
                         'previous_v1_training_and_waste_tokens':657528},
             'model_calibration':calibration,
             'stopping_implication':{'all_feasible_candidates_have_positive_upper_net':bool(radius is not None and radius-1>config['model']['cost_penalty']),
                'reason':'mean>=-1 and radius>1+penalty imply every supported upper net is positive; unsupported bounds are wider',
                'no_learning_success_claim_from_token_savings_alone':True},
             'interpretation':'development val, same60 questions in two densities, no test, historical baselines reused',
             'cached_savings_are_not_algorithm_savings':True}
    (ROOT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'metrics':metrics,'new_cost':summary['new_cost'],'training':summary['training']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
