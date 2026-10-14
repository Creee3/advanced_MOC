"""Fit on original train groups, calibrate separately, freeze before val."""
import hashlib
import json
import numpy as np
from common import ROOT,CONFIG,rows,save
from moc_adaptive_sampling.uncertainty import GroupUtilityModel

def main():
    committed=[q for p in (ROOT/'training').glob('shard*/questions.jsonl') for q in rows(p)]
    if len({q['question_id'] for q in committed})!=len(committed):
        raise RuntimeError('duplicate training question commitment')
    labels=[r for q in committed if q['status']=='ok' for r in q['labels']]
    fit=[r for r in labels if r['fit_or_calibration']=='fit']
    cal=[r for r in labels if r['fit_or_calibration']=='calibration']
    fit_ids={r['question_id'] for r in fit}; cal_ids={r['question_id'] for r in cal}
    if len(fit_ids)<30 or len(cal_ids)<8 or fit_ids & cal_ids:
        raise RuntimeError('insufficient independent fit/calibration groups; no substitute questions')
    params=CONFIG['model']
    model=GroupUtilityModel.fit([r['features'] for r in fit],[r['downstream_delta'] for r in fit],
        [r['question_id'] for r in fit],[r['feature_info'] for r in fit],
        regularization=params['regularization'],bootstrap_replicates=params['bootstrap_replicates'],seed=CONFIG['seed'])
    model.calibrate([r['features'] for r in cal],[r['downstream_delta'] for r in cal],
                    [r['question_id'] for r in cal],[r['feature_info'] for r in cal],alpha=params['alpha'])
    model.write(ROOT/'group_utility_model.json')
    predictions=[model.predict_interval(r['features'],r['feature_info']) for r in cal]
    y=np.asarray([r['downstream_delta'] for r in cal])
    report={'fit_questions':len(fit_ids),'calibration_questions':len(cal_ids),
            'fit_labels':len(fit),'calibration_labels':len(cal),
            'fit_delta_counts':{str(k):sum(r['downstream_delta']==k for r in fit) for k in (-1,0,1)},
            'calibration_delta_counts':{str(k):int((y==k).sum()) for k in (-1,0,1)},
            'calibration_gain_mse':float(np.mean((y-[p['mean'] for p in predictions])**2)),
            'constant_zero_gain_mse':float(np.mean(y*y)),
            'calibration_help_brier':float(np.mean(((y>0)-np.array([p['help'] for p in predictions]))**2)),
            'calibration_harm_brier':float(np.mean(((y<0)-np.array([p['harm'] for p in predictions]))**2)),
            'interval_radius':model.radius,'finite_calibration':model.radius is not None,
            'supported_calibration_rows':sum(p['supported'] for p in predictions),
            'role_counts':{role:sum(r['receiver_role']==role for r in labels) for role in ('Inspector','Math Solver','Programming Expert','Mathematical Analyst')},
            'target_scope':'fixed-material receiver intervention, original Vanilla downstream continuation',
            'val_used':False,'parameters_tuned_on_val':False}
    save(ROOT/'training_report.json',report)
    frozen=json.loads((ROOT/'experiment_lock.json').read_text())
    lock={'frozen_before_val':True,'experiment_lock_sha256':hashlib.sha256((ROOT/'experiment_lock.json').read_bytes()).hexdigest(),
          'config_sha256':hashlib.sha256((ROOT/'config.json').read_bytes()).hexdigest(),
          'model_sha256':hashlib.sha256((ROOT/'group_utility_model.json').read_bytes()).hexdigest(),
          'training_commitments_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in (ROOT/'training').glob('shard*/questions.jsonl')},
          'val_inputs_sha256':hashlib.sha256((ROOT/'data/val_inputs.jsonl').read_bytes()).hexdigest(),
          'source_hashes':frozen['source_hashes']}
    save(ROOT/'policy_lock.json',lock)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    main()
