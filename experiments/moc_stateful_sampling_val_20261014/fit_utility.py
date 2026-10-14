"""Question-group calibration, then freeze checkpoint before touching val."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/"config.json").read_text())
sys.path.insert(0,CONFIG["extension_directory"])
from moc_adaptive_sampling.stateful import MarginalUtilityModel, FEATURE_NAMES

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    if (ROOT/"policy_lock.json").exists():
        raise RuntimeError("policy already frozen; never refit using validation outcomes")
    labels,questions,excluded=[],[],[]
    for shard in range(CONFIG["workers"]):
        folder=ROOT/"training"/f"shard{shard}"
        assert json.loads((folder/"finished.json").read_text())["status"]=="complete"
        questions+=rows(folder/"questions.jsonl")
        labels+=rows(folder/"labels.jsonl")
        if (folder/"excluded_questions.jsonl").exists():
            excluded+=rows(folder/"excluded_questions.jsonl")
    valid_ids={q["question_id"] for q in questions}
    excluded_ids={q["question_id"] for q in excluded}
    assert not valid_ids&excluded_ids and len(valid_ids|excluded_ids)==48
    assert len(valid_ids)==len(questions) and all(r["question_id"] in valid_ids for r in labels)
    assert all(r["question_index"]<48 for r in labels)
    train=[r for r in labels if r["question_index"]<36]
    calibration=[r for r in labels if r["question_index"]>=36]
    assert train and calibration
    fit_groups={r["question_id"] for r in train}
    calibration_groups={r["question_id"] for r in calibration}
    assert len(fit_groups)>=30 and len(calibration_groups)>=8,"too many invalid training questions; do not proceed to val"
    scores=[]
    for regularization in (10.,100.,1000.):
        model=MarginalUtilityModel.fit([r["features"] for r in train],[r["label"] for r in train],regularization)
        mse=float(np.mean([(model.predict(np.asarray(r["features"]))-r["label"])**2 for r in calibration]))
        scores.append({"regularization":regularization,"calibration_mse":mse})
    best=min(scores,key=lambda r:(r["calibration_mse"],-r["regularization"]))
    model=MarginalUtilityModel.fit([r["features"] for r in train],[r["label"] for r in train],best["regularization"],
        metadata={"fit_questions":len(fit_groups),"calibration_questions":len(calibration_groups),"fit_pairs":len(train),"calibration_pairs":len(calibration),
                  "regularization":best["regularization"],"validation_used":False,
                  "label_scope":"paired local receiver-answer surrogate"})
    model.write(ROOT/"utility_model.json")
    hist=lambda records:{str(y):sum(r["label"]==y for r in records) for y in (-1,0,1)}
    predictions=np.asarray([model.predict(np.asarray(r["features"])) for r in calibration])
    report={"schema":"v1_training_report","paired_labels":len(labels),"fit_pairs":len(train),
            "planned_training_questions":48,"valid_training_questions":len(valid_ids),
            "excluded_questions":excluded,"fit_question_groups":len(fit_groups),"calibration_question_groups":len(calibration_groups),
            "calibration_pairs":len(calibration),"fit_label_histogram":hist(train),
            "calibration_label_histogram":hist(calibration),"regularization_candidates":scores,
            "selected_regularization":best["regularization"],"calibration_predictions_range":[float(predictions.min()),float(predictions.max())],
            "calibration_positive_predictions":int((predictions>0).sum()),
            "constant_zero_calibration_mse":float(np.mean([r["label"]**2 for r in calibration])),
            "underpowered_training_flag":sum(r["label"]!=0 for r in train)<20,
            "validation_used":False,"feature_names":list(FEATURE_NAMES)}
    (ROOT/"training_report.json").write_text(json.dumps(report,indent=2)+"\n")
    lock={"frozen_before_validation":True,"utility_model_sha256":hashlib.sha256((ROOT/"utility_model.json").read_bytes()).hexdigest(),
          "config_sha256":hashlib.sha256((ROOT/"config.json").read_bytes()).hexdigest(),
          "val_inputs_sha256":hashlib.sha256((ROOT/"data/val_inputs.jsonl").read_bytes()).hexdigest(),
          "train_labels_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT/"training").glob("shard*/labels.jsonl"))},
          "candidate":CONFIG["candidate"],"val_gold_on_server":False}
    paths=[ROOT/x for x in ("run_experiment.py","runtime_controls.py","trace_hooks.py","collect_training.py","fit_utility.py","training_integrity.py")]
    paths += list((ROOT/"coding_base/moc_adaptive_sampling").glob("*.py"))
    lock["execution_source_sha256"]={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (ROOT/"policy_lock.json").write_text(json.dumps(lock,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
