"""Isolated v1 deployment, agreed splits and official training source."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
OLD = ROOT.parent / "moc_adaptive_sampling_20261014"

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def fingerprint(row):
    return hashlib.sha256(json.dumps({"question": row["question"], "options": row["options"]},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

def write_rows(path, value):
    path.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in value), encoding="utf-8")

def main():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    sources = []
    for split in ("dev", "train"):
        path = data / f"official_{split}_LOCAL_ONLY.jsonl"
        url = f"https://raw.githubusercontent.com/google-deepmind/AQuA/master/{split}.json"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response, path.with_suffix(".tmp").open("wb") as stream:
                shutil.copyfileobj(response, stream)
            path.with_suffix(".tmp").replace(path)
        values = rows(path)
        sources.append({"split": split, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "records": len(values)})
    official_dev = rows(data / "official_dev_LOCAL_ONLY.jsonl")
    existing_dev = rows(OLD / "data/dev_inputs.jsonl")
    original_manifest = json.loads((ROOT.parent / "moc_component_ablation_20261011/data/manifest.json").read_text(encoding="utf-8"))
    assert len(official_dev) == 254
    assert hashlib.sha256((data / "official_dev_LOCAL_ONLY.jsonl").read_bytes()).hexdigest() == original_manifest["source_sha256"]
    selected = sorted(original_manifest["manifest"], key=lambda r:r["selection_rank"])
    assert {fingerprint(official_dev[x["official_dev_row_index"]]) for x in selected[:30]} == {fingerprint(x) for x in existing_dev}
    assert [x["question_id"] for x in selected[:30]] == [x["question_id"] for x in existing_dev]
    val_entries = selected[30:90]
    val = [official_dev[x["official_dev_row_index"]] for x in val_entries]
    official_test = rows(WORKSPACE / "coding_base/MOC-official/datasets/test/aqua_test_n254.jsonl")
    reserved = {fingerprint(x) for x in official_dev + official_test}
    train_pool = sorted(rows(data / "official_train_LOCAL_ONLY.jsonl"), key=fingerprint)
    deduplicated = {}
    for x in train_pool:
        if fingerprint(x) not in reserved:
            deduplicated.setdefault(fingerprint(x), x)
    train = list(deduplicated.values())[:48]
    assert len(train) == 48
    for name, values in (("train", train), ("val", val)):
        identities = [fingerprint(x) for x in values] if name == "train" else [x["question_id"] for x in val_entries]
        write_rows(data / f"{name}_inputs.jsonl", [{"question_id": identity, "question": x["question"], "options": x["options"]} for x,identity in zip(values,identities)])
        write_rows(data / f"{name}_evaluation_only_LOCAL_ONLY.jsonl", [{"question_id": identity, "correct": x["correct"]} for x,identity in zip(values,identities)])
    # Training-only labels are accessible solely to training workers; no val gold uploaded.
    shutil.copyfile(data / "train_evaluation_only_LOCAL_ONLY.jsonl", data / "train_labels.jsonl")
    shutil.copyfile(OLD / "data/dev_inputs.jsonl", data / "dev_inputs.jsonl")
    shutil.copyfile(OLD / "data/dev_evaluation_only_LOCAL_ONLY.jsonl", data / "dev_evaluation_only_LOCAL_ONLY.jsonl")
    manifest = {"dev": 30, "val": 60, "test": 254, "training": 48, "sources": sources,
                "val_rule": original_manifest["selection_method"], "original_seed": original_manifest["seed_text"],
                "original_manifest_sha256": hashlib.sha256((ROOT.parent / "moc_component_ablation_20261011/data/manifest.json").read_bytes()).hexdigest(),
                "train_rule": "first 48 unique content hashes of official train excluding all official dev/test",
                "training_fit_questions": 36, "training_calibration_questions": 12,
                "train_gold_upload": True, "val_gold_upload": False, "test_run": False,
                "cross_split_overlap": 0, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob("*.jsonl")}}
    write(data / "manifest.json", manifest)
    for name in ("runtime_controls.py", "trace_hooks.py", "frozen_topologies.json", "inspect_run.py", "snapshot.py", "supervise.py", "launch.py"):
        shutil.copyfile(OLD / name, ROOT / name)
    config = json.loads((OLD / "config.json").read_text())
    remote = "/home/liuxin/ICML2027/experiments/" + ROOT.name
    config.update(author_directory=remote+"/author_source", extension_directory=remote+"/coding_base",
                  main_ports=[11448,11449], merger_port=11450, conditions_total=540,
                  arms=["single_llm", "vanilla_mas", "moc", "moc_without_phi", "adaptive_sampling"],
                  training_questions=48, training_receivers=[1,3,4,5], utility_checkpoint=remote+"/utility_model.json")
    config["candidate"].update(cost_penalty=.02, projection_ridge=.1, local_state_weight=0.)
    write(ROOT / "config.json", config)
    destination = ROOT / "coding_base/moc_adaptive_sampling"
    shutil.copytree(WORKSPACE / "coding_base/moc_adaptive_sampling", destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "tests", "verify_local.py", "audit_assets.py"))
    print(json.dumps({"train": 48, "dev":30,"val":60,"official_train_records":sources[1]["records"],
                      "agreed_dev_match":True,"val_gold_upload":False}, ensure_ascii=False))

if __name__ == "__main__":
    main()
