"""Reproducible offline checks and an example author-path routing trace."""

from pathlib import Path
import asyncio
import contextlib
import io
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "coding_base"))
# Local, isolated dependency installation from this task; no model download.
sys.path.insert(0, str(ROOT / "maintenance/task_runtime/moc_sampling_20261014"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from moc_adaptive_sampling import MOCSamplingAdapter, SamplingConfig, LayerDependentSampler
from moc_adaptive_sampling.features import LocalTokenizerCounter
from moc_adaptive_sampling.audit_assets import audit
from source_harness import graph_fixture, fake_encode


async def example(counter):
    graph, _ = graph_fixture()
    adapter = MOCSamplingAdapter(graph, LayerDependentSampler(SamplingConfig(), counter),
                                 fake_encode, feature_space="OFFLINE_TEST_DOUBLE", counter_name=counter.name)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        answers, log_probs = await graph.arun({"task":"offline routing demonstration; not a benchmark item"})
    return {"scope":"offline service/embedding doubles; real author loop/prompt bodies and real local tokenizer",
            "ledger": adapter.ledger, "final_outputs": answers}


def main():
    destination = ROOT / "research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014"
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), pattern="test_*.py")
    with contextlib.redirect_stdout(output):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    (destination / "local_checks.log").write_text(output.getvalue(), encoding="utf-8")
    report = {"scope":"offline source-path integration, not model accuracy reproduction",
              "tests_run":result.testsRun, "failures":len(result.failures), "errors":len(result.errors),
              "status":"passed" if result.wasSuccessful() else "failed", "python":sys.version,
              "model_calls":0, "gpu_experiments":0, "cloud_jobs_resumed":0}
    if result.wasSuccessful():
        counter = LocalTokenizerCounter(ROOT / "experiments/moc_index_pull_20261013/tokenizer/tokenizer.json",
            "3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922")
        (destination / "offline_trace.json").write_text(json.dumps(asyncio.run(example(counter)),
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        source_report = audit(ROOT, destination / "asset_audit.json")
        report.update(upstream_assets=source_report["upstream_asset_count"],
                      author_source_files=len(source_report["moc_src_sha256"]), author_checkout="clean")
    (destination / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
