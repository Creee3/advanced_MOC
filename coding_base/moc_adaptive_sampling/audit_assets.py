"""Verify pinned source hashes and save provenance; does not invoke any model."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone


def audit(workspace, destination):
    workspace = Path(workspace).resolve(strict=True)
    assets = workspace / "research_ideation_20261013/moc_adaptive_neighborhood_sampling_mainline/code_migration_20261014"
    manifest = json.loads((assets / "source_manifest.json").read_text(encoding="utf-8"))
    verified = []
    for artifact in manifest["artifacts"]:
        path = assets / artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        verified.append(artifact["path"])
    author = workspace / "coding_base/MOC-official"
    commit = subprocess.check_output(["git", "-C", str(author), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(author), "status", "--porcelain=v1"], text=True).strip()
    if commit != "9c67c92507570704a7df73e452552a3f49e83897" or status:
        raise ValueError("author checkout is no longer the clean pinned version")
    sources = {str(p.relative_to(author)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((author / "src").rglob("*.py"))}
    port_root = workspace / "coding_base/moc_adaptive_sampling"
    port_sources = {str(p.relative_to(port_root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(port_root.rglob("*.py"))}
    citations = json.loads((assets / "citation_check.json").read_text(encoding="utf-8"))
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "upstream_assets_verified": verified, "upstream_asset_count": len(verified),
              "moc_commit": commit, "moc_git_status": "clean", "moc_src_sha256": sources,
              "port_python_sha256": port_sources,
              "citation_verdict_line": citations["verdict_line"],
              "citation_option_scope": "--no-canonical-instance; no same-title canonical lookup",
              "runtime_dependency": {"package": "tokenizers", "version": "0.22.2",
                  "local_target": "maintenance/task_runtime/moc_sampling_20261014", "model_downloads": 0},
              "gpu_or_cloud_experiments_started": 0}
    destination = Path(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = audit(sys.argv[1], sys.argv[2])
    print(f"passed: {result['upstream_asset_count']} pinned assets; {len(result['moc_src_sha256'])} unchanged author src files; clean MOC")
