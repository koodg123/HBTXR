from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


REQUIRED_FINAL_ARTIFACTS = {
    "dataset_split": ["experiment.train_manifest", "experiment.val_manifest", "experiment.test_manifest"],
    "checkpoint": ["experiment.init_checkpoint", "paper_artifacts.final_checkpoint"],
    "quantization": ["paper_artifacts.quant_tables"],
    "nonlinear_luts": ["paper_artifacts.nonlinear_luts"],
    "golden_vectors": ["paper_artifacts.golden_vectors"],
}


def _get_dotted(payload: dict[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _path_exists(value: Any, *, project_root: Path) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or text.lower() == "null":
        return False
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.exists()


def collect_missing_artifacts(cfg: dict[str, Any], *, project_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(project_root)
    missing: dict[str, dict[str, Any]] = {}
    for artifact, candidates in REQUIRED_FINAL_ARTIFACTS.items():
        present = [candidate for candidate in candidates if _path_exists(_get_dotted(cfg, candidate), project_root=root)]
        missing[artifact] = {
            "present": bool(present),
            "satisfied_by": present,
            "candidate_fields": list(candidates),
        }
    return missing


def paper_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(((cfg.get("paper") or {}).get("targets") or {}))


def build_reproduction_manifest(
    cfg: dict[str, Any],
    *,
    project_root: str | Path,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing = collect_missing_artifacts(cfg, project_root=project_root)
    exact_ready = all(item["present"] for item in missing.values())
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready_for_exact_metric_gate" if exact_ready else "structure_reproduction_only",
        "exact_metric_reproduction_ready": exact_ready,
        "paper": deepcopy(cfg.get("paper") or {}),
        "paper_targets": paper_targets(cfg),
        "actual_metrics": deepcopy(metrics or {}),
        "artifacts": deepcopy(artifacts or {}),
        "missing_artifacts": missing,
        "notes": [
            "Exact submitted-paper metrics require the final dataset split, trained checkpoint, quantization tables, nonlinear LUTs, and golden vectors.",
            "When those artifacts are absent, this manifest validates architecture/config/procedure reproduction only.",
        ],
    }


def write_reproduction_manifest(
    cfg: dict[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_reproduction_manifest(
        cfg,
        project_root=project_root,
        metrics=metrics,
        artifacts=artifacts,
    )
    path = out_dir / "reproduction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "REQUIRED_FINAL_ARTIFACTS",
    "build_reproduction_manifest",
    "collect_missing_artifacts",
    "paper_targets",
    "write_reproduction_manifest",
]
