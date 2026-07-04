from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def load_checker():
    script = Path("scripts/external/check_xr64_resume_artifacts.py").resolve()
    spec = importlib.util.spec_from_file_location("check_xr64_resume_artifacts", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_manifest(path: Path, sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in sample_ids), encoding="utf-8")


def write_required_files(root: Path) -> None:
    write_manifest(root / "manifests" / "train_manifest.jsonl", ["tr1", "tr2"])
    write_manifest(root / "manifests" / "val_manifest.jsonl", ["va1"])
    write_manifest(root / "manifests" / "test_manifest.jsonl", ["te1"])
    (root / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    write_json(root / "reference_config.json", {"training": {"batch_size": 1}})
    for rel in (
        "runs/xr62.pt",
        "runs/xr39.pt",
        "runs/xr56b.pt",
        "runs/xr58a.pt",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ckpt")


def args_for(root: Path, *, allow_missing: bool = False) -> Namespace:
    return Namespace(
        project_root=str(root),
        manifest_root="manifests",
        target_dir="targets",
        config="config.yaml",
        reference_config="reference_config.json",
        allow_missing_generated=allow_missing,
    )


def patch_checkpoints(module, monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "DEFAULT_CHECKPOINTS",
        {
            "xr62a": "runs/xr62.pt",
            "xr39": "runs/xr39.pt",
            "xr56b": "runs/xr56b.pt",
            "xr58a": "runs/xr58a.pt",
        },
    )


def eval_row(sample_id: str) -> dict:
    return {"sample_id": sample_id, "track_state": [1, 2, 3, 4, 0, 1]}


def override_row(sample_id: str, split: str, source: str = "xr62a") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "track_target_override_state": [1, 2, 3, 4, 0, 1],
        "track_target_override_weight": 1.0,
        "track_target_override_source": source,
        "selection_reason": "baseline_guard",
        "teacher_center_error_px": 0.0,
        "baseline_center_error_px": 0.0,
        "all_teacher_center_error_px": {"xr62a": 0.0, "xr39": 1.0, "xr56b": 2.0, "xr58a": 3.0},
    }


def write_generated(root: Path) -> None:
    target = root / "targets"
    for split, sample_ids in {"train": ["tr1", "tr2"], "val": ["va1"]}.items():
        for teacher in ("xr62a", "xr39", "xr56b", "xr58a"):
            write_json(target / "eval_rows" / split / teacher / "eval_rows.json", [eval_row(sample_id) for sample_id in sample_ids])
        for key, rule in {
            "xr64a_conservative": "conservative",
            "xr64b_threshold": "threshold_priority",
            "xr64c_minerror": "min_error",
        }.items():
            rows = [override_row(sample_id, split) for sample_id in sample_ids]
            write_json(
                target / f"{key}_{split}_overrides.json",
                {
                    "kind": "xr64_teacher_target_overrides",
                    "split": split,
                    "rule": rule,
                    "baseline": "xr62a",
                    "sample_count": len(rows),
                    "selection_counts": {"xr62a": len(rows), "xr39": 0, "xr56b": 0, "xr58a": 0},
                    "overrides": rows,
                },
            )


def test_xr64_resume_checker_allows_missing_generated(tmp_path: Path, monkeypatch) -> None:
    module = load_checker()
    patch_checkpoints(module, monkeypatch)
    write_required_files(tmp_path)

    code, report = module.run_check(args_for(tmp_path, allow_missing=True))

    assert code == 0
    assert report["ready"] is True
    assert report["generated_complete"] is False
    assert report["warnings"]


def test_xr64_resume_checker_passes_complete_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = load_checker()
    patch_checkpoints(module, monkeypatch)
    write_required_files(tmp_path)
    write_generated(tmp_path)

    code, report = module.run_check(args_for(tmp_path))

    assert code == 0
    assert report["ready"] is True
    assert report["generated_complete"] is True

    summary = module.format_summary(report, code)
    assert "resume_status: ready_to_train" in summary
    assert "ready_to_train: true" in summary
    assert "can_run_lane: true" in summary
    assert "leakage_risk: none" in summary
    assert "generation_matrix:" in summary
    assert "override_matrix:" in summary


def test_xr64_resume_checker_rejects_test_split_payload(tmp_path: Path, monkeypatch) -> None:
    module = load_checker()
    patch_checkpoints(module, monkeypatch)
    write_required_files(tmp_path)
    write_generated(tmp_path)
    write_json(tmp_path / "targets" / "xr64a_conservative_test_overrides.json", {"split": "test", "overrides": []})

    code, report = module.run_check(args_for(tmp_path))

    assert code == 1
    assert report["ready"] is False
    assert report["errors"]

    summary = module.format_summary(report, code)
    assert "resume_status: blocked" in summary
    assert "leakage_risk: high" in summary


def test_xr64_resume_checker_rejects_partial_eval_row_coverage(tmp_path: Path, monkeypatch) -> None:
    module = load_checker()
    patch_checkpoints(module, monkeypatch)
    write_required_files(tmp_path)
    write_generated(tmp_path)
    path = tmp_path / "targets" / "eval_rows" / "train" / "xr39" / "eval_rows.json"
    write_json(path, [eval_row("tr1")])

    code, report = module.run_check(args_for(tmp_path))

    assert code == 1
    assert report["ready"] is False
    assert report["generated_complete"] is True
    assert report["eval_rows"]["train"]["xr39"]["missing_manifest_sample_ids"] == 1
    assert any("eval rows missing ids from train manifest teacher=xr39: 1" in error for error in report["errors"])


def test_xr64_resume_checker_rejects_partial_override_coverage(tmp_path: Path, monkeypatch) -> None:
    module = load_checker()
    patch_checkpoints(module, monkeypatch)
    write_required_files(tmp_path)
    write_generated(tmp_path)
    path = tmp_path / "targets" / "xr64b_threshold_train_overrides.json"
    rows = [override_row("tr1", "train")]
    write_json(
        path,
        {
            "kind": "xr64_teacher_target_overrides",
            "split": "train",
            "rule": "threshold_priority",
            "baseline": "xr62a",
            "sample_count": len(rows),
            "selection_counts": {"xr62a": len(rows), "xr39": 0, "xr56b": 0, "xr58a": 0},
            "overrides": rows,
        },
    )

    code, report = module.run_check(args_for(tmp_path))

    assert code == 1
    assert report["ready"] is False
    assert report["generated_complete"] is True
    assert report["overrides"]["train"]["xr64b_threshold"]["missing_manifest_sample_ids"] == 1
    assert any("overrides missing ids from train manifest key=xr64b_threshold: 1" in error for error in report["errors"])
