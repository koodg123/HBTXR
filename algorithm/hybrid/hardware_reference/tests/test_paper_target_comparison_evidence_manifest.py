from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_manifest_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_comparison_evidence_manifest.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_comparison_evidence_manifest", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_manifest() -> dict:
    path = Path("docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_paper_target_comparison_evidence_manifest_passes() -> None:
    module = load_manifest_module()

    report = module.validate_manifest(read_current_manifest(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["evidence_state"] == "missing_required_evidence"
    assert report["required_evidence_complete"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["required_evidence_count"] == 7
    assert report["missing_evidence_count"] == 3
    assert report["partial_evidence_count"] == 4
    assert report["required_future_artifact_count"] == 10
    assert report["current_artifact_count"] == 4
    assert report["existing_required_future_artifact_count"] == 3
    assert report["artifact_schema_count"] == 10


def test_manifest_rejects_direct_comparison_claim() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["comparison_decision"]["direct_submission_comparison_allowed"] = True

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert "comparison_decision.direct_submission_comparison_allowed must be false" in report["errors"]


def test_manifest_rejects_missing_coordinate_artifact() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["required_evidence_items"][0]["required_artifacts"] = []

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("coordinate_frame_match must list required_artifacts" in error for error in report["errors"])


def test_manifest_rejects_wrong_current_artifact_presence() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["required_evidence_items"][2]["current_artifacts"][0]["exists_now"] = False

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("p1_metric_implementation.exists_now must be True" in error for error in report["errors"])


def test_manifest_rejects_missing_artifact_content_schema() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["artifact_content_schema"].pop("hybrid_eval_rows")

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("artifact_content_schema ids mismatch" in error for error in report["errors"])


def test_future_jsonl_payload_is_validated_when_file_exists(tmp_path: Path) -> None:
    module = load_manifest_module()
    payload_path = tmp_path / "bad_rows.jsonl"
    payload_path.write_text('{"sample_id": "s0"}\n', encoding="utf-8")

    errors = module.validate_artifact_payload(
        payload_path,
        "hybrid_eval_rows",
        {
            "format": "jsonl",
            "required_fields": [
                "sample_id",
                "split",
                "chosen_mode",
                "chosen_state",
                "center_error",
                "p10_hit",
                "p5_hit",
                "p1_hit",
                "target_definition",
            ],
        },
    )

    assert errors
    assert "hybrid_eval_rows JSONL row 1 missing fields" in errors[0]


def test_manifest_rejects_train_eval_validation_command() -> None:
    module = load_manifest_module()
    manifest = copy.deepcopy(read_current_manifest())
    manifest["validation_commands"].append(".venv/bin/python eval_hbtxr.py")

    report = module.validate_manifest(manifest, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])
