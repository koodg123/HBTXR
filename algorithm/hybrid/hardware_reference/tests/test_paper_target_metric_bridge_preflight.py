from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_preflight_module():
    script_dir = Path("scripts/external").resolve()
    sys.path.insert(0, str(script_dir))
    script = script_dir / "check_paper_target_metric_bridge_preflight.py"
    spec = importlib.util.spec_from_file_location("check_paper_target_metric_bridge_preflight", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_preflight() -> dict:
    return json.loads(Path("docs/resources/paper_target_metric_bridge_preflight_2026_06_21.json").read_text(encoding="utf-8"))


def test_current_preflight_passes() -> None:
    module = load_preflight_module()

    report = module.validate_preflight(read_preflight(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["required_evidence_count"] == 7
    assert report["required_output_schema_count"] == 10
    assert report["ptb1_evaluator_output_schema_count"] == 5
    assert report["ptb1_current_existing_output_count"] == 0
    assert report["xr64_ready_to_train"] is False


def test_preflight_rejects_direct_comparison_claim() -> None:
    module = load_preflight_module()
    preflight = read_preflight()
    preflight["direct_submission_comparison_allowed"] = True

    report = module.validate_preflight(preflight, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be false" in report["errors"]


def test_preflight_rejects_missing_requirement() -> None:
    module = load_preflight_module()
    preflight = read_preflight()
    preflight["preflight_requirements"] = [
        item for item in preflight["preflight_requirements"] if item["id"] != "hybrid_scheduler_eval"
    ]

    report = module.validate_preflight(preflight, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("preflight_requirements ids mismatch" in error for error in report["errors"])


def test_preflight_rejects_p1_status_drift() -> None:
    module = load_preflight_module()
    preflight = read_preflight()
    for item in preflight["preflight_requirements"]:
        if item["id"] == "p1_full_test_metric":
            item["status"] = "complete"

    report = module.validate_preflight(preflight, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("p1_full_test_metric.status" in error for error in report["errors"])


def test_preflight_rejects_schema_drift() -> None:
    module = load_preflight_module()
    preflight = copy.deepcopy(read_preflight())
    preflight["required_output_schema_ids"].remove("cur_state_mapping_audit")

    report = module.validate_preflight(preflight, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("required_output_schema_ids" in error for error in report["errors"])


def test_preflight_rejects_train_eval_validation_command() -> None:
    module = load_preflight_module()
    preflight = read_preflight()
    preflight["validation_commands"].append(".venv/bin/python scripts/external/eval_hbtxr.py")

    report = module.validate_preflight(preflight, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])
