from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_checklist_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_ablation_evidence_checklist.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_ablation_evidence_checklist", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_checklist() -> dict:
    path = Path("docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_ablation_evidence_checklist_passes() -> None:
    module = load_checklist_module()
    report = module.validate_checklist(read_current_checklist(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["do_not_execute_until_user_resumes"] is True
    assert report["experiment_count"] == 8
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["axis_count"] == 8
    assert report["software_promotion_allowed"] is False
    assert report["paper_level_completion_allowed"] is False
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6


def test_checklist_rejects_missing_p1_future_evidence() -> None:
    module = load_checklist_module()
    checklist = copy.deepcopy(read_current_checklist())
    checklist["promotion_policy"]["future_evidence_metrics"] = []

    report = module.validate_checklist(checklist, project_root=Path.cwd())

    assert report["ok"] is False
    assert "promotion_policy.future_evidence_metrics must be metric_track_p1_pct" in report["errors"]


def test_checklist_rejects_xr64_lane_without_override_free_test_eval() -> None:
    module = load_checklist_module()
    checklist = copy.deepcopy(read_current_checklist())
    lane = next(item for item in checklist["experiments"] if item["id"] == "XR-64A")
    lane["reject_if"] = [item for item in lane["reject_if"] if item != "test override path is non-null"]

    report = module.validate_checklist(checklist, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A: reject_if must forbid non-null test override path" in report["errors"]


def test_checklist_rejects_training_commands_as_validation() -> None:
    module = load_checklist_module()
    checklist = copy.deepcopy(read_current_checklist())
    checklist["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_checklist(checklist, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not execute" in error for error in report["errors"])
