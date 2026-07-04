from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_trace_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_objective_trace.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_objective_trace", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_trace() -> dict:
    path = Path("docs/resources/second_goal_objective_trace_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_second_goal_objective_trace_passes() -> None:
    module = load_trace_module()
    report = module.validate_trace(read_current_trace(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["active_goal_complete"] is False
    assert report["goal_complete"] is False
    assert report["requirement_count"] == 3
    assert report["requirement_status"]["REQ-1"] == "complete_as_planning_input"
    assert report["requirement_status"]["REQ-2"] == "planned_not_fully_executed"
    assert report["requirement_status"]["REQ-3"] == "incomplete"
    assert "head" in report["covered_axes"]
    assert "teacher_model_training" in report["covered_axes"]
    assert report["ablation_count"] == 8
    assert report["post_xr64_branch_count"] == 7
    assert report["post_xr64_minimal_next_worker"] == "XR-64-prep"
    assert report["post_xr64_direct_submission_comparison_allowed"] is False


def test_trace_rejects_false_goal_completion() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["completion_judgment"]["goal_complete"] = True

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "completion_judgment.goal_complete must be false" in report["errors"]


def test_trace_rejects_missing_experiment_axis() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["requirements"][1]["axis_coverage"].pop("optimizer")

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("REQ-2 axis_coverage missing" in error for error in report["errors"])


def test_trace_rejects_missing_ablation_matrix_row() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["ablation_matrix"] = [item for item in trace["ablation_matrix"] if item["id"] != "XR-66"]

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("ablation_matrix ids mismatch" in error for error in report["errors"])


def test_trace_rejects_missing_post_xr64_decision_tree_ref() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["requirements"][1].pop("post_xr64_decision_tree_ref")

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("post_xr64_decision_tree_ref.path" in error for error in report["errors"])


def test_trace_rejects_decision_tree_claimed_as_promotion_evidence() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["requirements"][1]["post_xr64_decision_tree_ref"]["allowed_claim_scope"] = "software_promotion"

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "REQ-2.post_xr64_decision_tree_ref.allowed_claim_scope must be experiment_planning_only" in report["errors"]


def test_trace_rejects_post_xr64_summary_mismatch() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["requirements"][1]["post_xr64_decision_tree_summary"]["branch_count"] = 6

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("post_xr64_decision_tree_summary.branch_count" in error for error in report["errors"])
