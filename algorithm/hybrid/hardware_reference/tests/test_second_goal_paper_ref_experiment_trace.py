from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_trace_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_paper_ref_experiment_trace.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_paper_ref_experiment_trace", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_trace() -> dict:
    path = Path("docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_paper_ref_experiment_trace_passes() -> None:
    module = load_trace_module()
    report = module.validate_trace(read_current_trace(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["allowed_while_paused"] is True
    assert report["experiment_count"] == 8
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["paper_group_count"] == 6
    assert report["axis_count"] == 8
    assert report["xr64_ready_to_train"] is False
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6
    assert report["completion_allowed"] is False
    assert report["direct_submission_comparison_allowed"] is False


def test_trace_rejects_missing_facet_analysis_path() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["paper_groups"]["geometry_state_supervision"]["papers"][0] = "anlaysis/paper-ref/papers/missing/analysis.md"

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("geometry_state_supervision" in error and "missing paper analysis" in error for error in report["errors"])


def test_trace_rejects_xr64a_without_override_free_test_rule() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    xr64a = next(item for item in trace["experiment_traces"] if item["id"] == "XR-64A")
    xr64a["rejection_rules"] = [rule for rule in xr64a["rejection_rules"] if rule != "test override path is non-null"]

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A: must reject non-null test override path" in report["errors"]


def test_trace_rejects_training_command_as_validation() -> None:
    module = load_trace_module()
    trace = copy.deepcopy(read_current_trace())
    trace["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_trace(trace, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not execute" in error for error in report["errors"])
