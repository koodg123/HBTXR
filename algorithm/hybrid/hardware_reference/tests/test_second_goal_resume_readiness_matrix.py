from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_resume_readiness_matrix.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_resume_readiness_matrix", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_matrix() -> dict:
    path = Path("docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_resume_readiness_matrix_passes() -> None:
    module = load_module()
    report = module.validate_matrix(read_matrix(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["axis_count"] == 6
    assert report["resume_gate_count"] == 5
    assert report["current_p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["xr64_ready_to_train"] is False
    assert report["xr64_missing_eval_rows"] == 8
    assert report["xr64_missing_overrides"] == 6
    assert report["next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["direct_submission_comparison_allowed"] is False
    assert report["software_promotion_allowed"] is False
    assert report["paper_level_completion_allowed"] is False


def test_rejects_false_direct_comparison() -> None:
    module = load_module()
    matrix = copy.deepcopy(read_matrix())
    matrix["direct_submission_comparison_allowed"] = True

    report = module.validate_matrix(matrix, project_root=Path.cwd())

    assert report["ok"] is False
    assert "direct_submission_comparison_allowed must be False" in report["errors"]


def test_rejects_source_count_drift() -> None:
    module = load_module()
    matrix = copy.deepcopy(read_matrix())
    matrix["current_state"]["xr64_missing_eval_rows"] = 7

    report = module.validate_matrix(matrix, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("current_state.xr64_missing_eval_rows" in error for error in report["errors"])


def test_rejects_missing_axis() -> None:
    module = load_module()
    matrix = copy.deepcopy(read_matrix())
    matrix["axis_resume_matrix"] = [
        row for row in matrix["axis_resume_matrix"] if row["axis"] != "optimizer"
    ]

    report = module.validate_matrix(matrix, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("axis_resume_matrix axes mismatch" in error for error in report["errors"])


def test_rejects_execution_validation_command() -> None:
    module = load_module()
    matrix = copy.deepcopy(read_matrix())
    matrix["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_matrix(matrix, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not contain execution marker" in error for error in report["errors"])


def test_rejects_missing_resume_gate() -> None:
    module = load_module()
    matrix = copy.deepcopy(read_matrix())
    matrix["resume_gates"] = [gate for gate in matrix["resume_gates"] if gate["id"] != "RG-5"]

    report = module.validate_matrix(matrix, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("resume_gates ids mismatch" in error for error in report["errors"])
