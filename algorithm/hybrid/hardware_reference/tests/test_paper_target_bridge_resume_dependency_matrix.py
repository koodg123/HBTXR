from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def load_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_paper_target_bridge_resume_dependency_matrix.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_paper_target_bridge_resume_dependency_matrix", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_matrix() -> dict:
    path = Path("docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_resume_dependency_matrix_passes() -> None:
    module = load_module()
    report = module.validate_matrix(read_current_matrix(), project_root=Path.cwd())
    assert report["ok"] is True
    assert report["dependency_gate_count"] == 6
    assert report["allowed_while_paused_gate_count"] == 2
    assert report["complete_now_gate_count"] == 2
    assert report["blocked_gate_count"] == 4
    assert report["execution_dependent_payload_missing_count"] == 8
    assert report["xr64_ready_to_train"] is False
    assert report["completion_allowed"] is False


def test_matrix_rejects_gate_reorder() -> None:
    module = load_module()
    matrix = read_current_matrix()
    matrix["dependency_gates"][2], matrix["dependency_gates"][3] = (
        matrix["dependency_gates"][3],
        matrix["dependency_gates"][2],
    )
    report = module.validate_matrix(matrix, project_root=Path.cwd())
    assert report["ok"] is False
    assert any("dependency_gates must preserve required order" in error for error in report["errors"])


def test_matrix_rejects_ptb1_payload_false_completion() -> None:
    module = load_module()
    matrix = read_current_matrix()
    gate = next(item for item in matrix["dependency_gates"] if item["id"] == "PBRD-2-PTB1-PAYLOADS-AFTER-RESUME")
    gate["complete_now"] = True
    gate["allowed_while_paused"] = True
    report = module.validate_matrix(matrix, project_root=Path.cwd())
    assert report["ok"] is False
    assert any("PBRD-2-PTB1-PAYLOADS-AFTER-RESUME must be blocked and incomplete" in error for error in report["errors"])


def test_matrix_rejects_source_count_drift() -> None:
    module = load_module()
    matrix = read_current_matrix()
    matrix["current_state"]["ptb1_payload_missing_count"] = 4
    report = module.validate_matrix(matrix, project_root=Path.cwd())
    assert report["ok"] is False
    assert any("current_state.ptb1_payload_missing_count" in error for error in report["errors"])


def test_matrix_rejects_executable_validation_command() -> None:
    module = load_module()
    matrix = read_current_matrix()
    matrix["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")
    report = module.validate_matrix(matrix, project_root=Path.cwd())
    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])
