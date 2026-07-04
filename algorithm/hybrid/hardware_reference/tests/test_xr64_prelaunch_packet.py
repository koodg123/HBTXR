from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_packet_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_xr64_prelaunch_packet.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_xr64_prelaunch_packet", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_packet() -> dict:
    path = Path("docs/resources/xr64_prelaunch_packet_2026_06_20.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_xr64_prelaunch_packet_passes() -> None:
    module = load_packet_module()

    report = module.validate_packet(read_current_packet(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["allowed_to_execute"] is False
    assert report["completion_allowed"] is False
    assert report["completion_blocker_count"] == 28
    assert report["phase_count"] == 6
    assert report["manifest_ok"] is True
    assert report["manifest_prep_command_count"] == 10
    assert report["manifest_eval_command_count"] == 8
    assert report["manifest_build_command_count"] == 2
    assert report["manifest_expected_eval_rows"] == 8
    assert report["manifest_expected_overrides"] == 6
    assert report["manifest_eval_inputs_ready"] is True
    assert report["manifest_missing_eval_required_input_count"] == 0
    assert report["manifest_build_inputs_ready"] is False
    assert report["manifest_missing_build_required_input_count"] == 8
    assert report["manifest_postrun_command_count"] == 2
    assert report["manifest_launch_lanes"] == ["XR-64A", "XR-64B"]
    assert report["resume_matrix_axis_count"] == 6
    assert report["resume_matrix_resume_gate_count"] == 5
    assert report["resume_matrix_current_p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["prep_ledger_ready_after_resume_count"] == 8
    assert report["prep_ledger_blocked_count"] == 2
    assert report["blocker_map_blocker_count"] == 28
    assert report["paper_target_preflight_required_evidence_count"] == 7
    assert report["paper_target_preflight_required_output_schema_count"] == 10


def test_prelaunch_packet_rejects_execution_enabled_state() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["allowed_to_execute"] = True
    packet["command_review"]["execute_supported"] = True

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert "allowed_to_execute must be false" in report["errors"]
    assert "command_review.execute_supported must be false" in report["errors"]


def test_prelaunch_packet_rejects_missing_source_artifact() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["source_artifacts"].pop("completion_gate")

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("source_artifacts missing keys" in error for error in report["errors"])


def test_prelaunch_packet_rejects_validation_train_launcher() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not execute run_xr64_teacher_target_construction.sh" in error for error in report["errors"])


def test_prelaunch_packet_rejects_missing_postrun_review_command() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["command_review"].pop("postrun_command_review")

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("command_review.postrun_command_review must use emit_xr64_resume_commands.py" in error for error in report["errors"])


def test_prelaunch_packet_rejects_missing_postrun_phase() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["prelaunch_phases"] = [
        phase for phase in packet["prelaunch_phases"] if phase["id"] != "PHASE-5-POSTRUN-CANDIDATE-REVIEW"
    ]

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("prelaunch_phases must preserve required phase order" in error for error in report["errors"])
    assert any("PHASE-5 expected_count must match manifest postrun command count" in error for error in report["errors"])


def test_prelaunch_packet_rejects_input_readiness_mismatch() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["current_state"]["eval_inputs_ready"] = False

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_state.eval_inputs_ready must match command manifest: True" in report["errors"]


def test_prelaunch_packet_rejects_build_readiness_mismatch() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["current_state"]["build_inputs_ready"] = True

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_state.build_inputs_ready must match command manifest: False" in report["errors"]


def test_prelaunch_packet_rejects_resume_matrix_mismatch() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["current_state"]["resume_matrix_axis_count"] = 5

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_state.resume_matrix_axis_count must match resume readiness matrix: 6" in report["errors"]


def test_prelaunch_packet_rejects_missing_latest_status_source() -> None:
    module = load_packet_module()
    packet = copy.deepcopy(read_current_packet())
    packet["source_artifacts"].pop("resume_readiness_matrix")

    report = module.validate_packet(packet, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("source_artifacts missing keys" in error for error in report["errors"])
