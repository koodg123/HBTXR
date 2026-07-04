from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_blocker_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_blocker_artifact_map.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_blocker_artifact_map", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_map() -> dict:
    return json.loads(Path("docs/resources/second_goal_blocker_artifact_map_2026_06_21.json").read_text())


def test_current_blocker_artifact_map_passes() -> None:
    module = load_blocker_module()

    report = module.validate_map(read_current_map(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execute_supported"] is False
    assert report["allowed_to_run_now"] is False
    assert report["xr64_missing_eval_rows"] == 8
    assert report["xr64_missing_overrides"] == 6
    assert report["ptb1_missing_payload_count"] == 5
    assert report["ptb2_missing_payload_count"] == 2
    assert report["ptb3_missing_payload_count"] == 1
    assert report["ablation_axis_count"] == 6


def test_blocker_map_rejects_execution_enablement() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["execute_supported"] = True
    payload["blocked_execution_units"][0]["allowed_to_run_now"] = True

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert "execute_supported must be false" in report["errors"]
    assert "blocked execution units must not be allowed to run now" in report["errors"]


def test_blocker_map_rejects_missing_eval_path_drift() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["xr64_missing_artifacts"]["eval_rows"].pop()

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert "xr64_missing_artifacts.eval_rows must match XR-64 command manifest" in report["errors"]


def test_blocker_map_rejects_test_artifact_path() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["xr64_missing_artifacts"]["eval_rows"][0] = (
        "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/test/xr62a/eval_rows.json"
    )

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("test artifacts" in error for error in report["errors"])


def test_blocker_map_rejects_ptb1_allow_while_paused() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["paper_target_bridge_missing_artifacts"]["ptb1_evaluator_payloads"][0][
        "allowed_while_paused"
    ] = True

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert "PTB-1 payloads must be blocked while paused" in report["errors"]


def test_blocker_map_rejects_train_eval_validation_command() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0")

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation command must not launch train/eval" in error for error in report["errors"])


def test_blocker_map_rejects_missing_ablation_axis() -> None:
    module = load_blocker_module()
    payload = read_current_map()
    payload["ablation_matrix"] = [
        item for item in payload["ablation_matrix"] if item["axis"] != "optimizer"
    ]

    report = module.validate_map(payload, project_root=Path.cwd())

    assert report["ok"] is False
    assert "ablation_matrix missing axis: optimizer" in report["errors"]
