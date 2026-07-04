from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_ladder_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_accuracy_lift_decision_ladder.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_accuracy_lift_decision_ladder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_ladder() -> dict:
    path = Path("docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def experiment(ladder: dict, exp_id: str) -> dict:
    return next(item for item in ladder["ablation_matrix"] if item["id"] == exp_id)


def test_current_accuracy_lift_decision_ladder_passes() -> None:
    module = load_ladder_module()
    report = module.validate_decision_ladder(read_current_ladder(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["execute_supported"] is False
    assert report["allowed_while_paused"] is True
    assert report["single_model_sota_claim_allowed"] is False
    assert report["oracle_promotable"] is False
    assert report["direct_submission_comparison_allowed"] is False
    assert report["stage_order"] == [
        "current_gates",
        "oracle_signal",
        "xr64_transfer",
        "post_xr64_branching",
        "paper_target_bridge",
    ]
    assert report["experiment_count"] == 8
    assert report["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert report["followup_ids"] == ["XR-64C", "XR-65", "XR-66", "XR-67", "XR-68"]
    assert report["axis_count"] == 6
    assert report["blocked_launch_count"] == 8
    assert report["expected_evidence_count"] == 8
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6
    assert report["xr64_postrun_candidate_count"] == 0
    assert report["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert report["current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"


def test_ladder_rejects_oracle_promotion() -> None:
    module = load_ladder_module()
    ladder = copy.deepcopy(read_current_ladder())
    ladder["oracle_signal"]["promotable_as_result"] = True

    report = module.validate_decision_ladder(ladder, project_root=Path.cwd())

    assert report["ok"] is False
    assert "oracle_signal.promotable_as_result must be false" in report["errors"]


def test_ladder_rejects_launch_allowed_while_paused() -> None:
    module = load_ladder_module()
    ladder = copy.deepcopy(read_current_ladder())
    experiment(ladder, "XR-64A")["launch_allowed_now"] = True

    report = module.validate_decision_ladder(ladder, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-64A: launch_allowed_now must be false" in report["errors"]


def test_ladder_rejects_missing_temporal_gate() -> None:
    module = load_ladder_module()
    ladder = copy.deepcopy(read_current_ladder())
    experiment(ladder, "XR-65")["blocked_until"] = ["XR-64 non-collapsing checkpoint"]

    report = module.validate_decision_ladder(ladder, project_root=Path.cwd())

    assert report["ok"] is False
    assert "XR-65 must require same-scene temporal evidence" in report["errors"]


def test_ladder_rejects_training_command_in_validation() -> None:
    module = load_ladder_module()
    ladder = copy.deepcopy(read_current_ladder())
    ladder["validation_commands"].append("bash scripts/external/run_xr64_teacher_target_construction.sh")

    report = module.validate_decision_ladder(ladder, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("validation_commands must not execute" in error for error in report["errors"])
