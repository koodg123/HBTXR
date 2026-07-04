from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


def load_gate_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_completion_gate.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_completion_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def incomplete_status() -> dict:
    return {
        "execution_state": "paused_by_user_directive",
        "active_goal_complete": False,
        "current_gates": {
            "center_lt": 16.468481131962367,
            "p10_gt": 35.02295998845781,
            "p5_gt": 12.133503770828247,
        },
        "objective_trace": {
            "exists": True,
            "goal_complete": False,
            "req2_status": "planned_not_fully_executed",
            "req3_status": "incomplete",
        },
        "current_result_synthesis": {
            "exists": True,
            "single_model_sota_claim_allowed": False,
            "oracle_promotable": False,
            "direct_submission_comparison_valid": False,
            "xr64_ready_to_train": False,
        },
        "metric_protocol_bridge": {
            "exists": True,
            "bridge_status": "blocked_until_metric_frame_and_protocol_match",
            "direct_submission_comparison_allowed": False,
        },
        "submission_target_gap": {
            "direct_comparison_valid": False,
        },
        "paper_target_comparison_evidence_manifest": {
            "exists": True,
            "evidence_state": "missing_required_evidence",
            "required_evidence_complete": False,
            "direct_submission_comparison_allowed": False,
            "required_future_artifact_count": 10,
            "existing_required_future_artifact_count": 3,
            "artifact_schema_count": 10,
        },
        "xr64": {
            "ready_to_train": False,
            "can_run_lane": False,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
            "leakage_risk": "none",
        },
        "indexed_xr64_generated_state": {
            "train_val_eval_rows_exist": False,
            "train_val_override_json_exist": False,
            "xr64_training_runs_exist": False,
        },
        "experiment_queue": {
            "paused": True,
            "do_not_execute_until_user_resumes": True,
        },
        "authority_links": {
            "missing": [],
        },
        "xr64_resume_command_emitter": {
            "execute_supported": False,
        },
        "xr64_resume_command_manifest": {
            "default_allowed_to_run": False,
            "postrun_command_count": 2,
        },
        "xr64_prelaunch_packet": {
            "exists": True,
            "phase_count": 6,
            "postrun_command_count": 2,
        },
        "xr64_postrun_promotion_decision": {
            "exists": True,
            "ok": True,
            "error_count": 0,
            "candidate_count": 0,
            "software_promotion_allowed": False,
            "single_model_sota_claim_allowed": False,
            "axis_claims_allowed": False,
            "control_variables_checked": False,
            "p1_evidence_candidate_count": 0,
        },
    }


def complete_status() -> dict:
    status = copy.deepcopy(incomplete_status())
    status["execution_state"] = "complete_verified"
    status["active_goal_complete"] = True
    status["objective_trace"]["goal_complete"] = True
    status["objective_trace"]["req2_status"] = "complete_verified"
    status["objective_trace"]["req3_status"] = "complete_verified"
    status["current_result_synthesis"]["single_model_sota_claim_allowed"] = True
    status["current_result_synthesis"]["direct_submission_comparison_valid"] = True
    status["current_result_synthesis"]["xr64_ready_to_train"] = True
    status["metric_protocol_bridge"]["bridge_status"] = "metric_frame_and_protocol_matched"
    status["metric_protocol_bridge"]["direct_submission_comparison_allowed"] = True
    status["submission_target_gap"]["direct_comparison_valid"] = True
    status["paper_target_comparison_evidence_manifest"]["evidence_state"] = "complete"
    status["paper_target_comparison_evidence_manifest"]["required_evidence_complete"] = True
    status["paper_target_comparison_evidence_manifest"]["direct_submission_comparison_allowed"] = True
    status["paper_target_comparison_evidence_manifest"]["existing_required_future_artifact_count"] = 10
    status["paper_target_comparison_evidence_manifest"]["artifact_schema_count"] = 10
    status["xr64"]["ready_to_train"] = True
    status["xr64"]["can_run_lane"] = True
    status["xr64"]["missing_eval_rows"] = 0
    status["xr64"]["missing_overrides"] = 0
    status["indexed_xr64_generated_state"]["train_val_eval_rows_exist"] = True
    status["indexed_xr64_generated_state"]["train_val_override_json_exist"] = True
    status["indexed_xr64_generated_state"]["xr64_training_runs_exist"] = True
    status["xr64_postrun_promotion_decision"]["candidate_count"] = 2
    status["xr64_postrun_promotion_decision"]["software_promotion_allowed"] = True
    status["xr64_postrun_promotion_decision"]["single_model_sota_claim_allowed"] = True
    status["xr64_postrun_promotion_decision"]["axis_claims_allowed"] = True
    status["xr64_postrun_promotion_decision"]["control_variables_checked"] = True
    status["xr64_postrun_promotion_decision"]["p1_evidence_candidate_count"] = 1
    status["experiment_queue"]["paused"] = False
    status["experiment_queue"]["do_not_execute_until_user_resumes"] = False
    return status


def blocker_ids(report: dict) -> set[str]:
    return {str(item["id"]) for item in report["blockers"]}


def test_completion_gate_rejects_current_incomplete_state() -> None:
    module = load_gate_module()

    report = module.evaluate_completion_gate(incomplete_status())

    assert report["ok"] is False
    assert report["completion_allowed"] is False
    ids = blocker_ids(report)
    assert "execution_paused" in ids
    assert "active_goal_not_marked_complete" in ids
    assert "objective_trace_incomplete" in ids
    assert "accuracy_closure_incomplete" in ids
    assert "metric_protocol_bridge_blocked" in ids
    assert "direct_submission_comparison_blocked" in ids
    assert "paper_target_required_evidence_incomplete" in ids
    assert "paper_target_direct_comparison_blocked" in ids
    assert "paper_target_future_artifacts_missing" in ids
    assert "xr64_not_ready_to_train" in ids
    assert "xr64_eval_rows_missing" in ids
    assert "xr64_overrides_missing" in ids
    assert "xr64_training_runs_missing" in ids
    assert "xr64_postrun_candidates_missing" in ids
    assert "xr64_postrun_software_promotion_blocked" in ids
    assert "xr64_postrun_axis_claims_blocked" in ids
    assert "xr64_postrun_p1_evidence_missing" in ids
    assert "xr64_postrun_promotion_decision_invalid" not in ids


def test_completion_gate_passes_only_when_all_strict_conditions_are_met() -> None:
    module = load_gate_module()

    report = module.evaluate_completion_gate(complete_status())

    assert report["ok"] is True
    assert report["completion_allowed"] is True
    assert report["blocker_count"] == 0
    guard_ids = {str(item["id"]) for item in report["guards_passed"]}
    assert "oracle_not_promotable" in guard_ids
    assert "leakage_risk_none" in guard_ids
    assert "authority_links_resolve" in guard_ids
    assert "command_emitter_non_executing" in guard_ids
    assert "xr64_postrun_promotion_decision_valid" in guard_ids


def test_completion_gate_rejects_leakage_even_if_other_conditions_pass() -> None:
    module = load_gate_module()
    status = complete_status()
    status["xr64"]["leakage_risk"] = "high"

    report = module.evaluate_completion_gate(status)

    assert report["ok"] is False
    assert "xr64_leakage_risk" in blocker_ids(report)


def test_completion_gate_rejects_corrupt_paper_target_schema_count() -> None:
    module = load_gate_module()
    status = complete_status()
    status["paper_target_comparison_evidence_manifest"]["artifact_schema_count"] = 9

    report = module.evaluate_completion_gate(status)

    assert report["ok"] is False
    assert "paper_target_artifact_schema_incomplete" in blocker_ids(report)


def test_completion_gate_rejects_missing_postrun_promotion_candidate() -> None:
    module = load_gate_module()
    status = complete_status()
    status["xr64_postrun_promotion_decision"]["candidate_count"] = 0

    report = module.evaluate_completion_gate(status)

    assert report["ok"] is False
    assert "xr64_postrun_candidates_missing" in blocker_ids(report)


def test_completion_gate_rejects_invalid_postrun_promotion_decision_artifact() -> None:
    module = load_gate_module()
    status = complete_status()
    status["xr64_postrun_promotion_decision"]["ok"] = False
    status["xr64_postrun_promotion_decision"]["error_count"] = 2

    report = module.evaluate_completion_gate(status)

    assert report["ok"] is False
    assert "xr64_postrun_promotion_decision_invalid" in blocker_ids(report)
    assert report["xr64_postrun_promotion_decision"]["ok"] is False
    assert report["xr64_postrun_promotion_decision"]["error_count"] == 2


def test_completion_gate_rejects_stale_prelaunch_postrun_contract() -> None:
    module = load_gate_module()
    status = complete_status()
    status["xr64_prelaunch_packet"]["postrun_command_count"] = 1

    report = module.evaluate_completion_gate(status)

    assert report["ok"] is False
    assert "xr64_prelaunch_postrun_review_missing" in blocker_ids(report)


def test_completion_gate_summary_preserves_blocker_ids() -> None:
    module = load_gate_module()
    report = module.evaluate_completion_gate(incomplete_status())

    summary = module.format_summary(report, allow_incomplete=True)

    assert "completion_allowed: false" in summary
    assert "allow_incomplete: true" in summary
    assert "blocker_count:" in summary
    assert "execution_paused" in summary
    assert "metric_protocol_bridge_blocked" in summary
