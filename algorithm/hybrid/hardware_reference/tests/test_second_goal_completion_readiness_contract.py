from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


def load_readiness_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "check_second_goal_completion_readiness_contract.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("check_second_goal_completion_readiness_contract", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_current_contract() -> dict:
    path = Path("docs/resources/second_goal_completion_readiness_contract_2026_06_21.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_completion_readiness_contract_passes() -> None:
    module = load_readiness_module()
    report = module.validate_contract(read_current_contract(), project_root=Path.cwd())

    assert report["ok"] is True
    assert report["execution_state"] == "paused_by_user_directive"
    assert report["active_goal_complete"] is False
    assert report["completion_allowed"] is False
    assert report["blocker_count"] == 28
    assert report["xr64_ready_to_train"] is False
    assert report["xr64_can_run_lane"] is False
    assert report["missing_eval_rows"] == 8
    assert report["missing_overrides"] == 6
    assert report["leakage_risk"] == "none"
    assert report["paper_target_artifact_schema_count"] == 10
    assert report["paper_target_bridge_evaluator_schema_execute_supported"] is False
    assert report["paper_target_bridge_evaluator_schema_output_count"] == 5
    assert report["paper_target_bridge_evaluator_schema_current_existing_output_count"] == 0
    assert report["paper_target_bridge_trained_candidate_execute_supported"] is False
    assert report["paper_target_bridge_trained_candidate_schema_allowed_while_paused"] is True
    assert report["paper_target_bridge_trained_candidate_payload_allowed_while_paused"] is False
    assert report["paper_target_bridge_trained_candidate_output_count"] == 2
    assert report["paper_target_bridge_trained_candidate_current_existing_output_count"] == 0
    assert report["paper_target_bridge_decision_execute_supported"] is False
    assert report["paper_target_bridge_decision_schema_allowed_while_paused"] is True
    assert report["paper_target_bridge_decision_payload_allowed_while_paused"] is False
    assert report["paper_target_bridge_decision_output_count"] == 1
    assert report["paper_target_bridge_decision_current_existing_output_count"] == 0
    assert report["xr64_command_manifest_postrun_command_count"] == 2
    assert report["xr64_command_manifest_full_eval_coverage_required"] is True
    assert report["xr64_command_manifest_full_override_coverage_required"] is True
    assert report["xr64_command_manifest_duplicate_sample_ids_allowed"] is False
    assert report["xr64_command_manifest_opposite_split_sample_ids_allowed"] is False
    assert report["xr64_command_manifest_subset_artifacts_satisfy_ready_to_train"] is False
    assert report["xr64_command_manifest_strict_generated_artifact_checker"] == (
        "scripts/external/check_xr64_resume_artifacts.py"
    )
    assert report["xr64_next_prep_selector_ok"] is True
    assert report["xr64_next_prep_selector_ready_after_resume_count"] == 8
    assert report["xr64_next_prep_selector_completed_count"] == 0
    assert report["xr64_next_prep_selector_blocked_count"] == 2
    assert report["xr64_next_prep_selector_invalid_count"] == 0
    assert report["xr64_next_prep_selector_next_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["xr64_next_prep_selector_next_status"] == "ready_after_resume"
    assert report["xr64_next_prep_selector_next_allowed_to_run_now"] is False
    assert report["xr64_next_prep_selector_execute_supported"] is False
    assert report["xr64_prelaunch_phase_count"] == 6
    assert report["xr64_prelaunch_postrun_command_count"] == 2
    assert report["post_xr64_decision_tree_execute_supported"] is False
    assert report["post_xr64_decision_tree_claim_level_count"] == 4
    assert report["post_xr64_decision_tree_decision_input_count"] == 3
    assert report["post_xr64_decision_tree_branch_count"] == 7
    assert report["post_xr64_decision_tree_minimal_next_worker"] == "XR-64-prep"
    assert report["post_xr64_decision_tree_current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert report["post_xr64_decision_tree_xr64_ready_to_train"] is False
    assert report["post_xr64_decision_tree_direct_submission_comparison_allowed"] is False
    assert report["post_xr64_followup_contract_execute_supported"] is False
    assert report["post_xr64_followup_contract_experiment_count"] == 5
    assert report["post_xr64_followup_contract_blocked_launch_count"] == 5
    assert report["post_xr64_followup_contract_followup_launch_allowed_now"] is False
    assert report["xr64_postrun_candidate_count"] == 0
    assert "metric_protocol_bridge_blocked" in report["blocker_ids"]
    assert "paper_target_required_evidence_incomplete" in report["blocker_ids"]
    assert "paper_target_direct_comparison_blocked" in report["blocker_ids"]
    assert "paper_target_future_artifacts_missing" in report["blocker_ids"]
    assert "xr64_eval_rows_missing" in report["blocker_ids"]
    assert "command_emitter_non_executing" in report["guard_ids"]
    assert "xr64_postrun_promotion_decision_valid" in report["guard_ids"]


def test_readiness_contract_rejects_unpaused_state() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["execution_state"] = "running"

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "execution_state must be paused_by_user_directive" in report["errors"]


def test_readiness_contract_rejects_missing_source_artifact_key() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["source_artifacts"].pop("metric_protocol_bridge")

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("source_artifacts missing keys" in error for error in report["errors"])


def test_readiness_contract_rejects_readiness_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["missing_eval_rows"] = 0

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.missing_eval_rows must match current status: 8" in report["errors"]


def test_readiness_contract_rejects_paper_target_manifest_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_required_evidence_complete"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.paper_target_required_evidence_complete must match current status: False" in report["errors"]


def test_readiness_contract_rejects_paper_target_schema_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_artifact_schema_count"] = 9

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.paper_target_artifact_schema_count must match current status: 10" in report["errors"]


def test_readiness_contract_rejects_evaluator_schema_output_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_evaluator_schema_output_count"] = 4

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_evaluator_schema_output_count must match current status: 5"
        in report["errors"]
    )


def test_readiness_contract_rejects_evaluator_schema_payload_presence_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_evaluator_schema_current_existing_output_count"] = 1

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_evaluator_schema_current_existing_output_count "
        "must match current status: 0"
    ) in report["errors"]


def test_readiness_contract_rejects_trained_candidate_contract_payload_presence_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_trained_candidate_current_existing_output_count"] = 1

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_trained_candidate_current_existing_output_count "
        "must match current status: 0"
    ) in report["errors"]


def test_readiness_contract_rejects_trained_candidate_contract_output_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_trained_candidate_output_count"] = 1

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_trained_candidate_output_count must match current status: 2"
        in report["errors"]
    )


def test_readiness_contract_rejects_bridge_decision_payload_presence_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_decision_current_existing_output_count"] = 1

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_decision_current_existing_output_count "
        "must match current status: 0"
    ) in report["errors"]


def test_readiness_contract_rejects_bridge_decision_execute_supported_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_decision_execute_supported"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_decision_execute_supported must match current status: False"
        in report["errors"]
    )


def test_readiness_contract_rejects_bridge_decision_schema_allowed_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_decision_schema_allowed_while_paused"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_decision_schema_allowed_while_paused "
        "must match current status: True"
    ) in report["errors"]


def test_readiness_contract_rejects_bridge_decision_output_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_decision_output_count"] = 0

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_decision_output_count must match current status: 1"
        in report["errors"]
    )


def test_readiness_contract_rejects_bridge_decision_paused_payload_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["paper_target_bridge_decision_payload_allowed_while_paused"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.paper_target_bridge_decision_payload_allowed_while_paused "
        "must match current status: False"
    ) in report["errors"]


def test_readiness_contract_rejects_prelaunch_phase_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_prelaunch_phase_count"] = 5

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.xr64_prelaunch_phase_count must match current status: 6" in report["errors"]


def test_readiness_contract_rejects_next_prep_selector_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_next_prep_selector_next_id"] = "XR64-EVAL-VAL-XR62A"

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_next_prep_selector_next_id must match current status: 'XR64-EVAL-TRAIN-XR62A'"
        in report["errors"]
    )


def test_readiness_contract_rejects_next_prep_selector_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_next_prep_selector_ready_after_resume_count"] = 7

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.xr64_next_prep_selector_ready_after_resume_count must match current status: 8" in report["errors"]


def test_readiness_contract_rejects_next_prep_selector_status_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_next_prep_selector_next_status"] = "blocked_missing_inputs"

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.xr64_next_prep_selector_next_status must match current status: 'ready_after_resume'" in report["errors"]


def test_readiness_contract_rejects_next_prep_selector_execute_supported_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_next_prep_selector_execute_supported"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.xr64_next_prep_selector_execute_supported must match current status: False" in report["errors"]


def test_readiness_contract_rejects_missing_next_prep_selector_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "emit_xr64_next_prep_command.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("emit_xr64_next_prep_command.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_next_prep_selector_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_emit_xr64_next_prep_command.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("tests/test_emit_xr64_next_prep_command.py" in error for error in report["errors"])


def test_readiness_contract_rejects_prelaunch_postrun_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_prelaunch_postrun_command_count"] = 1

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "current_readiness.xr64_prelaunch_postrun_command_count must match current status: 2" in report["errors"]


def test_readiness_contract_rejects_subset_artifact_ready_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_subset_artifacts_satisfy_ready_to_train"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_subset_artifacts_satisfy_ready_to_train must match current status: False"
        in report["errors"]
    )


def test_readiness_contract_rejects_full_eval_coverage_policy_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_full_eval_coverage_required"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_full_eval_coverage_required must match current status: True"
        in report["errors"]
    )


def test_readiness_contract_rejects_full_override_coverage_policy_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_full_override_coverage_required"] = False

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_full_override_coverage_required must match current status: True"
        in report["errors"]
    )


def test_readiness_contract_rejects_duplicate_sample_policy_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_duplicate_sample_ids_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_duplicate_sample_ids_allowed must match current status: False"
        in report["errors"]
    )


def test_readiness_contract_rejects_opposite_split_policy_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_opposite_split_sample_ids_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_opposite_split_sample_ids_allowed must match current status: False"
        in report["errors"]
    )


def test_readiness_contract_rejects_strict_checker_path_drift() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["xr64_command_manifest_strict_generated_artifact_checker"] = (
        "scripts/external/loose_checker.py"
    )

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert (
        "current_readiness.xr64_command_manifest_strict_generated_artifact_checker must match current status: "
        "'scripts/external/check_xr64_resume_artifacts.py'"
        in report["errors"]
    )


def test_readiness_contract_rejects_post_xr64_minimal_worker_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["post_xr64_decision_tree_minimal_next_worker"] = "XR-65"

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert "current_readiness.post_xr64_decision_tree_minimal_next_worker must match current status: 'XR-64-prep'" in report["errors"]


def test_readiness_contract_rejects_post_xr64_branch_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["post_xr64_decision_tree_branch_count"] = 6

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert "current_readiness.post_xr64_decision_tree_branch_count must match current status: 7" in report["errors"]


def test_readiness_contract_rejects_post_xr64_direct_comparison_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["post_xr64_decision_tree_direct_submission_comparison_allowed"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert (
        "current_readiness.post_xr64_decision_tree_direct_submission_comparison_allowed must match current status: False"
        in report["errors"]
    )


def test_readiness_contract_rejects_post_xr64_followup_launch_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["post_xr64_followup_contract_followup_launch_allowed_now"] = True

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert (
        "current_readiness.post_xr64_followup_contract_followup_launch_allowed_now "
        "must match current status: False"
    ) in report["errors"]


def test_readiness_contract_rejects_post_xr64_followup_count_mismatch() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["current_readiness"]["post_xr64_followup_contract_blocked_launch_count"] = 4

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert "current_readiness.post_xr64_followup_contract_blocked_launch_count must match current status: 5" in report["errors"]


def test_readiness_contract_rejects_missing_post_xr64_decision_tree_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_second_goal_post_xr64_decision_tree.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert any("check_second_goal_post_xr64_decision_tree.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_post_xr64_followup_contract_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_second_goal_post_xr64_followup_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert any("check_second_goal_post_xr64_followup_contract.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_post_xr64_followup_contract_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_second_goal_post_xr64_followup_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert any("tests/test_second_goal_post_xr64_followup_contract.py" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_prelaunch_packet_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_xr64_prelaunch_packet.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("check_xr64_prelaunch_packet.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_forbidden_train_command() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["forbidden_train_commands"] = ["train_hbtxr.py", "eval_hbtxr.py"]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert "validation_contract.forbidden_train_commands missing required forbidden commands" in report["errors"]


def test_readiness_contract_rejects_missing_paper_target_manifest_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_paper_target_comparison_evidence_manifest.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("check_paper_target_comparison_evidence_manifest.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_evaluator_schema_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_paper_target_bridge_evaluator_schema_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any(
        "check_paper_target_bridge_evaluator_schema_contract.py --format summary" in error
        for error in report["errors"]
    )


def test_readiness_contract_rejects_missing_evaluator_schema_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_paper_target_bridge_evaluator_schema_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any(
        "tests/test_paper_target_bridge_evaluator_schema_contract.py" in error
        for error in report["errors"]
    )


def test_readiness_contract_rejects_missing_trained_candidate_contract_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_paper_target_bridge_trained_candidate_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any(
        "check_paper_target_bridge_trained_candidate_contract.py --format summary" in error
        for error in report["errors"]
    )


def test_readiness_contract_rejects_missing_trained_candidate_contract_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_paper_target_bridge_trained_candidate_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any(
        "tests/test_paper_target_bridge_trained_candidate_contract.py" in error
        for error in report["errors"]
    )


def test_readiness_contract_rejects_missing_bridge_decision_contract_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_paper_target_bridge_decision_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any(
        "check_paper_target_bridge_decision_contract.py --format summary" in error
        for error in report["errors"]
    )


def test_readiness_contract_rejects_missing_bridge_decision_contract_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_paper_target_bridge_decision_contract.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("tests/test_paper_target_bridge_decision_contract.py" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_payload_checker_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "check_paper_target_bridge_payloads.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("check_paper_target_bridge_payloads.py --format summary" in error for error in report["errors"])


def test_readiness_contract_rejects_missing_payload_checker_test_check() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"] = [
        item
        for item in contract["validation_contract"]["required_checks"]
        if "tests/test_paper_target_bridge_payloads.py" not in item
    ]

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("tests/test_paper_target_bridge_payloads.py" in error for error in report["errors"])


def test_readiness_contract_rejects_train_command_in_required_checks() -> None:
    module = load_readiness_module()
    contract = copy.deepcopy(read_current_contract())
    contract["validation_contract"]["required_checks"].append(
        "bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0"
    )

    report = module.validate_contract(contract, project_root=Path.cwd())

    assert report["ok"] is False
    assert any("required_checks must not execute" in error for error in report["errors"])
