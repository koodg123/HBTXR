from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def load_status_module():
    script_dir = Path("scripts/external").resolve()
    script = script_dir / "report_second_goal_status.py"
    sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("report_second_goal_status", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def write_index(root: Path) -> Path:
    index = {
        "date": "2026-06-18",
        "execution_state": "paused_by_user_directive",
        "active_goal_complete": False,
        "current_gates": {
            "center_lt": 16.468481131962367,
            "p10_gt": 35.02295998845781,
            "p5_gt": 12.133503770828247,
        },
        "authority": {
            "pause_state": "docs/resources/experiment_pause_documentation_2026_06_18.md",
            "completion_audit": "docs/resources/second_goal_completion_audit_2026_06_18.md",
        },
        "analysis_artifacts": {
            "paper_ref_map": "anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md",
            "paper_ref_per_paper_index": "anlaysis/paper-ref/papers/index.md",
            "paper_ref_per_paper_root": "anlaysis/paper-ref/papers",
            "paper_ref_per_paper_analysis_count": 31,
            "per_codebase_analysis_count": 18,
            "per_paper_analysis_count": 21,
            "per_item_analysis_count": 39,
            "per_item_analysis_total_lines": 15590,
        },
        "xr63_xr64_artifacts": {
            "resume_checker": "scripts/external/check_xr64_resume_artifacts.py",
            "status_cli": "scripts/external/report_second_goal_status.py",
        },
        "experiment_queue": {
            "queue_md": "docs/resources/second_goal_experiment_queue_2026_06_20.md",
            "queue_json": "docs/resources/second_goal_experiment_queue_2026_06_20.json",
            "queue_checker": "scripts/external/check_second_goal_experiment_queue.py",
            "queue_checker_tests": "tests/test_second_goal_experiment_queue.py",
        },
        "xr64_experiment_design_contract": {
            "contract_md": "docs/resources/xr64_experiment_design_contract_2026_06_21.md",
            "contract_json": "docs/resources/xr64_experiment_design_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_xr64_experiment_design_contract.py",
            "contract_checker_tests": "tests/test_xr64_experiment_design_contract.py",
        },
        "xr64_resume_execution_dag": {
            "dag_md": "docs/resources/xr64_resume_execution_dag_2026_06_21.md",
            "dag_json": "docs/resources/xr64_resume_execution_dag_2026_06_21.json",
            "dag_checker": "scripts/external/check_xr64_resume_execution_dag.py",
            "dag_checker_tests": "tests/test_xr64_resume_execution_dag.py",
        },
        "ablation_evidence_checklist": {
            "checklist_md": "docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md",
            "checklist_json": "docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.json",
            "checklist_checker": "scripts/external/check_second_goal_ablation_evidence_checklist.py",
            "checklist_checker_tests": "tests/test_second_goal_ablation_evidence_checklist.py",
        },
        "paper_ref_experiment_trace": {
            "trace_md": "docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.md",
            "trace_json": "docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json",
            "trace_checker": "scripts/external/check_second_goal_paper_ref_experiment_trace.py",
            "trace_checker_tests": "tests/test_second_goal_paper_ref_experiment_trace.py",
        },
        "paper_ref_current_ablation_plan": {
            "plan_md": "docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md",
            "plan_checker": "scripts/external/check_second_goal_paper_ref_current_ablation_plan.py",
            "plan_checker_tests": "tests/test_second_goal_paper_ref_current_ablation_plan.py",
        },
        "post_xr64_decision_tree": {
            "decision_tree_md": "docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.md",
            "decision_tree_json": "docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json",
            "decision_tree_checker": "scripts/external/check_second_goal_post_xr64_decision_tree.py",
            "decision_tree_checker_tests": "tests/test_second_goal_post_xr64_decision_tree.py",
        },
        "post_xr64_followup_experiment_contract": {
            "contract_md": "docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.md",
            "contract_json": "docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_second_goal_post_xr64_followup_contract.py",
            "contract_checker_tests": "tests/test_second_goal_post_xr64_followup_contract.py",
        },
        "completion_readiness_contract": {
            "contract_md": "docs/resources/second_goal_completion_readiness_contract_2026_06_21.md",
            "contract_json": "docs/resources/second_goal_completion_readiness_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_second_goal_completion_readiness_contract.py",
            "contract_checker_tests": "tests/test_second_goal_completion_readiness_contract.py",
        },
        "submission_target_gap": {
            "gap_audit_md": "docs/resources/second_goal_submission_target_gap_audit_2026_06_20.md",
            "gap_audit_json": "docs/resources/second_goal_submission_target_gap_audit_2026_06_20.json",
            "gap_checker": "scripts/external/check_submission_target_gap.py",
            "gap_checker_tests": "tests/test_submission_target_gap.py",
            "submission_center_target_px": 0.1812,
            "submission_p10_target_pct": 99.97,
            "submission_p5_target_pct": 99.72,
            "direct_comparison_valid": False,
        },
        "submission_target_source_trace": {
            "trace_md": "docs/resources/submission_target_source_trace_2026_06_21.md",
            "trace_json": "docs/resources/submission_target_source_trace_2026_06_21.json",
            "trace_checker": "scripts/external/check_submission_target_source_trace.py",
            "trace_checker_tests": "tests/test_submission_target_source_trace.py",
        },
        "metric_protocol_bridge": {
            "bridge_md": "docs/resources/second_goal_metric_protocol_bridge_2026_06_20.md",
            "bridge_json": "docs/resources/second_goal_metric_protocol_bridge_2026_06_20.json",
            "bridge_checker": "scripts/external/check_metric_protocol_bridge.py",
            "bridge_checker_tests": "tests/test_metric_protocol_bridge.py",
            "bridge_status": "blocked_until_metric_frame_and_protocol_match",
            "direct_submission_comparison_allowed": False,
        },
        "metric_protocol_unblock_contract": {
            "contract_md": "docs/resources/metric_protocol_unblock_contract_2026_06_21.md",
            "contract_json": "docs/resources/metric_protocol_unblock_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_metric_protocol_unblock_contract.py",
            "contract_checker_tests": "tests/test_metric_protocol_unblock_contract.py",
            "contract_status": "blocked",
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
        },
        "paper_target_comparison_evidence_manifest": {
            "manifest_md": "docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md",
            "manifest_json": "docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json",
            "manifest_checker": "scripts/external/check_paper_target_comparison_evidence_manifest.py",
            "manifest_checker_tests": "tests/test_paper_target_comparison_evidence_manifest.py",
            "evidence_state": "missing_required_evidence",
            "required_evidence_complete": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "artifact_schema_count": 10,
        },
        "paper_target_bridge_artifact_workplan": {
            "workplan_md": "docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md",
            "workplan_json": "docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json",
            "workplan_checker": "scripts/external/check_paper_target_bridge_artifact_workplan.py",
            "workplan_checker_tests": "tests/test_paper_target_bridge_artifact_workplan.py",
            "execute_supported": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "work_package_count": 4,
            "future_artifact_count": 10,
            "allowed_while_paused_package_count": 2,
            "doc_prefill_artifact_count": 3,
            "trained_candidate_artifact_count": 2,
        },
        "paper_target_bridge_evaluator_schema_contract": {
            "contract_md": "docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.md",
            "contract_json": "docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_paper_target_bridge_evaluator_schema_contract.py",
            "contract_checker_tests": "tests/test_paper_target_bridge_evaluator_schema_contract.py",
            "execute_supported": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "evaluator_output_count": 5,
            "jsonl_output_schema_count": 2,
            "json_output_schema_count": 3,
            "blocked_output_count": 5,
            "current_existing_output_count": 0,
        },
        "paper_target_bridge_ptb1_resume_runbook": {
            "runbook_md": "docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.md",
            "runbook_json": "docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json",
            "runbook_checker": "scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py",
            "runbook_checker_tests": "tests/test_paper_target_bridge_ptb1_resume_runbook.py",
            "execute_supported": False,
            "allowed_while_paused": True,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "sequence_count": 5,
            "ptb1_output_count": 5,
            "ptb1_current_existing_output_count": 0,
            "ptb1_payload_missing_count": 5,
            "future_payload_validation_requires_allow_present": True,
        },
        "paper_target_bridge_trained_candidate_contract": {
            "contract_md": "docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.md",
            "contract_json": "docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_paper_target_bridge_trained_candidate_contract.py",
            "contract_checker_tests": "tests/test_paper_target_bridge_trained_candidate_contract.py",
            "execute_supported": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "schema_contract_allowed_while_paused": True,
            "trained_candidate_payload_allowed_while_paused": False,
            "trained_candidate_output_count": 2,
            "json_output_schema_count": 2,
            "blocked_output_count": 2,
            "current_existing_output_count": 0,
        },
        "paper_target_bridge_decision_contract": {
            "contract_md": "docs/resources/paper_target_bridge_decision_contract_2026_06_21.md",
            "contract_json": "docs/resources/paper_target_bridge_decision_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_paper_target_bridge_decision_contract.py",
            "contract_checker_tests": "tests/test_paper_target_bridge_decision_contract.py",
            "execute_supported": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "schema_contract_allowed_while_paused": True,
            "bridge_decision_payload_allowed_while_paused": False,
            "decision_output_count": 1,
            "json_output_schema_count": 1,
            "blocked_output_count": 1,
            "current_existing_output_count": 0,
            "signoff_rule_count": 6,
        },
        "paper_target_bridge_resume_dependency_matrix": {
            "matrix_md": "docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.md",
            "matrix_json": "docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json",
            "matrix_checker": "scripts/external/check_paper_target_bridge_resume_dependency_matrix.py",
            "matrix_checker_tests": "tests/test_paper_target_bridge_resume_dependency_matrix.py",
            "execute_supported": False,
            "allowed_while_paused": True,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
            "dependency_gate_count": 6,
            "allowed_while_paused_gate_count": 2,
            "complete_now_gate_count": 2,
            "blocked_gate_count": 4,
            "execution_dependent_payload_missing_count": 8,
            "completion_blocker_count": 28,
        },
        "paper_target_metric_bridge_preflight": {
            "preflight_md": "docs/resources/paper_target_metric_bridge_preflight_2026_06_21.md",
            "preflight_json": "docs/resources/paper_target_metric_bridge_preflight_2026_06_21.json",
            "preflight_checker": "scripts/external/check_paper_target_metric_bridge_preflight.py",
            "preflight_checker_tests": "tests/test_paper_target_metric_bridge_preflight.py",
        },
        "second_goal_blocker_artifact_map": {
            "map_md": "docs/resources/second_goal_blocker_artifact_map_2026_06_21.md",
            "map_json": "docs/resources/second_goal_blocker_artifact_map_2026_06_21.json",
            "map_checker": "scripts/external/check_second_goal_blocker_artifact_map.py",
            "map_checker_tests": "tests/test_second_goal_blocker_artifact_map.py",
        },
        "resume_readiness_matrix": {
            "matrix_md": "docs/resources/second_goal_resume_readiness_matrix_2026_06_21.md",
            "matrix_json": "docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json",
            "matrix_checker": "scripts/external/check_second_goal_resume_readiness_matrix.py",
            "matrix_checker_tests": "tests/test_second_goal_resume_readiness_matrix.py",
        },
        "xr64_prep_progress_ledger": {
            "ledger_md": "docs/resources/xr64_prep_progress_ledger_2026_06_21.md",
            "ledger_json": "docs/resources/xr64_prep_progress_ledger_2026_06_21.json",
            "ledger_checker": "scripts/external/check_xr64_prep_progress_ledger.py",
            "ledger_checker_tests": "tests/test_xr64_prep_progress_ledger.py",
        },
        "current_result_synthesis": {
            "synthesis_md": "docs/resources/second_goal_current_result_synthesis_2026_06_20.md",
            "synthesis_json": "docs/resources/second_goal_current_result_synthesis_2026_06_20.json",
            "synthesis_checker": "scripts/external/check_current_result_synthesis.py",
            "synthesis_checker_tests": "tests/test_current_result_synthesis.py",
        },
        "accuracy_lift_decision_ladder": {
            "ladder_md": "docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.md",
            "ladder_json": "docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.json",
            "ladder_checker": "scripts/external/check_second_goal_accuracy_lift_decision_ladder.py",
            "ladder_checker_tests": "tests/test_second_goal_accuracy_lift_decision_ladder.py",
        },
        "objective_trace": {
            "trace_md": "docs/resources/second_goal_objective_trace_2026_06_20.md",
            "trace_json": "docs/resources/second_goal_objective_trace_2026_06_20.json",
            "trace_checker": "scripts/external/check_second_goal_objective_trace.py",
            "trace_checker_tests": "tests/test_second_goal_objective_trace.py",
        },
        "xr64_resume_command_manifest": {
            "manifest_md": "docs/resources/xr64_resume_command_manifest_2026_06_20.md",
            "manifest_json": "docs/resources/xr64_resume_command_manifest_2026_06_20.json",
            "manifest_checker": "scripts/external/check_xr64_resume_command_manifest.py",
            "manifest_checker_tests": "tests/test_xr64_resume_command_manifest.py",
        },
        "xr64_resume_command_emitter": {
            "script": "scripts/external/emit_xr64_resume_commands.py",
            "tests": "tests/test_emit_xr64_resume_commands.py",
        },
        "xr64_next_prep_command_selector": {
            "script": "scripts/external/emit_xr64_next_prep_command.py",
            "tests": "tests/test_emit_xr64_next_prep_command.py",
            "manifest_json": "docs/resources/xr64_resume_command_manifest_2026_06_20.json",
        },
        "xr64_prelaunch_packet": {
            "packet_md": "docs/resources/xr64_prelaunch_packet_2026_06_20.md",
            "packet_json": "docs/resources/xr64_prelaunch_packet_2026_06_20.json",
            "packet_checker": "scripts/external/check_xr64_prelaunch_packet.py",
            "packet_checker_tests": "tests/test_xr64_prelaunch_packet.py",
        },
        "xr64_postrun_evidence_contract": {
            "contract_md": "docs/resources/xr64_postrun_evidence_contract_2026_06_21.md",
            "contract_json": "docs/resources/xr64_postrun_evidence_contract_2026_06_21.json",
            "contract_checker": "scripts/external/check_xr64_postrun_evidence_contract.py",
            "contract_checker_tests": "tests/test_xr64_postrun_evidence_contract.py",
        },
        "xr64_postrun_promotion_decision": {
            "decision_md": "docs/resources/xr64_postrun_promotion_decision_2026_06_21.md",
            "decision_json": "docs/resources/xr64_postrun_promotion_decision_2026_06_21.json",
            "candidate_collector": "scripts/external/collect_xr64_postrun_candidates.py",
            "candidate_collector_tests": "tests/test_collect_xr64_postrun_candidates.py",
            "decision_helper": "scripts/external/decide_xr64_postrun_promotion.py",
            "decision_helper_tests": "tests/test_decide_xr64_postrun_promotion.py",
            "decision_checker": "scripts/external/check_xr64_postrun_promotion_decision.py",
            "decision_checker_tests": "tests/test_xr64_postrun_promotion_decision.py",
        },
        "xr64_generated_state": {
            "train_val_eval_rows_exist": False,
            "train_val_override_json_exist": False,
            "xr64_training_runs_exist": False,
        },
        "next_resume_gate": {
            "required_resume_status": "ready_to_train",
            "required_can_run_lane": True,
        },
    }
    result_synthesis = {
        "result_interpretation": {"single_model_sota_claim_allowed": False},
        "oracle_diagnostic": {"promotable_as_result": False},
        "submission_target_relation": {"direct_comparison_valid": False},
        "xr64_resume_gate": {
            "ready_to_train": False,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
        },
    }
    objective_trace = {
        "requirements": [
            {"id": "REQ-1", "status": "complete_as_planning_input"},
            {
                "id": "REQ-2",
                "status": "planned_not_fully_executed",
                "axis_coverage": {
                    "head": ["XR-64A"],
                    "loss": ["XR-64A"],
                    "lr": ["XR-64A"],
                    "optimizer": ["XR-68"],
                    "self_supervised_distillation": ["XR-65"],
                    "teacher_model_training": ["XR-64-prep"],
                },
                "post_xr64_decision_tree_summary": {
                    "branch_count": 7,
                    "minimal_next_worker_after_resume": "XR-64-prep",
                    "current_next_prep_id": "XR64-EVAL-TRAIN-XR62A",
                    "direct_submission_comparison_allowed": False,
                    "next_experiments": ["XR-64C", "XR-65", "XR-66", "XR-67", "XR-68"],
                },
            },
            {"id": "REQ-3", "status": "incomplete"},
        ],
        "completion_judgment": {"goal_complete": False},
    }
    command_manifest = {
        "source_artifacts": {
            "prep_runner": "scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh",
        },
        "do_not_execute_until_user_resumes": True,
        "default_allowed_to_run": False,
        "prep_commands": [
            *[
                {
                    "action": "eval",
                    "required_inputs": [
                        f"data/_internal/manifests/manifest1/{split}_manifest.jsonl",
                        f"runs/checkpoints/{teacher}.pt",
                    ],
                }
                for split in ("train", "val")
                for teacher in ("xr62a", "xr39", "xr56b", "xr58a")
            ],
            {
                "action": "build",
                "required_inputs": [
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json",
                ],
            },
            {
                "action": "build",
                "required_inputs": [
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json",
                    "data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json",
                ],
            },
        ],
        "expected_generated_files": {
            "eval_rows": [f"eval-{idx}" for idx in range(8)],
            "overrides": [f"override-{idx}" for idx in range(6)],
        },
        "launch_commands_after_strict_ready": [
            {"lane": "XR-64A"},
            {"lane": "XR-64B"},
        ],
        "postrun_commands_after_launch": [
            {"id": "XR64-POSTRUN-COLLECT-CANDIDATES"},
            {"id": "XR64-POSTRUN-PRINT-DECISION-COMMAND"},
        ],
    }
    prelaunch_packet = {
        "allowed_to_execute": False,
        "current_state": {
            "completion_allowed": False,
            "completion_blocker_count": 28,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
            "eval_required_input_count": 6,
            "existing_eval_required_input_count": 6,
            "missing_eval_required_input_count": 0,
            "eval_inputs_ready": True,
            "build_required_input_count": 8,
            "existing_build_required_input_count": 0,
            "missing_build_required_input_count": 8,
            "build_inputs_ready": False,
            "postrun_command_count": 2,
        },
        "prelaunch_phases": [
            {"id": "PHASE-0-READONLY-STATUS"},
            {"id": "PHASE-1-XR64-PREP-EVAL-ROWS"},
            {"id": "PHASE-2-XR64-PREP-OVERRIDES"},
            {"id": "PHASE-3-STRICT-READY"},
            {"id": "PHASE-4-XR64A-XR64B-LAUNCH"},
            {"id": "PHASE-5-POSTRUN-CANDIDATE-REVIEW"},
        ],
    }
    postrun_contract = {
        "evidence_state": "missing_postrun_evidence",
        "allowed_to_claim_promotion": False,
        "allowed_to_mark_second_goal_complete": False,
        "required_lanes": [{"id": "XR-64A"}, {"id": "XR-64B"}],
        "required_checkpoint_kinds": ["best_track_p10", "best_track_p5", "best_metric_track_center_px"],
        "required_metrics": [
            "metric_track_center_px",
            "metric_track_p10_pct",
            "metric_track_p5_pct",
            "metric_track_p1_pct",
        ],
        "ablation_evidence_contract": {
            "required_axes": ["head", "loss", "lr", "teacher_model_training"],
            "lane_axis_map": {
                "XR-64A": {
                    "head": "heatmap/state heads plus conservative teacher-target override",
                    "loss": "conservative override losses",
                    "lr": "low-LR conservative continuation",
                    "teacher_model_training": "XR-62A init with XR-39 teacher",
                },
                "XR-64B": {
                    "head": "heatmap/state heads plus threshold-priority teacher-target override",
                    "loss": "threshold-priority override losses",
                    "lr": "low-LR threshold-priority continuation",
                    "teacher_model_training": "XR-56B init with XR-39 teacher",
                },
            },
        },
        "current_postrun_evidence": {
            "xr64a_eval_summary_count": 0,
            "xr64b_eval_summary_count": 0,
        },
        "decision_template": {"decision_status": "pending_postrun_evidence"},
    }
    promotion_decision = json.loads(
        Path("docs/resources/xr64_postrun_promotion_decision_2026_06_21.json").read_text(encoding="utf-8")
    )
    queue = {
        "execution_state": "paused_by_user_directive",
        "do_not_execute_until_user_resumes": True,
        "queue": [
            {"id": "XR-64-prep", "priority": "P0"},
            {"id": "XR-64A", "priority": "P0"},
            {"id": "XR-64B", "priority": "P0"},
            {"id": "XR-64C", "priority": "P1"},
        ],
    }
    ablation_checklist = {
        "experiments": [
            {"id": "XR-64-prep", "priority": "P0"},
            {"id": "XR-64A", "priority": "P0"},
            {"id": "XR-64B", "priority": "P0"},
            {"id": "XR-64C", "priority": "P1"},
            {"id": "XR-65", "priority": "P1"},
            {"id": "XR-66", "priority": "P1"},
            {"id": "XR-67", "priority": "P2"},
            {"id": "XR-68", "priority": "P2"},
        ],
        "axis_coverage": {
            "head": ["XR-64A"],
            "loss": ["XR-64A"],
            "lr": ["XR-64A"],
            "optimizer": ["XR-68"],
            "self_supervised_distillation": ["XR-65"],
            "teacher_model_training": ["XR-64-prep"],
            "data_protocol": ["XR-64-prep"],
            "hardware_export": ["XR-68"],
        },
        "current_state": {
            "software_promotion_allowed": False,
            "paper_level_completion_allowed": False,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
        },
    }
    post_xr64_tree = {
        "execute_supported": False,
        "claim_levels": [
            {"level": "diagnostic_improvement"},
            {"level": "software_promotion"},
            {"level": "clean_single_model_promotion"},
            {"level": "paper_level_completion"},
        ],
        "decision_inputs": [
            {"id": "xr64_readiness"},
            {"id": "xr64a_b_full_test"},
            {"id": "postrun_contract"},
        ],
        "branches": [
            {"result_pattern": "validation_gain_without_full_test_gate", "next_experiment": "XR-64C"},
            {"result_pattern": "gate_improvement_with_bounded_tradeoff", "next_experiment": "XR-65"},
            {"result_pattern": "center_gain_threshold_collapse", "next_experiment": "XR-66"},
            {"result_pattern": "dense_segments_available", "next_experiment": "XR-67"},
            {"result_pattern": "stable_full_width_teacher_improves_gates", "next_experiment": "XR-68"},
            {"result_pattern": "p10_gain_p5_collapse", "next_experiment": "bounded XR-64B continuation"},
            {"result_pattern": "no_validation_movement", "next_experiment": "data/protocol diagnostics"},
        ],
        "source_artifacts": [
            "anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md",
            "docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md",
            "docs/resources/second_goal_experiment_queue_2026_06_20.md",
            "docs/resources/xr64_postrun_evidence_contract_2026_06_21.md",
        ],
        "signoff_rules": [
            "Do not bypass XR-64-prep readiness.",
            "Do not promote XR-63 oracle or validation-only gains.",
            "Do not compare against 10_submission_initial targets until direct submission comparison is allowed.",
            "Do not start XR-65, XR-66, XR-67, or XR-68 unless the matching result pattern and evidence exist.",
        ],
        "minimal_next_worker_after_resume": "XR-64-prep",
        "current_next_prep_id": "XR64-EVAL-TRAIN-XR62A",
        "current_blockers": {
            "xr64_ready_to_train": False,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
            "xr64_postrun_candidate_count": 0,
            "direct_submission_comparison_allowed": False,
        },
    }
    post_xr64_followup = {
        "execute_supported": False,
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "current_state": {
            "minimal_next_worker_after_resume": "XR-64-prep",
            "xr64_postrun_candidate_count": 0,
            "followup_launch_allowed_now": False,
        },
        "followup_experiments": [
            {"id": "XR-64C", "priority": "P1", "launch_allowed": False},
            {"id": "XR-65", "priority": "P1", "launch_allowed": False},
            {"id": "XR-66", "priority": "P1", "launch_allowed": False},
            {"id": "XR-67", "priority": "P2", "launch_allowed": False},
            {"id": "XR-68", "priority": "P2", "launch_allowed": False},
        ],
    }
    completion_readiness = {
        "current_readiness": {
            "completion_allowed": False,
            "blocker_count": 28,
            "xr64_ready_to_train": False,
            "xr64_can_run_lane": False,
            "missing_eval_rows": 8,
            "missing_overrides": 6,
            "leakage_risk": "none",
            "paper_target_evidence_state": "missing_required_evidence",
            "paper_target_required_evidence_complete": False,
            "paper_target_direct_submission_comparison_allowed": False,
            "paper_target_required_future_artifact_count": 10,
            "paper_target_existing_required_future_artifact_count": 3,
            "paper_target_bridge_evaluator_schema_execute_supported": False,
            "paper_target_bridge_evaluator_schema_output_count": 5,
            "paper_target_bridge_evaluator_schema_current_existing_output_count": 0,
            "paper_target_bridge_trained_candidate_execute_supported": False,
            "paper_target_bridge_trained_candidate_schema_allowed_while_paused": True,
            "paper_target_bridge_trained_candidate_payload_allowed_while_paused": False,
            "paper_target_bridge_trained_candidate_output_count": 2,
            "paper_target_bridge_trained_candidate_current_existing_output_count": 0,
            "paper_target_bridge_decision_execute_supported": False,
            "paper_target_bridge_decision_schema_allowed_while_paused": True,
            "paper_target_bridge_decision_payload_allowed_while_paused": False,
            "paper_target_bridge_decision_output_count": 1,
            "paper_target_bridge_decision_current_existing_output_count": 0,
            "post_xr64_followup_contract_execute_supported": False,
            "post_xr64_followup_contract_experiment_count": 5,
            "post_xr64_followup_contract_blocked_launch_count": 5,
            "post_xr64_followup_contract_followup_launch_allowed_now": False,
            "xr64_command_manifest_postrun_command_count": 2,
            "xr64_command_manifest_full_eval_coverage_required": True,
            "xr64_command_manifest_full_override_coverage_required": True,
            "xr64_command_manifest_duplicate_sample_ids_allowed": False,
            "xr64_command_manifest_opposite_split_sample_ids_allowed": False,
            "xr64_command_manifest_subset_artifacts_satisfy_ready_to_train": False,
            "xr64_command_manifest_strict_generated_artifact_checker": "scripts/external/check_xr64_resume_artifacts.py",
            "xr64_next_prep_selector_ok": True,
            "xr64_next_prep_selector_ready_after_resume_count": 8,
            "xr64_next_prep_selector_completed_count": 0,
            "xr64_next_prep_selector_blocked_count": 2,
            "xr64_next_prep_selector_invalid_count": 0,
            "xr64_next_prep_selector_next_id": "XR64-EVAL-TRAIN-XR62A",
            "xr64_next_prep_selector_next_status": "ready_after_resume",
            "xr64_next_prep_selector_next_allowed_to_run_now": False,
            "xr64_next_prep_selector_execute_supported": False,
            "xr64_prelaunch_phase_count": 6,
            "xr64_prelaunch_postrun_command_count": 2,
            "xr64_postrun_candidate_count": 0,
        },
        "next_allowed_actions": [
            "read current status with report_second_goal_status.py",
            "validate experiment queue and ablation evidence checklist",
            "validate completion gate with --allow-incomplete",
            "after explicit user resume only: generate XR-64 train/val eval rows",
            "after strict XR-64 readiness only: launch XR-64A/B lanes",
        ],
    }
    target_gap = {
        "direct_comparison_valid": False,
        "submission_claims": {
            "hybrid_pixel_error_px": 0.1812,
            "hybrid_p10_pct": 99.97,
            "hybrid_p5_pct": 99.72,
        },
        "direct_numeric_gap_if_same_metric_frame": {
            "center_error_ratio_current_over_submission": 90.88565746116096,
            "p10_gap_pct_points": 64.94704001154219,
            "p5_gap_pct_points": 87.58649622917176,
        },
    }
    source_trace = {
        "source_file": "docs/resources/fake_submission_main.tex",
        "source_targets": {
            "mode_table_hybrid": {
                "pixel_error_px": 0.1812,
                "p10_pct": 99.97,
                "p5_pct": 99.72,
                "p1_pct": 99.61,
                "latency_ms": 0.43,
            },
            "accelerator_table_range": {"is_single_accuracy_target": False},
        },
        "current_software_target_relation": {
            "direct_comparison_valid": False,
            "required_before_direct_comparison": [
                "coordinate-frame match",
                "hybrid scheduler-mode evaluation",
                "paper-frame full-test P1 evidence",
            ],
        },
    }
    bridge = {
        "bridge_status": "blocked_until_metric_frame_and_protocol_match",
        "direct_submission_comparison_allowed": False,
        "coordinate_sanity_evidence": {
            "metric_frame": "post-transform input coordinate frame",
            "directly_sensor_px": False,
        },
        "hybrid_protocol_match": {
            "match_status": "not_proven",
            "current_p1_metric_available": True,
            "current_p1_full_test_evidence_available": False,
        },
    }
    unblock_contract = {
        "contract_status": "blocked",
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "required_evidence": [
            {"id": "coordinate_frame_match"},
            {"id": "hybrid_scheduler_eval"},
            {"id": "p1_metric_coverage"},
            {"id": "split_protocol_match"},
            {"id": "target_definition_match"},
            {"id": "latency_separation"},
            {"id": "xr64_result_dependency"},
        ],
        "current_blockers": [
            "metric frame",
            "sensor-space rows",
            "hybrid scheduler",
            "P1",
            "split protocol",
            "XR-64 evidence",
        ],
    }
    paper_target_manifest = {
        "comparison_decision": {
            "evidence_state": "missing_required_evidence",
            "required_evidence_complete": False,
            "direct_submission_comparison_allowed": False,
            "paper_level_completion_allowed": False,
        },
        "required_evidence_items": [
            {"id": "coordinate_frame_match", "status": "missing", "required_artifacts": [{"path": "future/1"}]},
            {"id": "hybrid_scheduler_eval", "status": "missing", "required_artifacts": [{"path": "future/2"}]},
            {
                "id": "p1_full_test_metric",
                "status": "partial",
                "current_artifacts": [{"path": "src/metrics"}, {"path": "tests/metrics"}],
                "required_artifacts": [{"path": "future/3"}],
            },
            {
                "id": "split_protocol_match",
                "status": "partial",
                "required_artifacts": [
                    {"path": "docs/resources/future/paper_target_bridge/split_protocol_audit.json"}
                ],
            },
            {
                "id": "target_definition_match",
                "status": "partial",
                "current_artifacts": [{"path": "docs/trace"}],
                "required_artifacts": [
                    {"path": "docs/resources/future/paper_target_bridge/cur_state_target_mapping_audit.json"}
                ],
            },
            {
                "id": "latency_separation",
                "status": "partial",
                "current_artifacts": [{"path": "docs/latency"}],
                "required_artifacts": [
                    {"path": "docs/resources/future/paper_target_bridge/accuracy_latency_separation_note.json"}
                ],
            },
            {"id": "xr64_or_later_trained_result", "status": "missing", "required_artifacts": [{"path": "future/7"}, {"path": "future/8"}, {"path": "future/9"}, {"path": "future/10"}]},
        ],
        "artifact_content_schema": {f"schema-{index}": {} for index in range(10)},
    }
    paper_target_workplan = {
        "execute_supported": False,
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "counts": {
            "work_package_count": 4,
            "future_artifact_count": 10,
            "allowed_while_paused_package_count": 2,
            "doc_prefill_artifact_count": 3,
            "evaluator_schema_artifact_count": 5,
            "trained_candidate_artifact_count": 2,
            "current_existing_future_artifact_count": 3,
        },
    }
    paper_target_evaluator_schema = {
        "execute_supported": False,
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "evaluator_outputs": [
            {
                "id": "sensor_space_eval_rows",
                "path": "docs/resources/future/paper_target_bridge/sensor_space_eval_rows.jsonl",
            },
            {
                "id": "coordinate_transform_audit",
                "path": "docs/resources/future/paper_target_bridge/coordinate_transform_audit.json",
            },
            {
                "id": "hybrid_eval_rows",
                "path": "docs/resources/future/paper_target_bridge/hybrid_scheduler_eval_rows.jsonl",
            },
            {
                "id": "hybrid_metric_report",
                "path": "docs/resources/future/paper_target_bridge/hybrid_scheduler_metric_report.json",
            },
            {
                "id": "paper_frame_full_test_p1",
                "path": "docs/resources/future/paper_target_bridge/paper_frame_full_test_p1_report.json",
            },
        ],
        "counts": {
            "evaluator_output_count": 5,
            "jsonl_output_schema_count": 2,
            "json_output_schema_count": 3,
            "blocked_output_count": 5,
            "current_existing_output_count": 0,
        },
    }
    for group in ("authority", "analysis_artifacts", "xr63_xr64_artifacts"):
        for key, value in index[group].items():
            if isinstance(value, str):
                if key.endswith("_root"):
                    (root / value).mkdir(parents=True, exist_ok=True)
                else:
                    write_file(root / value)
    for value in index["experiment_queue"].values():
        write_file(root / value)
    queue_path = root / index["experiment_queue"]["queue_json"]
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    for value in index["xr64_experiment_design_contract"].values():
        write_file(root / value)
    xr64_design_contract = json.loads(
        Path("docs/resources/xr64_experiment_design_contract_2026_06_21.json").read_text(encoding="utf-8")
    )
    xr64_design_path = root / index["xr64_experiment_design_contract"]["contract_json"]
    xr64_design_path.write_text(json.dumps(xr64_design_contract), encoding="utf-8")
    for value in index["xr64_resume_execution_dag"].values():
        write_file(root / value)
    xr64_resume_dag = json.loads(
        Path("docs/resources/xr64_resume_execution_dag_2026_06_21.json").read_text(encoding="utf-8")
    )
    xr64_dag_path = root / index["xr64_resume_execution_dag"]["dag_json"]
    xr64_dag_path.write_text(json.dumps(xr64_resume_dag), encoding="utf-8")
    for value in index["ablation_evidence_checklist"].values():
        write_file(root / value)
    ablation_path = root / index["ablation_evidence_checklist"]["checklist_json"]
    ablation_path.write_text(json.dumps(ablation_checklist), encoding="utf-8")
    for value in index["paper_ref_experiment_trace"].values():
        write_file(root / value)
    paper_ref_trace = json.loads(
        Path("docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json").read_text(encoding="utf-8")
    )
    paper_ref_trace_path = root / index["paper_ref_experiment_trace"]["trace_json"]
    paper_ref_trace_path.write_text(json.dumps(paper_ref_trace), encoding="utf-8")
    for value in index["paper_ref_current_ablation_plan"].values():
        write_file(root / value)
    paper_ref_plan = Path("docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md").read_text(
        encoding="utf-8"
    )
    paper_ref_plan_path = root / index["paper_ref_current_ablation_plan"]["plan_md"]
    paper_ref_plan_path.write_text(paper_ref_plan, encoding="utf-8")
    for value in index["post_xr64_decision_tree"].values():
        write_file(root / value)
    post_xr64_tree_path = root / index["post_xr64_decision_tree"]["decision_tree_json"]
    post_xr64_tree_path.write_text(json.dumps(post_xr64_tree), encoding="utf-8")
    for value in index["post_xr64_followup_experiment_contract"].values():
        write_file(root / value)
    post_xr64_followup_path = root / index["post_xr64_followup_experiment_contract"]["contract_json"]
    post_xr64_followup_path.write_text(json.dumps(post_xr64_followup), encoding="utf-8")
    for value in index["completion_readiness_contract"].values():
        write_file(root / value)
    readiness_path = root / index["completion_readiness_contract"]["contract_json"]
    readiness_path.write_text(json.dumps(completion_readiness), encoding="utf-8")
    for value in index["submission_target_gap"].values():
        if isinstance(value, str):
            write_file(root / value)
    target_gap_path = root / index["submission_target_gap"]["gap_audit_json"]
    target_gap_path.write_text(json.dumps(target_gap), encoding="utf-8")
    for value in index["submission_target_source_trace"].values():
        write_file(root / value)
    write_file(root / "docs/resources/fake_submission_main.tex")
    source_trace_path = root / index["submission_target_source_trace"]["trace_json"]
    source_trace_path.write_text(json.dumps(source_trace), encoding="utf-8")
    for value in index["metric_protocol_bridge"].values():
        if isinstance(value, str):
            write_file(root / value)
    bridge_path = root / index["metric_protocol_bridge"]["bridge_json"]
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    for value in index["metric_protocol_unblock_contract"].values():
        if isinstance(value, str):
            write_file(root / value)
    unblock_contract_path = root / index["metric_protocol_unblock_contract"]["contract_json"]
    unblock_contract_path.write_text(json.dumps(unblock_contract), encoding="utf-8")
    for value in index["paper_target_comparison_evidence_manifest"].values():
        if isinstance(value, str):
            write_file(root / value)
    paper_target_manifest_path = root / index["paper_target_comparison_evidence_manifest"]["manifest_json"]
    paper_target_manifest_path.write_text(json.dumps(paper_target_manifest), encoding="utf-8")
    write_file(root / "docs/resources/future/paper_target_bridge/split_protocol_audit.json")
    write_file(root / "docs/resources/future/paper_target_bridge/cur_state_target_mapping_audit.json")
    write_file(root / "docs/resources/future/paper_target_bridge/accuracy_latency_separation_note.json")
    for value in index["paper_target_bridge_artifact_workplan"].values():
        if isinstance(value, str):
            write_file(root / value)
    paper_target_workplan_path = root / index["paper_target_bridge_artifact_workplan"]["workplan_json"]
    paper_target_workplan_path.write_text(json.dumps(paper_target_workplan), encoding="utf-8")
    for value in index["paper_target_bridge_evaluator_schema_contract"].values():
        if isinstance(value, str):
            write_file(root / value)
    paper_target_evaluator_schema_path = root / index["paper_target_bridge_evaluator_schema_contract"]["contract_json"]
    paper_target_evaluator_schema_path.write_text(json.dumps(paper_target_evaluator_schema), encoding="utf-8")
    for value in index["paper_target_bridge_ptb1_resume_runbook"].values():
        if isinstance(value, str):
            write_file(root / value)
    ptb1_runbook = json.loads(
        Path("docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json").read_text(encoding="utf-8")
    )
    ptb1_runbook_path = root / index["paper_target_bridge_ptb1_resume_runbook"]["runbook_json"]
    ptb1_runbook_path.write_text(json.dumps(ptb1_runbook), encoding="utf-8")
    paper_target_trained_candidate = {
        "execute_supported": False,
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "schema_contract_allowed_while_paused": True,
        "trained_candidate_payload_allowed_while_paused": False,
        "trained_candidate_outputs": [
            {
                "id": "trained_candidate_eval_summary",
                "path": "docs/resources/future/paper_target_bridge/trained_candidate_test_eval_summary.json",
            },
            {
                "id": "trained_candidate_leakage_audit",
                "path": "docs/resources/future/paper_target_bridge/trained_candidate_leakage_audit.json",
            },
        ],
        "counts": {
            "trained_candidate_output_count": 2,
            "json_output_schema_count": 2,
            "blocked_output_count": 2,
            "current_existing_output_count": 0,
        },
    }
    for value in index["paper_target_bridge_trained_candidate_contract"].values():
        if isinstance(value, str):
            write_file(root / value)
    paper_target_trained_candidate_path = root / index["paper_target_bridge_trained_candidate_contract"]["contract_json"]
    paper_target_trained_candidate_path.write_text(json.dumps(paper_target_trained_candidate), encoding="utf-8")
    paper_target_bridge_decision = {
        "execute_supported": False,
        "direct_submission_comparison_allowed": False,
        "paper_level_completion_allowed": False,
        "schema_contract_allowed_while_paused": True,
        "bridge_decision_payload_allowed_while_paused": False,
        "bridge_decision_output": {
            "id": "bridge_decision",
            "path": "docs/resources/future/paper_target_bridge/bridge_decision.json",
        },
        "counts": {
            "decision_output_count": 1,
            "json_output_schema_count": 1,
            "blocked_output_count": 1,
            "current_existing_output_count": 0,
            "signoff_rule_count": 6,
        },
    }
    for value in index["paper_target_bridge_decision_contract"].values():
        if isinstance(value, str):
            write_file(root / value)
    paper_target_bridge_decision_path = root / index["paper_target_bridge_decision_contract"]["contract_json"]
    paper_target_bridge_decision_path.write_text(json.dumps(paper_target_bridge_decision), encoding="utf-8")
    for value in index["paper_target_bridge_resume_dependency_matrix"].values():
        if isinstance(value, str):
            write_file(root / value)
    resume_dependency_matrix = json.loads(
        Path("docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json").read_text(
            encoding="utf-8"
        )
    )
    resume_dependency_matrix_path = root / index["paper_target_bridge_resume_dependency_matrix"]["matrix_json"]
    resume_dependency_matrix_path.write_text(json.dumps(resume_dependency_matrix), encoding="utf-8")
    for value in index["paper_target_metric_bridge_preflight"].values():
        if isinstance(value, str):
            write_file(root / value)
    preflight = json.loads(Path("docs/resources/paper_target_metric_bridge_preflight_2026_06_21.json").read_text(encoding="utf-8"))
    preflight_path = root / index["paper_target_metric_bridge_preflight"]["preflight_json"]
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    for value in index["second_goal_blocker_artifact_map"].values():
        if isinstance(value, str):
            write_file(root / value)
    blocker_map = json.loads(Path("docs/resources/second_goal_blocker_artifact_map_2026_06_21.json").read_text(encoding="utf-8"))
    blocker_map_path = root / index["second_goal_blocker_artifact_map"]["map_json"]
    blocker_map_path.write_text(json.dumps(blocker_map), encoding="utf-8")
    for value in index["resume_readiness_matrix"].values():
        if isinstance(value, str):
            write_file(root / value)
    resume_matrix = json.loads(Path("docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json").read_text(encoding="utf-8"))
    resume_matrix_path = root / index["resume_readiness_matrix"]["matrix_json"]
    resume_matrix_path.write_text(json.dumps(resume_matrix), encoding="utf-8")
    for value in index["current_result_synthesis"].values():
        if isinstance(value, str):
            write_file(root / value)
    synthesis_path = root / index["current_result_synthesis"]["synthesis_json"]
    synthesis_path.write_text(json.dumps(result_synthesis), encoding="utf-8")
    for value in index["accuracy_lift_decision_ladder"].values():
        if isinstance(value, str):
            write_file(root / value)
    accuracy_ladder = json.loads(
        Path("docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.json").read_text(
            encoding="utf-8"
        )
    )
    accuracy_ladder_path = root / index["accuracy_lift_decision_ladder"]["ladder_json"]
    accuracy_ladder_path.write_text(json.dumps(accuracy_ladder), encoding="utf-8")
    for value in index["objective_trace"].values():
        if isinstance(value, str):
            write_file(root / value)
    trace_path = root / index["objective_trace"]["trace_json"]
    trace_path.write_text(json.dumps(objective_trace), encoding="utf-8")
    command_manifest = json.loads(Path("docs/resources/xr64_resume_command_manifest_2026_06_20.json").read_text(encoding="utf-8"))
    for value in index["xr64_resume_command_manifest"].values():
        if isinstance(value, str):
            write_file(root / value)
    for raw_path in (command_manifest.get("source_artifacts") or {}).values():
        path = root / raw_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path == "scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh":
            path.write_text(
                Path("scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        elif path.exists():
            continue
        else:
            path.write_text("ok\n", encoding="utf-8")
    for item in command_manifest["prep_commands"]:
        if item.get("action") == "eval":
            for raw_path in item.get("required_inputs") or []:
                write_file(root / raw_path)
    for split, count in (command_manifest.get("expected_generated_counts") or {}).get(
        "split_manifest_rows", {}
    ).items():
        split_manifest = root / f"data/_internal/manifests/manifest1/{split}_manifest.jsonl"
        split_manifest.parent.mkdir(parents=True, exist_ok=True)
        split_manifest.write_text("".join("{}\n" for _ in range(int(count))), encoding="utf-8")
    command_manifest_path = root / index["xr64_resume_command_manifest"]["manifest_json"]
    command_manifest_path.write_text(json.dumps(command_manifest), encoding="utf-8")
    for value in index["xr64_resume_command_emitter"].values():
        if isinstance(value, str):
            write_file(root / value)
    for key, value in index["xr64_next_prep_command_selector"].items():
        if key == "manifest_json":
            continue
        if isinstance(value, str):
            write_file(root / value)
    for value in index["xr64_prep_progress_ledger"].values():
        if isinstance(value, str):
            write_file(root / value)
    ledger = json.loads(Path("docs/resources/xr64_prep_progress_ledger_2026_06_21.json").read_text(encoding="utf-8"))
    ledger_path = root / index["xr64_prep_progress_ledger"]["ledger_json"]
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    for value in index["xr64_prelaunch_packet"].values():
        if isinstance(value, str):
            write_file(root / value)
    prelaunch_packet_path = root / index["xr64_prelaunch_packet"]["packet_json"]
    prelaunch_packet_path.write_text(json.dumps(prelaunch_packet), encoding="utf-8")
    for value in index["xr64_postrun_evidence_contract"].values():
        if isinstance(value, str):
            write_file(root / value)
    postrun_contract_path = root / index["xr64_postrun_evidence_contract"]["contract_json"]
    postrun_contract_path.write_text(json.dumps(postrun_contract), encoding="utf-8")
    for value in index["xr64_postrun_promotion_decision"].values():
        if isinstance(value, str):
            write_file(root / value)
    helper_path = root / index["xr64_postrun_promotion_decision"]["decision_helper"]
    helper_path.write_text(
        Path("scripts/external/decide_xr64_postrun_promotion.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    promotion_decision_path = root / index["xr64_postrun_promotion_decision"]["decision_json"]
    promotion_decision_path.write_text(json.dumps(promotion_decision), encoding="utf-8")
    for i in range(31):
        write_file(root / f"anlaysis/paper-ref/papers/paper-{i:02d}/analysis.md")
    index_path = root / "docs/resources/second_goal_artifact_index_2026_06_18.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def patch_checker(module, monkeypatch, *, checker_code: int = 0, generated_complete: bool = False) -> None:
    report = {
        "ready": True,
        "generated_complete": generated_complete,
        "warnings": ["missing generated artifacts"],
        "errors": [],
        "config_errors": [],
        "leakage": {"test_named_files": [], "test_split_payloads": []},
    }
    monkeypatch.setattr(module, "run_xr64_check", lambda _args: (checker_code, report))
    monkeypatch.setattr(module, "count_missing_generated", lambda _report: (8, 6))
    monkeypatch.setattr(module, "format_xr64_summary", lambda _report, _code: "resume_status: generated_incomplete\n")


def test_second_goal_status_summary_reports_paused_generated_incomplete(tmp_path: Path, monkeypatch) -> None:
    module = load_status_module()
    index_path = write_index(tmp_path)
    patch_checker(module, monkeypatch)

    code, status = module.build_status(
        Namespace(project_root=str(tmp_path), index=str(index_path), strict_xr64=False, format="summary")
    )
    summary = module.format_summary(status, code)

    assert code == 0
    assert status["execution_state"] == "paused_by_user_directive"
    assert status["active_goal_complete"] is False
    assert status["xr64"]["resume_status"] == "generated_incomplete"
    assert status["xr64"]["can_run_lane"] is False
    assert status["analysis_inventory"]["paper_ref_per_paper_actual_count"] == 31
    assert status["experiment_queue"]["experiment_count"] == 4
    assert status["experiment_queue"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["xr64_experiment_design_contract"]["exists"] is True
    assert status["xr64_experiment_design_contract"]["ok"] is True
    assert status["xr64_experiment_design_contract"]["execute_supported"] is False
    assert status["xr64_experiment_design_contract"]["allowed_while_paused"] is True
    assert status["xr64_experiment_design_contract"]["lane_count"] == 4
    assert status["xr64_experiment_design_contract"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["xr64_experiment_design_contract"]["axis_count"] == 6
    assert status["xr64_experiment_design_contract"]["current_ready_to_train"] is False
    assert status["xr64_experiment_design_contract"]["current_missing_eval_rows"] == 8
    assert status["xr64_experiment_design_contract"]["current_missing_overrides"] == 6
    assert status["xr64_experiment_design_contract"]["completion_allowed"] is False
    assert status["xr64_experiment_design_contract"]["paper_level_completion_allowed"] is False
    assert status["xr64_experiment_design_contract"]["error_count"] == 0
    assert status["xr64_resume_execution_dag"]["exists"] is True
    assert status["xr64_resume_execution_dag"]["ok"] is True
    assert status["xr64_resume_execution_dag"]["execute_supported"] is False
    assert status["xr64_resume_execution_dag"]["allowed_while_paused"] is True
    assert status["xr64_resume_execution_dag"]["node_count"] == 17
    assert status["xr64_resume_execution_dag"]["phase_count"] == 6
    assert status["xr64_resume_execution_dag"]["prep_nodes"] == 10
    assert status["xr64_resume_execution_dag"]["eval_nodes"] == 8
    assert status["xr64_resume_execution_dag"]["build_nodes"] == 2
    assert status["xr64_resume_execution_dag"]["launch_nodes"] == 2
    assert status["xr64_resume_execution_dag"]["postrun_nodes"] == 2
    assert status["xr64_resume_execution_dag"]["ready_to_train"] is False
    assert status["xr64_resume_execution_dag"]["missing_eval_rows"] == 8
    assert status["xr64_resume_execution_dag"]["missing_overrides"] == 6
    assert status["xr64_resume_execution_dag"]["current_next_node"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["xr64_resume_execution_dag"]["completion_allowed"] is False
    assert status["xr64_resume_execution_dag"]["manifest_ok"] is True
    assert status["xr64_resume_execution_dag"]["design_contract_ok"] is True
    assert status["xr64_resume_execution_dag"]["error_count"] == 0
    assert status["ablation_evidence_checklist"]["exists"] is True
    assert status["ablation_evidence_checklist"]["experiment_count"] == 8
    assert status["ablation_evidence_checklist"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["ablation_evidence_checklist"]["axis_count"] == 8
    assert status["ablation_evidence_checklist"]["software_promotion_allowed"] is False
    assert status["ablation_evidence_checklist"]["paper_level_completion_allowed"] is False
    assert status["ablation_evidence_checklist"]["missing_eval_rows"] == 8
    assert status["ablation_evidence_checklist"]["missing_overrides"] == 6
    assert status["paper_ref_experiment_trace"]["exists"] is True
    assert status["paper_ref_experiment_trace"]["execute_supported"] is False
    assert status["paper_ref_experiment_trace"]["allowed_while_paused"] is True
    assert status["paper_ref_experiment_trace"]["experiment_count"] == 8
    assert status["paper_ref_experiment_trace"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["paper_ref_experiment_trace"]["paper_group_count"] == 6
    assert status["paper_ref_experiment_trace"]["axis_count"] == 8
    assert status["paper_ref_experiment_trace"]["xr64_ready_to_train"] is False
    assert status["paper_ref_experiment_trace"]["missing_eval_rows"] == 8
    assert status["paper_ref_experiment_trace"]["missing_overrides"] == 6
    assert status["paper_ref_experiment_trace"]["completion_allowed"] is False
    assert status["paper_ref_experiment_trace"]["direct_submission_comparison_allowed"] is False
    assert status["paper_ref_current_ablation_plan"]["exists"] is True
    assert status["paper_ref_current_ablation_plan"]["ok"] is True
    assert status["paper_ref_current_ablation_plan"]["execute_supported"] is False
    assert status["paper_ref_current_ablation_plan"]["allowed_while_paused"] is True
    assert status["paper_ref_current_ablation_plan"]["experiment_count"] == 8
    assert status["paper_ref_current_ablation_plan"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["paper_ref_current_ablation_plan"]["paper_signal_count"] == 18
    assert status["paper_ref_current_ablation_plan"]["mapping_row_count"] == 10
    assert status["paper_ref_current_ablation_plan"]["lane_trace_count"] == 8
    assert status["paper_ref_current_ablation_plan"]["axis_count"] == 8
    assert status["paper_ref_current_ablation_plan"]["task_card_count"] >= 2
    assert status["paper_ref_current_ablation_plan"]["completion_allowed"] is False
    assert status["paper_ref_current_ablation_plan"]["direct_submission_comparison_allowed"] is False
    assert status["paper_ref_current_ablation_plan"]["error_count"] == 0
    assert status["post_xr64_decision_tree"]["exists"] is True
    assert status["post_xr64_decision_tree"]["execute_supported"] is False
    assert status["post_xr64_decision_tree"]["claim_level_count"] == 4
    assert status["post_xr64_decision_tree"]["decision_input_count"] == 3
    assert status["post_xr64_decision_tree"]["branch_count"] == 7
    assert status["post_xr64_decision_tree"]["source_artifact_count"] == 4
    assert status["post_xr64_decision_tree"]["signoff_rule_count"] == 4
    assert status["post_xr64_decision_tree"]["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert status["post_xr64_decision_tree"]["current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["post_xr64_decision_tree"]["xr64_ready_to_train"] is False
    assert status["post_xr64_decision_tree"]["missing_eval_rows"] == 8
    assert status["post_xr64_decision_tree"]["missing_overrides"] == 6
    assert status["post_xr64_decision_tree"]["postrun_candidate_count"] == 0
    assert status["post_xr64_decision_tree"]["direct_submission_comparison_allowed"] is False
    assert status["post_xr64_decision_tree"]["next_experiments"] == ["XR-64C", "XR-65", "XR-66", "XR-67", "XR-68"]
    assert status["post_xr64_followup_experiment_contract"]["exists"] is True
    assert status["post_xr64_followup_experiment_contract"]["execute_supported"] is False
    assert status["post_xr64_followup_experiment_contract"]["direct_submission_comparison_allowed"] is False
    assert status["post_xr64_followup_experiment_contract"]["paper_level_completion_allowed"] is False
    assert status["post_xr64_followup_experiment_contract"]["experiment_count"] == 5
    assert status["post_xr64_followup_experiment_contract"]["experiment_ids"] == [
        "XR-64C",
        "XR-65",
        "XR-66",
        "XR-67",
        "XR-68",
    ]
    assert status["post_xr64_followup_experiment_contract"]["blocked_launch_count"] == 5
    assert status["post_xr64_followup_experiment_contract"]["p1_count"] == 3
    assert status["post_xr64_followup_experiment_contract"]["p2_count"] == 2
    assert status["post_xr64_followup_experiment_contract"]["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert status["post_xr64_followup_experiment_contract"]["xr64_postrun_candidate_count"] == 0
    assert status["post_xr64_followup_experiment_contract"]["followup_launch_allowed_now"] is False
    assert status["completion_readiness_contract"]["exists"] is True
    assert status["completion_readiness_contract"]["completion_allowed"] is False
    assert status["completion_readiness_contract"]["blocker_count"] == 28
    assert status["completion_readiness_contract"]["xr64_ready_to_train"] is False
    assert status["completion_readiness_contract"]["xr64_can_run_lane"] is False
    assert status["completion_readiness_contract"]["missing_eval_rows"] == 8
    assert status["completion_readiness_contract"]["missing_overrides"] == 6
    assert status["completion_readiness_contract"]["leakage_risk"] == "none"
    assert status["completion_readiness_contract"]["paper_target_bridge_evaluator_schema_execute_supported"] is False
    assert status["completion_readiness_contract"]["paper_target_bridge_evaluator_schema_output_count"] == 5
    assert status["completion_readiness_contract"][
        "paper_target_bridge_evaluator_schema_current_existing_output_count"
    ] == 0
    assert status["completion_readiness_contract"]["paper_target_bridge_trained_candidate_execute_supported"] is False
    assert status["completion_readiness_contract"]["paper_target_bridge_trained_candidate_schema_allowed_while_paused"] is True
    assert status["completion_readiness_contract"]["paper_target_bridge_trained_candidate_payload_allowed_while_paused"] is False
    assert status["completion_readiness_contract"]["paper_target_bridge_trained_candidate_output_count"] == 2
    assert status["completion_readiness_contract"]["paper_target_bridge_trained_candidate_current_existing_output_count"] == 0
    assert status["completion_readiness_contract"]["paper_target_bridge_decision_execute_supported"] is False
    assert status["completion_readiness_contract"]["paper_target_bridge_decision_schema_allowed_while_paused"] is True
    assert status["completion_readiness_contract"]["paper_target_bridge_decision_payload_allowed_while_paused"] is False
    assert status["completion_readiness_contract"]["paper_target_bridge_decision_output_count"] == 1
    assert status["completion_readiness_contract"]["paper_target_bridge_decision_current_existing_output_count"] == 0
    assert status["completion_readiness_contract"]["post_xr64_followup_contract_execute_supported"] is False
    assert status["completion_readiness_contract"]["post_xr64_followup_contract_experiment_count"] == 5
    assert status["completion_readiness_contract"]["post_xr64_followup_contract_blocked_launch_count"] == 5
    assert status["completion_readiness_contract"]["post_xr64_followup_contract_followup_launch_allowed_now"] is False
    assert status["completion_readiness_contract"]["xr64_command_manifest_postrun_command_count"] == 2
    assert status["completion_readiness_contract"]["xr64_command_manifest_full_eval_coverage_required"] is True
    assert status["completion_readiness_contract"]["xr64_command_manifest_full_override_coverage_required"] is True
    assert status["completion_readiness_contract"]["xr64_command_manifest_duplicate_sample_ids_allowed"] is False
    assert status["completion_readiness_contract"]["xr64_command_manifest_opposite_split_sample_ids_allowed"] is False
    assert status["completion_readiness_contract"]["xr64_command_manifest_subset_artifacts_satisfy_ready_to_train"] is False
    assert (
        status["completion_readiness_contract"]["xr64_command_manifest_strict_generated_artifact_checker"]
        == "scripts/external/check_xr64_resume_artifacts.py"
    )
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_ok"] is True
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_ready_after_resume_count"] == 8
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_completed_count"] == 0
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_blocked_count"] == 2
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_invalid_count"] == 0
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_next_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_next_status"] == "ready_after_resume"
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_next_allowed_to_run_now"] is False
    assert status["completion_readiness_contract"]["xr64_next_prep_selector_execute_supported"] is False
    assert status["completion_readiness_contract"]["xr64_prelaunch_phase_count"] == 6
    assert status["completion_readiness_contract"]["xr64_prelaunch_postrun_command_count"] == 2
    assert status["completion_readiness_contract"]["xr64_postrun_candidate_count"] == 0
    assert status["completion_readiness_contract"]["next_allowed_action_count"] == 5
    assert status["submission_target_gap"]["exists"] is True
    assert status["submission_target_gap"]["direct_comparison_valid"] is False
    assert status["submission_target_gap"]["submission_center_target_px"] == 0.1812
    assert status["submission_target_source_trace"]["exists"] is True
    assert status["submission_target_source_trace"]["source_file_exists"] is True
    assert status["submission_target_source_trace"]["direct_comparison_valid"] is False
    assert status["submission_target_source_trace"]["mode_hybrid_pixel_error_px"] == 0.1812
    assert status["submission_target_source_trace"]["mode_hybrid_p10_pct"] == 99.97
    assert status["submission_target_source_trace"]["mode_hybrid_p5_pct"] == 99.72
    assert status["submission_target_source_trace"]["mode_hybrid_p1_pct"] == 99.61
    assert status["submission_target_source_trace"]["mode_hybrid_latency_ms"] == 0.43
    assert status["submission_target_source_trace"]["accelerator_range_is_single_target"] is False
    assert status["submission_target_source_trace"]["alignment_requirement_count"] == 3
    assert status["metric_protocol_bridge"]["exists"] is True
    assert status["metric_protocol_bridge"]["bridge_status"] == "blocked_until_metric_frame_and_protocol_match"
    assert status["metric_protocol_bridge"]["direct_submission_comparison_allowed"] is False
    assert status["metric_protocol_bridge"]["metric_frame"] == "post-transform input coordinate frame"
    assert status["metric_protocol_bridge"]["current_p1_metric_available"] is True
    assert status["metric_protocol_bridge"]["current_p1_full_test_evidence_available"] is False
    assert status["metric_protocol_unblock_contract"]["exists"] is True
    assert status["metric_protocol_unblock_contract"]["contract_status"] == "blocked"
    assert status["metric_protocol_unblock_contract"]["direct_submission_comparison_allowed"] is False
    assert status["metric_protocol_unblock_contract"]["paper_level_completion_allowed"] is False
    assert status["metric_protocol_unblock_contract"]["required_evidence_count"] == 7
    assert status["metric_protocol_unblock_contract"]["current_blocker_count"] == 6
    assert status["paper_target_comparison_evidence_manifest"]["exists"] is True
    assert status["paper_target_comparison_evidence_manifest"]["evidence_state"] == "missing_required_evidence"
    assert status["paper_target_comparison_evidence_manifest"]["required_evidence_complete"] is False
    assert status["paper_target_comparison_evidence_manifest"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_comparison_evidence_manifest"]["paper_level_completion_allowed"] is False
    assert status["paper_target_comparison_evidence_manifest"]["required_evidence_count"] == 7
    assert status["paper_target_comparison_evidence_manifest"]["missing_evidence_count"] == 3
    assert status["paper_target_comparison_evidence_manifest"]["partial_evidence_count"] == 4
    assert status["paper_target_comparison_evidence_manifest"]["required_future_artifact_count"] == 10
    assert status["paper_target_comparison_evidence_manifest"]["current_artifact_count"] == 4
    assert status["paper_target_comparison_evidence_manifest"]["existing_required_future_artifact_count"] == 3
    assert status["paper_target_comparison_evidence_manifest"]["artifact_schema_count"] == 10
    assert status["paper_target_bridge_artifact_workplan"]["exists"] is True
    assert status["paper_target_bridge_artifact_workplan"]["execute_supported"] is False
    assert status["paper_target_bridge_artifact_workplan"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_artifact_workplan"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_artifact_workplan"]["work_package_count"] == 4
    assert status["paper_target_bridge_artifact_workplan"]["future_artifact_count"] == 10
    assert status["paper_target_bridge_artifact_workplan"]["allowed_while_paused_package_count"] == 2
    assert status["paper_target_bridge_artifact_workplan"]["doc_prefill_artifact_count"] == 3
    assert status["paper_target_bridge_artifact_workplan"]["evaluator_schema_artifact_count"] == 5
    assert status["paper_target_bridge_artifact_workplan"]["trained_candidate_artifact_count"] == 2
    assert status["paper_target_bridge_artifact_workplan"]["current_existing_future_artifact_count"] == 3
    assert status["paper_target_bridge_evaluator_schema_contract"]["exists"] is True
    assert status["paper_target_bridge_evaluator_schema_contract"]["execute_supported"] is False
    assert status["paper_target_bridge_evaluator_schema_contract"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_evaluator_schema_contract"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_evaluator_schema_contract"]["evaluator_output_count"] == 5
    assert status["paper_target_bridge_evaluator_schema_contract"]["jsonl_output_schema_count"] == 2
    assert status["paper_target_bridge_evaluator_schema_contract"]["json_output_schema_count"] == 3
    assert status["paper_target_bridge_evaluator_schema_contract"]["blocked_output_count"] == 5
    assert status["paper_target_bridge_evaluator_schema_contract"]["current_existing_output_count"] == 0
    assert status["paper_target_bridge_ptb1_resume_runbook"]["exists"] is True
    assert status["paper_target_bridge_ptb1_resume_runbook"]["execute_supported"] is False
    assert status["paper_target_bridge_ptb1_resume_runbook"]["allowed_while_paused"] is True
    assert status["paper_target_bridge_ptb1_resume_runbook"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_ptb1_resume_runbook"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_ptb1_resume_runbook"]["sequence_count"] == 5
    assert status["paper_target_bridge_ptb1_resume_runbook"]["ptb1_output_count"] == 5
    assert status["paper_target_bridge_ptb1_resume_runbook"]["ptb1_current_existing_output_count"] == 0
    assert status["paper_target_bridge_ptb1_resume_runbook"]["ptb1_payload_missing_count"] == 5
    assert status["paper_target_bridge_ptb1_resume_runbook"]["future_payload_validation_requires_allow_present"] is True
    assert status["paper_target_bridge_trained_candidate_contract"]["exists"] is True
    assert status["paper_target_bridge_trained_candidate_contract"]["execute_supported"] is False
    assert status["paper_target_bridge_trained_candidate_contract"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_trained_candidate_contract"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_trained_candidate_contract"]["schema_contract_allowed_while_paused"] is True
    assert status["paper_target_bridge_trained_candidate_contract"]["trained_candidate_payload_allowed_while_paused"] is False
    assert status["paper_target_bridge_trained_candidate_contract"]["trained_candidate_output_count"] == 2
    assert status["paper_target_bridge_trained_candidate_contract"]["json_output_schema_count"] == 2
    assert status["paper_target_bridge_trained_candidate_contract"]["blocked_output_count"] == 2
    assert status["paper_target_bridge_trained_candidate_contract"]["current_existing_output_count"] == 0
    assert status["paper_target_bridge_decision_contract"]["exists"] is True
    assert status["paper_target_bridge_decision_contract"]["execute_supported"] is False
    assert status["paper_target_bridge_decision_contract"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_decision_contract"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_decision_contract"]["schema_contract_allowed_while_paused"] is True
    assert status["paper_target_bridge_decision_contract"]["bridge_decision_payload_allowed_while_paused"] is False
    assert status["paper_target_bridge_decision_contract"]["decision_output_count"] == 1
    assert status["paper_target_bridge_decision_contract"]["json_output_schema_count"] == 1
    assert status["paper_target_bridge_decision_contract"]["blocked_output_count"] == 1
    assert status["paper_target_bridge_decision_contract"]["current_existing_output_count"] == 0
    assert status["paper_target_bridge_decision_contract"]["signoff_rule_count"] == 6
    assert status["paper_target_bridge_resume_dependency_matrix"]["exists"] is True
    assert status["paper_target_bridge_resume_dependency_matrix"]["execute_supported"] is False
    assert status["paper_target_bridge_resume_dependency_matrix"]["allowed_while_paused"] is True
    assert status["paper_target_bridge_resume_dependency_matrix"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_bridge_resume_dependency_matrix"]["paper_level_completion_allowed"] is False
    assert status["paper_target_bridge_resume_dependency_matrix"]["dependency_gate_count"] == 6
    assert status["paper_target_bridge_resume_dependency_matrix"]["allowed_while_paused_gate_count"] == 2
    assert status["paper_target_bridge_resume_dependency_matrix"]["complete_now_gate_count"] == 2
    assert status["paper_target_bridge_resume_dependency_matrix"]["blocked_gate_count"] == 4
    assert status["paper_target_bridge_resume_dependency_matrix"]["execution_dependent_payload_missing_count"] == 8
    assert status["paper_target_bridge_resume_dependency_matrix"]["completion_blocker_count"] == 28
    assert status["paper_target_bridge_resume_dependency_matrix"]["xr64_ready_to_train"] is False
    assert status["paper_target_bridge_resume_dependency_matrix"]["completion_allowed"] is False
    assert status["paper_target_metric_bridge_preflight"]["exists"] is True
    assert status["paper_target_metric_bridge_preflight"]["execute_supported"] is False
    assert status["paper_target_metric_bridge_preflight"]["allowed_while_paused"] is True
    assert status["paper_target_metric_bridge_preflight"]["direct_submission_comparison_allowed"] is False
    assert status["paper_target_metric_bridge_preflight"]["paper_level_completion_allowed"] is False
    assert status["paper_target_metric_bridge_preflight"]["required_evidence_count"] == 7
    assert status["paper_target_metric_bridge_preflight"]["required_output_schema_count"] == 10
    assert status["paper_target_metric_bridge_preflight"]["ptb1_evaluator_output_schema_count"] == 5
    assert status["paper_target_metric_bridge_preflight"]["ptb1_current_existing_output_count"] == 0
    assert status["paper_target_metric_bridge_preflight"]["current_p1_metric_available"] is True
    assert status["paper_target_metric_bridge_preflight"]["current_p1_full_test_evidence_available"] is False
    assert status["paper_target_metric_bridge_preflight"]["xr64_ready_to_train"] is False
    assert status["paper_target_metric_bridge_preflight"]["xr64_missing_eval_rows"] == 8
    assert status["paper_target_metric_bridge_preflight"]["xr64_missing_overrides"] == 6
    assert status["second_goal_blocker_artifact_map"]["exists"] is True
    assert status["second_goal_blocker_artifact_map"]["execute_supported"] is False
    assert status["second_goal_blocker_artifact_map"]["allowed_to_run_now"] is False
    assert status["second_goal_blocker_artifact_map"]["blocker_count"] == 28
    assert status["second_goal_blocker_artifact_map"]["direct_submission_comparison_allowed"] is False
    assert status["second_goal_blocker_artifact_map"]["paper_level_completion_allowed"] is False
    assert status["second_goal_blocker_artifact_map"]["xr64_missing_eval_rows"] == 8
    assert status["second_goal_blocker_artifact_map"]["xr64_missing_overrides"] == 6
    assert status["second_goal_blocker_artifact_map"]["xr64_blocked_build_command_count"] == 2
    assert status["second_goal_blocker_artifact_map"]["ptb1_missing_payload_count"] == 5
    assert status["second_goal_blocker_artifact_map"]["ptb2_missing_payload_count"] == 2
    assert status["second_goal_blocker_artifact_map"]["ptb3_missing_payload_count"] == 1
    assert status["second_goal_blocker_artifact_map"]["blocked_execution_unit_count"] == 5
    assert status["second_goal_blocker_artifact_map"]["ablation_axis_count"] == 6
    assert status["second_goal_resume_readiness_matrix"]["exists"] is True
    assert status["second_goal_resume_readiness_matrix"]["execute_supported"] is False
    assert status["second_goal_resume_readiness_matrix"]["allowed_while_paused"] is True
    assert status["second_goal_resume_readiness_matrix"]["direct_submission_comparison_allowed"] is False
    assert status["second_goal_resume_readiness_matrix"]["software_promotion_allowed"] is False
    assert status["second_goal_resume_readiness_matrix"]["paper_level_completion_allowed"] is False
    assert status["second_goal_resume_readiness_matrix"]["axis_count"] == 6
    assert status["second_goal_resume_readiness_matrix"]["resume_gate_count"] == 5
    assert status["second_goal_resume_readiness_matrix"]["current_p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["second_goal_resume_readiness_matrix"]["goal_complete"] is False
    assert status["second_goal_resume_readiness_matrix"]["req1_status"] == "complete_as_planning_input"
    assert status["second_goal_resume_readiness_matrix"]["req2_status"] == "planned_not_fully_executed"
    assert status["second_goal_resume_readiness_matrix"]["req3_status"] == "incomplete"
    assert status["second_goal_resume_readiness_matrix"]["completion_allowed"] is False
    assert status["second_goal_resume_readiness_matrix"]["blocker_count"] == 28
    assert status["second_goal_resume_readiness_matrix"]["xr64_ready_to_train"] is False
    assert status["second_goal_resume_readiness_matrix"]["xr64_can_run_lane"] is False
    assert status["second_goal_resume_readiness_matrix"]["xr64_missing_eval_rows"] == 8
    assert status["second_goal_resume_readiness_matrix"]["xr64_missing_overrides"] == 6
    assert status["second_goal_resume_readiness_matrix"]["xr64_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["second_goal_resume_readiness_matrix"]["xr64_prep_unit_count"] == 10
    assert status["second_goal_resume_readiness_matrix"]["xr64_prep_ready_after_resume_count"] == 8
    assert status["second_goal_resume_readiness_matrix"]["xr64_prep_blocked_count"] == 2
    assert status["second_goal_resume_readiness_matrix"]["xr64_postrun_candidate_count"] == 0
    assert status["second_goal_resume_readiness_matrix"]["paper_target_required_evidence_count"] == 7
    assert status["second_goal_resume_readiness_matrix"]["paper_target_required_output_schema_count"] == 10
    assert status["second_goal_resume_readiness_matrix"]["paper_target_ptb1_existing_payload_count"] == 0
    assert status["current_result_synthesis"]["exists"] is True
    assert status["current_result_synthesis"]["single_model_sota_claim_allowed"] is False
    assert status["current_result_synthesis"]["oracle_promotable"] is False
    assert status["current_result_synthesis"]["direct_submission_comparison_valid"] is False
    assert status["current_result_synthesis"]["xr64_ready_to_train"] is False
    assert status["accuracy_lift_decision_ladder"]["exists"] is True
    assert status["accuracy_lift_decision_ladder"]["execute_supported"] is False
    assert status["accuracy_lift_decision_ladder"]["allowed_while_paused"] is True
    assert status["accuracy_lift_decision_ladder"]["single_model_sota_claim_allowed"] is False
    assert status["accuracy_lift_decision_ladder"]["oracle_promotable"] is False
    assert status["accuracy_lift_decision_ladder"]["direct_submission_comparison_allowed"] is False
    assert status["accuracy_lift_decision_ladder"]["stage_count"] == 5
    assert status["accuracy_lift_decision_ladder"]["experiment_count"] == 8
    assert status["accuracy_lift_decision_ladder"]["p0_ids"] == ["XR-64-prep", "XR-64A", "XR-64B"]
    assert status["accuracy_lift_decision_ladder"]["followup_ids"] == [
        "XR-64C",
        "XR-65",
        "XR-66",
        "XR-67",
        "XR-68",
    ]
    assert status["accuracy_lift_decision_ladder"]["axis_count"] == 6
    assert status["accuracy_lift_decision_ladder"]["blocked_launch_count"] == 8
    assert status["accuracy_lift_decision_ladder"]["expected_evidence_count"] == 8
    assert status["accuracy_lift_decision_ladder"]["missing_eval_rows"] == 8
    assert status["accuracy_lift_decision_ladder"]["missing_overrides"] == 6
    assert status["accuracy_lift_decision_ladder"]["xr64_postrun_candidate_count"] == 0
    assert status["accuracy_lift_decision_ladder"]["minimal_next_worker_after_resume"] == "XR-64-prep"
    assert status["accuracy_lift_decision_ladder"]["current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["objective_trace"]["exists"] is True
    assert status["objective_trace"]["goal_complete"] is False
    assert status["objective_trace"]["requirement_count"] == 3
    assert status["objective_trace"]["req3_status"] == "incomplete"
    assert "teacher_model_training" in status["objective_trace"]["covered_axes"]
    assert status["objective_trace"]["post_xr64_branch_count"] == 7
    assert status["objective_trace"]["post_xr64_minimal_next_worker"] == "XR-64-prep"
    assert status["objective_trace"]["post_xr64_current_next_prep_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["objective_trace"]["post_xr64_direct_submission_comparison_allowed"] is False
    assert status["objective_trace"]["post_xr64_next_experiments"] == [
        "XR-64C",
        "XR-65",
        "XR-66",
        "XR-67",
        "XR-68",
    ]
    assert status["xr64_resume_command_manifest"]["exists"] is True
    assert status["xr64_resume_command_manifest"]["do_not_execute_until_user_resumes"] is True
    assert status["xr64_resume_command_manifest"]["default_allowed_to_run"] is False
    assert status["xr64_resume_command_manifest"]["prep_command_count"] == 10
    assert status["xr64_resume_command_manifest"]["eval_command_count"] == 8
    assert status["xr64_resume_command_manifest"]["build_command_count"] == 2
    assert status["xr64_resume_command_manifest"]["expected_eval_rows"] == 8
    assert status["xr64_resume_command_manifest"]["expected_overrides"] == 6
    assert status["xr64_resume_command_manifest"]["expected_split_train_rows"] == 5929
    assert status["xr64_resume_command_manifest"]["expected_split_val_rows"] == 844
    assert status["xr64_resume_command_manifest"]["expected_teacher_count"] == 4
    assert status["xr64_resume_command_manifest"]["expected_override_rule_count"] == 3
    assert status["xr64_resume_command_manifest"]["expected_total_eval_row_records"] == 27092
    assert status["xr64_resume_command_manifest"]["expected_total_override_records"] == 20319
    assert status["xr64_resume_command_manifest"]["eval_required_input_count"] == 6
    assert status["xr64_resume_command_manifest"]["existing_eval_required_input_count"] == 6
    assert status["xr64_resume_command_manifest"]["missing_eval_required_input_count"] == 0
    assert status["xr64_resume_command_manifest"]["eval_inputs_ready"] is True
    assert status["xr64_resume_command_manifest"]["build_required_input_count"] == 8
    assert status["xr64_resume_command_manifest"]["existing_build_required_input_count"] == 0
    assert status["xr64_resume_command_manifest"]["missing_build_required_input_count"] == 8
    assert status["xr64_resume_command_manifest"]["build_inputs_ready"] is False
    assert status["xr64_resume_command_manifest"]["postrun_command_count"] == 2
    assert status["xr64_resume_command_manifest"]["prep_runner_contract_ok"] is True
    assert status["xr64_resume_command_manifest"]["prep_runner_contract_check_count"] == 17
    assert status["xr64_resume_command_emitter"]["script_exists"] is True
    assert status["xr64_resume_command_emitter"]["tests_exist"] is True
    assert status["xr64_resume_command_emitter"]["execute_supported"] is False
    assert status["xr64_next_prep_command_selector"]["script_exists"] is True
    assert status["xr64_next_prep_command_selector"]["tests_exist"] is True
    assert status["xr64_next_prep_command_selector"]["execute_supported"] is False
    assert status["xr64_next_prep_command_selector"]["ok"] is True
    assert status["xr64_next_prep_command_selector"]["ready_after_resume_count"] == 8
    assert status["xr64_next_prep_command_selector"]["completed_count"] == 0
    assert status["xr64_next_prep_command_selector"]["blocked_count"] == 2
    assert status["xr64_next_prep_command_selector"]["invalid_count"] == 0
    assert status["xr64_next_prep_command_selector"]["next_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["xr64_next_prep_command_selector"]["next_status"] == "ready_after_resume"
    assert status["xr64_next_prep_command_selector"]["next_allowed_to_run_now"] is False
    assert status["xr64_prep_progress_ledger"]["exists"] is True
    assert status["xr64_prep_progress_ledger"]["execute_supported"] is False
    assert status["xr64_prep_progress_ledger"]["allowed_to_run_now"] is False
    assert status["xr64_prep_progress_ledger"]["prep_unit_count"] == 10
    assert status["xr64_prep_progress_ledger"]["eval_unit_count"] == 8
    assert status["xr64_prep_progress_ledger"]["build_unit_count"] == 2
    assert status["xr64_prep_progress_ledger"]["ready_after_resume_count"] == 8
    assert status["xr64_prep_progress_ledger"]["completed_count"] == 0
    assert status["xr64_prep_progress_ledger"]["blocked_count"] == 2
    assert status["xr64_prep_progress_ledger"]["invalid_count"] == 0
    assert status["xr64_prep_progress_ledger"]["next_id"] == "XR64-EVAL-TRAIN-XR62A"
    assert status["xr64_prep_progress_ledger"]["next_status"] == "ready_after_resume"
    assert status["xr64_prep_progress_ledger"]["next_allowed_to_run_now"] is False
    assert status["xr64_prep_progress_ledger"]["missing_eval_rows"] == 8
    assert status["xr64_prep_progress_ledger"]["missing_overrides"] == 6
    assert status["xr64_prep_progress_ledger"]["eval_inputs_ready"] is True
    assert status["xr64_prep_progress_ledger"]["build_inputs_ready"] is False
    assert status["xr64_prep_progress_ledger"]["total_eval_row_records"] == 27092
    assert status["xr64_prep_progress_ledger"]["total_override_records"] == 20319
    assert status["xr64_prelaunch_packet"]["exists"] is True
    assert status["xr64_prelaunch_packet"]["allowed_to_execute"] is False
    assert status["xr64_prelaunch_packet"]["completion_allowed"] is False
    assert status["xr64_prelaunch_packet"]["completion_blocker_count"] == 28
    assert status["xr64_prelaunch_packet"]["phase_count"] == 6
    assert status["xr64_prelaunch_packet"]["missing_eval_rows"] == 8
    assert status["xr64_prelaunch_packet"]["missing_overrides"] == 6
    assert status["xr64_prelaunch_packet"]["eval_inputs_ready"] is True
    assert status["xr64_prelaunch_packet"]["missing_eval_required_input_count"] == 0
    assert status["xr64_prelaunch_packet"]["build_inputs_ready"] is False
    assert status["xr64_prelaunch_packet"]["missing_build_required_input_count"] == 8
    assert status["xr64_prelaunch_packet"]["postrun_command_count"] == 2
    assert status["xr64_postrun_evidence_contract"]["exists"] is True
    assert status["xr64_postrun_evidence_contract"]["evidence_state"] == "missing_postrun_evidence"
    assert status["xr64_postrun_evidence_contract"]["allowed_to_claim_promotion"] is False
    assert status["xr64_postrun_evidence_contract"]["allowed_to_mark_second_goal_complete"] is False
    assert status["xr64_postrun_evidence_contract"]["required_lane_count"] == 2
    assert status["xr64_postrun_evidence_contract"]["required_checkpoint_kind_count"] == 3
    assert status["xr64_postrun_evidence_contract"]["required_metric_count"] == 4
    assert status["xr64_postrun_evidence_contract"]["ablation_axis_count"] == 4
    assert status["xr64_postrun_evidence_contract"]["ablation_lane_axis_count"] == 2
    assert status["xr64_postrun_evidence_contract"]["xr64a_eval_summary_count"] == 0
    assert status["xr64_postrun_evidence_contract"]["xr64b_eval_summary_count"] == 0
    assert status["xr64_postrun_evidence_contract"]["decision_status"] == "pending_postrun_evidence"
    assert status["xr64_postrun_promotion_decision"]["exists"] is True
    assert status["xr64_postrun_promotion_decision"]["ok"] is True
    assert status["xr64_postrun_promotion_decision"]["error_count"] == 0
    assert status["xr64_postrun_promotion_decision"]["decision_status"] == "missing_postrun_evidence"
    assert status["xr64_postrun_promotion_decision"]["candidate_count"] == 0
    assert status["xr64_postrun_promotion_decision"]["software_promotion_allowed"] is False
    assert status["xr64_postrun_promotion_decision"]["single_model_sota_claim_allowed"] is False
    assert status["xr64_postrun_promotion_decision"]["paper_level_completion_allowed"] is False
    assert status["xr64_postrun_promotion_decision"]["future_evidence_metrics"] == ["metric_track_p1_pct"]
    assert status["xr64_postrun_promotion_decision"]["ablation_required_axes"] == [
        "head",
        "loss",
        "lr",
        "teacher_model_training",
    ]
    assert status["xr64_postrun_promotion_decision"]["axis_certification_required"] is True
    assert status["xr64_postrun_promotion_decision"]["axis_certified_candidate_count"] == 0
    assert status["xr64_postrun_promotion_decision"]["axis_claims_allowed"] is False
    assert status["xr64_postrun_promotion_decision"]["control_variables_checked"] is False
    assert status["xr64_postrun_promotion_decision"]["bounded_tradeoff_candidate_count"] == 0
    assert status["xr64_postrun_promotion_decision"]["unbounded_tradeoff_candidate_count"] == 0
    assert status["xr64_postrun_promotion_decision"]["rejection_reason"] == "missing_postrun_evidence"
    assert status["xr64_postrun_promotion_decision"]["p1_evidence_candidate_count"] == 0
    assert status["xr64_postrun_candidate_collector"]["exists"] is True
    assert status["xr64_postrun_candidate_collector"]["tests_exist"] is True
    assert status["xr64_postrun_candidate_collector"]["default_latest_per_lane_checkpoint"] is True
    assert status["xr64_postrun_candidate_collector"]["include_all_option"] is True
    assert status["xr64_postrun_candidate_collector"]["current_candidate_count"] == 0
    assert status["xr64_postrun_candidate_collector"]["current_decision_status"] == "missing_postrun_evidence"
    assert "date: 2026-06-18" in summary
    assert "paper_ref_per_paper_analysis_actual: 31" in summary
    assert "experiment_queue_count: 4" in summary
    assert "experiment_queue_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "xr64_experiment_design_contract_exists: true" in summary
    assert "xr64_experiment_design_contract_ok: true" in summary
    assert "xr64_experiment_design_contract_execute_supported: false" in summary
    assert "xr64_experiment_design_contract_allowed_while_paused: true" in summary
    assert "xr64_experiment_design_contract_lane_count: 4" in summary
    assert "xr64_experiment_design_contract_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "xr64_experiment_design_contract_axis_count: 6" in summary
    assert "xr64_experiment_design_contract_ready_to_train: false" in summary
    assert "xr64_experiment_design_contract_missing_eval_rows: 8" in summary
    assert "xr64_experiment_design_contract_missing_overrides: 6" in summary
    assert "xr64_experiment_design_contract_completion_allowed: false" in summary
    assert "xr64_experiment_design_contract_paper_level_completion_allowed: false" in summary
    assert "xr64_experiment_design_contract_error_count: 0" in summary
    assert "xr64_resume_execution_dag_exists: true" in summary
    assert "xr64_resume_execution_dag_ok: true" in summary
    assert "xr64_resume_execution_dag_execute_supported: false" in summary
    assert "xr64_resume_execution_dag_allowed_while_paused: true" in summary
    assert "xr64_resume_execution_dag_node_count: 17" in summary
    assert "xr64_resume_execution_dag_phase_count: 6" in summary
    assert "xr64_resume_execution_dag_prep_nodes: 10" in summary
    assert "xr64_resume_execution_dag_eval_nodes: 8" in summary
    assert "xr64_resume_execution_dag_build_nodes: 2" in summary
    assert "xr64_resume_execution_dag_launch_nodes: 2" in summary
    assert "xr64_resume_execution_dag_postrun_nodes: 2" in summary
    assert "xr64_resume_execution_dag_ready_to_train: false" in summary
    assert "xr64_resume_execution_dag_missing_eval_rows: 8" in summary
    assert "xr64_resume_execution_dag_missing_overrides: 6" in summary
    assert "xr64_resume_execution_dag_current_next_node: XR64-EVAL-TRAIN-XR62A" in summary
    assert "xr64_resume_execution_dag_completion_allowed: false" in summary
    assert "xr64_resume_execution_dag_manifest_ok: true" in summary
    assert "xr64_resume_execution_dag_design_contract_ok: true" in summary
    assert "xr64_resume_execution_dag_error_count: 0" in summary
    assert "ablation_evidence_checklist_exists: true" in summary
    assert "ablation_evidence_experiment_count: 8" in summary
    assert "ablation_evidence_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "ablation_evidence_axis_count: 8" in summary
    assert "ablation_evidence_software_promotion_allowed: false" in summary
    assert "ablation_evidence_missing_eval_rows: 8" in summary
    assert "paper_ref_experiment_trace_exists: true" in summary
    assert "paper_ref_experiment_trace_execute_supported: false" in summary
    assert "paper_ref_experiment_trace_allowed_while_paused: true" in summary
    assert "paper_ref_experiment_trace_experiment_count: 8" in summary
    assert "paper_ref_experiment_trace_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "paper_ref_experiment_trace_paper_group_count: 6" in summary
    assert "paper_ref_experiment_trace_axis_count: 8" in summary
    assert "paper_ref_experiment_trace_xr64_ready_to_train: false" in summary
    assert "paper_ref_experiment_trace_missing_eval_rows: 8" in summary
    assert "paper_ref_experiment_trace_missing_overrides: 6" in summary
    assert "paper_ref_experiment_trace_completion_allowed: false" in summary
    assert "paper_ref_experiment_trace_direct_submission_comparison_allowed: false" in summary
    assert "paper_ref_current_ablation_plan_exists: true" in summary
    assert "paper_ref_current_ablation_plan_ok: true" in summary
    assert "paper_ref_current_ablation_plan_execute_supported: false" in summary
    assert "paper_ref_current_ablation_plan_allowed_while_paused: true" in summary
    assert "paper_ref_current_ablation_plan_experiment_count: 8" in summary
    assert "paper_ref_current_ablation_plan_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "paper_ref_current_ablation_plan_paper_signal_count: 18" in summary
    assert "paper_ref_current_ablation_plan_mapping_row_count: 10" in summary
    assert "paper_ref_current_ablation_plan_lane_trace_count: 8" in summary
    assert "paper_ref_current_ablation_plan_axis_count: 8" in summary
    assert "paper_ref_current_ablation_plan_completion_allowed: false" in summary
    assert "paper_ref_current_ablation_plan_direct_submission_comparison_allowed: false" in summary
    assert "paper_ref_current_ablation_plan_error_count: 0" in summary
    assert "post_xr64_decision_tree_exists: true" in summary
    assert "post_xr64_decision_tree_execute_supported: false" in summary
    assert "post_xr64_decision_tree_claim_level_count: 4" in summary
    assert "post_xr64_decision_tree_decision_input_count: 3" in summary
    assert "post_xr64_decision_tree_branch_count: 7" in summary
    assert "post_xr64_decision_tree_minimal_next_worker: XR-64-prep" in summary
    assert "post_xr64_decision_tree_current_next_prep_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "post_xr64_decision_tree_direct_submission_comparison_allowed: false" in summary
    assert "post_xr64_decision_tree_next_experiments: XR-64C,XR-65,XR-66,XR-67,XR-68" in summary
    assert "post_xr64_followup_contract_exists: true" in summary
    assert "post_xr64_followup_contract_execute_supported: false" in summary
    assert "post_xr64_followup_contract_direct_submission_comparison_allowed: false" in summary
    assert "post_xr64_followup_contract_paper_level_completion_allowed: false" in summary
    assert "post_xr64_followup_contract_experiment_count: 5" in summary
    assert "post_xr64_followup_contract_experiment_ids: XR-64C,XR-65,XR-66,XR-67,XR-68" in summary
    assert "post_xr64_followup_contract_blocked_launch_count: 5" in summary
    assert "post_xr64_followup_contract_p1_count: 3" in summary
    assert "post_xr64_followup_contract_p2_count: 2" in summary
    assert "post_xr64_followup_contract_minimal_next_worker: XR-64-prep" in summary
    assert "post_xr64_followup_contract_xr64_postrun_candidate_count: 0" in summary
    assert "post_xr64_followup_contract_launch_allowed_now: false" in summary
    assert "completion_readiness_contract_exists: true" in summary
    assert "completion_readiness_allowed: false" in summary
    assert "completion_readiness_blocker_count: 28" in summary
    assert "completion_readiness_xr64_ready_to_train: false" in summary
    assert "completion_readiness_missing_eval_rows: 8" in summary
    assert "completion_readiness_leakage_risk: none" in summary
    assert "completion_readiness_paper_target_bridge_evaluator_schema_execute_supported: false" in summary
    assert "completion_readiness_paper_target_bridge_evaluator_schema_output_count: 5" in summary
    assert "completion_readiness_paper_target_bridge_evaluator_schema_current_existing_output_count: 0" in summary
    assert "completion_readiness_paper_target_bridge_trained_candidate_execute_supported: false" in summary
    assert "completion_readiness_paper_target_bridge_trained_candidate_schema_allowed_while_paused: true" in summary
    assert "completion_readiness_paper_target_bridge_trained_candidate_payload_allowed_while_paused: false" in summary
    assert "completion_readiness_paper_target_bridge_trained_candidate_output_count: 2" in summary
    assert "completion_readiness_paper_target_bridge_trained_candidate_current_existing_output_count: 0" in summary
    assert "completion_readiness_paper_target_bridge_decision_execute_supported: false" in summary
    assert "completion_readiness_paper_target_bridge_decision_schema_allowed_while_paused: true" in summary
    assert "completion_readiness_paper_target_bridge_decision_payload_allowed_while_paused: false" in summary
    assert "completion_readiness_paper_target_bridge_decision_output_count: 1" in summary
    assert "completion_readiness_paper_target_bridge_decision_current_existing_output_count: 0" in summary
    assert "completion_readiness_post_xr64_followup_contract_execute_supported: false" in summary
    assert "completion_readiness_post_xr64_followup_contract_experiment_count: 5" in summary
    assert "completion_readiness_post_xr64_followup_contract_blocked_launch_count: 5" in summary
    assert "completion_readiness_post_xr64_followup_contract_launch_allowed_now: false" in summary
    assert "completion_readiness_xr64_command_manifest_postrun_command_count: 2" in summary
    assert "completion_readiness_xr64_command_manifest_full_eval_coverage_required: true" in summary
    assert "completion_readiness_xr64_command_manifest_full_override_coverage_required: true" in summary
    assert "completion_readiness_xr64_command_manifest_duplicate_sample_ids_allowed: false" in summary
    assert "completion_readiness_xr64_command_manifest_opposite_split_sample_ids_allowed: false" in summary
    assert "completion_readiness_xr64_command_manifest_subset_artifacts_satisfy_ready_to_train: false" in summary
    assert (
        "completion_readiness_xr64_command_manifest_strict_generated_artifact_checker: "
        "scripts/external/check_xr64_resume_artifacts.py"
        in summary
    )
    assert "completion_readiness_xr64_next_prep_selector_ok: true" in summary
    assert "completion_readiness_xr64_next_prep_selector_ready_after_resume_count: 8" in summary
    assert "completion_readiness_xr64_next_prep_selector_completed_count: 0" in summary
    assert "completion_readiness_xr64_next_prep_selector_blocked_count: 2" in summary
    assert "completion_readiness_xr64_next_prep_selector_invalid_count: 0" in summary
    assert "completion_readiness_xr64_next_prep_selector_next_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "completion_readiness_xr64_next_prep_selector_next_status: ready_after_resume" in summary
    assert "completion_readiness_xr64_next_prep_selector_next_allowed_to_run_now: false" in summary
    assert "completion_readiness_xr64_next_prep_selector_execute_supported: false" in summary
    assert "completion_readiness_xr64_prelaunch_phase_count: 6" in summary
    assert "completion_readiness_xr64_prelaunch_postrun_command_count: 2" in summary
    assert "completion_readiness_xr64_postrun_candidate_count: 0" in summary
    assert "submission_target_gap_exists: true" in summary
    assert "submission_direct_comparison_valid: false" in summary
    assert "submission_target_source_trace_exists: true" in summary
    assert "submission_target_source_file_exists: true" in summary
    assert "submission_target_source_direct_comparison_valid: false" in summary
    assert "submission_target_source_hybrid_center_px: 0.1812" in summary
    assert "submission_target_source_hybrid_p1_pct: 99.61" in summary
    assert "submission_target_source_accel_range_single_target: false" in summary
    assert "metric_protocol_bridge_exists: true" in summary
    assert "metric_protocol_direct_submission_comparison_allowed: false" in summary
    assert "metric_protocol_current_p1_metric_available: true" in summary
    assert "metric_protocol_current_p1_full_test_evidence_available: false" in summary
    assert "metric_protocol_unblock_contract_exists: true" in summary
    assert "metric_protocol_unblock_status: blocked" in summary
    assert "metric_protocol_unblock_direct_submission_comparison_allowed: false" in summary
    assert "metric_protocol_unblock_required_evidence_count: 7" in summary
    assert "paper_target_evidence_manifest_exists: true" in summary
    assert "paper_target_evidence_state: missing_required_evidence" in summary
    assert "paper_target_required_evidence_complete: false" in summary
    assert "paper_target_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_required_future_artifact_count: 10" in summary
    assert "paper_target_existing_required_future_artifact_count: 3" in summary
    assert "paper_target_artifact_schema_count: 10" in summary
    assert "paper_target_bridge_workplan_exists: true" in summary
    assert "paper_target_bridge_workplan_execute_supported: false" in summary
    assert "paper_target_bridge_workplan_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_workplan_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_workplan_work_package_count: 4" in summary
    assert "paper_target_bridge_workplan_future_artifact_count: 10" in summary
    assert "paper_target_bridge_workplan_allowed_while_paused_package_count: 2" in summary
    assert "paper_target_bridge_workplan_doc_prefill_artifact_count: 3" in summary
    assert "paper_target_bridge_workplan_evaluator_schema_artifact_count: 5" in summary
    assert "paper_target_bridge_workplan_trained_candidate_artifact_count: 2" in summary
    assert "paper_target_bridge_workplan_current_existing_future_artifact_count: 3" in summary
    assert "paper_target_bridge_evaluator_schema_exists: true" in summary
    assert "paper_target_bridge_evaluator_schema_execute_supported: false" in summary
    assert "paper_target_bridge_evaluator_schema_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_evaluator_schema_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_evaluator_schema_output_count: 5" in summary
    assert "paper_target_bridge_evaluator_schema_jsonl_count: 2" in summary
    assert "paper_target_bridge_evaluator_schema_json_count: 3" in summary
    assert "paper_target_bridge_evaluator_schema_blocked_output_count: 5" in summary
    assert "paper_target_bridge_evaluator_schema_current_existing_output_count: 0" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_exists: true" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_execute_supported: false" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_allowed_while_paused: true" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_sequence_count: 5" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_output_count: 5" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_current_existing_output_count: 0" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_payload_missing_count: 5" in summary
    assert "paper_target_bridge_ptb1_resume_runbook_future_payload_validation_requires_allow_present: true" in summary
    assert "paper_target_bridge_trained_candidate_exists: true" in summary
    assert "paper_target_bridge_trained_candidate_execute_supported: false" in summary
    assert "paper_target_bridge_trained_candidate_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_trained_candidate_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_trained_candidate_schema_allowed_while_paused: true" in summary
    assert "paper_target_bridge_trained_candidate_payload_allowed_while_paused: false" in summary
    assert "paper_target_bridge_trained_candidate_output_count: 2" in summary
    assert "paper_target_bridge_trained_candidate_json_count: 2" in summary
    assert "paper_target_bridge_trained_candidate_blocked_output_count: 2" in summary
    assert "paper_target_bridge_trained_candidate_current_existing_output_count: 0" in summary
    assert "paper_target_bridge_decision_exists: true" in summary
    assert "paper_target_bridge_decision_execute_supported: false" in summary
    assert "paper_target_bridge_decision_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_decision_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_decision_schema_allowed_while_paused: true" in summary
    assert "paper_target_bridge_decision_payload_allowed_while_paused: false" in summary
    assert "paper_target_bridge_decision_output_count: 1" in summary
    assert "paper_target_bridge_decision_json_count: 1" in summary
    assert "paper_target_bridge_decision_blocked_output_count: 1" in summary
    assert "paper_target_bridge_decision_current_existing_output_count: 0" in summary
    assert "paper_target_bridge_decision_signoff_rule_count: 6" in summary
    assert "paper_target_bridge_resume_dependency_matrix_exists: true" in summary
    assert "paper_target_bridge_resume_dependency_matrix_execute_supported: false" in summary
    assert "paper_target_bridge_resume_dependency_matrix_allowed_while_paused: true" in summary
    assert "paper_target_bridge_resume_dependency_matrix_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_bridge_resume_dependency_matrix_paper_level_completion_allowed: false" in summary
    assert "paper_target_bridge_resume_dependency_matrix_dependency_gate_count: 6" in summary
    assert "paper_target_bridge_resume_dependency_matrix_allowed_while_paused_gate_count: 2" in summary
    assert "paper_target_bridge_resume_dependency_matrix_complete_now_gate_count: 2" in summary
    assert "paper_target_bridge_resume_dependency_matrix_blocked_gate_count: 4" in summary
    assert "paper_target_bridge_resume_dependency_matrix_execution_dependent_payload_missing_count: 8" in summary
    assert "paper_target_bridge_resume_dependency_matrix_completion_blocker_count: 28" in summary
    assert "paper_target_bridge_resume_dependency_matrix_xr64_ready_to_train: false" in summary
    assert "paper_target_bridge_resume_dependency_matrix_completion_allowed: false" in summary
    assert "paper_target_metric_preflight_exists: true" in summary
    assert "paper_target_metric_preflight_execute_supported: false" in summary
    assert "paper_target_metric_preflight_allowed_while_paused: true" in summary
    assert "paper_target_metric_preflight_direct_submission_comparison_allowed: false" in summary
    assert "paper_target_metric_preflight_required_evidence_count: 7" in summary
    assert "paper_target_metric_preflight_required_output_schema_count: 10" in summary
    assert "paper_target_metric_preflight_ptb1_evaluator_output_schema_count: 5" in summary
    assert "paper_target_metric_preflight_current_p1_metric_available: true" in summary
    assert "paper_target_metric_preflight_xr64_ready_to_train: false" in summary
    assert "second_goal_resume_matrix_exists: true" in summary
    assert "second_goal_resume_matrix_execute_supported: false" in summary
    assert "second_goal_resume_matrix_allowed_while_paused: true" in summary
    assert "second_goal_resume_matrix_direct_submission_comparison_allowed: false" in summary
    assert "second_goal_resume_matrix_software_promotion_allowed: false" in summary
    assert "second_goal_resume_matrix_paper_level_completion_allowed: false" in summary
    assert "second_goal_resume_matrix_axis_count: 6" in summary
    assert "second_goal_resume_matrix_resume_gate_count: 5" in summary
    assert "second_goal_resume_matrix_current_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "second_goal_resume_matrix_goal_complete: false" in summary
    assert "second_goal_resume_matrix_req3_status: incomplete" in summary
    assert "second_goal_resume_matrix_completion_allowed: false" in summary
    assert "second_goal_resume_matrix_blocker_count: 28" in summary
    assert "second_goal_resume_matrix_xr64_ready_to_train: false" in summary
    assert "second_goal_resume_matrix_xr64_missing_eval_rows: 8" in summary
    assert "second_goal_resume_matrix_xr64_missing_overrides: 6" in summary
    assert "second_goal_resume_matrix_xr64_next_prep_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "second_goal_resume_matrix_xr64_prep_unit_count: 10" in summary
    assert "second_goal_resume_matrix_xr64_prep_ready_after_resume_count: 8" in summary
    assert "second_goal_resume_matrix_paper_target_required_output_schema_count: 10" in summary
    assert "current_result_synthesis_exists: true" in summary
    assert "current_result_oracle_promotable: false" in summary
    assert "accuracy_lift_decision_ladder_exists: true" in summary
    assert "accuracy_lift_decision_ladder_execute_supported: false" in summary
    assert "accuracy_lift_decision_ladder_allowed_while_paused: true" in summary
    assert "accuracy_lift_decision_ladder_single_model_sota_claim_allowed: false" in summary
    assert "accuracy_lift_decision_ladder_oracle_promotable: false" in summary
    assert "accuracy_lift_decision_ladder_direct_submission_comparison_allowed: false" in summary
    assert "accuracy_lift_decision_ladder_stage_count: 5" in summary
    assert "accuracy_lift_decision_ladder_experiment_count: 8" in summary
    assert "accuracy_lift_decision_ladder_p0_ids: XR-64-prep,XR-64A,XR-64B" in summary
    assert "accuracy_lift_decision_ladder_followup_ids: XR-64C,XR-65,XR-66,XR-67,XR-68" in summary
    assert "accuracy_lift_decision_ladder_axis_count: 6" in summary
    assert "accuracy_lift_decision_ladder_blocked_launch_count: 8" in summary
    assert "accuracy_lift_decision_ladder_expected_evidence_count: 8" in summary
    assert "accuracy_lift_decision_ladder_missing_eval_rows: 8" in summary
    assert "accuracy_lift_decision_ladder_missing_overrides: 6" in summary
    assert "accuracy_lift_decision_ladder_xr64_postrun_candidate_count: 0" in summary
    assert "accuracy_lift_decision_ladder_minimal_next_worker: XR-64-prep" in summary
    assert "accuracy_lift_decision_ladder_current_next_prep_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "objective_trace_exists: true" in summary
    assert "objective_trace_goal_complete: false" in summary
    assert "objective_trace_req3_status: incomplete" in summary
    assert "objective_trace_post_xr64_branch_count: 7" in summary
    assert "objective_trace_post_xr64_minimal_next_worker: XR-64-prep" in summary
    assert "objective_trace_post_xr64_current_next_prep_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "objective_trace_post_xr64_direct_submission_comparison_allowed: false" in summary
    assert "objective_trace_post_xr64_next_experiments: XR-64C,XR-65,XR-66,XR-67,XR-68" in summary
    assert "xr64_command_manifest_exists: true" in summary
    assert "xr64_command_manifest_eval_command_count: 8" in summary
    assert "xr64_command_manifest_build_command_count: 2" in summary
    assert "xr64_command_manifest_expected_split_train_rows: 5929" in summary
    assert "xr64_command_manifest_expected_split_val_rows: 844" in summary
    assert "xr64_command_manifest_expected_teacher_count: 4" in summary
    assert "xr64_command_manifest_expected_override_rule_count: 3" in summary
    assert "xr64_command_manifest_expected_total_eval_row_records: 27092" in summary
    assert "xr64_command_manifest_expected_total_override_records: 20319" in summary
    assert "xr64_command_manifest_missing_eval_required_input_count: 0" in summary
    assert "xr64_command_manifest_eval_inputs_ready: true" in summary
    assert "xr64_command_manifest_missing_build_required_input_count: 8" in summary
    assert "xr64_command_manifest_build_inputs_ready: false" in summary
    assert "xr64_command_manifest_postrun_command_count: 2" in summary
    assert "xr64_command_manifest_prep_runner_contract_ok: true" in summary
    assert "xr64_command_manifest_prep_runner_contract_check_count: 17" in summary
    assert "xr64_command_manifest_full_eval_coverage_required: true" in summary
    assert "xr64_command_manifest_full_override_coverage_required: true" in summary
    assert "xr64_command_manifest_duplicate_sample_ids_allowed: false" in summary
    assert "xr64_command_manifest_opposite_split_sample_ids_allowed: false" in summary
    assert "xr64_command_manifest_subset_artifacts_satisfy_ready_to_train: false" in summary
    assert "xr64_command_manifest_strict_generated_artifact_checker: scripts/external/check_xr64_resume_artifacts.py" in summary
    assert "xr64_command_emitter_script_exists: true" in summary
    assert "xr64_command_emitter_execute_supported: false" in summary
    assert "xr64_next_prep_selector_script_exists: true" in summary
    assert "xr64_next_prep_selector_execute_supported: false" in summary
    assert "xr64_next_prep_selector_ok: true" in summary
    assert "xr64_next_prep_selector_ready_after_resume_count: 8" in summary
    assert "xr64_next_prep_selector_blocked_count: 2" in summary
    assert "xr64_next_prep_selector_next_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "xr64_next_prep_selector_next_status: ready_after_resume" in summary
    assert "xr64_next_prep_selector_next_allowed_to_run_now: false" in summary
    assert "xr64_prep_ledger_exists: true" in summary
    assert "xr64_prep_ledger_prep_unit_count: 10" in summary
    assert "xr64_prep_ledger_ready_after_resume_count: 8" in summary
    assert "xr64_prep_ledger_blocked_count: 2" in summary
    assert "xr64_prep_ledger_next_id: XR64-EVAL-TRAIN-XR62A" in summary
    assert "xr64_prep_ledger_missing_eval_rows: 8" in summary
    assert "xr64_prep_ledger_total_eval_row_records: 27092" in summary
    assert "xr64_prelaunch_packet_exists: true" in summary
    assert "xr64_prelaunch_completion_allowed: false" in summary
    assert "xr64_prelaunch_completion_blocker_count: 28" in summary
    assert "xr64_prelaunch_eval_inputs_ready: true" in summary
    assert "xr64_prelaunch_missing_eval_required_input_count: 0" in summary
    assert "xr64_prelaunch_build_inputs_ready: false" in summary
    assert "xr64_prelaunch_missing_build_required_input_count: 8" in summary
    assert "xr64_prelaunch_postrun_command_count: 2" in summary
    assert "xr64_postrun_contract_exists: true" in summary
    assert "xr64_postrun_evidence_state: missing_postrun_evidence" in summary
    assert "xr64_postrun_allowed_to_claim_promotion: false" in summary
    assert "xr64_postrun_decision_status: pending_postrun_evidence" in summary
    assert "xr64_postrun_required_metric_count: 4" in summary
    assert "xr64_postrun_ablation_axis_count: 4" in summary
    assert "xr64_postrun_ablation_lane_axis_count: 2" in summary
    assert "xr64_promotion_decision_exists: true" in summary
    assert "xr64_promotion_decision_ok: true" in summary
    assert "xr64_promotion_decision_error_count: 0" in summary
    assert "xr64_promotion_decision_status: missing_postrun_evidence" in summary
    assert "xr64_promotion_software_allowed: false" in summary
    assert "xr64_promotion_future_evidence_metrics: metric_track_p1_pct" in summary
    assert "xr64_promotion_ablation_required_axes: head,loss,lr,teacher_model_training" in summary
    assert "xr64_promotion_axis_certification_required: true" in summary
    assert "xr64_promotion_axis_certified_candidate_count: 0" in summary
    assert "xr64_promotion_axis_claims_allowed: false" in summary
    assert "xr64_promotion_control_variables_checked: false" in summary
    assert "xr64_promotion_bounded_tradeoff_candidate_count: 0" in summary
    assert "xr64_promotion_unbounded_tradeoff_candidate_count: 0" in summary
    assert "xr64_promotion_rejection_reason: missing_postrun_evidence" in summary
    assert "xr64_promotion_p1_evidence_candidate_count: 0" in summary
    assert "xr64_candidate_collector_exists: true" in summary
    assert "xr64_candidate_collector_tests_exist: true" in summary
    assert "xr64_candidate_collector_current_candidate_count: 0" in summary
    assert "xr64_candidate_collector_current_decision_status: missing_postrun_evidence" in summary
    assert "xr64_can_run_lane: false" in summary
    assert "next_action: generate XR-64 train/val eval rows" in summary


def test_second_goal_status_blocks_launch_when_command_manifest_build_inputs_missing(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_status_module()
    index_path = write_index(tmp_path)
    patch_checker(module, monkeypatch, generated_complete=True)

    code, status = module.build_status(
        Namespace(project_root=str(tmp_path), index=str(index_path), strict_xr64=False, format="json")
    )

    assert code == 0
    assert status["xr64"]["generated_complete"] is True
    assert status["xr64_resume_command_manifest"]["eval_inputs_ready"] is True
    assert status["xr64_resume_command_manifest"]["build_inputs_ready"] is False
    assert status["xr64"]["ready_to_train"] is False
    assert status["xr64"]["can_run_lane"] is False
    assert status["next_action"] != "launch XR-64A/B only after user resumes experiments"


def test_second_goal_status_strict_propagates_checker_failure(tmp_path: Path, monkeypatch) -> None:
    module = load_status_module()
    index_path = write_index(tmp_path)
    patch_checker(module, monkeypatch, checker_code=1)

    code, status = module.build_status(
        Namespace(project_root=str(tmp_path), index=str(index_path), strict_xr64=True, format="json")
    )

    assert code == 1
    assert status["xr64"]["checker_exit_code"] == 1
    assert status["xr64"]["ready_to_train"] is False
