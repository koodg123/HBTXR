#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"

export PYTHONPATH="$FACET_ROOT"
export PYTHONPYCACHEPREFIX=/tmp/facet_validation_smoke_suite_pycache

echo "[1/4] Python syntax checks"
"$PY" -m py_compile \
  "$FACET_ROOT/EvEye/utils/scripts/build_reproduction_summary.py" \
  "$FACET_ROOT/EvEye/utils/scripts/check_reproduction_status.py" \
  "$FACET_ROOT/EvEye/utils/scripts/compare_model_evaluation_results.py" \
  "$FACET_ROOT/EvEye/utils/scripts/sync_reproduction_status_summary.py" \
  "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
  "$REPORT_ROOT/audit_reproduction_completion_2026-06-26.py" \
  "$REPORT_ROOT/summarize_missing_gates_2026-06-26.py"

echo "[2/4] Shell syntax checks"
bash -n \
  "$REPORT_ROOT/run_full_checkpoint_evaluation_2026-06-26.sh" \
  "$REPORT_ROOT/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh" \
  "$REPORT_ROOT/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh" \
  "$REPORT_ROOT/watch_full_training_jobs_2026-06-26.sh" \
  "$REPORT_ROOT/watch_full_checkpoints_and_evaluate_2026-06-26.sh" \
  "$REPORT_ROOT/watch_followup_training_jobs_2026-06-26.sh" \
  "$REPORT_ROOT/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh" \
  "$REPORT_ROOT/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh" \
  "$REPORT_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh" \
  "$REPORT_ROOT/run_next_hourly_refresh_once_2026-06-26.sh" \
  "$REPORT_ROOT/start_next_hourly_refresh_once_2026-06-26.sh" \
  "$REPORT_ROOT/run_guarded_missing_gate_summary_2026-06-26.sh" \
  "$REPORT_ROOT/check_goal_completion_after_guard_2026-06-26.sh" \
  "$REPORT_ROOT/test_artifact_validation_smoke_2026-06-26.sh" \
  "$REPORT_ROOT/test_pairwise_input_validation_smoke_2026-06-26.sh" \
  "$REPORT_ROOT/test_training_completion_marker_2026-06-26.sh" \
  "$REPORT_ROOT/test_evaluation_runner_completion_gate_2026-06-26.sh" \
  "$REPORT_ROOT/test_hourly_refresh_guard_routing_2026-06-26.sh" \
  "$REPORT_ROOT/test_hourly_guard_skip_order_2026-06-26.sh" \
  "$REPORT_ROOT/test_monitoring_interval_defaults_2026-06-26.sh" \
  "$REPORT_ROOT/test_next_hourly_refresh_once_2026-06-26.sh" \
  "$REPORT_ROOT/test_missing_gate_summary_2026-06-26.sh" \
  "$REPORT_ROOT/test_guarded_missing_gate_summary_2026-06-26.sh" \
  "$REPORT_ROOT/test_goal_completion_check_2026-06-26.sh" \
  "$REPORT_ROOT/test_followup_waiter_completion_gate_2026-06-26.sh" \
  "$REPORT_ROOT/test_completion_audit_fail_gate_2026-06-26.sh" \
  "$REPORT_ROOT/test_completion_audit_pass_gate_2026-06-26.sh"

echo "[3/4] Artifact validation smoke"
"$REPORT_ROOT/test_artifact_validation_smoke_2026-06-26.sh"

echo "[4/4] Pairwise, completion marker, and routing smokes"
"$REPORT_ROOT/test_pairwise_input_validation_smoke_2026-06-26.sh"
"$REPORT_ROOT/test_training_completion_marker_2026-06-26.sh"
"$REPORT_ROOT/test_evaluation_runner_completion_gate_2026-06-26.sh"
"$REPORT_ROOT/test_hourly_refresh_guard_routing_2026-06-26.sh"
"$REPORT_ROOT/test_hourly_guard_skip_order_2026-06-26.sh"
"$REPORT_ROOT/test_monitoring_interval_defaults_2026-06-26.sh"
"$REPORT_ROOT/test_next_hourly_refresh_once_2026-06-26.sh"
"$REPORT_ROOT/test_missing_gate_summary_2026-06-26.sh"
"$REPORT_ROOT/test_guarded_missing_gate_summary_2026-06-26.sh"
"$REPORT_ROOT/test_goal_completion_check_2026-06-26.sh"
"$REPORT_ROOT/test_followup_waiter_completion_gate_2026-06-26.sh"
"$REPORT_ROOT/test_completion_audit_fail_gate_2026-06-26.sh"
"$REPORT_ROOT/test_completion_audit_pass_gate_2026-06-26.sh"

echo "FACET validation smoke suite passed"
