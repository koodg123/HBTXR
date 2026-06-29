#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"

check_runner() {
  local script="$1"
  local env_name="$2"
  local log_name="$3"

  if ! rg -q "${env_name}:-1" "$script"; then
    echo "$script does not default $env_name to require completion" >&2
    exit 1
  fi
  if ! rg -q "training_complete\\(\\)" "$script"; then
    echo "$script does not define training_complete()" >&2
    exit 1
  fi
  if ! rg -q "refusing final evaluation without completion marker" "$script"; then
    echo "$script does not refuse final evaluation without completion marker" >&2
    exit 1
  fi
  if ! rg -q "$log_name" "$script"; then
    echo "$script does not check expected training log $log_name" >&2
    exit 1
  fi
  if ! rg -q 'sync_reproduction_status_summary\.py' "$script"; then
    echo "$script does not synchronize status after final evaluation" >&2
    exit 1
  fi
  if ! rg -q 'audit_reproduction_completion_2026-06-26\.py' "$script"; then
    echo "$script does not refresh completion audit after final evaluation" >&2
    exit 1
  fi
  if ! rg -q 'FACET_reproduction_completion_audit_2026-06-26\.json' "$script"; then
    echo "$script does not write the expected completion audit JSON" >&2
    exit 1
  fi
}

check_runner \
  "$OPERATIONS_ROOT/run_full_checkpoint_evaluation_2026-06-26.sh" \
  "FACET_FULL_EVAL_REQUIRE_COMPLETED" \
  "EPNet_full_unet_gpu0_train_2026-06-26.log"
check_runner \
  "$OPERATIONS_ROOT/run_full_checkpoint_evaluation_2026-06-26.sh" \
  "FACET_FULL_EVAL_REQUIRE_COMPLETED" \
  "HBTXR_full_unet_gpu1_train_2026-06-26.log"
check_runner \
  "$OPERATIONS_ROOT/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh" \
  "FACET_FPN_DW_EVAL_REQUIRE_COMPLETED" \
  "EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log"
check_runner \
  "$OPERATIONS_ROOT/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh" \
  "FACET_EFFBS32_EVAL_REQUIRE_COMPLETED" \
  "HBTXR_full_unet_effbs32_gpu1_train_2026-06-26.log"

echo "evaluation runner completion gate smoke passed"
