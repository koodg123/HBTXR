#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"

check_waiter() {
  local script="$1"
  local wait_env="$2"
  local baseline_log="$3"
  local wait_message="$4"

  if ! rg -q "${wait_env}:-1" "$script"; then
    echo "$script does not default $wait_env to require baseline completion" >&2
    exit 1
  fi

  if ! rg -q "training_complete\\(\\)" "$script"; then
    echo "$script does not define training_complete()" >&2
    exit 1
  fi

  if ! rg -q "$baseline_log" "$script"; then
    echo "$script does not point at expected baseline log $baseline_log" >&2
    exit 1
  fi

  if ! rg -q "$wait_message" "$script"; then
    echo "$script does not log the expected baseline wait message" >&2
    exit 1
  fi

  if ! rg -q 'while ! training_complete "\$\{BASELINE_LOG\}"' "$script"; then
    echo "$script does not loop until the baseline completion marker exists" >&2
    exit 1
  fi
}

check_waiter \
  "$REPORT_ROOT/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh" \
  "FACET_FPN_DW_WAIT_BASELINE_COMPLETE" \
  "EPNet_full_unet_gpu0_train_2026-06-26.log" \
  "waiting for baseline EPNet completion marker"

check_waiter \
  "$REPORT_ROOT/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh" \
  "FACET_EFFBS32_WAIT_BASELINE_COMPLETE" \
  "HBTXR_full_unet_gpu1_train_2026-06-26.log" \
  "waiting for baseline HBTXR completion marker"

echo "follow-up waiter completion gate smoke passed"
