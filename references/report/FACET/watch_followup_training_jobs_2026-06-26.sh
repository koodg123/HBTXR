#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"

FPN_DW_SESSION="facet_epnet_fpn_dw_gpu0_waiter"
FPN_DW_EVAL_SESSION="facet_epnet_fpn_dw_eval_watcher"
EFFBS32_SESSION="facet_hbtxr_effbs32_gpu1_waiter"
EFFBS32_EVAL_SESSION="facet_hbtxr_effbs32_eval_watcher"

FPN_DW_LAUNCHER="$REPORT_ROOT/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh"
FPN_DW_EVAL_WATCHER="$REPORT_ROOT/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh"
EFFBS32_LAUNCHER="$REPORT_ROOT/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh"
EFFBS32_EVAL_WATCHER="$REPORT_ROOT/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh"

FPN_DW_TRAIN_LOG="$REPORT_ROOT/EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log"
EFFBS32_TRAIN_LOG="$REPORT_ROOT/HBTXR_full_unet_effbs32_gpu1_train_2026-06-26.log"
WATCHDOG_LOG="$REPORT_ROOT/FACET_followup_training_watchdog_2026-06-26.log"

HOURLY_REFRESH_GUARD="$REPORT_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"

INTERVAL_SECONDS="${FACET_FOLLOWUP_WATCHDOG_INTERVAL_SECONDS:-3600}"
MAX_LOOPS="${FACET_FOLLOWUP_WATCHDOG_MAX_LOOPS:-0}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

has_session() {
  local session="$1"
  tmux ls 2>/dev/null | awk -F: '{print $1}' | grep -Fxq "$session"
}

training_complete() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    return 1
  fi
  rg -q '`max_epochs=70` reached|max_epochs=70 reached|Trainer\.fit stopped:.*max_epochs=70.*reached' "$log_file"
}

artifact_valid() {
  local artifact_type="$1"
  local artifact_path="$2"
  "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
    --type "$artifact_type" \
    --path "$artifact_path" \
    >/dev/null 2>&1
}

fpn_dw_final_artifacts_exist() {
  artifact_valid eval "$REPORT_ROOT/FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_summary_2026-06-26.json" ]]
}

effbs32_final_artifacts_exist() {
  artifact_valid eval "$REPORT_ROOT/FACET_hbtxr_effbs32_reproduction_results_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_hbtxr_effbs32_reproduction_results_2026-06-26.md" ]] \
    && artifact_valid comparison "$REPORT_ROOT/FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_summary_2026-06-26.json" ]]
}

restart_waiter_if_needed() {
  local session="$1"
  local launcher="$2"
  local log_file="$3"
  local label="$4"
  local final_check="$5"

  if has_session "$session"; then
    echo "[$(timestamp)] $label waiter alive: $session" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  if "$final_check"; then
    echo "[$(timestamp)] $label final artifacts exist; not restarting waiter" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  if training_complete "$log_file"; then
    echo "[$(timestamp)] $label training complete; not restarting waiter" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  echo "[$(timestamp)] $label waiter missing; restarting with $launcher" | tee -a "$WATCHDOG_LOG"
  tmux new-session -d -s "$session" "bash '$launcher'"
}

restart_eval_if_needed() {
  local session="$1"
  local watcher="$2"
  local label="$3"
  local final_check="$4"

  if has_session "$session"; then
    echo "[$(timestamp)] $label eval watcher alive: $session" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  if "$final_check"; then
    echo "[$(timestamp)] $label final artifacts exist; not restarting eval watcher" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  echo "[$(timestamp)] $label eval watcher missing; restarting with $watcher" | tee -a "$WATCHDOG_LOG"
  tmux new-session -d -s "$session" "bash '$watcher'"
}

refresh_status() {
  bash "$HOURLY_REFRESH_GUARD" >>"$WATCHDOG_LOG" 2>&1 || true
}

mkdir -p "$REPORT_ROOT"
touch "$WATCHDOG_LOG"

loop=0
while true; do
  loop=$((loop + 1))
  echo "[$(timestamp)] followup watchdog loop=$loop" | tee -a "$WATCHDOG_LOG"

  restart_waiter_if_needed \
    "$FPN_DW_SESSION" \
    "$FPN_DW_LAUNCHER" \
    "$FPN_DW_TRAIN_LOG" \
    "EPNet fpn_dw" \
    fpn_dw_final_artifacts_exist
  restart_eval_if_needed \
    "$FPN_DW_EVAL_SESSION" \
    "$FPN_DW_EVAL_WATCHER" \
    "EPNet fpn_dw" \
    fpn_dw_final_artifacts_exist

  restart_waiter_if_needed \
    "$EFFBS32_SESSION" \
    "$EFFBS32_LAUNCHER" \
    "$EFFBS32_TRAIN_LOG" \
    "HBTXR effbs32" \
    effbs32_final_artifacts_exist
  restart_eval_if_needed \
    "$EFFBS32_EVAL_SESSION" \
    "$EFFBS32_EVAL_WATCHER" \
    "HBTXR effbs32" \
    effbs32_final_artifacts_exist

  refresh_status

  if [[ "$MAX_LOOPS" != "0" && "$loop" -ge "$MAX_LOOPS" ]]; then
    echo "[$(timestamp)] max loops reached; exiting followup watchdog" | tee -a "$WATCHDOG_LOG"
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done
