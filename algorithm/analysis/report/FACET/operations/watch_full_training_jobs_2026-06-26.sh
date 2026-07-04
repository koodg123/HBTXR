#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"

EP_SESSION="facet_epnet_full_gpu0"
HB_SESSION="facet_hbtxr_full_gpu1"
EVAL_SESSION="facet_full_eval_watcher"

EP_LAUNCHER="$OPERATIONS_ROOT/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh"
HB_LAUNCHER="$OPERATIONS_ROOT/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh"
EVAL_WATCHER="$OPERATIONS_ROOT/watch_full_checkpoints_and_evaluate_2026-06-26.sh"

EP_TRAIN_LOG="$REPORT_ROOT/EPNet_full_unet_gpu0_train_2026-06-26.log"
HB_TRAIN_LOG="$REPORT_ROOT/HBTXR_full_unet_gpu1_train_2026-06-26.log"
WATCHDOG_LOG="$REPORT_ROOT/FACET_full_training_watchdog_2026-06-26.log"

HOURLY_REFRESH_GUARD="$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"

INTERVAL_SECONDS="${FACET_TRAINING_WATCHDOG_INTERVAL_SECONDS:-3600}"
MAX_LOOPS="${FACET_TRAINING_WATCHDOG_MAX_LOOPS:-0}"
START_EVAL_WATCHER="${FACET_TRAINING_WATCHDOG_START_EVAL_WATCHER:-1}"

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

final_artifacts_exist() {
  artifact_valid eval "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_summary_2026-06-26.json" ]] \
    && artifact_valid eval "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.md" ]] \
    && artifact_valid comparison "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_table2_comparison_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.md" ]]
}

restart_if_needed() {
  local session="$1"
  local launcher="$2"
  local log_file="$3"
  local label="$4"

  if has_session "$session"; then
    echo "[$(timestamp)] $label session alive: $session" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  if training_complete "$log_file"; then
    echo "[$(timestamp)] $label training complete; not restarting $session" | tee -a "$WATCHDOG_LOG"
    return 0
  fi

  echo "[$(timestamp)] $label session missing; restarting with $launcher" | tee -a "$WATCHDOG_LOG"
  tmux new-session -d -s "$session" "bash '$launcher'"
}

refresh_status() {
  bash "$HOURLY_REFRESH_GUARD" >>"$WATCHDOG_LOG" 2>&1 || true
}

mkdir -p "$REPORT_ROOT"
touch "$WATCHDOG_LOG"

loop=0
while true; do
  loop=$((loop + 1))
  echo "[$(timestamp)] watchdog loop=$loop" | tee -a "$WATCHDOG_LOG"

  restart_if_needed "$EP_SESSION" "$EP_LAUNCHER" "$EP_TRAIN_LOG" "EPNet"
  restart_if_needed "$HB_SESSION" "$HB_LAUNCHER" "$HB_TRAIN_LOG" "HBTXR"

  if [[ "$START_EVAL_WATCHER" == "1" ]]; then
    if has_session "$EVAL_SESSION"; then
      echo "[$(timestamp)] evaluation watcher alive: $EVAL_SESSION" | tee -a "$WATCHDOG_LOG"
    elif final_artifacts_exist; then
      echo "[$(timestamp)] final artifacts exist; not restarting evaluation watcher" | tee -a "$WATCHDOG_LOG"
    else
      echo "[$(timestamp)] evaluation watcher missing; restarting" | tee -a "$WATCHDOG_LOG"
      tmux new-session -d -s "$EVAL_SESSION" "bash '$EVAL_WATCHER'"
    fi
  fi

  refresh_status

  if [[ "$MAX_LOOPS" != "0" && "$loop" -ge "$MAX_LOOPS" ]]; then
    echo "[$(timestamp)] max loops reached; exiting watchdog" | tee -a "$WATCHDOG_LOG"
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done
