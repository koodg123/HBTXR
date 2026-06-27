#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
EVAL_SCRIPT="$REPORT_ROOT/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh"
PY="$ROOT/.facet-train-venv/bin/python"
HOURLY_REFRESH_GUARD="$REPORT_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"

INTERVAL_SECONDS="${FACET_FPN_DW_WATCH_INTERVAL_SECONDS:-3600}"
MAX_LOOPS="${FACET_FPN_DW_WATCH_MAX_LOOPS:-0}"
REQUIRE_COMPLETED="${FACET_FPN_DW_WATCH_REQUIRE_COMPLETED:-1}"
LOG="$REPORT_ROOT/FACET_epnet_fpn_dw_checkpoint_watch_2026-06-26.log"

CKPT_ROOT="$FACET_ROOT/runs/logs/EPNet_fpn_dw_full_unet"
TRAIN_LOG="$REPORT_ROOT/EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

ckpt_count() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    echo 0
    return 0
  fi
  find "$root" -path '*/checkpoints/*.ckpt' -type f \
    ! -path '*/step_checkpoints/*' \
    | wc -l
}

latest_ckpt() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -path '*/checkpoints/*.ckpt' -type f \
    ! -path '*/step_checkpoints/*' \
    ! -name 'last.ckpt' -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
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
  artifact_valid eval "$REPORT_ROOT/FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json" \
    && [[ -f "$REPORT_ROOT/FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.md" ]] \
    && [[ -f "$REPORT_ROOT/FACET_reproduction_summary_2026-06-26.json" ]]
}

loop=0
mkdir -p "$REPORT_ROOT"
touch "$LOG"

while true; do
  loop=$((loop + 1))
  count="$(ckpt_count "$CKPT_ROOT")"
  latest="$(latest_ckpt "$CKPT_ROOT" || true)"
  done_flag=0
  if training_complete "$TRAIN_LOG"; then
    done_flag=1
  fi

  {
    echo "[$(timestamp)] loop=$loop fpn_dw_ckpt_count=$count fpn_dw_done=$done_flag require_completed=$REQUIRE_COMPLETED"
    echo "  fpn_dw_latest=${latest:-missing}"
  } | tee -a "$LOG"

  bash "$HOURLY_REFRESH_GUARD" >>"$LOG" 2>&1 || true

  if final_artifacts_exist; then
    echo "[$(timestamp)] fpn_dw final artifacts already exist; exiting" | tee -a "$LOG"
    exit 0
  fi

  ready_for_eval=0
  if [[ -n "${latest:-}" ]]; then
    if [[ "$REQUIRE_COMPLETED" == "0" || "$done_flag" == "1" ]]; then
      ready_for_eval=1
    fi
  fi

  if [[ "$ready_for_eval" == "1" ]]; then
    echo "[$(timestamp)] fpn_dw checkpoint ready; running evaluation" | tee -a "$LOG"
    "$EVAL_SCRIPT" 2>&1 | tee -a "$LOG"
    echo "[$(timestamp)] fpn_dw evaluation completed" | tee -a "$LOG"
    exit 0
  fi

  if [[ "$MAX_LOOPS" != "0" && "$loop" -ge "$MAX_LOOPS" ]]; then
    echo "[$(timestamp)] max loops reached before fpn_dw checkpoint was ready" | tee -a "$LOG"
    exit 3
  fi

  sleep "$INTERVAL_SECONDS"
done
