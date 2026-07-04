#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
EVAL_SCRIPT="$OPERATIONS_ROOT/run_full_checkpoint_evaluation_2026-06-26.sh"
PY="$ROOT/.facet-train-venv/bin/python"
HOURLY_REFRESH_GUARD="$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"

INTERVAL_SECONDS="${FACET_WATCH_INTERVAL_SECONDS:-3600}"
MAX_LOOPS="${FACET_WATCH_MAX_LOOPS:-0}"
REQUIRE_COMPLETED="${FACET_WATCH_REQUIRE_COMPLETED:-1}"
LOG="$REPORT_ROOT/FACET_full_checkpoint_watch_2026-06-26.log"

EP_CKPT_ROOT="$FACET_ROOT/runs/logs/EPNet_full_unet"
HB_CKPT_ROOT="$FACET_ROOT/runs/logs/HBTXR_full_unet"
EP_TRAIN_LOG="$REPORT_ROOT/EPNet_full_unet_gpu0_train_2026-06-26.log"
HB_TRAIN_LOG="$REPORT_ROOT/HBTXR_full_unet_gpu1_train_2026-06-26.log"

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

loop=0
mkdir -p "$REPORT_ROOT"
touch "$LOG"

while true; do
  loop=$((loop + 1))
  ep_count="$(ckpt_count "$EP_CKPT_ROOT")"
  hb_count="$(ckpt_count "$HB_CKPT_ROOT")"
  ep_latest="$(latest_ckpt "$EP_CKPT_ROOT" || true)"
  hb_latest="$(latest_ckpt "$HB_CKPT_ROOT" || true)"
  ep_done=0
  hb_done=0
  if training_complete "$EP_TRAIN_LOG"; then
    ep_done=1
  fi
  if training_complete "$HB_TRAIN_LOG"; then
    hb_done=1
  fi

  {
    echo "[$(timestamp)] loop=$loop ep_ckpt_count=$ep_count hb_ckpt_count=$hb_count ep_done=$ep_done hb_done=$hb_done require_completed=$REQUIRE_COMPLETED"
    echo "  ep_latest=${ep_latest:-missing}"
    echo "  hb_latest=${hb_latest:-missing}"
  } | tee -a "$LOG"

  bash "$HOURLY_REFRESH_GUARD" >>"$LOG" 2>&1 || true

  ready_for_eval=0
  if [[ -n "${ep_latest:-}" && -n "${hb_latest:-}" ]]; then
    if [[ "$REQUIRE_COMPLETED" == "0" || "$ep_done" == "1" && "$hb_done" == "1" ]]; then
      ready_for_eval=1
    fi
  fi

  if [[ "$ready_for_eval" == "1" ]]; then
    echo "[$(timestamp)] both checkpoints found; running evaluation" | tee -a "$LOG"
    "$EVAL_SCRIPT" 2>&1 | tee -a "$LOG"
    echo "[$(timestamp)] evaluation completed" | tee -a "$LOG"
    exit 0
  fi

  if [[ "$MAX_LOOPS" != "0" && "$loop" -ge "$MAX_LOOPS" ]]; then
    echo "[$(timestamp)] max loops reached before both checkpoints existed" | tee -a "$LOG"
    exit 3
  fi

  sleep "$INTERVAL_SECONDS"
done
