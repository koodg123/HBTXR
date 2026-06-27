#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/project/PRJXR-HBTXR/HBTXR"
FACET_DIR="$ROOT/references/codebase/software/FACET"
DATASET_ROOT="/mnt/e/DATASET/DeanDataset_full_unet"
CONFIG_NAME="DavisEyeEllipse_EPNet_local_deandataset_full_unet.yaml"
VENV_PY="$FACET_DIR/.venv/bin/python"
REPORT_DIR="$ROOT/references/report/FACET"
WATCH_LOG="$REPORT_DIR/EPNet_local_deandataset_full_unet_waiter_2026-06-27.log"
TRAIN_LOG="$REPORT_DIR/EPNet_local_deandataset_full_unet_train_2026-06-27.log"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
REQUIRED_STABLE_CHECKS="${REQUIRED_STABLE_CHECKS:-3}"

mkdir -p "$REPORT_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

require_path() {
  local path="$1"
  [[ -e "$path" ]]
}

count_dataset_files() {
  find "$DATASET_ROOT" -type f 2>/dev/null | wc -l
}

dataset_ready() {
  require_path "$DATASET_ROOT/train/cached_data" || return 1
  require_path "$DATASET_ROOT/train/cached_ellipse" || return 1
  require_path "$DATASET_ROOT/val/cached_data" || return 1
  require_path "$DATASET_ROOT/val/cached_ellipse" || return 1
  find "$DATASET_ROOT/train/cached_data" -name 'events_batch_*.memmap' -print -quit | grep -q .
  find "$DATASET_ROOT/train/cached_data" -name 'events_indices_*.npy' -print -quit | grep -q .
  find "$DATASET_ROOT/train/cached_ellipse" -name 'ellipses_batch_*.memmap' -print -quit | grep -q .
  find "$DATASET_ROOT/train/cached_ellipse" -name 'ellipses_indices_*.npy' -print -quit | grep -q .
  find "$DATASET_ROOT/val/cached_data" -name 'events_batch_*.memmap' -print -quit | grep -q .
  find "$DATASET_ROOT/val/cached_data" -name 'events_indices_*.npy' -print -quit | grep -q .
  find "$DATASET_ROOT/val/cached_ellipse" -name 'ellipses_batch_*.memmap' -print -quit | grep -q .
  find "$DATASET_ROOT/val/cached_ellipse" -name 'ellipses_indices_*.npy' -print -quit | grep -q .
}

main() {
  log "waiter started"
  log "dataset_root=$DATASET_ROOT"
  log "config=$FACET_DIR/configs/$CONFIG_NAME"

  if [[ ! -x "$VENV_PY" ]]; then
    log "missing venv python: $VENV_PY"
    exit 2
  fi

  local last_count=""
  local stable_checks=0
  while true; do
    if dataset_ready; then
      local current_count
      current_count="$(count_dataset_files)"
      if [[ "$current_count" == "$last_count" ]]; then
        stable_checks=$((stable_checks + 1))
      else
        stable_checks=0
        last_count="$current_count"
      fi
      log "dataset structure ready; file_count=$current_count; stable_checks=$stable_checks/$REQUIRED_STABLE_CHECKS"
      if (( stable_checks >= REQUIRED_STABLE_CHECKS )); then
        break
      fi
    else
      local current_count
      current_count="$(count_dataset_files || true)"
      log "dataset not ready yet; observed_file_count=$current_count"
      stable_checks=0
      last_count="$current_count"
    fi
    sleep "$CHECK_INTERVAL_SECONDS"
  done

  log "dataset ready and stable; starting training"
  cd "$FACET_DIR"
  export PYTHONPATH="$FACET_DIR"
  export NO_ALBUMENTATIONS_UPDATE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-facet-local}"
  export FACET_DEVICES="${FACET_DEVICES:-0}"
  exec "$VENV_PY" tools/train.py -c "$CONFIG_NAME" >> "$TRAIN_LOG" 2>&1
}

main "$@"
