#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"
LOG="$REPORT_ROOT/HBTXR_batch_probe_gpu0_2026-06-26.log"
OUT_JSON="$REPORT_ROOT/HBTXR_batch_probe_gpu0_2026-06-26.json"

INTERVAL_SECONDS="${FACET_HBTXR_PROBE_INTERVAL_SECONDS:-3600}"
MAX_LOOPS="${FACET_HBTXR_PROBE_MAX_LOOPS:-0}"
BATCH_SIZES="${FACET_HBTXR_PROBE_BATCH_SIZES:-2,4,6,8}"
STEPS="${FACET_HBTXR_PROBE_STEPS:-2}"

mkdir -p "$REPORT_ROOT"
touch "$LOG"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

gpu0_has_compute_app() {
  local apps
  if ! apps="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name --format=csv,noheader 2>/dev/null)"; then
    # Fail closed. If GPU state cannot be read, do not start an extra probe.
    return 0
  fi
  [[ "$apps" == *"00000000:02:00.0,"* ]]
}

loop=0
while true; do
  loop=$((loop + 1))
  if gpu0_has_compute_app; then
    echo "[$(timestamp)] GPU0 busy; waiting before HBTXR batch probe" | tee -a "$LOG"
  else
    echo "[$(timestamp)] GPU0 free; starting HBTXR batch probe" | tee -a "$LOG"
    cd "$FACET_ROOT"
    export PYTHONPATH=.
    export FACET_DISABLE_CUDNN=1
    export NO_ALBUMENTATIONS_UPDATE=1
    "$PY" EvEye/utils/scripts/probe_hbtxr_batch_size.py \
      --config DavisEyeEllipse_HBTXR_full_unet.yaml \
      --device cuda:0 \
      --batch-sizes "$BATCH_SIZES" \
      --steps "$STEPS" \
      --num-workers 0 \
      --output-json "$OUT_JSON" \
      2>&1 | tee -a "$LOG"
    echo "[$(timestamp)] HBTXR batch probe completed: $OUT_JSON" | tee -a "$LOG"
    exit 0
  fi

  if [[ "$MAX_LOOPS" != "0" && "$loop" -ge "$MAX_LOOPS" ]]; then
    echo "[$(timestamp)] max loops reached before GPU0 became free" | tee -a "$LOG"
    exit 3
  fi
  sleep "$INTERVAL_SECONDS"
done
