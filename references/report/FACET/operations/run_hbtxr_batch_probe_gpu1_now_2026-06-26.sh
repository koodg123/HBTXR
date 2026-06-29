#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"
LOG="$REPORT_ROOT/HBTXR_batch_probe_gpu1_2026-06-26.log"
OUT_JSON="${FACET_HBTXR_PROBE_OUTPUT_JSON:-$REPORT_ROOT/HBTXR_batch_probe_gpu1_2026-06-26.json}"

BATCH_SIZES="${FACET_HBTXR_PROBE_BATCH_SIZES:-2,4,8,12,16}"
STEPS="${FACET_HBTXR_PROBE_STEPS:-3}"
PRECISION="${FACET_HBTXR_PROBE_PRECISION:-fp32}"

mkdir -p "$REPORT_ROOT"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

{
  echo "[$(timestamp)] starting HBTXR batch probe on GPU1"
  cd "$FACET_ROOT"
  export PYTHONPATH=.
  export FACET_DISABLE_CUDNN=1
  export NO_ALBUMENTATIONS_UPDATE=1
  export PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_probe_gpu1_pycache
  "$PY" EvEye/utils/scripts/probe_hbtxr_batch_size.py \
    --config DavisEyeEllipse_HBTXR_full_unet.yaml \
    --device cuda:1 \
    --batch-sizes "$BATCH_SIZES" \
    --steps "$STEPS" \
    --num-workers 0 \
    --precision "$PRECISION" \
    --output-json "$OUT_JSON"
  echo "[$(timestamp)] HBTXR batch probe completed: $OUT_JSON"
} 2>&1 | tee -a "$LOG"
