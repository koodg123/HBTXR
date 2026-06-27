#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

DEFAULT_JSONL="${PROJECT_ROOT}/runs/comparison/fecet_hbtxr/infer/runtime_trace.jsonl"
DEFAULT_SUMMARY="${PROJECT_ROOT}/runs/comparison/fecet_hbtxr/infer/runtime_summary.json"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/infer.py" \
    --config "${PROJECT_ROOT}/configs/stage2_hybrid.yaml" \
    --output-jsonl "${DEFAULT_JSONL}" \
    --output-summary "${DEFAULT_SUMMARY}" \
    "$@"

