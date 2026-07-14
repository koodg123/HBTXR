#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

DEFAULT_OUTPUT="${PROJECT_ROOT}/runs/comparison/fecet_hbtxr/eval/fecet_eval_summary.json"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/eval.py" \
    --config "${PROJECT_ROOT}/configs/stage2_hybrid.yaml" \
    --output "${DEFAULT_OUTPUT}" \
    "$@"

