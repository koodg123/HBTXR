#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/runs/comparison/fecet_hbtxr/stage1"
DEFAULT_EXPERIMENT_NAME="${EXPERIMENT_NAME:-fecet_compare_stage1}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
    --config "${PROJECT_ROOT}/configs/stage1_search.yaml" \
    --stage stage1 \
    --output-dir "${DEFAULT_OUTPUT_DIR}" \
    --experiment-name "${DEFAULT_EXPERIMENT_NAME}" \
    "$@"

