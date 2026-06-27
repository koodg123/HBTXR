#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/runs/comparison/fecet_hbtxr/stage2"
DEFAULT_EXPERIMENT_NAME="${EXPERIMENT_NAME:-fecet_compare_stage2}"

if [ -n "${STAGE1_CHECKPOINT:-}" ]; then
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
        --config "${PROJECT_ROOT}/configs/stage2_hybrid.yaml" \
        --stage stage2 \
        --output-dir "${DEFAULT_OUTPUT_DIR}" \
        --experiment-name "${DEFAULT_EXPERIMENT_NAME}" \
        --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
        "$@"
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
    --config "${PROJECT_ROOT}/configs/stage2_hybrid.yaml" \
    --stage stage2 \
    --output-dir "${DEFAULT_OUTPUT_DIR}" \
    --experiment-name "${DEFAULT_EXPERIMENT_NAME}" \
    "$@"

