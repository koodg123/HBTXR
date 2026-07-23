#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/stage2_hybrid.yaml}

if [ -n "${STAGE1_CHECKPOINT:-}" ]; then
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
        --config "${CONFIG_PATH}" \
        --stage stage2 \
        --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
        "$@"
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/train.py" \
    --config "${CONFIG_PATH}" \
    --stage stage2 \
    "$@"

