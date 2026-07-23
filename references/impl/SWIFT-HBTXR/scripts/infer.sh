#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/stage2_hybrid.yaml}

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/infer.py" \
    --config "${CONFIG_PATH}" \
    "$@"

