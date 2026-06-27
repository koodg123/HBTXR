#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/facet_train.py" "$@"

