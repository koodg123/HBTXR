#!/usr/bin/env sh
set -eu

if [ -z "${SCRIPT_DIR:-}" ]; then
    echo "[ERROR] SCRIPT_DIR must be set before sourcing _bootstrap.sh" >&2
    exit 1
fi

PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[ERROR] No usable Python interpreter found." >&2
        exit 1
    fi
fi

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_ROOT/src"
fi
