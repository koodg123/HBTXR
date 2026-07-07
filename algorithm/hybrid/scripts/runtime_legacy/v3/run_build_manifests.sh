#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
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

PATHS_CONFIG_DEFAULT="$PROJECT_ROOT/configs/paths/ev_eye_groundedsam_paths.json"
DEFAULT_ARGS=()
if [ -f "$PATHS_CONFIG_DEFAULT" ]; then
    DEFAULT_ARGS+=(--paths-config "$PATHS_CONFIG_DEFAULT")
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/build_mode_manifests.py" "${DEFAULT_ARGS[@]}" "$@"
