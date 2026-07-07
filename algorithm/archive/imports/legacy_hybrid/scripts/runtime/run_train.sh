#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    exec bash "$SCRIPT_DIR/hbtxr_mode_pipeline.sh" --help
fi
exec bash "$SCRIPT_DIR/hbtxr_mode_pipeline.sh" train "$@"
