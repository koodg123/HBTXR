#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/_bootstrap.sh"

exec "$PYTHON_BIN" "$SCRIPT_DIR/build_target_fps_canonical.py" "$@"
