#!/bin/sh
. "$(dirname "$0")/_common.sh"
exec "$PYTHON_BIN" hardware/tools/overlay_preview.py "$@"

