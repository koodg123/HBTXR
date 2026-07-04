#!/bin/sh
. "$(dirname "$0")/_common.sh"
exec "$PYTHON_BIN" software/tools/infer.py "$@"

