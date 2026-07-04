#!/bin/sh
. "$(dirname "$0")/_common.sh"
exec "$PYTHON_BIN" hardware/tools/compare_hw_sw.py --refs-root hardware/refs "$@"

