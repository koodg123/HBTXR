#!/bin/sh
. "$(dirname "$0")/_common.sh"
"$PYTHON_BIN" hardware/tools/export_test_vectors.py --output-root hardware/refs
"$PYTHON_BIN" hardware/tools/export_weights.py --output-root hardware/refs/weights

