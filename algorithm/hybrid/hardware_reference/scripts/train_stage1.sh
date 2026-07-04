#!/bin/sh
. "$(dirname "$0")/_common.sh"
exec "$PYTHON_BIN" software/tools/train.py --config software/configs/stage1_search.yaml "$@"

