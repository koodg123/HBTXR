#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/_bootstrap.sh"

usage() {
    cat <<'USAGE'
Usage:
  sh exps/scripts/run_mode0_train_stage1.sh [options]

This is a mode0 Stage1 convenience wrapper for:
  sh scripts/run_train.sh --config exps/configs/mode0_stage1.yaml --mode mode0 --stage stage1 ...

Defaults:
  - always uses `exps/configs/mode0_stage1.yaml`
  - always uses `--mode mode0`
  - always uses `--stage stage1`
  - uses exactly two GPUs by default via `--device cuda:0,cuda:1`

Examples:
  sh exps/scripts/run_mode0_train_stage1.sh

  sh exps/scripts/run_mode0_train_stage1.sh \
    --experiment-name mode0_stage1_debug \
    --override training.epochs=5

All other options are forwarded to `scripts/run_train.sh`.
Pass `--device` explicitly if you need to override the default two-GPU setting.
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

DEFAULT_ARGS=(
    --config exps/configs/mode0_stage1.yaml
    --mode mode0
    --stage stage1
    --device cuda:0,cuda:1
)

exec bash "$PROJECT_ROOT/scripts/run_train.sh" "${DEFAULT_ARGS[@]}" "$@"
