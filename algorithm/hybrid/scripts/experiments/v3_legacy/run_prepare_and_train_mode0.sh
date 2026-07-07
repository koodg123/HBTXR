#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/_bootstrap.sh"

DEFAULT_PATHS_CONFIG="$PROJECT_ROOT/configs/paths/ev_eye_groundedsam_paths.json"

usage() {
    cat <<'USAGE'
Usage:
  sh exps/scripts/run_prepare_and_train_mode0.sh [options]

This is a mode0 convenience wrapper for:
  sh scripts/run_prepare_and_train.sh --mode mode0 --annotation-mode manual_csv ...

Defaults:
  - always uses `--mode mode0`
  - always uses `--annotation-mode manual_csv`
  - always uses `--stage1-device cuda:0,cuda:1`
  - always uses `--stage2-device cuda:0,cuda:1`
  - automatically adds `--paths-config configs/paths/ev_eye_groundedsam_paths.json` when that file exists

Examples:
  sh exps/scripts/run_prepare_and_train_mode0.sh

  sh exps/scripts/run_prepare_and_train_mode0.sh \
    --skip-prepare \
    --skip-stage1 \
    --stage1-checkpoint runs/mode0_stage1_YYYYMMDD_HHMMSS/train/best_search_p10.pt

All other options are forwarded to `scripts/run_prepare_and_train.sh`.
Pass `--stage1-device` or `--stage2-device` explicitly if you need to override the default two-GPU setting.
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

DEFAULT_ARGS=(
    --mode mode0
    --annotation-mode manual_csv
    --stage1-config exps/configs/mode0_stage1.yaml
    --stage2-config exps/configs/mode0_stage2.yaml
    --stage1-device cuda:0,cuda:1
    --stage2-device cuda:0,cuda:1
)
if [ -f "$DEFAULT_PATHS_CONFIG" ]; then
    DEFAULT_ARGS+=(--paths-config "$DEFAULT_PATHS_CONFIG")
fi

exec sh "$PROJECT_ROOT/scripts/run_prepare_and_train.sh" "${DEFAULT_ARGS[@]}" "$@"
