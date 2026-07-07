#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

DEFAULT_PATHS_CONFIG="$PROJECT_ROOT/configs/paths/ev_eye_groundedsam_paths.json"

usage() {
    cat <<'USAGE'
Usage:
  sh scripts/run_prepare_and_train_mode2.sh [options]

This is a mode2 convenience wrapper for:
  sh scripts/run_prepare_and_train.sh --mode mode2 ...

Defaults:
  - always uses `--mode mode2`
  - automatically adds `--paths-config configs/paths/ev_eye_groundedsam_paths.json` when that file exists

Examples:
  sh scripts/run_prepare_and_train_mode2.sh

  sh scripts/run_prepare_and_train_mode2.sh \
    --prepare-arg=--interp-backend=timelens \
    --prepare-arg=--interp-target-fps=60

  sh scripts/run_prepare_and_train_mode2.sh \
    --skip-prepare \
    --skip-stage1 \
    --stage1-checkpoint runs/mode2_stage1_YYYYMMDD_HHMMSS/train/best_search_p10.pt

All other options are forwarded to `scripts/run_prepare_and_train.sh`.
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

DEFAULT_ARGS=(--mode mode2)
if [ -f "$DEFAULT_PATHS_CONFIG" ]; then
    DEFAULT_ARGS+=(--paths-config "$DEFAULT_PATHS_CONFIG")
fi

exec sh "$SCRIPT_DIR/run_prepare_and_train.sh" "${DEFAULT_ARGS[@]}" "$@"
