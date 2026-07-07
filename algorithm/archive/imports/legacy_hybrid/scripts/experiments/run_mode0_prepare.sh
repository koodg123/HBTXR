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
  sh exps/scripts/run_mode0_prepare.sh [options]

This is a mode0 convenience wrapper for:
  sh scripts/run_prepare.sh --annotation-mode manual_csv --data-mode mode0 ...

Purpose:
  - canonical generation
  - manifest generation
  - raw-CSV-based mode0 dataset preparation

Defaults:
  - always uses `--annotation-mode manual_csv`
  - always uses `--data-mode mode0`
  - always uses `--canonical-name canonical0`
  - always uses `--manifest-name manifest0`
  - always uses `--frame-source original`

All other options are forwarded to `scripts/run_prepare.sh`.
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

DEFAULT_ARGS=(
    --annotation-mode manual_csv
    --data-mode mode0
    --canonical-name canonical0
    --manifest-name manifest0
    --frame-source original
)

exec sh "$PROJECT_ROOT/scripts/run_prepare.sh" "${DEFAULT_ARGS[@]}" "$@"
