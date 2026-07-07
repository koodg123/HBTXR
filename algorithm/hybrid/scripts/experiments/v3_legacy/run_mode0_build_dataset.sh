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
  sh exps/scripts/run_mode0_build_dataset.sh [options]

Alias for:
  sh exps/scripts/run_mode0_prepare.sh [options]

Purpose:
  - build the mode0 canonical dataset
  - build the mode0 manifests
USAGE
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 0
fi

exec sh "$SCRIPT_DIR/run_mode0_prepare.sh" "$@"
