#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
SCRIPT="$REPORT_ROOT/run_guarded_missing_gate_summary_2026-06-26.sh"

if ! rg -q 'run_hourly_status_refresh_guard_2026-06-26\.sh' "$SCRIPT"; then
  echo "guarded summary wrapper does not call hourly guard" >&2
  exit 1
fi

if ! rg -q 'FACET_SKIP_HOURLY_GUARD' "$SCRIPT"; then
  echo "guarded summary wrapper does not expose the hourly guard skip option" >&2
  exit 1
fi

if rg -q -- '--force' "$SCRIPT"; then
  echo "guarded summary wrapper must not force the hourly guard" >&2
  exit 1
fi

if ! rg -q 'summarize_missing_gates_2026-06-26\.py' "$SCRIPT"; then
  echo "guarded summary wrapper does not call missing gate summary" >&2
  exit 1
fi

if ! rg -q -- '--output-md "\$OUTPUT_MD"' "$SCRIPT"; then
  echo "guarded summary wrapper does not write the summary Markdown artifact" >&2
  exit 1
fi

echo "guarded missing gate summary smoke passed"
