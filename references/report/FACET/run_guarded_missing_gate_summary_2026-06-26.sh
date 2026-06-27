#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
GUARD="$REPORT_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"
SUMMARY="$REPORT_ROOT/summarize_missing_gates_2026-06-26.py"
OUTPUT_MD="$REPORT_ROOT/FACET_missing_gate_summary_2026-06-26.md"
LOG="$REPORT_ROOT/FACET_guarded_missing_gate_summary_2026-06-26.log"
SKIP_HOURLY_GUARD="${FACET_SKIP_HOURLY_GUARD:-0}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mkdir -p "$REPORT_ROOT"
touch "$LOG"

if [[ "$SKIP_HOURLY_GUARD" == "1" ]]; then
  echo "[$(timestamp)] skipping hourly status refresh guard by FACET_SKIP_HOURLY_GUARD=1" | tee -a "$LOG"
else
  echo "[$(timestamp)] running hourly status refresh guard" | tee -a "$LOG"
  bash "$GUARD" 2>&1 | tee -a "$LOG"
fi

echo "[$(timestamp)] writing no-log missing gate summary: $OUTPUT_MD" | tee -a "$LOG"
"$PY" "$SUMMARY" --output-md "$OUTPUT_MD" 2>&1 | tee -a "$LOG"

echo "[$(timestamp)] guarded missing gate summary complete" | tee -a "$LOG"
