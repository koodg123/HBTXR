#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
GUARDED_SUMMARY="$REPORT_ROOT/run_guarded_missing_gate_summary_2026-06-26.sh"
AUDIT_JSON="$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.json"
LOG="$REPORT_ROOT/FACET_goal_completion_check_2026-06-26.log"
ALLOW_INCOMPLETE="${FACET_GOAL_CHECK_ALLOW_INCOMPLETE:-0}"
SKIP_GUARDED_SUMMARY="${FACET_SKIP_GUARDED_SUMMARY:-0}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mkdir -p "$REPORT_ROOT"
touch "$LOG"

if [[ "$SKIP_GUARDED_SUMMARY" == "1" ]]; then
  echo "[$(timestamp)] skipping guarded summary by FACET_SKIP_GUARDED_SUMMARY=1" | tee -a "$LOG"
else
  echo "[$(timestamp)] running guarded summary before completion check" | tee -a "$LOG"
  bash "$GUARDED_SUMMARY" 2>&1 | tee -a "$LOG"
fi

decision="$("$PY" - "$AUDIT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
audit = json.loads(path.read_text())
complete = bool(audit.get("can_mark_goal_complete")) and audit.get("completion_decision") == "complete"
print("complete" if complete else "incomplete")
PY
)"

if [[ "$decision" == "complete" ]]; then
  echo "[$(timestamp)] FACET reproduction goal is complete according to completion audit" | tee -a "$LOG"
  exit 0
fi

echo "[$(timestamp)] FACET reproduction goal is incomplete according to completion audit" | tee -a "$LOG"
if [[ "$ALLOW_INCOMPLETE" == "1" ]]; then
  exit 0
fi
exit 2
