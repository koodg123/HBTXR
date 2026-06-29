#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"
OUT_DIR="/tmp/facet_completion_audit_fail_gate_2026-06-26"
OUT_JSON="$OUT_DIR/completion_audit.json"
OUT_MD="$OUT_DIR/completion_audit.md"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

set +e
"$PY" "$OPERATIONS_ROOT/audit_reproduction_completion_2026-06-26.py" \
  --status-json "$REPORT_ROOT/FACET_reproduction_status_2026-06-26.json" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  --fail-on-incomplete \
  >/tmp/facet_completion_audit_fail_gate_2026-06-26.stdout \
  2>/tmp/facet_completion_audit_fail_gate_2026-06-26.stderr
exit_code=$?
set -e

if [[ "$exit_code" -eq 0 ]]; then
  echo "completion audit unexpectedly passed on incomplete status" >&2
  exit 1
fi

if [[ ! -s "$OUT_JSON" || ! -s "$OUT_MD" ]]; then
  echo "completion audit did not write expected /tmp outputs" >&2
  exit 1
fi

"$PY" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("can_mark_goal_complete") is not False:
    raise SystemExit("can_mark_goal_complete should be false")
if audit.get("completion_decision") != "incomplete":
    raise SystemExit("completion_decision should be incomplete")
if audit.get("status_overall") != "incomplete":
    raise SystemExit("status_overall should be incomplete")
if not audit.get("non_passed_status_items"):
    raise SystemExit("non_passed_status_items should not be empty")
PY

rg -q 'Can mark goal complete: `False`' "$OUT_MD"
rg -q 'Completion decision: `incomplete`' "$OUT_MD"

echo "completion audit fail gate smoke passed"
