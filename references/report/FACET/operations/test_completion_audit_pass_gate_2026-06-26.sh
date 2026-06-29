#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"
OUT_DIR="/tmp/facet_completion_audit_pass_gate_2026-06-26"
STATUS_JSON="$OUT_DIR/status_all_passed.json"
OUT_JSON="$OUT_DIR/completion_audit.json"
OUT_MD="$OUT_DIR/completion_audit.md"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

"$PY" - "$OPERATIONS_ROOT/audit_reproduction_completion_2026-06-26.py" "$STATUS_JSON" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("facet_completion_audit", audit_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

items = [
    {
        "name": name,
        "state": "passed",
        "missing": [],
        "note": "synthetic pass gate smoke",
    }
    for name in module.EXPECTED_STATUS_ITEMS
]
status_path.write_text(
    json.dumps(
        {
            "overall_status": "passed",
            "counts": {"passed": len(items), "missing": 0},
            "items": items,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

"$PY" "$OPERATIONS_ROOT/audit_reproduction_completion_2026-06-26.py" \
  --status-json "$STATUS_JSON" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  --fail-on-incomplete \
  >/tmp/facet_completion_audit_pass_gate_2026-06-26.stdout \
  2>/tmp/facet_completion_audit_pass_gate_2026-06-26.stderr

"$PY" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("can_mark_goal_complete") is not True:
    raise SystemExit("can_mark_goal_complete should be true")
if audit.get("completion_decision") != "complete":
    raise SystemExit("completion_decision should be complete")
if audit.get("status_overall") != "passed":
    raise SystemExit("status_overall should be passed")
if audit.get("non_passed_status_items"):
    raise SystemExit("non_passed_status_items should be empty")
PY

rg -q 'Can mark goal complete: `True`' "$OUT_MD"
rg -q 'Completion decision: `complete`' "$OUT_MD"

echo "completion audit pass gate smoke passed"
