#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
GUARD="$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"

python3 - "$GUARD" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()

def first_line_containing(text, start=0):
    for index, line in enumerate(lines[start:], start=start):
        if text in line:
            return index + 1
    raise SystemExit(f"missing pattern in hourly guard: {text}")

skip_line = first_line_containing("skip refresh: latest artifact age=")
exit_line = first_line_containing("exit 0", start=skip_line)
status_line = first_line_containing('"$PY" "$STATUS_PY"')
progress_line = first_line_containing('"$PY" "$PROGRESS_PY"')
audit_line = first_line_containing('"$PY" "$AUDIT_PY"')
summary_line = first_line_containing('"$PY" "$SUMMARY_PY"')

for label, refresh_line in {
    "status": status_line,
    "progress": progress_line,
    "audit": audit_line,
    "summary": summary_line,
}.items():
    if skip_line >= refresh_line:
        raise SystemExit(f"skip log appears after {label} refresh call")
    if exit_line >= refresh_line:
        raise SystemExit(f"skip exit appears after {label} refresh call")

condition_text = " ".join(lines[max(0, skip_line - 4) : skip_line])
for required in ['"$FORCE" != "1"', '"$latest" != "0"', '"$age" -lt "$MIN_INTERVAL_SECONDS"']:
    if required not in condition_text:
        raise SystemExit(f"skip condition missing: {required}")

if "next_due_in=" not in lines[skip_line - 1]:
    raise SystemExit("skip log no longer reports next_due_in")
PY

echo "hourly guard skip order smoke passed"
