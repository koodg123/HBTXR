#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
SCRIPT="$OPERATIONS_ROOT/check_goal_completion_after_guard_2026-06-26.sh"

if ! rg -q 'run_guarded_missing_gate_summary_2026-06-26\.sh' "$SCRIPT"; then
  echo "goal completion check does not run the guarded summary first" >&2
  exit 1
fi

if ! rg -q 'FACET_SKIP_GUARDED_SUMMARY' "$SCRIPT"; then
  echo "goal completion check does not expose the guarded summary skip option" >&2
  exit 1
fi

if ! rg -q 'FACET_reproduction_completion_audit_2026-06-26\.json' "$SCRIPT"; then
  echo "goal completion check does not read the completion audit JSON" >&2
  exit 1
fi

if ! rg -q 'can_mark_goal_complete' "$SCRIPT"; then
  echo "goal completion check does not inspect can_mark_goal_complete" >&2
  exit 1
fi

if ! rg -q 'completion_decision.*complete' "$SCRIPT"; then
  echo "goal completion check does not require completion_decision=complete" >&2
  exit 1
fi

if ! rg -q 'exit 2' "$SCRIPT"; then
  echo "goal completion check does not fail closed while incomplete" >&2
  exit 1
fi

FACET_GOAL_CHECK_ALLOW_INCOMPLETE=1 "$SCRIPT" >/tmp/facet_goal_completion_check_smoke_2026-06-26.out
if ! rg -q 'FACET reproduction goal is incomplete according to completion audit' /tmp/facet_goal_completion_check_smoke_2026-06-26.out; then
  echo "goal completion check smoke did not report current incomplete state" >&2
  exit 1
fi

FACET_GOAL_CHECK_ALLOW_INCOMPLETE=1 FACET_SKIP_GUARDED_SUMMARY=1 "$SCRIPT" >/tmp/facet_goal_completion_check_skip_smoke_2026-06-26.out
if ! rg -q 'skipping guarded summary by FACET_SKIP_GUARDED_SUMMARY=1' /tmp/facet_goal_completion_check_skip_smoke_2026-06-26.out; then
  echo "goal completion check skip smoke did not skip guarded summary" >&2
  exit 1
fi

echo "goal completion check smoke passed"
