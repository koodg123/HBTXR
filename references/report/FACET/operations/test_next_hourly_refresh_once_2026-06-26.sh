#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
SCRIPT="$OPERATIONS_ROOT/run_next_hourly_refresh_once_2026-06-26.sh"
STARTER="$OPERATIONS_ROOT/start_next_hourly_refresh_once_2026-06-26.sh"

if ! rg -q 'GUARD=.*run_hourly_status_refresh_guard_2026-06-26\.sh' "$SCRIPT"; then
  echo "one-shot refresh script does not bind the hourly guard" >&2
  exit 1
fi

if ! rg -q 'bash "\$GUARD"' "$SCRIPT"; then
  echo "one-shot refresh script does not call the hourly guard" >&2
  exit 1
fi

if ! rg -q 'GOAL_CHECK=.*check_goal_completion_after_guard_2026-06-26\.sh' "$SCRIPT"; then
  echo "one-shot refresh script does not bind the goal completion check" >&2
  exit 1
fi

if rg -q 'GUARDED_SUMMARY=.*run_guarded_missing_gate_summary_2026-06-26\.sh' "$SCRIPT"; then
  echo "one-shot refresh script should rely on the hourly guard for summary generation" >&2
  exit 1
fi

if ! rg -q 'FACET_GOAL_CHECK_ALLOW_INCOMPLETE=1 FACET_SKIP_GUARDED_SUMMARY=1 bash "\$GOAL_CHECK"' "$SCRIPT"; then
  echo "one-shot refresh script does not run the goal completion check in record-only mode" >&2
  exit 1
fi

if rg -q -- '--force' "$SCRIPT"; then
  echo "one-shot refresh script must not force the hourly guard" >&2
  exit 1
fi

if ! rg -q 'wait_seconds=\$\(\(MIN_INTERVAL_SECONDS - age\)\)' "$SCRIPT"; then
  echo "one-shot refresh script does not compute wait_seconds from remaining interval" >&2
  exit 1
fi

if ! rg -q 'sleep "\$wait_seconds"' "$SCRIPT"; then
  echo "one-shot refresh script does not sleep until the guarded refresh is due" >&2
  exit 1
fi

if ! rg -q 'SESSION="facet_next_hourly_refresh_once"' "$STARTER"; then
  echo "one-shot starter does not bind the expected tmux session name" >&2
  exit 1
fi

if ! rg -q 'tmux ls' "$STARTER"; then
  echo "one-shot starter does not inspect existing tmux sessions" >&2
  exit 1
fi

if ! rg -q 'unable to inspect tmux sessions; not starting duplicate-prone one-shot refresh' "$STARTER"; then
  echo "one-shot starter does not fail closed when tmux sessions cannot be inspected" >&2
  exit 1
fi

if ! rg -q 'grep -Fxq "\$SESSION"' "$STARTER"; then
  echo "one-shot starter does not use an exact session-name guard" >&2
  exit 1
fi

if ! rg -q 'tmux new-session -d -s "\$SESSION"' "$STARTER"; then
  echo "one-shot starter does not create the guarded tmux session" >&2
  exit 1
fi

echo "next hourly refresh once smoke passed"
