#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
RUNNER="$OPERATIONS_ROOT/run_next_hourly_refresh_once_2026-06-26.sh"
LOG="$REPORT_ROOT/FACET_next_hourly_refresh_once_2026-06-26.log"
SESSION="facet_next_hourly_refresh_once"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

mkdir -p "$REPORT_ROOT"
touch "$LOG"

if ! tmux_sessions="$(tmux ls 2>>"$LOG")"; then
  echo "[$(timestamp)] unable to inspect tmux sessions; not starting duplicate-prone one-shot refresh" | tee -a "$LOG"
  exit 1
fi

if printf '%s\n' "$tmux_sessions" | awk -F: '{print $1}' | grep -Fxq "$SESSION"; then
  echo "[$(timestamp)] one-shot hourly refresh session already alive: $SESSION" | tee -a "$LOG"
  exit 0
fi

echo "[$(timestamp)] starting one-shot hourly refresh session: $SESSION" | tee -a "$LOG"
tmux new-session -d -s "$SESSION" "bash '$RUNNER'"
