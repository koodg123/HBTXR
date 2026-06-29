#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
GUARD="$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"
GOAL_CHECK="$OPERATIONS_ROOT/check_goal_completion_after_guard_2026-06-26.sh"
LOG="$REPORT_ROOT/FACET_next_hourly_refresh_once_2026-06-26.log"

STATUS_JSON="$REPORT_ROOT/FACET_reproduction_status_2026-06-26.json"
STATUS_MD="$REPORT_ROOT/FACET_reproduction_status_2026-06-26.md"
PROGRESS_JSON="$REPORT_ROOT/FACET_full_training_progress_snapshot_2026-06-26.json"
PROGRESS_MD="$REPORT_ROOT/FACET_full_training_progress_snapshot_2026-06-26.md"

MIN_INTERVAL_SECONDS="${FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS:-3600}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

latest_artifact_mtime() {
  local latest=0
  local path
  local mtime
  for path in "$STATUS_JSON" "$STATUS_MD" "$PROGRESS_JSON" "$PROGRESS_MD"; do
    if [[ -e "$path" ]]; then
      mtime="$(stat -c '%Y' "$path")"
      if (( mtime > latest )); then
        latest="$mtime"
      fi
    fi
  done
  echo "$latest"
}

mkdir -p "$REPORT_ROOT"
touch "$LOG"

now="$(date '+%s')"
latest="$(latest_artifact_mtime)"
if [[ "$latest" == "0" ]]; then
  wait_seconds=0
else
  age=$((now - latest))
  if (( age >= MIN_INTERVAL_SECONDS )); then
    wait_seconds=0
  else
    wait_seconds=$((MIN_INTERVAL_SECONDS - age))
  fi
fi

echo "[$(timestamp)] next hourly refresh once scheduled wait_seconds=${wait_seconds} min_interval=${MIN_INTERVAL_SECONDS}" | tee -a "$LOG"

if (( wait_seconds > 0 )); then
  sleep "$wait_seconds"
fi

echo "[$(timestamp)] invoking hourly refresh guard" | tee -a "$LOG"
bash "$GUARD" | tee -a "$LOG"
echo "[$(timestamp)] checking goal completion decision" | tee -a "$LOG"
FACET_GOAL_CHECK_ALLOW_INCOMPLETE=1 FACET_SKIP_GUARDED_SUMMARY=1 bash "$GOAL_CHECK" | tee -a "$LOG"
echo "[$(timestamp)] one-shot hourly refresh and completion check finished" | tee -a "$LOG"
