#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"

mapfile -t routine_scripts < <(
  find "$OPERATIONS_ROOT" -maxdepth 1 -type f -name 'watch_*_2026-06-26.sh' | sort
)

if [[ "${#routine_scripts[@]}" -eq 0 ]]; then
  echo "no routine watcher scripts found" >&2
  exit 1
fi

for script in "${routine_scripts[@]}"; do
  if ! rg -q 'HOURLY_REFRESH_GUARD=.*run_hourly_status_refresh_guard_2026-06-26\.sh' "$script"; then
    echo "missing hourly guard binding: $script" >&2
    exit 1
  fi
  if ! rg -q 'bash "\$HOURLY_REFRESH_GUARD"' "$script"; then
    echo "missing hourly guard call: $script" >&2
    exit 1
  fi
  if rg -q 'check_reproduction_status\.py|write_full_training_progress_snapshot\.py' "$script"; then
    echo "routine watcher directly refreshes status/progress: $script" >&2
    exit 1
  fi
done

if ! rg -q 'check_reproduction_status\.py' "$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"; then
  echo "hourly guard no longer refreshes status" >&2
  exit 1
fi

if ! rg -q 'write_full_training_progress_snapshot\.py' "$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"; then
  echo "hourly guard no longer refreshes progress" >&2
  exit 1
fi

if ! rg -q 'audit_reproduction_completion_2026-06-26\.py' "$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"; then
  echo "hourly guard no longer refreshes completion audit" >&2
  exit 1
fi

if ! rg -q 'summarize_missing_gates_2026-06-26\.py' "$OPERATIONS_ROOT/run_hourly_status_refresh_guard_2026-06-26.sh"; then
  echo "hourly guard no longer refreshes missing-gate summary" >&2
  exit 1
fi

echo "hourly refresh guard routing smoke passed"
