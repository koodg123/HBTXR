#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"

STATUS_PY="$FACET_ROOT/EvEye/utils/scripts/check_reproduction_status.py"
PROGRESS_PY="$FACET_ROOT/EvEye/utils/scripts/write_full_training_progress_snapshot.py"
AUDIT_PY="$REPORT_ROOT/audit_reproduction_completion_2026-06-26.py"
SUMMARY_PY="$REPORT_ROOT/summarize_missing_gates_2026-06-26.py"

STATUS_JSON="$REPORT_ROOT/FACET_reproduction_status_2026-06-26.json"
STATUS_MD="$REPORT_ROOT/FACET_reproduction_status_2026-06-26.md"
PROGRESS_JSON="$REPORT_ROOT/FACET_full_training_progress_snapshot_2026-06-26.json"
PROGRESS_MD="$REPORT_ROOT/FACET_full_training_progress_snapshot_2026-06-26.md"
AUDIT_JSON="$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.json"
AUDIT_MD="$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.md"
MISSING_SUMMARY_MD="$REPORT_ROOT/FACET_missing_gate_summary_2026-06-26.md"
LOG="$REPORT_ROOT/FACET_hourly_status_refresh_guard_2026-06-26.log"

EP_TRAIN_LOG="$REPORT_ROOT/EPNet_full_unet_gpu0_train_2026-06-26.log"
HB_TRAIN_LOG="$REPORT_ROOT/HBTXR_full_unet_gpu1_train_2026-06-26.log"
EP_RUN_ROOT="$FACET_ROOT/runs/logs/EPNet_full_unet"
HB_RUN_ROOT="$FACET_ROOT/runs/logs/HBTXR_full_unet"

MIN_INTERVAL_SECONDS="${FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS:-3600}"
FORCE=0

usage() {
  cat <<'EOF'
Usage: run_hourly_status_refresh_guard_2026-06-26.sh [--force]

Refresh FACET reproduction status/progress only when the latest status or
progress artifact is older than FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS.

Environment:
  FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS  default: 3600
EOF
}

for arg in "$@"; do
  case "$arg" in
    --force)
      FORCE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
age=$((now - latest))

if [[ "$FORCE" != "1" && "$latest" != "0" && "$age" -lt "$MIN_INTERVAL_SECONDS" ]]; then
  next_due=$((MIN_INTERVAL_SECONDS - age))
  echo "[$(timestamp)] skip refresh: latest artifact age=${age}s, next_due_in=${next_due}s, min_interval=${MIN_INTERVAL_SECONDS}s" | tee -a "$LOG"
  exit 0
fi

echo "[$(timestamp)] refreshing FACET status/progress artifacts force=${FORCE} age=${age}s min_interval=${MIN_INTERVAL_SECONDS}s" | tee -a "$LOG"

"$PY" "$STATUS_PY" \
  --output-json "$STATUS_JSON" \
  --output-md "$STATUS_MD" \
  >>"$LOG" 2>&1

"$PY" "$PROGRESS_PY" \
  --epnet-log "$EP_TRAIN_LOG" \
  --hbtxr-log "$HB_TRAIN_LOG" \
  --epnet-run-root "$EP_RUN_ROOT" \
  --hbtxr-run-root "$HB_RUN_ROOT" \
  --output-json "$PROGRESS_JSON" \
  --output-md "$PROGRESS_MD" \
  >>"$LOG" 2>&1

"$PY" "$AUDIT_PY" \
  --status-json "$STATUS_JSON" \
  --output-json "$AUDIT_JSON" \
  --output-md "$AUDIT_MD" \
  >>"$LOG" 2>&1

"$PY" "$SUMMARY_PY" \
  --output-md "$MISSING_SUMMARY_MD" \
  >>"$LOG" 2>&1

echo "[$(timestamp)] refresh complete, completion audit and missing-gate summary updated" | tee -a "$LOG"
