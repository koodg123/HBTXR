#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
PY="$ROOT/.facet-train-venv/bin/python"
SCRIPT="$OPERATIONS_ROOT/summarize_missing_gates_2026-06-26.py"
TMP_MD="/tmp/FACET_missing_gate_summary_test_2026-06-26.md"

output="$("$PY" "$SCRIPT")"

if ! grep -Fq '# FACET Missing Gate Summary' <<<"$output"; then
  echo "missing gate summary title not found" >&2
  exit 1
fi

if ! grep -Fq 'can_mark_goal_complete' <<<"$output"; then
  echo "missing gate summary lacks completion decision" >&2
  exit 1
fi

if ! grep -Fq 'Phase 4 full EPNet training completion' <<<"$output"; then
  echo "missing gate summary lacks EPNet completion gate" >&2
  exit 1
fi

if ! grep -Fq 'This summary only reads status/progress/audit JSON artifacts' <<<"$output"; then
  echo "missing gate summary does not state log-scan boundary" >&2
  exit 1
fi

"$PY" "$SCRIPT" --output-md "$TMP_MD" >/tmp/FACET_missing_gate_summary_stdout_2026-06-26.txt
if [[ ! -f "$TMP_MD" ]]; then
  echo "missing gate summary did not write --output-md file" >&2
  exit 1
fi

if ! rg -q 'refresh_next_due_in_seconds' "$TMP_MD"; then
  echo "missing gate summary output file lacks refresh due field" >&2
  exit 1
fi

if ! rg -q 'latest_artifact_age_seconds' "$TMP_MD"; then
  echo "missing gate summary output file lacks artifact age field" >&2
  exit 1
fi

if ! rg -q 'refresh_state' "$TMP_MD"; then
  echo "missing gate summary output file lacks refresh state field" >&2
  exit 1
fi

runtime_output="$("$PY" "$SCRIPT" --include-runtime)"
if ! grep -Fq '## Runtime Snapshot' <<<"$runtime_output"; then
  echo "missing gate summary runtime output lacks runtime section" >&2
  exit 1
fi

if ! grep -Fq 'facet_epnet_full_gpu0' <<<"$runtime_output"; then
  echo "missing gate summary runtime output lacks EPNet tmux session" >&2
  exit 1
fi

if ! grep -Fq 'tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml' <<<"$runtime_output"; then
  echo "missing gate summary runtime output lacks EPNet training process pattern" >&2
  exit 1
fi

if ! grep -Eq 'alive|unavailable' <<<"$runtime_output"; then
  echo "missing gate summary runtime output lacks observable or unavailable session state" >&2
  exit 1
fi

if ! grep -Eq 'present|unavailable' <<<"$runtime_output"; then
  echo "missing gate summary runtime output lacks observable or unavailable process state" >&2
  exit 1
fi

echo "missing gate summary smoke passed"
