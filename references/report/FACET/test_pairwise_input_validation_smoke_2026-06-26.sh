#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
REPORT_ROOT="$ROOT/references/report/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
SMOKE_JSON="$REPORT_ROOT/FACET_phase4_epnet_eval_smoke_2026-06-25.json"
OUT_JSON="/tmp/facet_pairwise_invalid_input_debug.json"
OUT_MD="/tmp/facet_pairwise_invalid_input_debug.md"

export PYTHONPATH="$FACET_ROOT"
export PYTHONPYCACHEPREFIX=/tmp/facet_pairwise_input_validation_pycache

if "$PY" "$FACET_ROOT/EvEye/utils/scripts/compare_model_evaluation_results.py" \
  --left-json "$SMOKE_JSON" \
  --right-json "$SMOKE_JSON" \
  --left-label smoke_left \
  --right-label smoke_right \
  --output-json /tmp/facet_pairwise_should_not_exist.json \
  --output-md /tmp/facet_pairwise_should_not_exist.md \
  >/tmp/facet_pairwise_reject_stdout.json 2>/tmp/facet_pairwise_reject_stderr.txt; then
  echo "expected invalid pairwise inputs to be rejected, but comparison succeeded" >&2
  exit 1
fi

"$PY" "$FACET_ROOT/EvEye/utils/scripts/compare_model_evaluation_results.py" \
  --left-json "$SMOKE_JSON" \
  --right-json "$SMOKE_JSON" \
  --left-label smoke_left \
  --right-label smoke_right \
  --allow-invalid-inputs \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  >/tmp/facet_pairwise_allow_invalid_stdout.json

"$PY" - "$OUT_JSON" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("left_label") != "smoke_left":
    raise SystemExit("unexpected left label")
if data.get("right_label") != "smoke_right":
    raise SystemExit("unexpected right label")
if not data.get("rows"):
    raise SystemExit("expected comparison rows in debug output")
print("pairwise input validation smoke passed")
PY
