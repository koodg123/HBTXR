#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
REPORT_ROOT="$ROOT/references/report/FACET"
OPERATIONS_ROOT="$REPORT_ROOT/operations"
AUDIT_PY="$OPERATIONS_ROOT/audit_reproduction_completion_2026-06-26.py"
FPN_DW_TRAIN_LOG="$REPORT_ROOT/EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log"
REQUIRE_COMPLETED="${FACET_FPN_DW_EVAL_REQUIRE_COMPLETED:-1}"

export PYTHONPATH="$FACET_ROOT"
export FACET_DISABLE_CUDNN=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_epnet_fpn_dw_checkpoint_eval_pycache

cd "$FACET_ROOT"

find_best_or_latest_ckpt() {
  local run_root="$1"
  "$PY" - "$run_root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [
    path
    for path in root.glob("**/checkpoints/*.ckpt")
    if "step_checkpoints" not in path.parts and path.name != "last.ckpt"
]

def val_mean_distance(path: Path):
    patterns = [
        r"val_mean_distance=([0-9]+(?:\.[0-9]+)?)\.ckpt$",
        r"[-_]([0-9]+(?:\.[0-9]+)?)\.ckpt$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            return float(match.group(1))
    return None

scored = [(val_mean_distance(path), path.stat().st_mtime, path) for path in candidates]
with_metric = [item for item in scored if item[0] is not None]
if with_metric:
    print(min(with_metric, key=lambda item: (item[0], -item[1]))[2])
elif scored:
    print(max(scored, key=lambda item: item[1])[2])
PY
}

artifact_valid() {
  local artifact_type="$1"
  local artifact_path="$2"
  "$PY" "$FACET_ROOT/EvEye/utils/scripts/validate_reproduction_artifact.py" \
    --type "$artifact_type" \
    --path "$artifact_path" \
    >/dev/null 2>&1
}

training_complete() {
  local log_file="$1"
  if [[ ! -f "$log_file" ]]; then
    return 1
  fi
  rg -q '`max_epochs=70` reached|max_epochs=70 reached|Trainer\.fit stopped:.*max_epochs=70.*reached' "$log_file"
}

if [[ "$REQUIRE_COMPLETED" == "1" ]]; then
  if ! training_complete "$FPN_DW_TRAIN_LOG"; then
    echo "EPNet fpn_dw training is not complete; refusing final evaluation without completion marker" >&2
    echo "Set FACET_FPN_DW_EVAL_REQUIRE_COMPLETED=0 only for explicit debugging." >&2
    exit 3
  fi
fi

FPN_DW_CKPT="$(find_best_or_latest_ckpt "$FACET_ROOT/runs/logs/EPNet_fpn_dw_full_unet")"

if [[ -z "${FPN_DW_CKPT:-}" ]]; then
  echo "missing EPNet fpn_dw full checkpoint" >&2
  exit 2
fi

echo "EPNet fpn_dw checkpoint: $FPN_DW_CKPT"

if artifact_valid eval "$REPORT_ROOT/FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json" \
  && [[ -f "$REPORT_ROOT/FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md" ]]; then
  echo "EPNet fpn_dw evaluation artifacts already exist; skipping fpn_dw re-evaluation"
else
  FACET_DEVICES=0 "$PY" "$FACET_ROOT/EvEye/utils/scripts/evaluate_epnet_checkpoint.py" \
    --config DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml \
    --checkpoint "$FPN_DW_CKPT" \
    --output-json "$REPORT_ROOT/FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json" \
    --output-md "$REPORT_ROOT/FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md" \
    --device cuda:0
fi

"$PY" "$FACET_ROOT/EvEye/utils/scripts/sync_reproduction_status_summary.py" \
  --date 2026-06-26

"$PY" "$AUDIT_PY" \
  --status-json "$REPORT_ROOT/FACET_reproduction_status_2026-06-26.json" \
  --output-json "$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.json" \
  --output-md "$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.md"
