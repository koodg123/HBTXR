#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="$ROOT/references/codebase/software/FACET"
PY="$ROOT/.facet-train-venv/bin/python"
REPORT_ROOT="$ROOT/references/report/FACET"
AUDIT_PY="$REPORT_ROOT/audit_reproduction_completion_2026-06-26.py"
EP_TRAIN_LOG="$REPORT_ROOT/EPNet_full_unet_gpu0_train_2026-06-26.log"
HB_TRAIN_LOG="$REPORT_ROOT/HBTXR_full_unet_gpu1_train_2026-06-26.log"
REQUIRE_COMPLETED="${FACET_FULL_EVAL_REQUIRE_COMPLETED:-1}"

export PYTHONPATH="$FACET_ROOT"
export FACET_DISABLE_CUDNN=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_full_checkpoint_eval_pycache

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
  if ! training_complete "$EP_TRAIN_LOG"; then
    echo "EPNet full training is not complete; refusing final evaluation without completion marker" >&2
    echo "Set FACET_FULL_EVAL_REQUIRE_COMPLETED=0 only for explicit debugging." >&2
    exit 3
  fi
  if ! training_complete "$HB_TRAIN_LOG"; then
    echo "HBTXR full training is not complete; refusing final evaluation without completion marker" >&2
    echo "Set FACET_FULL_EVAL_REQUIRE_COMPLETED=0 only for explicit debugging." >&2
    exit 3
  fi
fi

EP_CKPT="$(find_best_or_latest_ckpt "$FACET_ROOT/runs/logs/EPNet_full_unet")"
HB_CKPT="$(find_best_or_latest_ckpt "$FACET_ROOT/runs/logs/HBTXR_full_unet")"

if [[ -z "${EP_CKPT:-}" ]]; then
  echo "missing EPNet full checkpoint" >&2
  exit 2
fi

if [[ -z "${HB_CKPT:-}" ]]; then
  echo "missing HBTXR full checkpoint" >&2
  exit 2
fi

echo "EPNet checkpoint: $EP_CKPT"
echo "HBTXR checkpoint: $HB_CKPT"

if artifact_valid eval "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.json" \
  && [[ -f "$REPORT_ROOT/FACET_table2_comparison_2026-06-26.md" ]]; then
  echo "EPNet evaluation artifacts already exist; skipping EPNet re-evaluation"
else
  FACET_DEVICES=0 "$PY" "$FACET_ROOT/EvEye/utils/scripts/evaluate_epnet_checkpoint.py" \
    --config DavisEyeEllipse_EPNet_full_unet.yaml \
    --checkpoint "$EP_CKPT" \
    --output-json "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.json" \
    --output-md "$REPORT_ROOT/FACET_table2_comparison_2026-06-26.md" \
    --device cuda:0
fi

if artifact_valid eval "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.json" \
  && [[ -f "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.md" ]]; then
  echo "HBTXR evaluation artifacts already exist; skipping HBTXR re-evaluation"
else
  FACET_DEVICES=1 "$PY" "$FACET_ROOT/EvEye/utils/scripts/evaluate_epnet_checkpoint.py" \
    --config DavisEyeEllipse_HBTXR_full_unet.yaml \
    --checkpoint "$HB_CKPT" \
    --output-json "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.json" \
    --output-md "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.md" \
    --device cuda:1
fi

if artifact_valid comparison "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.json" \
  && [[ -f "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.md" ]]; then
  echo "EPNet-vs-HBTXR comparison artifacts already exist; skipping pairwise regeneration"
else
  "$PY" "$FACET_ROOT/EvEye/utils/scripts/compare_model_evaluation_results.py" \
    --left-json "$REPORT_ROOT/FACET_reproduction_results_2026-06-26.json" \
    --right-json "$REPORT_ROOT/FACET_hbtxr_reproduction_results_2026-06-26.json" \
    --left-label EPNet_full_unet \
    --right-label HBTXR_full_unet \
    --output-json "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.json" \
    --output-md "$REPORT_ROOT/FACET_epnet_vs_hbtxr_comparison_2026-06-26.md"
fi

"$PY" "$FACET_ROOT/EvEye/utils/scripts/sync_reproduction_status_summary.py" \
  --date 2026-06-26

"$PY" "$AUDIT_PY" \
  --status-json "$REPORT_ROOT/FACET_reproduction_status_2026-06-26.json" \
  --output-json "$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.json" \
  --output-md "$REPORT_ROOT/FACET_reproduction_completion_audit_2026-06-26.md"
