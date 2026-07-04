#!/usr/bin/env bash
set -euo pipefail

FACET_ROOT="/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET"
PROJECT_ROOT="/home/kjm26/project/PRJXR/HBTXR"
PYTHON="${PROJECT_ROOT}/.facet-train-venv/bin/python"
REPORT_ROOT="${PROJECT_ROOT}/references/report/FACET"
EVALUATION_ROOT="${REPORT_ROOT}/evaluation"
CONFIG="${FACET_ROOT}/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml"
RUN_NAME="HBTXR_subject_independent_img64_patch4"
LOG_ROOT="${FACET_ROOT}/runs/logs/${RUN_NAME}"
EVAL_LOG="${REPORT_ROOT}/${RUN_NAME}_eval_after_training_2026-06-28.log"

mkdir -p "${REPORT_ROOT}" "${EVALUATION_ROOT}"

echo "[$(date --iso-8601=seconds)] waiting for ${RUN_NAME} training to finish" | tee -a "${EVAL_LOG}"
while pgrep -af "[t]ools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml" >/dev/null; do
  sleep 600
done

echo "[$(date --iso-8601=seconds)] validating completed training checkpoint in ${LOG_ROOT}" | tee -a "${EVAL_LOG}"
CKPT="$("${PYTHON}" - <<'PY'
from pathlib import Path
import re
import sys

import torch
import yaml

root = Path("/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/HBTXR_subject_independent_img64_patch4")
config = Path("/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml")
with config.open("r", encoding="utf-8") as f:
    max_epochs = int(yaml.safe_load(f)["train"]["max_epochs"])
versions = sorted(root.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1)
if not versions:
    raise SystemExit("no version_* directory found")
version = versions[-1]
ckpt_dir = version / "checkpoints"
last = ckpt_dir / "last.ckpt"
if not last.exists():
    raise SystemExit(f"training did not produce epoch checkpoint: {last}")
last_state = torch.load(last, map_location="cpu")
last_epoch = int(last_state.get("epoch", -1))
required_epoch = max_epochs - 1
if last_epoch < required_epoch:
    raise SystemExit(
        "training appears incomplete; refusing final val/test evaluation: "
        f"last_epoch={last_epoch}, required_epoch>={required_epoch}, last={last}"
    )

best = []
for path in ckpt_dir.glob("*.ckpt"):
    m = re.search(r"val_mean_distance=([0-9.]+)", path.name)
    if m:
        best.append((float(m.group(1).rstrip(".")), path))
if best:
    print(min(best, key=lambda x: x[0])[1])
elif last.exists():
    print(last)
else:
    raise SystemExit(f"no checkpoint found in {ckpt_dir}")
PY
)"
echo "[$(date --iso-8601=seconds)] checkpoint=${CKPT}" | tee -a "${EVAL_LOG}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${FACET_ROOT}"
export FACET_DEVICES=1
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_subject_independent_eval_pycache

for SPLIT in val test; do
  OUT_DIR="${EVALUATION_ROOT}/${RUN_NAME}_${SPLIT}_motion_eval"
  REPORT_NAME="${RUN_NAME}_${SPLIT}_motion_eval_2026-06-28.md"
  echo "[$(date --iso-8601=seconds)] evaluating ${SPLIT}; output=${OUT_DIR}" | tee -a "${EVAL_LOG}"
  "${PYTHON}" "${FACET_ROOT}/EvEye/utils/scripts/evaluate_hbtxr_val_motion.py" \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --run-name "${RUN_NAME}" \
    --dataset-label "DeanDataset_full_unet_subject_independent" \
    --report-name "${REPORT_NAME}" \
    --device "cuda:1" \
    --batch-size 256 \
    --num-workers 8 \
    2>&1 | tee -a "${EVAL_LOG}"
done

echo "[$(date --iso-8601=seconds)] validating val/test motion evaluation artifacts" | tee -a "${EVAL_LOG}"
"${PYTHON}" "${FACET_ROOT}/EvEye/utils/scripts/validate_hbtxr_motion_eval_artifacts.py" \
  --report-root "${EVALUATION_ROOT}" \
  --run-name "${RUN_NAME}" \
  --date "2026-06-28" \
  --splits val test \
  --output-json "${EVALUATION_ROOT}/${RUN_NAME}_motion_eval_validation_2026-06-28.json" \
  2>&1 | tee -a "${EVAL_LOG}"

echo "[$(date --iso-8601=seconds)] building final subject-independent results report" | tee -a "${EVAL_LOG}"
"${PYTHON}" "${FACET_ROOT}/EvEye/utils/scripts/build_hbtxr_subject_independent_results_report.py" \
  --report-root "${EVALUATION_ROOT}" \
  --run-name "${RUN_NAME}" \
  --date "2026-06-28" \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --validation-json "${EVALUATION_ROOT}/${RUN_NAME}_motion_eval_validation_2026-06-28.json" \
  --output "${EVALUATION_ROOT}/${RUN_NAME}_results_2026-06-28.md" \
  2>&1 | tee -a "${EVAL_LOG}"

echo "[$(date --iso-8601=seconds)] evaluation complete" | tee -a "${EVAL_LOG}"
