#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="${PROJECT_ROOT}/references/codebase/software/FACET"
PYTHON="${PROJECT_ROOT}/.facet-train-venv/bin/python"
REPORT_ROOT="${PROJECT_ROOT}/references/report/FACET"
LOG_FILE="${REPORT_ROOT}/EPNet_subject_independent_img64_gpu1_train_2026-06-30.log"
CONFIG="DavisEyeEllipse_EPNet_subject_independent_img64.yaml"

mkdir -p "${REPORT_ROOT}" /tmp/matplotlib-facet

export PYTHONPATH="${FACET_ROOT}"
export FACET_DEVICES=1
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_epnet_subject_independent_img64_gpu1_pycache

{
  echo "[$(date --iso-8601=seconds)] starting EPNet subject-independent img64 training on GPU1"
  echo "[$(date --iso-8601=seconds)] config=${CONFIG}"
  echo "[$(date --iso-8601=seconds)] matched baseline: HBTXR_subject_independent_img64_patch4 split/resolution/batch/epoch/lr/decay/optimizer/scheduler"
} | tee -a "${LOG_FILE}"

echo "[$(date --iso-8601=seconds)] running config gate" | tee -a "${LOG_FILE}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from pathlib import Path
import yaml

root = Path("/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/configs")
cfg = yaml.safe_load((root / "DavisEyeEllipse_EPNet_subject_independent_img64.yaml").read_text())
assert cfg["dataloader"]["train"]["dataset"]["root_path"].endswith("DeanDataset_full_unet_subject_independent")
assert cfg["dataloader"]["val"]["dataset"]["root_path"].endswith("DeanDataset_full_unet_subject_independent")
assert cfg["dataloader"]["train"]["dataset"]["default_resolution"] == [64, 64]
assert cfg["dataloader"]["val"]["dataset"]["default_resolution"] == [64, 64]
assert cfg["dataloader"]["train"]["batch_size"] == 32
assert cfg["dataloader"]["val"]["batch_size"] == 32
assert cfg["dataloader"]["train"]["num_workers"] == 4
assert cfg["dataloader"]["val"]["num_workers"] == 4
assert cfg["train"]["max_epochs"] == 70
assert float(cfg["train"]["optimizer"]["learning_rate"]) == 1.0e-3
assert float(cfg["train"]["optimizer"]["weight_decay"]) == 1.0e-5
print("config gate ok")
PY

cd "${FACET_ROOT}"
"${PYTHON}" tools/train.py -c "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
