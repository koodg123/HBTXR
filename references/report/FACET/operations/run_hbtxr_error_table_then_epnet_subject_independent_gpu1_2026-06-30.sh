#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="${PROJECT_ROOT}/references/codebase/software/FACET"
PYTHON="${PROJECT_ROOT}/.facet-train-venv/bin/python"
REPORT_ROOT="${PROJECT_ROOT}/references/report/FACET"
LOG_FILE="${REPORT_ROOT}/HBTXR_error_table_then_EPNet_subject_independent_gpu1_2026-06-30.log"
HBTXR_FILL_SCRIPT="${PROJECT_ROOT}/analysis/scripts/fill_hbtxr_error_distribution.py"
EPNET_CONFIG="DavisEyeEllipse_EPNet_subject_independent_img64.yaml"
HBTXR_CONFIG="${FACET_ROOT}/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml"
EPNET_CONFIG_PATH="${FACET_ROOT}/configs/${EPNET_CONFIG}"

mkdir -p "${REPORT_ROOT}" /tmp/matplotlib-facet

{
  echo "[$(date --iso-8601=seconds)] pipeline start"
  echo "[$(date --iso-8601=seconds)] project=${PROJECT_ROOT}"
  echo "[$(date --iso-8601=seconds)] HBTXR error table fill first, then EPNet training on GPU1"
} | tee -a "${LOG_FILE}"

export PYTHONPATH="${FACET_ROOT}"
export FACET_DEVICES=1
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_error_epnet_gpu1_pycache

echo "[$(date --iso-8601=seconds)] validating matched HBTXR/EPNet subject-independent img64 settings" | tee -a "${LOG_FILE}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from pathlib import Path
import yaml

root = Path("/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/configs")
hbtxr = yaml.safe_load((root / "DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml").read_text())
epnet = yaml.safe_load((root / "DavisEyeEllipse_EPNet_subject_independent_img64.yaml").read_text())

checks = [
    ("train_root", hbtxr["dataloader"]["train"]["dataset"]["root_path"], epnet["dataloader"]["train"]["dataset"]["root_path"]),
    ("val_root", hbtxr["dataloader"]["val"]["dataset"]["root_path"], epnet["dataloader"]["val"]["dataset"]["root_path"]),
    ("train_resolution", hbtxr["dataloader"]["train"]["dataset"]["default_resolution"], epnet["dataloader"]["train"]["dataset"]["default_resolution"]),
    ("val_resolution", hbtxr["dataloader"]["val"]["dataset"]["default_resolution"], epnet["dataloader"]["val"]["dataset"]["default_resolution"]),
    ("train_batch", hbtxr["dataloader"]["train"]["batch_size"], epnet["dataloader"]["train"]["batch_size"]),
    ("val_batch", hbtxr["dataloader"]["val"]["batch_size"], epnet["dataloader"]["val"]["batch_size"]),
    ("train_workers", hbtxr["dataloader"]["train"]["num_workers"], epnet["dataloader"]["train"]["num_workers"]),
    ("val_workers", hbtxr["dataloader"]["val"]["num_workers"], epnet["dataloader"]["val"]["num_workers"]),
    ("epochs", hbtxr["train"]["max_epochs"], epnet["train"]["max_epochs"]),
    ("lr", hbtxr["train"]["optimizer"]["learning_rate"], epnet["train"]["optimizer"]["learning_rate"]),
    ("weight_decay", hbtxr["train"]["optimizer"]["weight_decay"], epnet["train"]["optimizer"]["weight_decay"]),
]
bad = [(name, a, b) for name, a, b in checks if a != b]
if bad:
    for name, a, b in bad:
        print(f"mismatch {name}: HBTXR={a!r} EPNet={b!r}")
    raise SystemExit("EPNet config is not matched to HBTXR subject-independent img64 settings")
print("config gate ok")
for name, a, _ in checks:
    print(f"{name}: {a}")
print("optimizer/scheduler gate: EPNet and HBTXR both use Adam + StepLRScheduler(decay_t=10, decay_rate=0.7, warmup_lr=1e-5, warmup_t=5) in model code")
PY

echo "[$(date --iso-8601=seconds)] filling HBTXR JETCAS Error-Distributions workbook on GPU1" | tee -a "${LOG_FILE}"
cd "${PROJECT_ROOT}"
"${PYTHON}" "${HBTXR_FILL_SCRIPT}" \
  --config "${HBTXR_CONFIG}" \
  --device cuda:1 \
  --batch-size 256 \
  --num-workers 4 \
  2>&1 | tee -a "${LOG_FILE}"

echo "[$(date --iso-8601=seconds)] HBTXR error table completed; starting EPNet subject-independent img64 training on GPU1" | tee -a "${LOG_FILE}"
cd "${FACET_ROOT}"
"${PYTHON}" tools/train.py -c "${EPNET_CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

echo "[$(date --iso-8601=seconds)] EPNet training command exited" | tee -a "${LOG_FILE}"
