#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
FACET_ROOT="${ROOT}/references/codebase/software/FACET"
TDTRACKER_ROOT="${ROOT}/references/codebase/software/ais2025/tdtracker"
PYTHON="${ROOT}/.facet-train-venv/bin/python"
DATASET_ROOT="/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"
TDTRACKER_DATA_DIR="${HBTXR_TDTRACKER_DIR:-/home/kjm26/project/dataset/XR/EV_Eye/target_data/tdtracker_hbtxr_img64_seq100}"
LOG_DIR="${ROOT}/references/report/FACET/operations"
LOG_FILE="${LOG_DIR}/tdtracker_subject_independent_img64_gpu1_2026-07-01.log"

mkdir -p "${TDTRACKER_DATA_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[start] $(date -Is)"
echo "[dataset] ${DATASET_ROOT}"
echo "[tdtracker_data] ${TDTRACKER_DATA_DIR}"
echo "[gpu] physical GPU1 via CUDA_VISIBLE_DEVICES"

if [[ ! -f "${TDTRACKER_DATA_DIR}/train_hbtxr_img64_seq100.h5" || \
      ! -f "${TDTRACKER_DATA_DIR}/val_hbtxr_img64_seq100.h5" || \
      ! -f "${TDTRACKER_DATA_DIR}/test_hbtxr_img64_seq100.h5" ]]; then
  echo "[export] TDTracker H5 files are missing; exporting seq100 stride100 64x64 data."
  cd "${ROOT}"
  PYTHONPATH="${FACET_ROOT}" "${PYTHON}" \
    "${FACET_ROOT}/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py" \
    --root-path "${DATASET_ROOT}" \
    --format tdtracker-h5 \
    --output-dir "${TDTRACKER_DATA_DIR}" \
    --sequence-length 100 \
    --stride 100
else
  echo "[export] TDTracker H5 files already exist; reusing them."
fi

echo "[train] launching TDTracker subject-independent img64 training."
cd "${TDTRACKER_ROOT}"
PYTHON="${PYTHON}" GPU=1 HBTXR_TDTRACKER_DIR="${TDTRACKER_DATA_DIR}" \
  bash ./run_hbtxr_subject_independent_img64.sh

echo "[done] $(date -Is)"
