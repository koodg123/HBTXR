#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/kjm26/project/PRJXR/HBTXR"
PYTHON="${ROOT}/.facet-train-venv/bin/python"
BRAT_DIR="${ROOT}/references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution"
EXPORTER="${ROOT}/references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py"
DATA_DIR="${BRAT_DIR}/event_data_hbtxr_img64"
LOG="${ROOT}/references/report/FACET/operations/brat_subject_independent_img64_gpu1_2026-07-02.log"

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${ROOT}/.facet-train-venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

exec > >(tee -a "${LOG}") 2>&1

echo "[BRAT] started $(date -Is)"
echo "[BRAT] root=${ROOT}"
echo "[BRAT] data_dir=${DATA_DIR}"

cd "${ROOT}"
if [[ ! -f "${DATA_DIR}/dataset/train_files.txt" || ! -f "${DATA_DIR}/dataset/val_files.txt" || ! -f "${DATA_DIR}/dataset/test_files.txt" ]]; then
  echo "[BRAT] exporting HBTXR subject-independent split to 3ET-style tree"
  "${PYTHON}" "${EXPORTER}" \
    --format threeet-tree \
    --output-dir "${DATA_DIR}"
else
  echo "[BRAT] existing export detected; skipping export"
fi

cd "${BRAT_DIR}"
echo "[BRAT] training on GPU1 $(date -Is)"
"${PYTHON}" train.py --config_file hbtxr_subject_independent_img64.json --device 1
echo "[BRAT] finished $(date -Is)"
