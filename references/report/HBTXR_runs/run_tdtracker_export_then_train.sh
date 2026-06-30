#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/project/PRJXR/HBTXR"
PY="${ROOT}/.venv/bin/python"
DATASET_ROOT="/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent"
EXPORT_DIR="/mnt/d/dataset/EV_Eye/target_data/hbtxr_exports/tdtracker_h5_img64"
TDTRACKER_ROOT="${ROOT}/references/codebase/software/ais2025/tdtracker"
LOG_DIR="${ROOT}/references/report/HBTXR_runs/logs"
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
mkdir -p "${LOG_DIR}" "${EXPORT_DIR}"
LOG="${LOG_DIR}/tdtracker_export_train_$(date +%Y%m%d_%H%M%S).log"

exec >>"${LOG}" 2>&1

echo "[TDTracker] log=${LOG}"
echo "[TDTracker] start=$(date -Is)"
echo "[TDTracker] root=${ROOT}"
echo "[TDTracker] dataset=${DATASET_ROOT}"
echo "[TDTracker] export_dir=${EXPORT_DIR}"
cd "${ROOT}"

if [[ ! -f "${EXPORT_DIR}/.hbtxr_tdtracker_export_complete" ]]; then
  "${PY}" references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
    --root-path "${DATASET_ROOT}" \
    --format tdtracker-h5 \
    --output-dir "${EXPORT_DIR}" \
    --sequence-length 100 \
    --stride 100
  touch "${EXPORT_DIR}/.hbtxr_tdtracker_export_complete"
else
  echo "[TDTracker] existing_export_complete_marker_found"
fi

while true; do
  if "${PY}" - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
if ok:
    print("cuda_ok", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
    sys.exit(0)
print("cuda_not_ready", torch.__version__, torch.version.cuda, "count", torch.cuda.device_count(), flush=True)
sys.exit(1)
PY
  then
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "[TDTracker] gpu_free_mib=${free_mib} required=${MIN_FREE_MIB} $(date -Is)"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      break
    fi
  fi
  echo "[TDTracker] waiting_for_cuda_or_free_memory $(date -Is)"
  sleep 60
done

nvidia-smi || true
cd "${TDTRACKER_ROOT}"
export PATH="${ROOT}/.venv/bin:${PATH}"
export HBTXR_TDTRACKER_DIR="${EXPORT_DIR}"
export GPU="${GPU:-0}"

exec bash run_hbtxr_subject_independent_img64.sh
