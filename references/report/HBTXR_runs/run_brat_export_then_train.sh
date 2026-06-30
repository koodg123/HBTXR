#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/project/PRJXR/HBTXR"
PY="${ROOT}/.venv/bin/python"
DATASET_ROOT="/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent"
EXPORT_DIR="/mnt/d/dataset/EV_Eye/target_data/hbtxr_exports/brat_event_data_hbtxr_img64"
BRAT_ROOT="${ROOT}/references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution"
LOG_DIR="${ROOT}/references/report/HBTXR_runs/logs"
mkdir -p "${LOG_DIR}" "${EXPORT_DIR}"
LOG="${LOG_DIR}/brat_export_train_$(date +%Y%m%d_%H%M%S).log"

exec >>"${LOG}" 2>&1

echo "[BRAT] log=${LOG}"
echo "[BRAT] start=$(date -Is)"
echo "[BRAT] root=${ROOT}"
echo "[BRAT] dataset=${DATASET_ROOT}"
echo "[BRAT] export_dir=${EXPORT_DIR}"
cd "${ROOT}"

if [[ ! -f "${EXPORT_DIR}/.hbtxr_threeet_export_complete" ]]; then
  "${PY}" references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
    --root-path "${DATASET_ROOT}" \
    --format threeet-tree \
    --output-dir "${EXPORT_DIR}"
  touch "${EXPORT_DIR}/.hbtxr_threeet_export_complete"
else
  echo "[BRAT] existing_export_complete_marker_found"
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
    break
  fi
  echo "[BRAT] waiting_for_cuda $(date -Is)"
  sleep 60
done

nvidia-smi || true
cd "${BRAT_ROOT}"

exec "${PY}" train.py \
  --config_file hbtxr_subject_independent_img64.json \
  --device 0 \
  --data_dir "${EXPORT_DIR}" \
  --data_list_dir "${EXPORT_DIR}/dataset"
