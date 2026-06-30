#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/project/PRJXR/HBTXR"
PY="${ROOT}/.venv/bin/python"
DATASET_ROOT="/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent"
LOG_DIR="${ROOT}/references/report/HBTXR_runs/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/ervt_train_$(date +%Y%m%d_%H%M%S).log"

exec >>"${LOG}" 2>&1

echo "[ERVT] log=${LOG}"
echo "[ERVT] start=$(date -Is)"
echo "[ERVT] root=${ROOT}"
echo "[ERVT] dataset=${DATASET_ROOT}"
cd "${ROOT}"

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
  echo "[ERVT] waiting_for_cuda $(date -Is)"
  sleep 60
done

nvidia-smi || true

exec "${PY}" references/codebase/software/ais2024/ERVT/train_hbtxr_subject_independent.py \
  --root-path "${DATASET_ROOT}" \
  --device cuda:0 \
  --batch-size 32 \
  --num-workers 4
