#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/user/project/PRJXR/HBTXR"
PY="${ROOT}/.venv/bin/python"
RETINA_ROOT="${ROOT}/references/codebase/software/retina"
LOG_DIR="${ROOT}/references/report/HBTXR_runs/logs"
mkdir -p "${LOG_DIR}" "${ROOT}/tmp/matplotlib"
LOG="${LOG_DIR}/retina_train_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${LOG}") 2>&1

echo "[Retina] log=${LOG}"
echo "[Retina] start=$(date -Is)"
echo "[Retina] root=${ROOT}"
echo "[Retina] config=${RETINA_ROOT}/configs/hbtxr_subject_independent_img64_patch4.yaml"
echo "[Retina] num_workers=${RETINA_NUM_WORKERS:-config_default}"
echo "[Retina] ckpt_path=${RETINA_CKPT_PATH:-none}"
echo "[Retina] disable_cudnn=${RETINA_DISABLE_CUDNN:-0}"
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
  echo "[Retina] waiting_for_cuda $(date -Is)"
  sleep 60
done

nvidia-smi || true
export MPLCONFIGDIR="${ROOT}/tmp/matplotlib"
CMD=(
  "${PY}" references/codebase/software/retina/scripts/train_hbtxr_subject_independent.py
  --device 0 \
  --run-name "${RETINA_RUN_NAME:-Retina_subject_independent_img64}" \
  --output-root "${RETINA_ROOT}/runs"
)
if [[ -n "${RETINA_NUM_WORKERS:-}" ]]; then
  CMD+=(--num-workers "${RETINA_NUM_WORKERS}")
fi
if [[ -n "${RETINA_CKPT_PATH:-}" ]]; then
  CMD+=(--ckpt-path "${RETINA_CKPT_PATH}")
fi
if [[ "${RETINA_DISABLE_CUDNN:-0}" == "1" ]]; then
  CMD+=(--disable-cudnn)
fi

printf '[Retina] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
