#!/usr/bin/env bash
set -euo pipefail

cd /home/kjm26/project/PRJXR/HBTXR

unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/tmp/hbtxr_ev_eye_pycache

LOG_DIR=references/report/EV-Eye/logs
OUT_DIR=analysis/RESULTS/EV-Eye_Hybrid_frame128_event64_train_ready
mkdir -p "$LOG_DIR" "$OUT_DIR"
LOG_FILE="$LOG_DIR/EV_Eye_Hybrid_frame128_event64_train_ready_gpu0_2026-07-04.log"

{
  date
  nvidia-smi
  .facet-cu128-venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cudnn", torch.backends.cudnn.version())
print("cudnn_enabled", torch.backends.cudnn.enabled)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY

  .facet-cu128-venv/bin/python analysis/scripts/train_eveye_exgaze_hybrid.py \
    --config analysis/configs/EV_Eye_Hybrid_frame128_event64_subject_independent_train_ready.json \
    --model ev-eye \
    --device cuda:0 \
    --output-dir "$OUT_DIR"
} 2>&1 | tee -a "$LOG_FILE"
