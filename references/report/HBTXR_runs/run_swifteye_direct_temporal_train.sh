#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/user/project/PRJXR/HBTXR}"
PYTHON="${PYTHON:-${REPO_ROOT}/tmp/venvs/mmrotate_py38/bin/python}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-${REPO_ROOT}/analysis/results/Swift-Eye/direct_hbtxr_img64}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/references/codebase/software/Swift-Eye/mmrotate:${PYTHONPATH:-}"

"${PYTHON}" analysis/scripts/swifteye_hbtxr_direct_train.py \
  --mode train-temporal \
  --split train \
  --input-height 64 \
  --input-width 64 \
  --batch-size "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-0}" \
  --epochs "${EPOCHS:-30}" \
  --device "${DEVICE}" \
  --output-root "${OUT}"
