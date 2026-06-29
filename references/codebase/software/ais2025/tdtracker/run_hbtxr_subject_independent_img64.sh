#!/usr/bin/env bash
set -euo pipefail

python3 train.py \
  --gpu "${GPU:-0}" \
  --train_h5_path "${HBTXR_TDTRACKER_DIR:-./data/hbtxr_img64}/train_hbtxr_img64_seq100.h5" \
  --test_h5_path "${HBTXR_TDTRACKER_DIR:-./data/hbtxr_img64}/val_hbtxr_img64_seq100.h5" \
  --save_path ./checkpoint \
  --eyetracking_log_path ./eyetracking_log \
  --log_name /HBTXR_subject_independent_img64 \
  --batch_size 32 \
  --sensor_width 64 \
  --sensor_height 64 \
  --spatial_factor 1.0 \
  --pixel_tolerances 1 3 5 10 15 \
  --epoch 70 \
  --learning_rate 0.001 \
  --optimizer Adam \
  --decay_rate 1e-5
