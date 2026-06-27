#!/usr/bin/env bash
set -euo pipefail

cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

export PYTHONPATH=.
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet

/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  EvEye/utils/scripts/build_full_dean_dataset_with_unet.py \
  --checkpoint /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=02-val_mean_distance=0.4997.ckpt \
  --output-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet \
  --device cuda:0 \
  --inference-batch-size 16 \
  --train-ratio 0.8 \
  --mask-threshold 0.5 \
  --events-per-sample 5000 \
  --sample-count 10 \
  --resume
