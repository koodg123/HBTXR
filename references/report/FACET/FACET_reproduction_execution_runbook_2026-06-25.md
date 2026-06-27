# FACET Reproduction Execution Runbook

Date: 2026-06-25

## Purpose

이 문서는 `FACET_reproduction_plan_2026-06-25.md`의 남은 Phase 2-4를 실제 GPU 환경에서 순서대로 실행하기 위한 runbook이다.

현재 완료된 준비:

- subset EPNet smoke
- U-Net labelled subset PNG dataset 생성
- U-Net train smoke
- full `Data_davis` expansion script smoke
- EPNet evaluation/comparison script smoke
- U-Net labelled dataset sample visualization

현재 미완료:

- full U-Net training
- full `DeanDataset_full_unet` generation
- full EPNet training
- final Table II comparison

## Environment

Repo:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
```

Report directory:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET
```

Recommended Python:

```text
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python
```

Important caveat:

- Current smoke venv resolved to `torch 2.12.1+cu130`, while the original FACET `requirements.txt` requests `torch==2.2`.
- For final reproduction, create a clean pinned environment if exact dependency reproducibility is required.
- In Codex, GPU commands must be run with host/escalated execution, or directly in the user's terminal. The non-escalated Codex sandbox may not expose `/dev/nvidia*`, even when the host shell sees the GPUs.

## Gate 0: Preflight

Run from FACET repo:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
import torch, lightning, cv2, albumentations, timm, tonic, h5py
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

Pass criteria:

- Python imports succeed.
- CUDA is available for full training.
- `nvidia-smi` returns at least one usable GPU.

Current status in this session:

- Non-escalated Codex sandbox previously could not see `/dev/nvidia*`.
- Host/escalated execution sees two `NVIDIA GeForce RTX 5080` GPUs.
- Host/escalated `.facet-train-venv` reports `torch.cuda.is_available() == True`.

## Gate 1: Existing Dataset Checks

Check subset EPNet dataset:

```bash
test -f /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/manifest.json
```

Expected:

```text
num_samples: 8911
num_train: 7128
num_val: 1783
```

Check U-Net labelled subset:

```bash
test -f /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset/manifest.json
```

Expected:

```text
num_samples: 9011
num_train: 7201
num_val: 1810
```

Check visual samples:

```bash
find /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/unet_dataset_samples -maxdepth 1 -type f | sort
```

Expected:

- `manifest.json`
- `README.md`
- 10 frame/mask/overlay sample triplets

## Phase 2: Full U-Net Training

Command:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

PYTHONPATH=. FACET_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_RGBUNet_local_subset.yaml
```

Expected outputs:

```text
runs/logs/RGBUNet_local_subset/version_*/checkpoints/*.ckpt
runs/logs/RGBUNet_local_subset/version_*/events.out.tfevents.*
```

Pass criteria:

- training completes without crash.
- `last.ckpt` exists.
- at least one monitored checkpoint exists.
- validation metrics are logged.
- a checkpoint path is selected for Phase 3.

Record results in:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_unet_training_results_<date>.md
```

Minimum report fields:

```text
config path
dataset root
checkpoint path
train/val sample counts
epochs
best monitored metric
sample prediction visualization path
runtime GPU
dependency versions
```

## Phase 3: Full Data_davis Expansion

Set checkpoint:

```bash
UNET_CKPT=/path/to/RGBUNet_local_subset/version_x/checkpoints/best_or_last.ckpt
```

Run full expansion:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_full_dean_dataset_with_unet.py \
  --checkpoint "$UNET_CKPT" \
  --output-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet \
  --device cuda:0 \
  --inference-batch-size 32 \
  --train-ratio 0.8 \
  --mask-threshold 0.5 \
  --events-per-sample 5000 \
  --overwrite
```

Expected input scale:

```text
Data_davis sessions: 384
Data_davis frames: 1490548
```

Expected output:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/manifest.json
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/train/cached_data
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/train/cached_ellipse
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/val/cached_data
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/val/cached_ellipse
```

Load check:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
root = '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet'
for split in ['train', 'val']:
    ds = DavisEyeEllipseDataset(
        root_path=root,
        split=split,
        accumulate_mode='fixed_count',
        sensor_size=[346, 260, 2],
        events_interpolation='causal_linear_ori',
        pupil_area=200,
        num_classes=1,
        default_resolution=[256, 256],
    )
    item = ds[0]
    print(split, len(ds), item['input'].shape, item['hm'].shape, item['ellipse'])
PY
```

Pass criteria:

- manifest exists.
- `valid_ellipse_count > 0`.
- train/val counts are recorded.
- both splits load through `DavisEyeEllipseDataset`.
- sample visualization is generated.

Record results in:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_full_expansion_results_<date>.md
```

## Phase 4: Full EPNet Training

Command:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

PYTHONPATH=. FACET_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
```

Expected outputs:

```text
runs/logs/EPNet_full_unet/version_*/checkpoints/*.ckpt
runs/logs/EPNet_full_unet/version_*/events.out.tfevents.*
```

Pass criteria:

- full training completes.
- checkpoint exists.
- validation metrics include:
  - `val_p10_acc`
  - `val_p5_acc`
  - `val_p3_acc`
  - `val_p1_acc`
  - `val_mean_distance`
  - `val_IoU`
  - `val_AP`

Record results in:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_results_<date>.md
```

## Phase 4 Evaluation And Table II Comparison

Set checkpoint:

```bash
EPNET_CKPT=/path/to/EPNet_full_unet/version_x/checkpoints/best_or_last.ckpt
```

Evaluate:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET

PYTHONPATH=. FACET_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/evaluate_epnet_checkpoint.py \
  --config DavisEyeEllipse_EPNet_full_unet.yaml \
  --checkpoint "$EPNET_CKPT" \
  --output-json /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_results_<date>.json \
  --output-md /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_table2_comparison_<date>.md \
  --device cuda:0 \
  --latency-warmup 50 \
  --latency-iterations 200
```

Pass criteria:

- JSON contains `metrics`, `params_m`, `flops_g`, `latency_ms`.
- Markdown table compares current values to paper reference values.
- report explains gaps by:
  - dataset size
  - split rule
  - U-Net label generation
  - EPNet mode
  - runtime/backend

## Known Risks

- The current smoke environment uses `torch 2.12.1+cu130`, not the original `torch==2.2`.
- Paper Table II latency likely uses an optimized runtime; Python CPU/GPU latency is not directly comparable.
- The local paper analysis recovered P1, PE, params, FLOPs, and latency reference values, but not all P10/P5/P3/IoU/AP table values.
- Full `Data_davis` expansion scans about 1.49M frames, so disk and runtime cost are significant.
- U-Net mask quality directly controls Phase 3 label quality and Phase 4 EPNet ceiling.

## Current Stop Point

As of this run:

- Work is ready up to GPU-gated full U-Net training.
- The current machine session did not expose a working NVIDIA driver.
- Do not mark FACET reproduction complete until:
  1. full U-Net checkpoint exists,
  2. `DeanDataset_full_unet` exists and loads,
  3. full EPNet checkpoint exists,
  4. final evaluation JSON/Markdown has been generated from the full checkpoint and full validation split.
