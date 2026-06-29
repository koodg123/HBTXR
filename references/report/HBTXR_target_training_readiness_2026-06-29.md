# HBTXR Subject-Independent Img64 Target Training Readiness

Date: 2026-06-29

## Shared Contract

All targets are prepared against the same experimental contract as
`HBTXR_subject_independent_img64_patch4`.

- Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Split: train subjects 1-32, val subjects 33-36, test subjects 37-48
- Input resolution: 64x64
- Event input: two polarity channels where the target model supports it
- Main comparable metric: center pixel error in 64x64 coordinates
- Batch size: 32
- DataLoader workers: 4
- Epochs: 70
- Optimizer target: Adam
- Learning rate: 1e-3
- Weight decay: 1e-5

## Scheduler Inventory

The optimizer-level options and scheduler were aligned to HBTXR where the local
code exposes a train loop or Lightning scheduler hook.

| Target | Optimizer after alignment | Scheduler currently used |
|---|---|---|
| HBTXR reference | Adam, lr 1e-3, weight_decay 1e-5 | `timm.scheduler.StepLRScheduler(decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5)` |
| EPNet/FECET | Adam, lr 1e-3, weight_decay 1e-5 | Same FACET `StepLRScheduler` |
| FACET TennSt | Adam, lr 1e-3, weight_decay 1e-5 | Same FACET `StepLRScheduler` |
| Retina | Adam, lr 1e-3, weight_decay 1e-5 | Same HBTXR `StepLRScheduler` |
| TDTracker | Adam, lr 1e-3, weight_decay 1e-5 | Same HBTXR `StepLRScheduler` |
| ERVT | Adam, lr 1e-3, weight_decay 1e-5 | Same HBTXR `StepLRScheduler` |
| TENNs-Eye | Adam, lr 1e-3, weight_decay 1e-5 | Same HBTXR `StepLRScheduler` |
| BRAT/CNN_GRU_base | Adam, lr 1e-3, weight_decay 1e-5 | Same HBTXR `StepLRScheduler` |

## Complexity Inventory

Measured on 2026-06-30 with:

```bash
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/measure_hbtxr_target_complexity.py
```

Detailed outputs:

- `references/report/HBTXR_target_model_complexity_2026-06-30.md`
- `references/report/HBTXR_target_model_complexity_2026-06-30.csv`

| Target | Input shape | Params | MACs | FLOPs |
|---|---:|---:|---:|---:|
| HBTXR | 1x2x64x64 | 4,368,393 | 1,111,665,456 | 2,223,330,912 |
| EPNet/FECET | 1x2x64x64 | 3,898,280 | 215,312,088 | 430,624,176 |
| FACET TennSt | 1x2x50x64x64 | 808,771 | 922,099,200 | 1,844,198,400 |
| Retina | 1x2x64x64 | 59,572 | 21,492,072 | 42,984,144 |
| TDTracker | 1x100x2x64x64 | 3,246,880 | 23,269,558,144 | 46,539,116,288 |
| ERVT | 1x30x3x64x64 | 143,938 | 1,387,069,440 | 2,774,138,880 |
| TENNs-Eye | 1x2x50x64x64 | 808,771 | 922,099,200 | 1,844,198,400 |
| BRAT/CNN_GRU_base | 1x30x2x64x64 | 12,892,898 | 4,509,919,680 | 9,019,839,360 |

Notes:

- MACs were measured with `thop.profile` on CPU dummy inputs.
- FLOPs are reported as `2 * MACs`.
- `EPNet/FECET` is listed this way because no separate `FECET` model path was
  found under `references/codebase/software`; the FACET EPNet baseline is used.

## Priority 1: FACET/FECET EPNet

Status: ready by config.

Config:

```bash
references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_subject_independent_img64.yaml
```

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
FACET_DEVICES=0 /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python tools/train.py \
  -c DavisEyeEllipse_EPNet_subject_independent_img64.yaml
```

Difference from original:

- Original EPNet full-unet config used `DeanDataset_full_unet` and 256x256.
- New config uses subject-independent HBTXR split and 64x64.

Difference from HBTXR:

- Same dataset contract.
- EPNet is CNN/FPN-based, while HBTXR is DeiT-style transformer-based.

## Priority 2: Retina

Status: ready by dataset helper and config.

Added files:

- `references/codebase/software/retina/data/datasets/hbtxr_dean/`
- `references/codebase/software/retina/configs/hbtxr_subject_independent_img64_patch4.yaml`

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/retina
OUTPUT_PATH=/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/retina/runs \
python3 scripts/train.py \
  --device 0 \
  --wandb_mode disabled \
  --project_name hbtxr \
  --run_name Retina_subject_independent_img64 \
  --path_to_config configs/hbtxr_subject_independent_img64_patch4.yaml
```

Difference from original:

- Original Retina supports 64x64 2-channel input but only `ini-30` and `3et-data`
  helpers.
- New helper reads FACET/HBTXR cache directly and emits Retina bbox labels.
- The HBTXR config sets Adam, lr 1e-3, and weight decay 1e-5.

Difference from HBTXR:

- Retina trains a bbox/center target, not native ellipse heatmap, axes, angle, or mask.

## Priority 3: TDTracker

Status: ready after H5 export.

Export:

```bash
cd /home/kjm26/project/PRJXR/HBTXR
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
  --format tdtracker-h5 \
  --output-dir references/codebase/software/ais2025/tdtracker/data/hbtxr_img64 \
  --sequence-length 100 \
  --stride 100
```

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2025/tdtracker
GPU=0 HBTXR_TDTRACKER_DIR=./data/hbtxr_img64 bash run_hbtxr_subject_independent_img64.sh
```

Difference from original:

- Original uses `train_aug.h5` and `test_aug.h5` with 80x60 output.
- TDTracker head/loss/decoder were patched to infer output width/height from config.
- The HBTXR runner uses Adam, lr 1e-3, and weight decay 1e-5.

Difference from HBTXR:

- TDTracker is center-only sequence tracking. It cannot report ellipse IoU natively.

## Priority 4: ERVT AIS2024

Status: ready after 3ET-style export.

Export:

```bash
cd /home/kjm26/project/PRJXR/HBTXR
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
  --format threeet-tree \
  --output-dir references/codebase/software/ais2024/ERVT/event_data_hbtxr_img64
```

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2024/ERVT
python3 train.py --config_file hbtxr_subject_independent_img64.json
```

Difference from original:

- Original assumes 640x480 source with `spatial_factor=0.125`, effective 80x60.
- New config uses exported 64x64 events with `spatial_factor=1.0`.
- The HBTXR config uses Adam, lr 1e-3, and weight decay 1e-5.

Difference from HBTXR:

- ERVT is recurrent/event-transformer center tracking, not ellipse modeling.

## Priority 5: TENNs-Eye AIS2024

Status: ready after 3ET-style export.

Export:

```bash
cd /home/kjm26/project/PRJXR/HBTXR
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
  --format threeet-tree \
  --output-dir references/codebase/software/ais2024/eye_track_spatiotemporal/event_data_hbtxr_img64
```

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2024/eye_track_spatiotemporal
python3 train.py --config-name config_hbtxr_subject_independent_img64
```

Difference from original:

- Original uses hardcoded validation files and 640x480-derived downsampling.
- Dataset now supports `sensor_size` and `data_list_dir` for HBTXR split lists.
- The HBTXR config uses Adam, lr 1e-3, and weight decay 1e-5 instead of the
  original AdamW, lr 2e-3, weight decay 1e-3 default.

Difference from HBTXR:

- TENNs-Eye is center-only temporal tracking with 50-frame segments.

## Priority 6: BRAT AIS2025

Status: ready after 3ET-style export.

Export:

```bash
cd /home/kjm26/project/PRJXR/HBTXR
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py \
  --format threeet-tree \
  --output-dir references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/event_data_hbtxr_img64
```

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution
python3 train.py --config_file hbtxr_subject_independent_img64.json --device 0
```

Difference from original:

- Original `CNN_GRU_base` config assumes 3ET+ 640x480 and challenge scaling.
- Train/test scaling now uses `sensor_width` and `sensor_height` from config.
- The HBTXR config uses Adam, lr 1e-3, and weight decay 1e-5.

Difference from HBTXR:

- BRAT predicts center coordinates only, not ellipse axes, angle, or mask.

## Priority 7: FACET TennSt

Status: ready by FACET sequence dataset adapter and config.

Added files:

- `DavisEyeEllipseCenterSequenceDataset`
- `DavisEyeEllipse_TennSt_subject_independent_img64.yaml`

Train:

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
FACET_DEVICES=0 /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python tools/train.py \
  -c DavisEyeEllipse_TennSt_subject_independent_img64.yaml
```

Difference from original:

- Original FACET TennSt uses `MemmapDavisEyeCenterDataset`, 346x260 input, and
  center labels.
- New adapter builds 50-frame center sequences from HBTXR `cached_ellipse`.

Difference from HBTXR:

- TennSt is center-only temporal tracking. It does not train HBTXR's ellipse heads.

## Export Smoke Validation

The shared exporter was smoke-tested with one output unit per split:

```bash
--format threeet-tree --max-frames-per-split 1
--format tdtracker-h5 --sequence-length 2 --stride 2 --max-sequences-per-split 1
```

Observed TDTracker smoke H5 shapes:

```text
frames: (1, 2, 2, 64, 64)
label:  (1, 2, 2)
```

## Remaining Practical Risk

- Full TDTracker H5 export can be large because it materializes sequence tensors.
- 3ET-style export creates synthetic 100 Hz recording timelines from HBTXR fixed-count
  samples; it preserves subject/session boundaries but is not the original raw EV-Eye
  timestamp stream.
- Center-only models should be compared on center pixel error. Ellipse IoU is only native
  for HBTXR/EPNet-like ellipse models.
