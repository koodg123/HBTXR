# Swift-Eye HBTXR Direct Training Preparation

Date: 2026-07-04

## Goal

Prepare Swift-Eye training up to the point immediately before launching full GPU training with `64x64` HBTXR event input and a 2-channel Swift-Eye backbone input.

## Decision

The prepared path does not export PNG files, DOTA annotations, or temporal pair pickles.

Instead, `analysis/scripts/swifteye_hbtxr_direct_train.py` reads the HBTXR subject-independent cache directly:

- Source: `/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent`
- Train split: subject 1-32
- Val split: subject 33-36
- Test split: subject 37-48
- Event input: structured event cache resized to `2x64x64`
- Backbone first layer: `SwinTransformer.patch_embed.projection` changed from `Conv2d(3,96,kernel=4,stride=4)` to `Conv2d(2,96,kernel=4,stride=4)`
- Detector RPN anchor scales: `[2,4]`, reduced for 64px pupil boxes
- Temporal fusion crop: `search_shape=15`, `template_shape=7`, reduced for a `16x16` P2 feature map
- Temporal tracking anchor scales: `[2,3,4]`, reduced for 64px pupil boxes
- Detector config: `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`
- Temporal config: `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_with_temporal_fusion_component/model_config.py`

The original Swift-Eye temporal tracking anchor generator hard-coded the original `33x33` search crop geometry. It is now parameterized with defaults preserving the original behavior, and the direct HBTXR path supplies the reduced `15/7` crop geometry for 64px training.

## Prepared Artifacts

- Direct training script: `analysis/scripts/swifteye_hbtxr_direct_train.py`
- Detector launch wrapper: `references/report/HBTXR_runs/run_swifteye_direct_detector_train.sh`
- Temporal launch wrapper: `references/report/HBTXR_runs/run_swifteye_direct_temporal_train.sh`
- Preparation metadata: `analysis/results/Swift-Eye/direct_hbtxr_img64/direct_hbtxr_training_preparation.json`

## Dataset Counts

Validated with:

```bash
PYTHONPATH=references/codebase/software/Swift-Eye/mmrotate \
tmp/venvs/mmrotate_py38/bin/python analysis/scripts/swifteye_hbtxr_direct_train.py \
  --mode inspect --output-root /tmp/swifteye_direct_img64_prep_smoke
```

Result:

| Split | Valid detector samples | Valid temporal pairs |
|---|---:|---:|
| train | 968,522 | 962,813 |
| val | 122,696 | 122,300 |
| test | 366,130 | 364,159 |

Temporal pairs are generated on the fly with `index -> index+1`, rejecting pairs with invalid ellipse labels, non-increasing timestamps, or timestamp gaps larger than `50,000 us`.

## Smoke Validation

All smoke checks were run in `tmp/venvs/mmrotate_py38` with local Swift-Eye MMRotate on CPU.

Passed checks:

- `py_compile` for `analysis/scripts/swifteye_hbtxr_direct_train.py`
- metadata write
- split/pair inspection
- detector forward loss
- detector backward and optimizer step
- temporal fusion forward loss
- temporal fusion backward and optimizer step
- train mode checkpoint/history creation with `--limit 1 --epochs 1`

Observed one-batch smoke losses:

| Stage | Check | Result |
|---|---|---:|
| Detector | backward smoke | loss around 1.69 |
| Detector | train mode checkpoint smoke | loss around 3.75 |
| Temporal fusion | backward smoke | loss around 150.87 |
| Temporal fusion | train mode checkpoint smoke | loss around 126.57 |

The absolute losses are not meaningful because these were one-sample randomly initialized smoke checks.

## Launch Commands

Detector stage:

```bash
PYTHON=tmp/venvs/mmrotate_py38/bin/python DEVICE=cpu \
bash references/report/HBTXR_runs/run_swifteye_direct_detector_train.sh
```

Temporal fusion stage:

```bash
PYTHON=tmp/venvs/mmrotate_py38/bin/python DEVICE=cpu \
bash references/report/HBTXR_runs/run_swifteye_direct_temporal_train.sh
```

For real training, replace `DEVICE=cpu` with `DEVICE=cuda` only after a GPU-enabled MMRotate environment is available.

## Remaining Blocker Before Full Training

The current `tmp/venvs/mmrotate_py38` stack is functional for build/smoke, but its PyTorch is `1.13.0+cpu`. It is not suitable for full Swift-Eye training.

Full GPU training needs one of these:

1. A GPU-enabled MMRotate 0.3.4/MMCV 1.x stack compatible with the installed RTX 50-series driver and CUDA runtime.
2. A carefully ported MMRotate runtime in the main `.venv` while preserving the original Swift-Eye model configs and custom heads.

Until that environment issue is solved, the code and data path are ready, but long training should not be launched.
