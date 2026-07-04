# HBTXR img64 patch4 Training Launch

Date: 2026-06-27

## Purpose

HBTXR `img_size=256, patch_size=4` produces a `64x64` token grid
(`4096` patch tokens), which makes DeiT attention training slow. This run is a
fast ablation that reduces the input to `64x64` while keeping `patch_size=4`.

## Configuration

- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_img64_patch4.yaml`
- Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet`
- Train input resolution: `64x64`
- Label/output downsample ratio: `4`
- Heatmap/output resolution: `16x16`
- Patch size: `4`
- Patch grid: `16x16`
- Patch tokens: `256`
- Batch size: `32`
- Logger name: `HBTXR_full_unet_img64_patch4`
- GPU target: GPU1 via `FACET_DEVICES=1`

## Interpretation

This is not the paper-equivalent `64x64` heatmap reproduction setting. The
paper/FACET baseline uses `256x256` input and `64x64` output. This run should
be interpreted as a low-resolution speed ablation with `16x16` heatmap output.

## Launcher

`references/report/FACET/run_hbtxr_img64_patch4_gpu1_2026-06-27.sh`

The launcher performs:

1. Dataset manifest/progress completion gate.
2. Shape and loss smoke check.
3. Training on GPU1 with `DavisEyeEllipse_HBTXR_full_unet_img64_patch4.yaml`.

## Validation Plan

- Verify sample input shape: `(2, 64, 64)`.
- Verify target `hm` and `mask` shape: `(1, 16, 16)`.
- Verify prediction heads match target spatial shape: `(B, C, 16, 16)`.
- Verify one loss computation succeeds before starting full training.
- Verify tmux session and GPU process after launch.
