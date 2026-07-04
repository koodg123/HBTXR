# HBTXR img128 patch4 GPU0 Training Launch

## Purpose

This experiment tests a middle-resolution HBTXR setting between the fast `64x64` ablation and the paper-equivalent `256x256` input setting.

## Configuration

- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_img128_patch4.yaml`
- Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet`
- GPU target: GPU0 through `FACET_DEVICES=0`
- Input resolution: `128x128`
- Patch size: `4`
- Token grid: `32x32`
- Token count: `1024`
- Target/output heatmap resolution: `32x32`
- Train batch size: `32`
- Validation batch size: `32`
- Workers: `8`
- Gradient accumulation: default `1`
- Logger name: `HBTXR_full_unet_img128_patch4`

## Interpretation

This is a speed/accuracy ablation. It is closer to the original `256x256` input run than the `64x64` ablation because it keeps four times more tokens than `64x64`, but it is still not paper-equivalent because the output heatmap is `32x32` instead of `64x64`.

## Launcher

`references/report/FACET/run_hbtxr_img128_patch4_gpu0_2026-06-27.sh`

The launcher performs:

1. Full dataset readiness check against `manifest.json` and `progress_state.json`.
2. Shape and loss smoke test for `128x128` input and `32x32` outputs.
3. Training on GPU0 with `DavisEyeEllipse_HBTXR_full_unet_img128_patch4.yaml`.

## Log

`references/report/FACET/HBTXR_img128_patch4_gpu0_train_2026-06-27.log`
