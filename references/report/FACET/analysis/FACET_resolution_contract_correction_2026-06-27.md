# FACET Resolution Contract Correction

Date: 2026-06-27

## Summary

The FACET paper uses `64x64` as the final feature map, heatmap, and metric
resolution. It does not mean that the raw event frame input to EPNet is
`64x64`.

For the current reproduction setup, the correct resolution contract is:

- input event tensor: `(B, 2, 256, 256)`
- label/output downsample ratio: `4`
- heatmap, mask, and detection heads: `(B, C, 64, 64)`

## Evidence

- Paper Fig. 2 shows fixed-count event binning into `(256x256x2)`.
- Paper network section describes the final FPN feature map as `(64, 64, 64)`.
- Paper heads section describes the heatmap head as `64x64`.
- Paper evaluation section says metrics are obtained at `64x64` resolution.
- Code config `DavisEyeEllipse_EPNet_full_unet.yaml` uses
  `default_resolution: [256, 256]`.
- `DavisEyeEllipseDataset.py` uses `down_ratio = 4`, therefore `256 / 4 = 64`.
- HBTXR with `img_size: 256` and `patch_size: 4` produces a `64x64` patch map.

## Correction

An intermediate HBTXR config change interpreted `img_size: 64` as the paper
resolution. That would make the model input `64x64` and the output target
`16x16`, because the dataset still applies a `down_ratio` of 4.

The HBTXR configs and launcher smoke check were restored to the paper/EPNet
contract:

- `default_resolution: [256, 256]`
- `img_size: 256`
- `patch_size: 4`

The dataset mask allocation was kept generalized to
`(num_classes, output_height, output_width)` instead of hard-coded `64x64`.
This preserves current `256 -> 64` behavior and avoids future shape bugs if a
separate low-resolution experiment is intentionally added.

## Decision

For FACET paper reproduction and fair EPNet-vs-HBTXR comparison, use `256x256`
input and `64x64` output/head/metric resolution.

A true `64x64` input experiment should be treated as a separate ablation, not
as the paper reproduction setting.
