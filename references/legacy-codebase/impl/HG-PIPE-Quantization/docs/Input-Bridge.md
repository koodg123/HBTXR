# ImageNet Input Bridge

## Purpose

This document records the current bridge from normalized ImageNet tensors to the integer patch input consumed by the HG-PIPE artifact graph.

## Evidence

The public ICCAD24-HG-PIPE checkout exposes the patch embedding input contract in case/PATCH_EMBED.cpp and src/patch_embed.h. The statistics files report patch_embed.input as signed 8-bit with observed range -102 to 127.

The checkout does not include the original image preprocessing, calibration, QAT checkpoint generation, or image-to-patch quantization script. Therefore this bridge is implemented as an explicit experimental utility, not as a paper-equivalent calibration reproduction.

## Implemented Utility

File: hgpipe_quantization/input_bridge.py

Implemented functions:

- extract_patch_vectors: converts a CHW or HWC 224x224 image tensor into 196 patch vectors of width 768.
- quantize_patches_symmetric: applies signed int8 symmetric quantization with explicit caller-provided scale.
- to_hgpipe_patch_input: inserts the HG-PIPE cls slot behavior by zeroing token 0 and shifting image patches into tokens 1 through 195.
- patch_input_contract: exposes the signed int8 and observed artifact range contract.

## Validation

Unit tests verify patch flatten order, cls slot insertion, last-patch drop behavior implied by PATCH_EMBED.cpp, signed int8 clamp behavior, and artifact range metadata.

## Remaining Limit

The bridge requires an explicit scale or externally validated calibration policy. Without the original HG-PIPE preprocessing and calibration script, arbitrary ImageNet accuracy from this artifact graph must be marked as experimental and not paper-equivalent.

## Experimental Runner Path

The CLI command run-compare-image-npy accepts a normalized CHW or HWC NumPy image tensor plus an explicit positive scale. It converts the tensor through to_hgpipe_patch_input, runs both the torch.int graph and the FakeQuant graph, and writes JSON plus Markdown comparison reports.

Example:

python3 -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-compare-image-npy --image-npy reports/smoke_zero_image.npy --scale 1.0 --json reports/run_compare_image_result.json --markdown reports/run_compare_image_result.md

The output includes input_bridge.paper_equivalent=false because this path does not recover the missing original preprocessing/calibration policy.

## Experimental Scale Estimate

The CLI command estimate-input-scale-npy estimates a symmetric max-absolute scale over normalized CHW/HWC image tensors stored in a single image .npy or batch .npy file.

Example:

python3 -m hgpipe_quantization.cli estimate-input-scale-npy --images-npy reports/smoke_scale_images.npy --json reports/input_scale_estimate.json

The estimate is experimental and has paper_equivalent=false because it is not the original HG-PIPE calibration pipeline.
