# Artifact Patch Matrix Manifest Preflight

## Summary

- Status: failed
- Passed: False
- Rows: 9/9
- Paper-equivalent rows: 9
- Valid-flow rows: 9
- Existing patch input files: 0
- Existing label files: 0
- Loadable patch input files: 0
- Loadable label files: 0
- Valid patch-shape rows: 0
- Valid label-shape rows: 0
- Integer patch dtype rows: 0
- Integer label dtype rows: 0
- Matching sample-count rows: 0
- Patch input width: 150528

## Missing Files

Patch input files:
- configs/configs/patch_inputs/deit_tiny_patch16_224_int8_patch_inputs.npy
- configs/configs/patch_inputs/deit_tiny_patch16_224_int4_patch_inputs.npy
- configs/configs/patch_inputs/deit_tiny_patch16_224_w4a8_patch_inputs.npy
- configs/configs/patch_inputs/deit_small_patch16_224_int8_patch_inputs.npy
- configs/configs/patch_inputs/deit_small_patch16_224_int4_patch_inputs.npy
- configs/configs/patch_inputs/deit_small_patch16_224_w4a8_patch_inputs.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_int8_patch_inputs.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_int4_patch_inputs.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_w4a8_patch_inputs.npy
Label files:
- configs/configs/patch_inputs/deit_tiny_patch16_224_int8_labels.npy
- configs/configs/patch_inputs/deit_tiny_patch16_224_int4_labels.npy
- configs/configs/patch_inputs/deit_tiny_patch16_224_w4a8_labels.npy
- configs/configs/patch_inputs/deit_small_patch16_224_int8_labels.npy
- configs/configs/patch_inputs/deit_small_patch16_224_int4_labels.npy
- configs/configs/patch_inputs/deit_small_patch16_224_w4a8_labels.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_int8_labels.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_int4_labels.npy
- configs/configs/patch_inputs/vit_tiny_patch16_224_w4a8_labels.npy

## Invalid Entries

- [0] deit_tiny_patch16_224 int8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [1] deit_tiny_patch16_224 int4: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [2] deit_tiny_patch16_224 w4a8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [3] deit_small_patch16_224 int8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [4] deit_small_patch16_224 int4: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [5] deit_small_patch16_224 w4a8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [6] vit_tiny_patch16_224 int8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [7] vit_tiny_patch16_224 int4: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count
- [8] vit_tiny_patch16_224 w4a8: patch_input_missing, label_missing, patch_input_shape, label_shape, patch_input_dtype, label_dtype, sample_count

## Missing Pairs

- none

## Errors

- integer_label_dtype_rows
- integer_patch_dtype_rows
- label_files
- loadable_label_files
- loadable_patch_input_files
- matching_sample_count_rows
- patch_input_files
- valid_label_shape_rows
- valid_patch_shape_rows
