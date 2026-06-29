# Retina Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/retina`

Retina is a low-power event-camera eye tracking model with ANN/SNN variants and hardware-oriented deployment utilities.

Evidence:

- `references/codebase/software/retina/README.md:1-8` identifies Retina and its paper.
- `references/codebase/software/retina/README.md:86-103` documents the 3ET dataset and training command.
- `references/codebase/software/retina/configs/default.yaml:39-49` defines dataset name and 64x64 2-channel input shape.
- `references/codebase/software/retina/data/module.py:7-13` supports only `ini-30` and `3et-data` dataset helpers.
- `references/codebase/software/retina/engine/models/retina/retina.py:25-34` reads `num_bins`, `input_channel`, `img_width`, and `img_height`.

## Original Protocol

Default config:

- `arch_name: retina_ann`
- `batch_size: 32`
- `dataset_name: ini-30`
- `num_bins: 1`
- `input_channel: 2`
- `img_width: 64`
- `img_height: 64`
- Events per frame: `1000`

The data module selects either `ini-30` or `3et-data`, then builds train and val datasets.

Evidence:

- `references/codebase/software/retina/configs/default.yaml:1-38`
- `references/codebase/software/retina/configs/default.yaml:39-67`
- `references/codebase/software/retina/data/module.py:15-46`

## HBTXR Contract Compatibility

Compatibility is medium.

Positive:

- Retina already uses 64x64.
- Retina already uses 2 input channels.
- Batch size 32 matches the HBTXR reference config.

Required changes:

1. Add a new dataset name, for example `hbtxr-dean-subject-independent`.
2. Implement `data/datasets/hbtxr_dean/helper.py` returning train/val/test datasets from `DeanDataset_full_unet_subject_independent`.
3. Convert FACET event segments to Retina input tensors with shape compatible with `(input_channel, img_width, img_height)`.
4. Convert ellipse center to Retina box/center target format used by its loss.
5. Preserve test split explicitly; current data module returns no test dataloader.
6. Verify whether Retina's `bbox_w` and grid settings remain valid at 64x64 for pupil-scale ellipses.

## Expected Output

Possible after adapter:

- Center pixel error.
- Possibly box error if labels are represented as boxes.

Not direct:

- Ellipse IoU, unless ellipse axes are represented or reconstructed.

## Readiness

Status: good candidate after adding a dataset helper.

Risk: medium. The resolution matches already, so most work is data/label mapping rather than model surgery.

