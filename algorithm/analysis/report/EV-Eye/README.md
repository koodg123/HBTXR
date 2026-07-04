# EV-Eye Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/EV-Eye`

EV-Eye is both a dataset repository and benchmark codebase. The local Python training path is a U-Net pupil segmentation benchmark, not a direct event-based center/ellipse tracker.

Evidence:

- `references/codebase/software/EV-Eye/README.md:42-55` describes the dataset organization.
- `references/codebase/software/EV-Eye/README.md:60-67` describes `Data_davis`.
- `references/codebase/software/EV-Eye/README.md:94-110` describes `Data_davis_labelled_with_mask`.
- `references/codebase/software/EV-Eye/README.md:135-140` describes benchmark metrics.
- `references/codebase/software/EV-Eye/README.md:191-208` documents Python U-Net training/evaluation.

## Original Protocol

The local `train.py`:

- Reads HDF5 frame/mask files from `Data_davis_labelled_with_mask`.
- Uses sessions `1_0_2`, `2_0_1`, and `2_0_2`.
- Runs a leave-one-subject-out loop over users 1-48.
- Trains on all users except the held-out user.
- Validates/tests on the held-out user.
- Uses 1-channel frame input of shape `1x260x346`.
- Uses U-Net segmentation with CrossEntropy plus Dice loss.

Evidence:

- `references/codebase/software/EV-Eye/train.py:62-70`
- `references/codebase/software/EV-Eye/train.py:80-116`
- `references/codebase/software/EV-Eye/train.py:120-154`
- `references/codebase/software/EV-Eye/train.py:156-189`
- `references/codebase/software/EV-Eye/utils/data_loading.py:12-23`
- `references/codebase/software/EV-Eye/utils/data_loading.py:61-80`

## HBTXR Contract Compatibility

Compatibility is low for direct HBTXR-equivalent model training.

Why:

- EV-Eye Python model trains segmentation masks, not event-frame pupil center/ellipse regression.
- It uses manually labelled mask HDF5 files, not the full pseudo-labelled `DeanDataset_full_unet_subject_independent` event cache.
- Original resolution is 260x346, not 64x64.
- Original subject protocol is leave-one-subject-out, not train 1-32 / val 33-36 / test 37-48.

Required changes if EV-Eye U-Net is still desired:

1. Decide whether the model should train on frame images/masks or event-frame/masks.
2. Export HBTXR split samples into frame/mask pairs at 64x64.
3. Replace the leave-one-subject-out loop with fixed subject-independent split loaders.
4. Preserve both left and right eye sessions according to the HBTXR split manifest.
5. Add center extraction from predicted masks if pixel error is required.
6. Add ellipse fitting from predicted masks if IoU/ellipse metrics are required.

## Expected Output

Direct output:

- Segmentation mask.
- Dice and IoU for masks.

Derived output:

- Center pixel error after mask-to-center conversion.
- Ellipse IoU after mask-to-ellipse fitting.

## Readiness

Status: special-purpose; not an immediate competitor to HBTXR unless the benchmark accepts mask segmentation models with post-processing.

