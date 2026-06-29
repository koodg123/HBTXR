# FECET Porting Review

Date: 2026-06-29

## Local Mapping

No local codebase, class, config, or README named `FECET` was found under `references/codebase/software`.

The closest and likely intended target is `FACET`:

- Source path: `references/codebase/software/FACET`
- FACET training entrypoint: `references/codebase/software/FACET/tools/train.py`
- FACET dataset path used by HBTXR: `DavisEyeEllipseDataset`
- FACET model registry includes `EPNet`, `HBTXR`, `TennSt`, `ElNet`, and `UNet`.

Evidence:

- `references/codebase/software/FACET/README.md:70-82` documents FACET's train/validate commands.
- `references/codebase/software/FACET/EvEye/dataset/dataset_factory.py:23-32` registers `DavisEyeEllipseDataset`.
- `references/codebase/software/FACET/EvEye/model/model_factory.py:5-13` registers FACET models including `TennSt`, `EPNet`, and `HBTXR`.
- `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml:5-18` is the current reference split/resolution config.

## Original Protocol

FACET uses Lightning-style YAML configs. The original `EPNet_full_unet` config uses:

- Dataset: `DavisEyeEllipseDataset`.
- Input root: `DeanDataset_full_unet`.
- Train split: `train`.
- Val split: `val`.
- Input resolution: 256x256 in the older full-unet EPNet config.
- Model: `EPNet`.
- Optimizer: learning rate `1e-3`, weight decay `1e-5`.

Evidence:

- `references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml:5-18`
- `references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml:21-37`
- `references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml:39-59`

## HBTXR Contract Compatibility

Compatibility is high if `FECET` means `FACET/EPNet`.

Required changes:

1. Copy the HBTXR subject-independent config.
2. Set `model.type: EPNet`.
3. Preserve `root_path: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`.
4. Preserve `default_resolution: [64, 64]`.
5. Preserve train/val split names and dataloader batch/worker settings unless GPU memory requires tuning.
6. Use a separate logger and run directory, for example `EPNet_subject_independent_img64`.

## Expected Output

FACET/EPNet can directly report:

- Center pixel error.
- Ellipse axes/angle estimates.
- Ellipse mask and IoU, if the validation loop decodes those outputs.
- Same validation cadence as HBTXR.

## Readiness

Status: ready after config creation.

Risk: low. This is the only target family already designed around the same cached event/ellipse data format as HBTXR.

