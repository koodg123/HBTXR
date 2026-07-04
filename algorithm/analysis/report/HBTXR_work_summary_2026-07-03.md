# HBTXR Work Summary

Date: 2026-07-03
Branch: `etri-server`

## Scope

This document summarizes the conversation context and current implementation,
training, inference, and packaging state for the HBTXR/FACET reproduction and
model-comparison workflow.

## Durable Decisions

- FACET-related reports and logs are kept under
  `references/report/FACET`.
- Model-comparison result packages are kept under `analysis/RESULTS`.
- Subject-independent evaluation uses the DeanDataset full-UNet split:
  train subjects `1-32`, validation subjects `33-36`, and test subjects
  `37-48`.
- Pixel errors are reported in the model input coordinate system unless a report
  explicitly states otherwise.
- Blink rows are excluded from error-distribution tables through
  `analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv`.
- `FECET` was confirmed as a typo and removed from the active target list.
- `EIDet` refers to the local FACET `ElNet` implementation.

## Dataset And Split State

- FACET reproduction initially used the labelled Davis eye data and then moved
  to the expanded subject-independent DeanDataset generated with U-Net labels.
- The active subject-independent dataset split is:
  - train: subjects `1-32`
  - val: subjects `33-36`
  - test: subjects `37-48`
- Motion labels are handled as row-level labels for Fixation, Saccade, and
  Smooth. Blink is treated as a separate exclusion flag for error tables rather
  than as a mutually exclusive motion class.
- The 4-state motion-distribution analysis artifacts were generated under
  `analysis/RESULTS` and `references/report/FACET`.

## Model Training And Inference State

### HBTXR

- HBTXR subject-independent `img64_patch4` training and evaluation were run
  earlier.
- HBTXR full-UNet `img128_patch4` was trained and packaged under:
  `analysis/RESULTS/HBTXR_full_unet_img128_patch4`.
- HBTXR predicts ellipse outputs and reports native ellipse-shape IoU.

### EPNet

- EPNet subject-independent `img64` training was launched from the FACET
  implementation with settings aligned to the HBTXR subject-independent setup.
- Its subject 37-48 test result package is under:
  `analysis/RESULTS/EPNet_subject_independent_img64`.

### TennSt And TENNs-Eye

- TennSt and TENNs-Eye were evaluated and packaged under:
  - `analysis/RESULTS/TennSt`
  - `analysis/RESULTS/TENNs-Eye`
- These are center-output workflows, so center pixel error is the primary metric.

### TDTracker

- TDTracker subject-independent `img64` training required H5 conversion and
  runtime fixes.
- Known blockers addressed earlier:
  - optionalized unconditional `open3d` import in `provider_data.py`
  - disabled cuDNN in `train.py` to avoid the local cuDNN sublibrary mismatch
- Its subject 37-48 test result package is under:
  `analysis/RESULTS/TDTracker`.

### EIDet

- EIDet uses the local FACET `ElNet` implementation.
- Native DCNv2 was not available in the environment, so a PyTorch/torchvision
  deform-convolution replacement path was added through:
  - `references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/ElNet/torchvision_dcn.py`
  - modifications in `ElNet.py`
- The subject-independent `img64` config is:
  `references/codebase/software/FACET/configs/DavisEyeEllipse_ElNet_subject_independent_img64.yaml`
- The result package is under `analysis/RESULTS/EIDet`.

### BRAT

- BRAT refers to
  `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution`.
- A right-eye-only duplication issue was found in the first BRAT test export:
  session ids did not include the eye side, causing left/right export overwrite.
- The exporter was fixed so BRAT/3ET sessions are named with explicit eye
  tokens such as `user37_L_1_0_1` and `user37_R_1_0_1`.
- A full-test BRAT export was generated under:
  `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/event_data_hbtxr_img64_fulltest`.
- The full-test config is:
  `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/configs/hbtxr_subject_independent_img64_fulltest.json`.
- `test.py` now supports `BRAT_DISABLE_CUDNN=1` to avoid the local cuDNN
  sublibrary mismatch during inference.
- BRAT was re-run on GPU1 using the same Subject 37-48 full test set used by
  EIDet.
- The corrected BRAT result package is under `analysis/RESULTS/BRAT`.

## Result Package Summary

The active model result packages follow the HBTXR reference package layout
locally:

- per-sample prediction CSV
- sample metadata CSV
- joined motion/error CSV
- dropped-blink CSV
- subject/motion summary CSV
- JETCAS-style Excel workbook
- Markdown report
- copied checkpoint/config provenance where available

For GitHub push, large row-level CSV/H5/weight/log files are excluded from git
tracking and retained only in the local workspace. The tracked package keeps
summary CSVs, workbooks, reports, configs, scripts, and lightweight label files.

Primary package summary:

- `analysis/RESULTS/model_inference_package_summary_2026-07-03.md`

Current EIDet and BRAT weighted test summary:

| Model | Valid N | Weighted Mean Error | Weighted Median Error | Weighted Mean IoU |
|---|---:|---:|---:|---:|
| EIDet | 360,255 | 3.849677 | 2.386649 | 0.357419 |
| BRAT | 360,495 | 0.879130 | 0.477422 | 0.770778 |

BRAT inference verification:

- corrected BRAT test file list: `96` entries, `96` unique
- raw BRAT prediction rows: `366,171`
- joined non-Blink rows: `360,495`
- inference command used GPU1 with `BRAT_DISABLE_CUDNN=1`
- average BRAT inference time reported by `test.py`: `4.2751710813288275 ms`

## Code And Script Changes

- `analysis/scripts/fill_facet_ellipse_error_distribution.py`
  generates FACET-style ellipse model error-distribution packages.
- `analysis/scripts/fill_brat_error_distribution.py`
  maps BRAT center predictions to the corrected Subject 37-48 full test
  metadata and fills BRAT error-distribution tables.
- `analysis/scripts/fill_tdtracker_error_distribution.py`
  packages TDTracker sequence-center outputs.
- `analysis/scripts/fill_sequence_center_error_distribution.py`
  handles sequence center-output models.
- `export_hbtxr_subject_independent_for_targets.py` now preserves eye side in
  BRAT/3ET session ids and supports split-selective export.
- BRAT `test.py` now supports a cuDNN-disable environment flag.
- TDTracker scripts were adjusted for local runtime compatibility.

## Verification Performed

- Verified active branch: `etri-server`.
- Verified BRAT corrected test export has `96` unique eye-session entries.
- Verified BRAT full-test `submission_test.csv` has `366,171` prediction rows.
- Verified BRAT package generation produced:
  `analysis/RESULTS/BRAT/JETCAS_REPLY_TABLES (Error-Distributions)_BRAT.xlsx`.
- Verified stale BRAT right-eye-only summary language was removed from the
  2026-07-03 model package summary.

## Remaining Risks

- BRAT was evaluated on the corrected full test set, but the checkpoint itself
  was trained before the corrected full L/R export was introduced. A fully fair
  BRAT comparison requires retraining BRAT with corrected train/val/test export.
- Center-only models do not predict ellipse axes/angle. Their IoU values are
  proxy values unless a report explicitly states a native ellipse reconstruction
  method.
- Generated result packages and exported BRAT full-test data are large. They
  should remain local unless explicitly moved to Git LFS or external artifact
  storage.
