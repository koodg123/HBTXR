# HBTXR Work Summary

Date: 2026-07-01

## Scope

This document summarizes the current workspace changes prepared for commit on the
`etri-server` branch.

## Analysis And Result Artifacts

- Added HBTXR subject-independent test error distribution artifacts under
  `analysis/RESULTS/HBTXR_subject37_48_error_distribution`.
- Filled `analysis/RESULTS/JETCAS_REPLY_TABLES (Error-Distributions).xlsx`
  using subject 37-48 row-level motion labels from the motion-label packages.
- Added TennSt and TENNs-Eye subject-independent test result folders under
  `analysis/RESULTS/TennSt` and `analysis/RESULTS/TENNs-Eye`.
- Generated per-model joined motion/error tables, dropped-blink logs,
  prediction metadata, and filled JETCAS-style error-distribution workbooks.
- Added data-distribution and motion-label package artifacts under:
  - `analysis/data-distribution`
  - `analysis/motion-label-packages`

## Evaluation Scripts

- Added HBTXR error-distribution fill script:
  `analysis/scripts/fill_hbtxr_error_distribution.py`.
- Added sequence center-model error-distribution fill script:
  `analysis/scripts/fill_sequence_center_error_distribution.py`.
- The sequence script is used for center-only models such as TennSt and
  TENNs-Eye, where ellipse IoU is not native and can only be reported through a
  proxy if needed.

## FACET/HBTXR Operation Scripts

- Added EPNet subject-independent GPU1 launch helper:
  `references/report/FACET/operations/run_epnet_subject_independent_img64_gpu1_2026-06-30.sh`.
- Added chained HBTXR error-table plus EPNet training helper:
  `references/report/FACET/operations/run_hbtxr_error_table_then_epnet_subject_independent_gpu1_2026-06-30.sh`.

## Model Target Naming Cleanup

- Removed the misspelled report-side target from the comparison list because it
  was confirmed to be a typo.
- Deleted the obsolete report entry for that misspelled target.
- Renamed the old combined EPNet labels to `EPNet`.
- Updated the target complexity measurement script to use `EPNet` as the FACET
  CNN/FPN ellipse baseline name.
- Kept the older separate implementation project under `references/impl`
  untouched because it is not the report-side target list.

## Target Comparison State

Current report-side target list after cleanup:

- HBTXR
- EPNet
- TennSt
- Retina
- EX-Gaze
- EV-Eye
- Swift-Eye
- E-Track
- ERVT
- TENNs-Eye
- BRAT
- TDTracker

Prediction-format classification:

- Native ellipse: HBTXR, EPNet.
- Rotated box or ellipse-compatible shape after adapter: EX-Gaze, Swift-Eye,
  E-Track.
- Mask-first segmentation: EV-Eye.
- Center-only tracking: TennSt, TENNs-Eye, Retina, ERVT, BRAT, TDTracker.

## Validation Performed

- Verified that `references/report` and the target complexity script no longer
  contain the misspelled target label or the old combined EPNet label.
- Verified the active git branch is `etri-server` before committing.

## Remaining Notes

- Some result artifacts are generated workbooks, CSVs, figures, caches, and
  Python bytecode files. They are included because the requested action was to
  track all current changes.
- Center-only model comparisons should be reported using center pixel error as
  the primary metric. Ellipse IoU for those models requires a clearly marked
  proxy or reconstruction rule.
