# HBTXR Model Result Packages

Date: 2026-07-03

This directory organizes model result artifacts under one folder per model.
The package layout follows the existing reference package:

`/mnt/d/dataset/EV_Eye/paper_works/RESULTS/HBTXR_subject37_48_error_distribution`

## Package Status

| Model | Status | Notes |
|---|---|---|
| HBTXR | packaged | Full subject37-48 test error-distribution package copied from the reference RESULTS tree. |
| Retina | packaged | Subject37-48 test pixel-error distribution CSVs and JETCAS-style Excel table generated from the best Retina checkpoint using the HBTXR joined motion label map. |
| ERVT | packaged | Subject37-48 test pixel-error distribution CSVs and JETCAS-style Excel table generated from the best ERVT checkpoint using the HBTXR joined motion label map. The label-map join leaves 4,587 unmatched rows. |
| EPNet_FECET | pending | No model-specific subject37-48 error-distribution output found in this workspace. |
| FACET_TennSt | pending | No model-specific subject37-48 error-distribution output found in this workspace. |
| TDTracker | pending | No completed model-specific subject37-48 error-distribution output found in this workspace. |
| TENNs-Eye | pending | No model-specific subject37-48 error-distribution output found in this workspace. |
| BRAT | pending | No completed model-specific subject37-48 error-distribution output found in this workspace. |

## Available Metric Summary

See `available_metrics.csv` for the current metric inventory. HBTXR, Retina,
and ERVT have subject37-48 test error-distribution packages at this time.

## Large Local-Only Files

Large per-sample CSV tables, logs, and the HBTXR checkpoint are intentionally
excluded from git tracking to satisfy GitHub file size limits. See
`LARGE_FILES_MANIFEST.md` for the excluded file list and regeneration notes.
