# Retina/ERVT Result Package Report

Date: 2026-07-03
Branch: etri-desktop

## What Was Built

Retina and ERVT now have HBTXR-style subject37-48 pixel-error distribution packages under `analysis/results`.

The package includes per-sample predictions, metadata joins, HBTXR joined-motion error joins, motion counts, motion-specific distribution tables, status files, and JETCAS-style Excel workbooks.

## Output Paths

- `analysis/results/Retina`
- `analysis/results/ERVT`
- `analysis/results/JETCAS_REPLY_TABLES (Error-Distributions)_Retina_ERVT_subject37_48.xlsx`

## Important Correction

The first Excel generation used raw metadata `motion_state`. That produced blank Saccade columns for some subjects because the raw metadata had zero Saccade labels for subjects 39, 42, 44, 46, and 48.

The final package uses the HBTXR joined motion label map:

`analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv`

This aligns Retina and ERVT motion grouping with the HBTXR reference Excel.

## Scripts

- `analysis/scripts/generate_retina_ervt_error_distribution.py`
- `analysis/scripts/rebuild_retina_ervt_with_hbtxr_motion_labels.py`
- `analysis/scripts/create_retina_ervt_jetcas_tables.py`

## Final State

Saccade rows are populated for every subject 37-48 in both Retina and ERVT Excel workbooks.

## GitHub Size Policy

Large per-sample CSV tables, the Retina training log, and the large HBTXR checkpoint are local-only and ignored for git push compatibility. The tracked package keeps compact distribution CSVs, joined motion counts, Excel tables, scripts, configs, and documentation. See `analysis/results/LARGE_FILES_MANIFEST.md`.
