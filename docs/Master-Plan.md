# HBTXR Retina/ERVT Result Packaging Master Plan

Date: 2026-07-03
Branch: etri-desktop

## Goal

Create reproducible HBTXR-style subject-independent evaluation packages for Retina and ERVT, then document and commit the generated artifacts.

## Scope

- Dataset: `/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent`
- Output root: `analysis/results`
- Target models for this package: Retina and ERVT
- Reference package: `analysis/results/HBTXR`
- Reference JETCAS table format: `/mnt/d/dataset/EV_Eye/paper_works/RESULTS/HBTXR_full_unet_img128_patch4/JETCAS_REPLY_TABLES (Error-Distributions)_HBTXR_full_unet_img128_patch4.xlsx`

## Strategy

1. Preserve HBTXR reference artifacts under `analysis/results/HBTXR`.
2. Generate Retina and ERVT per-sample predictions from their best available checkpoints.
3. Rebuild Retina and ERVT motion-specific error distributions using the HBTXR joined motion label map.
4. Generate JETCAS-style Excel workbooks for Retina, ERVT, and a combined Retina/ERVT workbook.
5. Record scripts, decisions, validation evidence, and progress in repo-local docs.

## Current Result

Retina and ERVT have generated subject37-48 CSV packages and Excel tables under:

- `analysis/results/Retina`
- `analysis/results/ERVT`

The final motion grouping uses `analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv` as the authoritative `sample_idx -> motion_state` label map.
