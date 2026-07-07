# Progress

Date: 2026-07-03

## Completed

- Set working repository context to `/home/user/project/PRJXR/HBTXR`.
- Switched analysis context to `etri-desktop`.
- Packaged HBTXR reference outputs under `analysis/results/HBTXR`.
- Packaged Retina training config/checkpoint artifacts.
- Packaged ERVT training config/checkpoint artifacts.
- Generated Retina subject37-48 predictions and pixel-error distribution files.
- Generated ERVT subject37-48 predictions and pixel-error distribution files.
- Found and fixed the Saccade blank issue by using the HBTXR joined motion label map.
- Regenerated Retina/ERVT JETCAS-style Excel workbooks.
- Added validation and execution documentation.

## Current Artifacts

- `analysis/results/Retina`
- `analysis/results/ERVT`
- `analysis/results/HBTXR`
- `analysis/scripts`
- `docs`

## Remaining Notes

- ERVT has fewer joined rows than Retina because sequence segmentation does not cover every sample index.
- Large per-sample CSVs, the Retina log, and the large HBTXR checkpoint are excluded from git tracking for GitHub push compatibility.
