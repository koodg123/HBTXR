# HBTXR

This directory is the reorganized HBTXR workspace assembled from the three
source branches in `HBTXR-Pool`.

## Layout

- `algorithm/`: training, evaluation, packaging, analysis, tests, and docs.
- `hardware/`: hardware-side reference documents and future implementation area.
- `quantization/`: quantization-side reference documents and future implementation area.
- `third/`: vendored third-party references, with nested repository metadata excluded.
- `references/`: paper references and legacy codebases preserved for traceability.

## Provenance

- Baseline branch: `HBTXR-etri-server/HBTXR`.
- Retina/ERVT result packaging additions: `HBTXR-etri-desktop/HBTXR`.
- Annotation and frame/crop experiment additions: `HBTXR-home/HBTXR`.

Large checkpoints, row-level prediction tables, HDF5 files, and local run
outputs are intentionally excluded or routed to artifact storage paths.
