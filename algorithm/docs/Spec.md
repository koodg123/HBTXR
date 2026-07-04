# HBTXR Reorganization Spec

## Required Root Layout

The root `HBTXR` directory must keep project domains separated:

- `hardware/`
- `quantization/`
- `algorithm/`
- `third/`
- `references/`
- `README.md`
- `.gitignore`

## Required Algorithm Layout

`algorithm/` contains runnable code and experiment assets:

- `src/EvEye/` for the preserved Python package.
- `configs/` for YAML experiment configs.
- `scripts/` for entrypoints and analysis helpers.
- `analysis/` for result summaries, reports, motion labels, and packaging.
- `tests/` for existing and future smoke tests.
- `docs/` for planning and validation artifacts.

## Artifact Policy

Large checkpoints, row-level CSVs, HDF5 files, tensorboard logs, and local run
outputs should not be treated as normal source files. They are excluded in
`.gitignore` and should be moved to explicit artifact storage if they must be
retained.
