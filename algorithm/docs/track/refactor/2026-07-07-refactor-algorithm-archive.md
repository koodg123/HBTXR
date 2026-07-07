# Refactor: Algorithm Archive Separation

**Date:** 2026-07-07
**Mode:** target
**Files changed:** archive relocation plus documentation

## Target

- `algorithm/analysis/conversations` moved to `algorithm/archive/imports/conversations`.
- `algorithm/analysis/imported-docs` moved to `algorithm/archive/imports/hgtxr/docs`.
- Legacy hybrid docs, scripts, configs, and reference tests moved to
  `algorithm/archive/imports/legacy_hybrid`.
- Active hybrid scripts/configs/tests now describe maintained surfaces only.

## Transformation

This was a behavior-preserving structural move. Runtime source files under
`algorithm/hybrid/src` were not changed in this refactor commit.

## Verification

- Baseline before the checkpoint commit:
  `PYTHONPATH=algorithm/hybrid python3 -m compileall -q algorithm/hybrid/src`
- Final verification planned:
  compile active hybrid source, inspect active legacy/imported directory
  patterns, and check Git status.

## Notes

The first commit on this branch is a checkpoint of the previously dirty import
state. This refactor is intentionally separate so future reviews can distinguish
source integration from archive organization.
