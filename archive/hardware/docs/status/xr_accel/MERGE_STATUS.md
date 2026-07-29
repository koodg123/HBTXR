# Repository Status

This document records the current repository status for `XR_Accel`.

## Current Scope

The active repository scope is:

- baseline `DeiT-Tiny` execution and validation,
- cyclic `DeiT-Tiny` classification hardware bring-up,
- board-oriented VCK190 and ZynqMP execution helpers,
- HBTXR architecture planning and follow-on implementation work.

## Current Structure

The canonical repository layout uses:

- `automation/`
- `configs/`
- `docs/`
- `workspace/flow/`
- `workspace/hardware/`
- `workspace/board/`
- `workspace/artifacts/`

The clustered `workspace/` layout is the current execution root for scripts,
hardware sources, generated build trees, and reports.

## Verified Areas

Repository areas that are already in active use:

- config-driven Python entrypoints under `workspace/flow/entrypoints/python/`
- board and target wrappers under `workspace/flow/scripts/targets/`
- Docker helpers under `workspace/flow/scripts/docker/`
- baseline hardware sources under `workspace/hardware/src/` and
  `workspace/hardware/case*/`
- documentation and runbooks under `docs/`

Recently confirmed items:

- VCK190 baseline flow coverage through `step4-export`
- ZynqMP Spinal export regeneration
- ZynqMP `step4-prepare zcu102` smoke coverage
- standalone exact-match regression for the cyclic `DeiT-Tiny` top
- checked-in synthesis reports for `PATCH_EMBED_IMAGE` and `DEIT_HEAD`

## Open Items

The main open technical items are:

- final top-level `CYCLIC_VIT_TOP` synthesis confirmation
- remaining Spinal `.dat` bundle recovery work
- final `vck190` implementation closure
- final `zcu102` bitstream closure
- later HBTXR architecture expansion after the cyclic DeiT milestone is fully
  closed
