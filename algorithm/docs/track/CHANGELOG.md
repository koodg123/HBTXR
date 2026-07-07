# Changelog

## 2026-07-07

- Checkpointed the pre-refactor imported HBTXR/HGTXR dirty tree in commit
  `51b18b6`.
- Archived imported conversation logs, HGTXR project docs, and legacy hybrid
  docs/scripts/configs/tests under `algorithm/archive/imports`.
- Added archive README files and moved the import manifest to
  `algorithm/archive/imports/IMPORT_MANIFEST_P0_P1_P2.md`.
- Added active-surface README files for `algorithm/hybrid/configs`,
  `algorithm/hybrid/scripts`, and `algorithm/hybrid/tests`.
- Moved the refactor summary from root `.agents` into
  `algorithm/docs/track/refactor` to preserve the requested root layout.

## 2026-07-05

- Reorganized three branch workspaces into the new `HBTXR` target directory.
- Established root-level `hardware`, `quantization`, `third`, `references`, and
  `algorithm` boundaries.
- Migrated the FACET/EvEye algorithm package under `algorithm/src/EvEye`.
