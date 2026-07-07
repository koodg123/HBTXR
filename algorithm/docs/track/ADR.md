# Architecture Decision Records

## ADR-001: Preserve `EvEye` Import Name

Decision: keep the migrated source package at `algorithm/src/EvEye`.

Rationale: existing training, evaluation, and exporter scripts import
`EvEye.*`. Renaming the package during the directory reorganization would create
large behavioral risk before smoke tests and dependency setup are stabilized.

## ADR-002: Archive Imported Legacy Materials Outside Active Algorithm Surfaces

Decision: keep imported conversation logs, HGTXR project docs, and legacy hybrid
docs/scripts/configs/tests under `algorithm/archive/imports`.

Rationale: the material is useful for provenance and future porting, but it is
not guaranteed to run against the current active APIs. Keeping it outside
`algorithm/hybrid` active surfaces prevents old branch-specific paths and test
contracts from being mistaken for maintained package behavior.
