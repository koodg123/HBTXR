# Architecture Decision Records

## ADR-001: Preserve `EvEye` Import Name

Decision: keep the migrated source package at `algorithm/src/EvEye`.

Rationale: existing training, evaluation, and exporter scripts import
`EvEye.*`. Renaming the package during the directory reorganization would create
large behavioral risk before smoke tests and dependency setup are stabilized.
