# Architecture Decision Records

## ADR-001: Use HBTXR Joined Motion Labels For Retina/ERVT Distribution Tables

Date: 2026-07-03

### Context

The raw test metadata contains `motion_state`, but it has near-zero Saccade counts and zero Saccade labels for several subjects. The HBTXR reference Excel uses a joined motion label result with nonzero Saccade counts for all subjects 37-48.

### Decision

Retina and ERVT motion-specific error distribution files must use:

`analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv`

as the authoritative `sample_idx -> motion_state` label map.

### Consequences

- Retina and ERVT JETCAS Excel Saccade columns are populated for every subject.
- Retina joined rows match the HBTXR label-map sample space.
- ERVT joined rows remain slightly lower because sequence segmentation does not predict every sample index.
