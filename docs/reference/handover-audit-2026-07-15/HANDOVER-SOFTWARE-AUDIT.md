# HANDOVER software semantic audit

## Decision summary

The HANDOVER software is useful primarily as behavioral evidence. It must not
be bulk-copied over `algorithm/hybrid/src`. The current HBTXR implementation is
the integration baseline; selected search, tracking, checkpoint, and evaluation
behaviors may become adapters only after contract and regression gates pass.
No active software is promoted by this audit.

## Evidence base

| Comparison | Evidence | Consequence |
| --- | --- | --- |
| `HANDOVER/HBTXR/analysis` → `algorithm/analysis` | 320 common, 320 byte-identical | `NO-OP`; the inspected content is already present. |
| `HANDOVER/HGTXR/software/src/hbtxr` → `algorithm/hybrid/src` | 113 common, 2 identical, 111 changed | Semantic extraction only; replacement would overwrite evolved interfaces. |
| HGTXR `scripts/v3` → current external pipeline | 35 common, all 35 changed | Treat scripts as workflow descriptions, not executable drop-ins. |
| HGTXR tests → current external pipeline tests | 5 common, all 5 changed | Mine assertions and fixtures; rewrite them against current APIs. |
| HGTXR configs → current configs | no normalized direct common path | Introduce an explicit schema translator; do not infer compatibility by filename. |
| HANDOVER/HBTXR `main..etri-server` | 4 commits, 46 paths; 36 identical in target, 9 different, 1 missing | Review only the ten non-identical paths; the branch is not a merge source. |

The non-identical HANDOVER/HBTXR branch surface contains four code/config
paths for ERVT/Retina subject-independent training, five experiment/readiness/
complexity reports, and a missing FECET README. These remain legacy-reference
inputs until their licenses and current API compatibility are established.

## Classification by software surface

| Source surface | Class | Concrete action | Promotion gate |
| --- | --- | --- | --- |
| Byte-identical HBTXR analysis and third-party content | `NO-OP` | Do not recopy or recommit. | Re-audit only if source revision changes. |
| HBTXR legacy reports and the non-identical ERVT/Retina files | `REFERENCE_ONLY` / `LICENSE_BLOCKED` | Preserve provenance and use reports to reconstruct intent. | License mapping plus current data/API tests. |
| HGTXR Stage-1 search recovery and ensemble behavior | `ADAPT` | Implement behind the current search interface, keeping current model ownership and output schema. | Deterministic fixture, ranking/recovery regression, latency ceiling. |
| HGTXR Stage-2 support-adaptive and track-state auxiliary behavior | `ADAPT` | Add optional strategy/state adapters; default behavior remains unchanged. | State-transition, lost-track recovery, reset, and sequence-boundary tests. |
| XR64–XR68 teacher-target and failure-bucket experiments | `ACTIVE_CANDIDATE` | Port experiment definitions and failure taxonomy, not old training entrypoints. | Reproducible config, seed, split, metric, and ablation evidence. |
| Checkpoint interpolation/averaging utilities | `ACTIVE_CANDIDATE` | Reimplement as a format-aware utility for current checkpoints. | Key/shape/dtype audit, optimizer-state policy, numerical smoke test. |
| Metric/paper-target bridge validators | `ACTIVE_CANDIDATE` | Normalize prediction/label records, then compute named metrics explicitly. | Cross-evaluator golden cases and coordinate-domain assertions. |
| Absolute-path launch scripts and machine manifests | `EXCLUDED` as executable code | Retain only as provenance; replace paths with CLI/env/config inputs if behavior is adapted. | Clean-room invocation from a new checkout. |
| Checkpoints, HDF5 datasets, runs, logs, caches, and virtual environments | `EXCLUDED` | Keep outside Git in artifact/data storage with hashes and manifests. | Artifact-store access and integrity policy. |

## Concrete adapter plan

### A1. Data-contract adapter

Create a single current-HBTXR boundary that emits a typed sample record:
`events`, optional `frames`, target coordinates/shape, subject, eye, sequence,
timestamp window, and source metadata. The adapter must reject ambiguous input.
The HANDOVER evidence includes at least these incompatible shapes:
`2x64x64`, `50x2x64x64`, `30x3x64x64`, and `100x2x64x64`.

Required tests:

- exact tensor rank, axis order, dtype, event polarity convention, value range,
  temporal-bin/window meaning, and missing-frame policy;
- subject-independent split equality and train/validation/test leakage checks;
- sequence boundary and eye-side preservation;
- explicit conversion fixtures for each accepted source shape, with rejection of
  silent reshape, truncation, or channel coercion.

### A2. Search and ensemble strategy adapter

Extract the Stage-1 recovery/ensemble decision rules into a strategy interface
that consumes current model outputs and returns a current-HBTXR search result.
It must not import HGTXR model construction or global configuration state.

Required tests cover stable ranking, empty candidates, ties, out-of-frame
predictions, recovery after failure, deterministic seeding, and bounded runtime.
The existing current strategy remains the default until an ablation demonstrates
an improvement under the same split and metric.

### A3. Tracking-state adapter

Model Stage-2 support-adaptive and track-state behavior as explicit states and
transitions: initialization, tracking, degraded confidence, recovery, reset,
and end-of-sequence. State must be scoped per sequence/subject and serializable
for debugging; no hidden module-global state is allowed.

Required tests cover first frame, discontinuous timestamps, missing detections,
long failure runs, subject switches, reset idempotence, and equivalence when the
adapter is disabled.

### A4. Checkpoint conversion/averaging utility

Accept an explicit source format and produce a new checkpoint plus a manifest.
Before interpolation or averaging, compare parameter keys, shapes, dtypes,
normalization buffers, quantization metadata, architecture identifier, and
training step. The caller must choose policies for optimizer/scaler/scheduler
state; silent dropping is forbidden. Validate endpoints (`alpha=0/1`), equal
model averaging, loadability, and a fixed-batch numerical smoke result.

### A5. Evaluation bridge

HANDOVER evaluations mix center-only, bounding-box, and ellipse semantics.
The bridge must name the task type and coordinate domain in every record. Only
center pixel error at a declared `64x64` domain is directly comparable across
the inspected variants; other metrics must stay task-specific. Conversions from
`80x60` or `640x480` to `64x64` require per-axis scale factors and golden corner/
center cases. Reports must include aggregation level, invalid-sample policy, and
subject weighting.

## Ordered implementation backlog

1. `P0` — land data-contract and evaluation golden tests before porting behavior.
2. `P1` — adapt Stage-1 search recovery/ensemble behind a feature flag.
3. `P1` — adapt Stage-2 state handling with explicit sequence ownership.
4. `P1` — implement checkpoint audit/conversion/averaging as an isolated tool.
5. `P2` — reproduce XR64–XR68 experiment definitions and failure buckets using
   current trainers and configs.
6. `P2` — evaluate legacy ERVT/Retina differences only after license clearance.

## Acceptance gates

An item can move from `ADAPT` or `ACTIVE_CANDIDATE` to active HBTXR only when:

- its source revision, original path, license, and adaptation owner are recorded;
- data, coordinate, metric, checkpoint, CLI/config, and deterministic-seed
  contracts are explicit and tested;
- it passes current unit/integration tests and adds focused regression cases;
- an ablation compares the same dataset split, task definition, metric, seed
  policy, and hardware/runtime budget;
- documentation does not claim reproducibility when external datasets or weights
  are unavailable.

## Conclusion

The highest-value HANDOVER software contribution is not its file layout but its
search/recovery, state, experiment, checkpoint, and evaluation behavior. Those
ideas should be re-expressed through current HBTXR interfaces after contract,
license, and regression gates. Byte-identical material remains a no-op, and
legacy entrypoints, absolute paths, and large artifacts remain non-active.
