# HANDOVER integration recommendations

## Executive recommendation

Adopt a **test-first selective adaptation** strategy. Keep current HBTXR as the
only active baseline, close byte-identical material as no-op, preserve historic
implementations as references, and adapt only behaviors that can pass current
contracts, licenses, and reproducible validation. The first implementation
proposal should combine HGTXR search/tracking regression tests with an XR cyclic
configuration/automation prototype; it should not copy either source tree.

## What should be reflected in HBTXR

### 1. Reflect now as design and test work

- A versioned data contract covering tensor rank/order, dtype, polarity,
  temporal window, subject/eye/sequence identity, and split provenance.
- A task/coordinate-aware evaluation contract separating center, bounding-box,
  and ellipse semantics and handling `80x60`, `640x480`, and `64x64` explicitly.
- HGTXR-derived regression fixtures for Stage-1 recovery/ensemble and Stage-2
  support/state behavior, rewritten against current interfaces.
- A checkpoint inspection/conversion/averaging utility with manifest output and
  strict key/shape/dtype/buffer policies.
- Portable XR experiment/config schemas and dry-run automation, especially for
  cyclic ZCU104 candidates.
- Hardware operator golden vectors with quantization/rounding/tolerance metadata.

These items are concrete reflections of HANDOVER knowledge without declaring
historic source executable or replacing evolved modules.

### 2. Reflect later, after evidence

- Activate Stage-1 and Stage-2 adapters only after same-split ablation and
  disabled-path equivalence tests.
- Recreate XR64–XR68 teacher-target and failure-bucket experiments using current
  trainers/configs, then compare under fixed seeds and metrics.
- Select one XR ZCU104 cyclic case for simulation/synthesis only after schema,
  golden, toolchain, resource, and timing gates are approved.
- Reuse ViT multi-board/report utilities only when multi-board support becomes
  an explicit requirement.

### 3. Do not reflect into active code

- the 320 byte-identical HBTXR analysis paths or 71 byte-identical common
  third-party paths;
- entire HGTXR software/hardware trees or HANDOVER/HBTXR branch history;
- absolute-path launchers, virtual environments, caches, logs, and run state;
- datasets, checkpoints, HDF5 exports, bitstreams, Vivado/Vitis generated trees,
  or large workspaces;
- ERVT, TENNs-Eye, TDTracker, or BRAT material before license clearance;
- the ICCAD24 SPINAL gap unless a current dependency is demonstrated.

## Proposed implementation epics

| Epic | Scope | Deliverables | Exit criteria |
| --- | --- | --- | --- |
| E1 Contract foundation | software data/evaluation/checkpoint boundaries | schemas, validators, golden fixtures, rejection tests | all known HANDOVER shapes/domains handled explicitly; ambiguity fails closed |
| E2 HGTXR behavioral regression | Stage-1 search and Stage-2 tracking | current-interface fixtures, state diagrams, feature-flagged adapters | current default unchanged; deterministic ablation and latency evidence |
| E3 Checkpoint tooling | inspect/convert/interpolate/average | CLI/library, provenance manifest, endpoint/load tests | no silent key/state loss; fixed-batch numerical smoke passes |
| E4 Hardware golden suite | quantized operators and mode boundaries | vectors, C/C++ tests, software/hardware match report | scale/rounding/tolerance and source revision recorded; current tests pass |
| E5 XR cyclic prototype | config schema, automation, one ZCU104 case | translator, dry-run, parser fixtures, design proposal | clean-checkout dry-run and C-sim/golden pass; no board claim |
| E6 Reproducible experiments | XR64–XR68/failure buckets | current configs, fixed split/seeds, report template | same-condition metrics and failure analysis; artifacts externalized |

Each epic should be a separate reviewed change set. E1 precedes E2, E3, and E6;
E4 precedes active hardware modification; E5 requires a separate approval before
synthesis, board transfer, or active source edits.

## Feasibility and tradeoffs

| Approach | Benefits | Costs / risks | Judgment |
| --- | --- | --- | --- |
| Bulk merge/copy | superficially fast; preserves old layout | overwrites evolved APIs, imports paths/artifacts, obscures provenance, high regression risk | Reject |
| Reference-only preservation | lowest runtime risk; keeps history | useful behaviors remain dormant; can accumulate documentation debt | Use for legacy/generated/uncleared material |
| Selective adaptation | fits current architecture; measurable and reviewable | requires contract/test engineering and may expose irreconcilable assumptions | Recommended |
| Clean reimplementation from behavior | strongest current fit and testability | more effort; still requires provenance/license awareness | Prefer for high-divergence HGTXR surfaces |

Feasibility is high for contracts, metrics, checkpoint inspection, regression
fixtures, and portable automation. Feasibility is medium for search/tracking
behavior because 111/113 common HGTXR source paths changed. Hardware activation
is medium-to-low until tool/board/golden evidence is reproduced. Claims depending
on unavailable datasets, checkpoints, or boards remain `확실하지 않음`.

## Validation gates G0–G10

| Gate | Requirement |
| --- | --- |
| G0 Scope | Exact source revision, destination, owner, and non-goals recorded. |
| G1 Provenance/license | File/behavior origin and compatible license/notice evidence complete. |
| G2 Data | Shape/order/dtype/polarity/window/split and privacy permissions validated. |
| G3 Coordinates/task | Center/bbox/ellipse and source/target coordinate domains explicit. |
| G4 Metrics | Aggregation, invalid samples, weighting, and golden evaluator cases fixed. |
| G5 Checkpoints | Keys/shapes/dtypes/buffers/architecture/state policy verified. |
| G6 Software regression | Unit/integration/determinism/default-equivalence and latency gates pass. |
| G7 Hardware numerics | Quantization, packing, rounding, saturation, tolerance and vectors pass. |
| G8 Build/implementation | Pinned tool/part/clock; simulation, synthesis and post-route evidence pass. |
| G9 Board/artifacts | Hash manifest, runner, physical smoke, logs and rollback exist. |
| G10 Comparative evidence | Same split/task/metric/seed/budget comparison supports promotion. |

G0–G6 apply to software candidates. Hardware candidates additionally require
G7–G9. Any accuracy/performance promotion requires G10. A failed or unknown gate
keeps the item `REFERENCE_ONLY`, `ADAPT`, or `LICENSE_BLOCKED`.

## Recommended next change set

Prepare a new, separately approved implementation plan with this bounded scope:

1. add only the E1 contract schemas/validators and golden rejection tests;
2. add a minimal Stage-1 recovery fixture and Stage-2 state-transition fixture;
3. add an XR config schema validator and command-generation dry-run fixture;
4. make no model architecture, active hardware module, dataset, checkpoint,
   synthesis, board, or remote-storage change;
5. report which HANDOVER behavior can be reproduced before requesting adapter
   activation.

This creates the evidence needed to decide whether active integration is worth
the cost while preserving rollback and current behavior.

## Final decision

The HANDOVER pool should influence HBTXR through contracts, tests, selective
behavioral adapters, and portable automation—not through repository merging.
C1–C3 preserve useful provenance and compact evidence. The next engineering
investment should validate HGTXR search/tracking ideas and XR cyclic automation
behind current interfaces, with licenses and external artifacts treated as
hard gates rather than cleanup tasks.
