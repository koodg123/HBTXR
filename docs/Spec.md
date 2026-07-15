# Specification: HBTXR Checkpoint And HANDOVER Analysis

## Status

Approved for sequential execution. Active code integration remains out of scope.

Spec-Kit CLI `0.10.2.dev0` is available and `specify check` passed. Project
initialization is deferred until the dirty baseline is checkpointed because
`specify init --here` would add scaffolding to the current change set. This
manual spec will be refined with Spec-Kit after that gate if approved.

## Functional Requirements

### FR-1 Repository Boundary

All commits MUST use `/HBTXR` as the Git root. Sibling directories MUST remain
unchanged. Push MUST NOT occur in this scope.

### FR-2 Checkpoint Atomicity

The existing import MUST be preserved as three semantic commits after the plan
commit: inactive algorithm archives; algorithm analysis material; hardware and
legacy evidence. Each commit MUST pass staged allowlist and static checks.

### FR-3 Provenance

Every HANDOVER source MUST record repository path, branch, full SHA, original
path, mapped HBTXR path, and exclusion status.

### FR-4 Candidate Classification

Each candidate MUST receive exactly one classification:

- `NO-OP`: already identical or superseded.
- `REFERENCE_ONLY`: useful evidence but not active code.
- `ADAPT`: behavior is useful but must be rewritten for current APIs.
- `ACTIVE_CANDIDATE`: compatible, licensed, and testable candidate.
- `EXCLUDED`: generated, cache, binary, duplicate, or irrelevant.
- `LICENSE_BLOCKED`: active use lacks sufficient license evidence.

### FR-5 Semantic Comparison

Changed descendants MUST be compared by symbols, interfaces, configuration
schema, behavior, test coverage, and runtime assumptions. Raw overwrite,
whole-tree copy, and branch cherry-pick MUST NOT be recommended.

### FR-6 Recommendation Row

Every final matrix row MUST contain source repo/SHA/path, target path,
classification, evidence, delta, benefit, risk, adaptation work, validation,
dependencies, priority, and rollback.

### FR-7 Artifact Policy

Venvs, caches, `.Xil`, generated builds, checkpoints, H5, large CSV, bit/hwh,
logs, runs, nested `.git`, and secrets MUST be excluded from Git. Required
external artifacts MUST be represented by URI/checksum manifests only after an
artifact policy is approved.

### FR-8 License Policy

License and NOTICE scope MUST be mapped before active promotion. Unclear ERVT,
TENNs-Eye, TDTracker, BRAT, or related third-party code remains reference-only
or license-blocked.

## Priority Analysis Targets

1. HGTXR software behavior missing from current `algorithm/hybrid`: recovery,
   ensemble, support-adaptive tracking, teacher/failure-bucket experiments,
   checkpoint interpolation, and metric/paper-target validators.
2. HGTXR hardware regression tests and golden vectors missing from current
   `hardware`; current HLS sources are evolved descendants and must not be
   replaced.
3. XR_Accel cyclic/ZCU104 automation, design configs, report parsers, and
   case-level interfaces that can be adapted to the active HBTXR hardware API.
4. ViT_Accel board-independent automation or report utilities only; generic
   full-DeiT and generated workspace content remains reference/future.
5. HANDOVER/HBTXR analysis and HG-PIPE Quantization are expected NO-OP except
   the small set of changed/missing reports, configs, scripts, and SPINAL files.

## Non-Functional Requirements

- Reproducible: all counts and recommendations have mechanical commands.
- Reviewable: commits and recommendation rows remain concern-sized.
- Safe: no external state change, secret exposure, or destructive Git action.
- Maintainable: adapters and tests are preferred over upstream source edits.
- Resumable: progress and decisions live under `docs/track`.

## Acceptance Criteria

```yaml
acceptance_criteria:
  - id: AC-1
    check: HBTXR-only commit history and sibling repos unchanged
  - id: AC-2
    check: four local commits with explicit staged scopes and passing gates
  - id: AC-3
    check: six source repositories pinned and inventoried
  - id: AC-4
    check: every candidate has one classification and complete evidence fields
  - id: AC-5
    check: identical/superseded content is not proposed for re-import
  - id: AC-6
    check: active candidates have license, API, data, test, artifact, and rollback gates
  - id: AC-7
    check: no HANDOVER code is implemented during the analysis phase
```

## Deferred Decision

The analysis output will be the input to a separate implementation plan. No
active HANDOVER port is authorized by this specification.

## Follow-up Planning Addendum — 2026-07-15

The concrete implementation plan is
`docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md`. It converts the
analysis requirements into exact file, test, dependency, review, rollback, and
approval boundaries. Plan creation is authorized; implementation, dependency
installation, training, FPGA tools/board access, artifact mutation, LFS,
Spec-Kit initialization, push, merge, and release remain separately gated.
