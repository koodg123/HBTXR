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

## Standalone HBTXR Selective Integration Addendum — 2026-07-15

### Functional requirements

- **FR-SI-1:** pin standalone source revision `2ff5262`; use committed blobs,
  not its CRLF-dirty worktree.
- **FR-SI-2:** never merge/cherry-pick/copy the standalone tree wholesale.
- **FR-SI-3:** every candidate uses the FR-4 classification vocabulary and an
  exact source path, target owner, dependency, verification, and rollback.
- **FR-SI-4:** unresolved source authorship/license keeps source-derived active
  code `LICENSE_BLOCKED`.
- **FR-SI-5:** resolve 346x260, 346x240, and 640x480 frame meanings through an
  ADR before modifying evaluation contracts.
- **FR-SI-6:** first-slice code is additive, standard-library-only, free of
  absolute paths/import-time I/O, and owned by the current evaluation package.
- **FR-SI-7:** samples, datasets, results, figures, tables, weights, lockfiles,
  environments, model downloads, and runtime experiments remain excluded.
- **FR-SI-8:** FECET/HG-PIPE/SWIFT archives and active quantization files are
  not overwritten; proven no-op content is not re-imported.

### Acceptance criteria

- The standalone matrix covers annotation and all three implementation roots.
- Rights and coordinate gates fail closed before source code adaptation.
- Focused annotation modules, if approved, pass standard-library unit tests and
  existing evaluation/provenance regressions.
- No dependency installation, reproduction, experiment, model/checkpoint/data
  ingestion, push, merge, or release occurs implicitly.

The canonical task plan is
`docs/aegis/plans/2026-07-15-standalone-hbtxr-selective-integration.md`.

## Semantic Census and Refactor/Cleanup Addendum — 2026-07-15

### Functional requirements

- **FR-SC-1:** cover every Git-tracked file in a machine-readable inventory.
- **FR-SC-2:** record every Python function/class with exact line spans without importing project code.
- **FR-SC-3:** separate maintained/test/analysis/artifact/archive/reference/vendor authority.
- **FR-SC-4:** type every material conclusion as fact, inference, or unknown with evidence.
- **FR-SC-5:** prioritize correctness and import safety before package or large-scale decomposition.
- **FR-SC-6:** preserve CLI, schema, data/model, validator, and hardware artifact contracts.
- **FR-SC-7:** treat heuristic dead-code and duplicate findings as candidates, not deletion authority.
- **FR-SC-8:** require source/license/hash/consumer/read-back evidence and separate approval before destructive non-reference cleanup.
- **FR-SC-9:** keep reference/vendor code outside active runtime ownership.
- **FR-SC-10:** create a validated baseline recon and a delta recon after execution.
- **FR-SC-11:** preserve every `references/**` path and blob as a permanent in-repository comparison asset; forbid deletion, rename, deduplication, and external-pointer replacement.
- **FR-SC-12:** preserve the modality layout and `algorithm/hybrid/src`; Hybrid namespace migration requires supported-workflow reproduction plus explicit approval, otherwise NO-OP.
- **FR-SC-13:** remove repository-owned `FACET/facet` paths, identifiers, environment variables, and runtime contract values from the active algorithm surface while preserving source-faithful reference, archive, analysis, provenance, history, and paper identity. Do not replace removed FACET controls with `HBTXR_*` or another project-prefixed environment namespace; reuse `runtime.disable_cudnn`, `trainer.devices`, `train.ckpt_path`, and the standalone builder's semantic `--disable-cudnn` option.
- **FR-SC-14 (superseded by FR-AM-6):** the earlier plan retained only the LR
  scheduler when removing `src.pools`. FR-AM-6 is the current pools authority.

### Acceptance criteria

- File, symbol, line-finding, duplicate, dead-code, and import-edge inventories exist.
- Baseline `codebase-recon.v1` passes the stored Python equivalent validator; the installed shell validator remains blocked by CRLF and missing `jq` and must not be reported as passed.
- The plan names exact owners, files, verification, compatibility, rollback,
  and retirement gates for each major slice.
- Reference deletion/rename count and reference hash drift are zero.
- Quantization package work does not imply Hybrid package migration.
- RC-120 completion has no active-name match outside its exact reference/history allowlist, no `(FACET|HBTXR)_(DISABLE_CUDNN|DEVICES|CKPT_PATH)` control, no branded temporary replacement path, no frozen-zone diff, and no `EvEye` or modality-layout migration.
- The earlier RC-130 acceptance applied only to the superseded pools policy;
  current acceptance is defined by FR-AM-6 and the AM plan.
- No code refactor, deletion, install, test/build/experiment, commit, push, or
  merge is implied by completion of the analysis.

## Session Handover Requirements — 2026-07-16

- **FR-HO-1:** record repository, full HEAD, branch, tracker, pawl disposition, helper outcome, dirty files, external worktree, and exact next action.
- **FR-HO-2:** summarize user decisions and completed/open task IDs with durable path or SHA evidence.
- **FR-HO-3:** keep the continuation prompt as a pointer to one authoritative HANDOVER.
- **FR-HO-4:** preserve every inherited dirty file; no reset, checkout, deletion, commit, push, install, experiment, or cleanup is implied.
- **Acceptance:** both files are non-empty, required headings/markers and metadata regexes pass, HEAD is live-verified, and an independent read-only reviewer reports no blocking inconsistency.

## Algorithm Modular Ownership Requirements — 2026-07-21

- **FR-AM-1:** preserve `algorithm/{analysis,archive,artifacts,requirements}` at
  their current paths with zero baseline path/blob drift.
- **FR-AM-2:** retain `algorithm/README.md` and `algorithm/pyproject.toml` at their
  current paths; content may change only to describe and package the approved
  active ownership tree.
- **FR-AM-3:** stage active owners as `common`, `dataset`, `utils`, `engine`,
  `docs`, `configs`, `tests`, `frame`, `event`, and `hybrid`.
- **FR-AM-4:** use a named `eveye.*` package for shared code and prohibit new bare
  `dataset`, `utils`, `engine`, or product-prefixed replacement imports.
- **FR-AM-5:** keep frame-only, event-only, and Hybrid-only behavior in its
  modality owner; Hybrid code moves to shared owners only after a second active
  modality consumer and parity fixtures exist.
- **FR-AM-6:** remove `src.pools` while retaining exact domain APIs at
  `src.loss.losses`, `src.optim.optimizer`, `src.optim.lr_schedulers`, and
  `src.runtime.runtime_schedulers`; preserve `src.optim.pool`.
- **FR-AM-7:** retire `EvEye` implementation/wrappers only after maintained and
  required external consumers are zero and isolated wheel/import validation
  passes.

Acceptance requires the target-tree manifest, package/import origin checks,
config/model/dataset contract parity, full Hybrid regression, absent
`src.pools`, all five retained Hybrid paths, and zero preserved-zone drift.
Planning completion does not imply implementation, deletion, commit, push, or
cleanup authorization.
