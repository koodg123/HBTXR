> **작성** 2026-07-14 · **갱신** 2026-07-23
> **상태** active
> **소유** repo

# Architecture Decision Records

## ADR-001: Semantic Integration Instead Of Bulk Copy

- Status: accepted
- Date: 2026-07-14
- Decision: classify HANDOVER content by provenance and behavior; do not copy
  entire repositories or cherry-pick source branches into HBTXR.
- Reason: large portions are already identical, while HGTXR software/hardware
  common paths are mostly evolved and byte-different.
- Consequence: analysis takes longer but avoids regressions and duplicate code.

## ADR-002: Separate Checkpoint Commits

- Status: accepted
- Decision: commit plan, inactive archives, algorithm analysis, and hardware
  evidence separately.
- Reason: independent review, validation, and rollback.

## ADR-003: Analysis Before Active Promotion

- Status: accepted
- Decision: this task produces recommendations only; active code ports require
  a new approved implementation plan.
- Reason: license, API, data, toolchain, and runtime gates are unresolved.

## ADR-004: Test-First Selective Adaptation

- Status: accepted for recommendation
- Decision: prioritize data/evaluation/checkpoint contracts, HGTXR behavioral
  regression tests, and XR cyclic schema/automation before active adapters.
- Reason: HGTXR common source paths are mostly byte-different and accelerator
  gaps contain generated/tool-specific content; contracts and golden tests make
  future choices measurable and reversible.
- Consequence: useful behavior is re-expressed through current interfaces;
  legacy trees, generated artifacts, and uncleared sources remain non-active.

## ADR-006: Authority-Zoned Refactoring And Manifest-First Cleanup

- Status: accepted for planning
- Date: 2026-07-15
- Decision: evaluate maintained/test/analysis separately from artifact/archive/reference/vendor;
  execute correctness repair before package/owner decomposition; require a
  fail-closed asset manifest before reference or binary retirement.
- Reason: reference/vendor content dominates repository size and would distort
  active-code metrics, while deletion without source/license/hash/consumer
  evidence would lose provenance.
- Consequence: cleanup takes multiple gated commits, but active refactoring and
  archival retention remain independently reviewable and reversible.

## ADR-007: Preserve Algorithm Modalities And Permanent Comparison References

- Status: accepted
- Date: 2026-07-15
- Decision: preserve `algorithm/{common,event,frame,hybrid}` and the current
  `algorithm/hybrid/src` implementation owner. Apply named-package cleanup to
  Quantization only. Hybrid namespace migration is conditional on a clean,
  supported-workflow collision reproduction plus explicit approval; otherwise it
  is NO-OP. Preserve `references/**` permanently in the repository for comparison
  experiments, with no deletion, rename, deduplication, or pointer substitution.
- Reason: event/frame/hybrid are intentional modality surfaces, no supported
  same-interpreter failure has been demonstrated, and reference snapshots are
  required comparison evidence.
- Consequence: ADR-006 is superseded only where it allowed reference retirement;
  manifest-first retirement remains applicable to approved non-reference binaries.

## ADR-009: Retire Active FACET Names And Hybrid Pool Facades

- Status: accepted for planning
- Date: 2026-07-21
- Decision: preserve `algorithm/{common,event,frame,hybrid}`,
  `algorithm/common/src/EvEye`, `algorithm/hybrid/src`, and source-faithful
  reference/history FACET identity, while retiring repository-owned FACET names
  from the active algorithm surface. Remove `algorithm/hybrid/src/pools` without
  a permanent shim. Move only its used LR scheduler to
  `src.optim.lr_schedulers`; reuse existing domain owners and keep
  `src.optim.pool` unchanged. Removed FACET environment controls are not renamed
  to `HBTXR_*`: common launchers use existing config keys, the config-driven
  evaluator uses `runtime.disable_cudnn`, and only the standalone dataset builder
  gains semantic `--disable-cudnn`.
- Reason: common script/test flattening has no basename collision; active legacy
  names obscure current ownership; five pool facades have no maintained consumer
  and the only consumed module has a natural `optim` owner. Recreating registries
  would duplicate existing model/loss/optimizer/runtime ownership.
- Consequence: old environment/API/config names and `src.pools` intentionally
  break after consumer gates. Required external consumers stop the applicable
  slice for a new compatibility decision. ADR-007 remains in force for modality,
  Hybrid root, and permanent reference retention. Existing HBTXR project/model/
  config identity is not a target of this replacement-policy decision.

## ADR-010: Stage Algorithm Modular Ownership And Retain Hybrid Domain APIs

- Status: accepted for planning
- Date: 2026-07-21
- Decision: preserve `algorithm/{analysis,archive,artifacts,requirements}` in
  place, retain root README/pyproject paths, and stage the active owners
  `common/dataset/utils/engine/docs/configs/tests/frame/event/hybrid`. Shared
  code uses `eveye.*`; bare generic packages are prohibited. Hybrid-special code
  remains Hybrid-local. Remove `src.pools`, but retain its loss, optimizer, LR
  scheduler and runtime scheduler APIs under their domain packages, and preserve
  `src.optim.pool`.
- Reason: current `EvEye` combines unrelated responsibilities, frame/event are
  config-only, and deleting all pool APIs would contradict the latest user
  requirement. Staged vertical slices reduce package and import risk.
- Consequence: ADR-007 is superseded where it permanently fixed `EvEye` and the
  current active ownership layout. ADR-009 is superseded where it kept only the
  LR scheduler. Reference retention and neutral runtime-control decisions remain
  valid. Implementation stays gated by consumer, wheel and regression evidence.
