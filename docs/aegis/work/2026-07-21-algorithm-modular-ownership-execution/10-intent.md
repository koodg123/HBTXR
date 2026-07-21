# Algorithm Modular Ownership Execution Intent

## Requested outcome

Execute AM-000 through AM-900 from the approved modular ownership plan in
strict sequence, preserving behavior while moving active algorithm ownership to
`eveye.*` packages and retiring active FACET, `EvEye`, and `src.pools` owners at
their explicit gates.

## Scope and non-goals

- Scope: `algorithm/{common,dataset,utils,engine,configs,tests,frame,event,hybrid}`
  plus required root packaging/docs updates.
- Frozen: `references`, `algorithm/{analysis,archive,artifacts,requirements}`,
  modality analysis docs, Hybrid hardware reference/handover, provenance and
  algorithm tracking history.
- Non-goals: reference/history rewrites, Hybrid namespace migration, speculative
  Hybrid-to-shared extraction, experiments, push, merge, release or cleanup.

## BaselineReadSetHint

- Baseline commit: `ebe862b11506e819c5bd5abc925299fc5fbb6f1a`.
- Parent plan authority: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`.
- Parent plan SHA-256: `c73cd5ee0e4f9dd0798b2dbd38bf3c9a50c25e81b996d634c4bc776b022b97b4`.
- Absolute worktree paths in these records are 2026-07-21 capture metadata, not portable path contracts.
- Build contract: `algorithm/pyproject.toml`.
- Runtime guide: `algorithm/README.md`.
- Current owners: `algorithm/common/src/EvEye` and `algorithm/hybrid/src`.

## BaselineUsageDraft

- Required refs: parent plan, live tree/import consumers, pyproject, preserved
  paths, focused tests.
- Acknowledged refs: parent plan and baseline SHA.
- Missing refs: repository-external `EvEye.*` and `src.*` consumers.
- Decision: proceed slice-by-slice; fail closed at compatibility retirement.

## ImpactStatementDraft

This is a behavior-preserving ownership migration. The main risks are package
discovery drift, module identity duplication, reverse dependencies, direct
script breakage, registry changes and preservation-zone mutation.

## Execution Readiness View

- Intent lock: implement only the approved AM task sequence.
- Scope fence: active algorithm ownership; frozen evidence stays untouched.
- Baseline lock: exact commit above and a clean isolated worktree.
- Owner constraints: utils is leaf; dataset/common do not import modality
  owners; engine does not import Hybrid; Hybrid retains special behavior.
- Compatibility boundary: dataset/model/CLI/checkpoint/Hybrid APIs remain
  behavior-compatible until explicit retirement gates pass.
- Retirement boundary: no early deletion of `EvEye` or `src.pools`.
- Test obligations: focused tests per slice, then full AM-900 checks.
- Review gates: spec compliance then code quality for every implementation task.
- Drift rule: pause on preserved-zone drift, new owner, unplanned adapter or
  failed baseline.
- Advisory boundary: evidence supports decisions but does not authorize push,
  merge or release.
