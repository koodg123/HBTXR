# HBTXR Commit And HANDOVER Analysis Master Plan

## Prompt Brief

- Goal: commit only the current `HBTXR` work in reviewable checkpoints, then
  analyze all `HANDOVER` repositories and identify concrete HBTXR integration
  candidates without implementing them yet.
- Inputs: live Git state; six HANDOVER repositories; current HBTXR code,
  tracking docs, tests, and import manifests.
- Assumptions: current HBTXR is the evolved target; HANDOVER is read-only
  evidence; identical files are no-op; push is not authorized.
- Unknowns: license scope for some model code, external artifact policy,
  current executability of historical runners, licensed FPGA tool availability.
- Safety: no sibling-repo writes, no whole-tree copy/cherry-pick, no secrets or
  generated builds, no active promotion without provenance and tests.
- Outputs: five local commits, six analysis documents, one prioritized
  integration matrix, and preserved validation evidence.

## Best-Output Criteria

| Criterion | Acceptance |
|---|---|
| Completeness | All six source repositories and all current checkpoint paths covered |
| Evidence | Every claim maps to path, SHA, count, diff, test, or document |
| Executability | Each slice has owner, dependencies, command, and rollback |
| Consistency | Source/target mappings use the current HBTXR package layout |
| Safety | Generated, licensed, secret, nested-repo, and push risks gated |
| Maintainability | No legacy overwrite; active candidates use adapters/tests |

## Scope

In scope:

1. Plan-only documentation commit.
2. Three commits for the existing 2026-07-08 selective import.
3. Read-only semantic analysis of `HANDOVER/HBTXR`, `HGTXR`, `HGPIPE`,
   `ViT_Accel`, and `XR_Accel`.
4. Recommendation artifacts and a decision-ready integration matrix.

Out of scope:

- Git push.
- Active code integration from HANDOVER.
- Training, full HLS/Vivado runs, board execution, or artifact migration.
- Git LFS/external storage adoption without a user decision.

## Expert Council

- Git/Release: isolate plan, archive, analysis, and hardware evidence commits.
- Software/Provenance: compare behavior and API; never overwrite evolved code.
- QA/Hardware: require license, artifact, data-contract, regression, and
  hardware-tool gates before promotion.

## Team Blueprint

| Team | Responsibility | Primary artifact |
|---|---|---|
| Release | staged allowlists, static checks, local commits | `docs/Execution.md` |
| Research | repository pins, counts, collision inventory | `docs/analysis/HANDOVER-INVENTORY.md` |
| Software | HGTXR software/API/test/config semantic diff | `docs/analysis/HANDOVER-SOFTWARE-AUDIT.md` |
| Hardware | HGTXR/XR_Accel HLS/automation/config diff | `docs/analysis/HANDOVER-HARDWARE-AUDIT.md` |
| QA | license, artifact, validation, rollback gates | `docs/analysis/HANDOVER-LICENSE-ARTIFACT-AUDIT.md` |
| Council | prioritization and final candidate matrix | `docs/analysis/HANDOVER-RECOMMENDATIONS.md` |

## Execution DAG

```text
Plan commit
  -> archive commit
  -> algorithm analysis commit
  -> hardware/reference commit
  -> [inventory | software audit | hardware audit | risk audit]
  -> integration matrix
  -> recommendation commit
  -> user decision for a separate implementation plan
```

## Current Evidence

- HBTXR baseline: `refactor/hbtxr-structure` at `087912c2b27f`.
- Baseline dirty state: 2 modified, 545 untracked, 0 staged.
- Current import static checks: 105 JSON, 81 Python, and 26 shell files passed.
- `HANDOVER/HBTXR/analysis`: 320/320 mapped files identical.
- HG-PIPE Quantization: all 125 source-unique blobs already represented.
- HGTXR software: 113 common core paths, only 2 byte-identical; semantic audit required.
- HGTXR hardware: 394 common paths, 349 changed; overwrite prohibited.

## Historical Analysis-Phase Approval Gates

These labels applied only to the completed 2026-07-14 import/analysis workflow.
The 2026-07-15 follow-up workstream uses the distinct A–I gate table in its
canonical plan; meanings must not be carried between the two phases.

- Gate A: plan approval — satisfied on 2026-07-14.
- Gate B: separate approval before push.
- Gate C: decide artifact storage and any license-blocked promotion.
- Gate D: approve a new implementation plan after analysis recommendations.

## Rollback

Use semantic commits and `git revert`; never destructive reset. Generated
artifacts stay outside Git so code rollback and artifact retention remain
independent.

## Follow-up Implementation Workstream — 2026-07-15

The analysis phase is complete. The concrete follow-up workstream is defined in
`docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md` and remains
awaiting execution approval. It sequences provenance/artifact governance,
software contracts, hardware golden/config/dry-run validation, tracker
characterization, static experiment reproducibility, and separately approved
training/HLS/synthesis/board gates. This pointer does not authorize execution.

## Standalone HBTXR Selective Integration Workstream — 2026-07-15

The separate repository `/mnt/d/dataset/EV_Eye/paper_works/HBTXR` is not a
branch/worktree of this checkout. Its `annotation` revision `2ff5262` is treated
as read-only evidence. The canonical plan is
`docs/aegis/plans/2026-07-15-standalone-hbtxr-selective-integration.md`; the
file-level evidence is
`docs/analysis/STANDALONE-HBTXR-SELECTIVE-INTEGRATION-MATRIX.md`.

The workstream excludes bulk merge, cherry-pick, data/results/weights,
dependency installation, reproduction, experiments, runtime model execution,
archive replacement, push, and merge. Future code execution is blocked until
source rights and the 346x260/346x240/640x480 coordinate contract are resolved.
FECET and HG-PIPE have no new active-code task; SWIFT runtime candidates require
a separate future plan.

## Semantic Refactor and Cleanup Workstream — 2026-07-15

A committed-blob semantic census at commit `ebe862b` covers all 12,856 tracked files,
attempts static parsing for all 3,721 Python files, and records 30,351 Python function/class symbols and 20,423
non-Python structural symbols. Canonical artifacts are under
`.agents/recon/2026-07-15-semantic-census/`; the human synthesis and decision
matrix are under `docs/analysis/`.

The execution plan is
`docs/aegis/plans/2026-07-15-hbtxr-semantic-refactor-cleanup.md`. Its order is
repair, package/API refactor, owner decomposition, then manifest-first cleanup.
It does not authorize code implementation, external archive mutation, deletion,
commit, push, merge, or release. The algorithm modality layout and
`algorithm/hybrid/src` remain canonical. Quantization named-package cleanup is
the default package task; Hybrid namespace migration is conditional and defaults
to NO-OP. `references/**` remains permanent in-repository comparison material and
must not be deleted, renamed, deduplicated, or replaced by external pointers.
Non-reference binary retirement still requires evidence and separate approval.

## Session Continuation Workstream — 2026-07-16

The authoritative pause point is `.agents/handoff/2026-07-16-hbtxr-semantic-refactor-planning.md`; its paired prompt is only a pointer. Continuation must verify HEAD and the inherited dirty worktree before choosing a lane. This workstream documents state only and does not authorize implementation, commit, push, install, experiment, or cleanup.

## Algorithm Active-Name And Pools Retirement Workstream — 2026-07-21

The user selected a bounded algorithm-only planning change. The canonical
semantic plan now includes RC-120 and RC-130. RC-120 flattens
`common/scripts/facet` and `common/tests/facet`, renames repository-owned active
FACET contracts, and preserves `algorithm/common/src/EvEye` plus source-faithful
reference/history identity. Removed FACET controls are absorbed by existing
config keys or one semantic CLI flag; no `HBTXR_*` replacement environment
variables are introduced. The original RC-130 removed
`algorithm/hybrid/src/pools` and retained only the LR scheduler plus
`src.optim.pool`; that pools disposition is superseded by the modular ownership
workstream below. This historical revision authorized no implementation.

## Algorithm Modular Ownership Workstream — 2026-07-21

The later user decision supersedes the `EvEye`-permanent and only-LR portions of
the preceding workstream. The authoritative plan is
`docs/aegis/plans/2026-07-21-algorithm-modular-ownership-refactor.md`.

The plan preserves `algorithm/{analysis,archive,artifacts,requirements}` in place
without path/blob drift and keeps `algorithm/{README.md,pyproject.toml}` at their
root paths. Active ownership is staged across `common`, `dataset`, `utils`,
`engine`, `docs`, `configs`, `tests`, `frame`, `event`, and `hybrid`. Shared code
uses the named `eveye.*` namespace; no bare `dataset`, `utils`, or `engine`
package is introduced. Hybrid-special behavior remains under Hybrid.

`src.pools` is retired, but its loss, optimizer, LR scheduler, and runtime
scheduler APIs move respectively to `src.loss.losses`, `src.optim.optimizer`,
`src.optim.lr_schedulers`, and `src.runtime.runtime_schedulers`. `src.optim.pool`
remains the separate optimizer experiment/report feature. All source moves are
gated; implementation has not started.
