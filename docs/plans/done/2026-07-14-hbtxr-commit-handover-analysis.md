---
id: plan-2026-07-14-hbtxr-commit-handover-analysis
type: plan
date: 2026-07-14
source: "live HBTXR and HANDOVER repositories"
intent_issue: "docs/Spec.md"
---

# Plan: HBTXR Checkpoint And HANDOVER Integration Analysis

## Context

`HBTXR` is the only commit target. Before this plan was written it was on
`refactor/hbtxr-structure` at `087912c2b27f`, with 2 modified files, 545
untracked files, no staged files, and no configured upstream. The uncommitted
work is a selective 2026-07-08 import of algorithm evidence, archived software,
hardware documentation/configuration, and legacy hardware references.

`HANDOVER` contains six independent source repositories. Mechanical comparison
shows that much of `HANDOVER/HBTXR`, HG-PIPE Quantization, and ICCAD24 material
is already present in `HBTXR`. The high-value unresolved work is a semantic
comparison of HGTXR software/hardware and XR_Accel cyclic/ZCU104 flows. Bulk
copying or branch-level cherry-picking would overwrite evolved HBTXR code.

Applied findings:

- `none` — no project `.agents/planning-rules` or findings registry existed.
- Memory guidance applied: checkpoint only the active `HBTXR` repository,
  verify large/generated artifacts, and require separate approval before push.

## Intent Issue

- **Intent issue:** `docs/Spec.md`
- **Bounded context:** HBTXR repository checkpoint and read-only HANDOVER gap analysis
- **Domain terms:** checkpoint, provenance, semantic diff, active promotion,
  reference-only, no-op, artifact exclusion, license gate

```gherkin
Feature: Safe HBTXR checkpoint and HANDOVER analysis

  Scenario: Commit only the intended repository
    Given HBTXR-Pool contains multiple independent Git repositories
    When the checkpoint is created
    Then only files under HBTXR are staged and committed
    And no push occurs without separate approval

  Scenario: Preserve current imports in reviewable commits
    Given HBTXR has algorithm archive, analysis, and hardware reference changes
    When the checkpoint is created
    Then each concern is committed separately after its validation gate passes

  Scenario: Avoid duplicate or regressive HANDOVER imports
    Given many HANDOVER blobs are already identical or have evolved descendants
    When candidates are classified
    Then identical content is NO-OP
    And evolved targets are compared semantically rather than overwritten

  Scenario: Produce concrete integration recommendations
    Given every candidate has provenance, license, API, data, and test evidence
    When the analysis is complete
    Then each candidate is classified with an exact source path, target path,
    required adaptation, validation command, risk, and priority
```

## Files to Modify

Planning phase files:

| File | Change |
|---|---|
| `.agents/plans/2026-07-14-hbtxr-commit-handover-analysis.md` | **NEW** canonical execution plan |
| `docs/Master-Plan.md` | **NEW** prompt brief, council judgment, DAG, gates |
| `docs/Sub-Plan.md` | **NEW** task cards and ownership |
| `docs/Spec.md` | **NEW** behavioral and data specification |
| `docs/Execution.md` | **NEW** execution ledger initialized as not started |
| `docs/Validation.md` | **NEW** validation matrix |
| `docs/track/*` | **NEW** durable progress, decisions, backlog, and handoff records |
| `docs/aegis/work/2026-07-14-hbtxr-commit-handover-analysis/*` | **NEW** checkpoint and evidence records |

Execution-phase analysis files:

| File | Change |
|---|---|
| `docs/analysis/HANDOVER-INVENTORY.md` | **NEW** repository pins, counts, exclusions |
| `docs/analysis/HANDOVER-INTEGRATION-MATRIX.md` | **NEW** source-to-target candidate matrix |
| `docs/analysis/HANDOVER-SOFTWARE-AUDIT.md` | **NEW** HGTXR software semantic diff |
| `docs/analysis/HANDOVER-HARDWARE-AUDIT.md` | **NEW** HGTXR/XR_Accel hardware semantic diff |
| `docs/analysis/HANDOVER-LICENSE-ARTIFACT-AUDIT.md` | **NEW** license and artifact policy |
| `docs/analysis/HANDOVER-RECOMMENDATIONS.md` | **NEW** prioritized final recommendations |

Existing checkpoint scopes are listed under Issues C1-C3 and must not be
expanded with `git add -A`.

## Boundaries

**Always:** operate from `HBTXR`; pin every source repo by branch and SHA;
preserve provenance; run static gates before each commit; keep generated data
outside Git; cite exact source and target paths.

**Ask First:** any push; Git LFS or external artifact-store adoption; activation
of code with unclear license; HLS/Vivado implementation requiring licensed tools
or a board; changes beyond analysis into active production code.

**Never:** stage sibling repositories; copy `.git`, venvs, caches, `.Xil`,
checkpoints, bitstreams, generated build trees, or local credentials; overwrite
current HBTXR descendants with older HANDOVER files; use destructive reset.

## Baseline Audit

| Metric | Command | Result |
|---|---|---|
| HBTXR branch and HEAD | `git branch --show-current && git rev-parse HEAD` | `refactor/hbtxr-structure`, `087912c2b27f...` |
| Worktree before plan files | `git status --porcelain=v1 -uall` | 2 modified, 545 untracked, 0 staged |
| Untracked size | `git ls-files --others --exclude-standard` plus byte count | 7.576 MiB; max 1,862,668 bytes |
| Large/nested risk | scan >10 MiB, nested `.git`, submodules, symlinks | none in checkpoint scope |
| Static import gate | JSON parse, Python compile, `bash -n` | 105 JSON, 81 Python, 26 shell files passed |
| HANDOVER/HBTXR | `git ls-files` and blob comparison | 6,336 tracked; analysis 320/320 identical |
| HANDOVER/HGTXR | `git ls-files` and normalized path join | 1,881 tracked; software 859, hardware 701 |
| HG-PIPE Quantization | blob comparison to HBTXR | all 125 unique blobs already present |
| ICCAD24-HG-PIPE | blob comparison to HBTXR | 928/964 unique blobs present; missing mostly SPINAL |
| Spec-Kit | `specify check` | CLI 0.10.2.dev0 available; project not initialized |

## Expert Council Judgment

1. **Git release perspective:** create a plan-only commit, then three explicit
   checkpoint commits; never stage all paths; no push in this scope.
2. **Software/provenance perspective:** treat already-identical material as
   NO-OP and compare HGTXR descendants by behavior/API/test, not raw file copy.
3. **QA/hardware perspective:** license, data-contract, coordinate/metric,
   runtime, and hardware-tool gates precede any active promotion.

## Implementation Slices

### S0. Persist and commit the plan

Stage only `.agents/plans/2026-07-14-hbtxr-commit-handover-analysis.md` and
`docs/**`, including tracking and Aegis work records. Commit message:
`docs(plan): define HBTXR handover analysis workflow`.

### S1. Checkpoint archived algorithm sources

Stage only `algorithm/archive/imports/fecet_hbtxr` and
`algorithm/archive/imports/swift_hbtxr`. Treat these as provenance-preserving,
inactive sources. Commit message:
`chore(algorithm): archive FECET and SWIFT porting sources`.

### S2. Checkpoint EV-Eye/EX-Gaze analysis material

Stage `algorithm/analysis/{RESULTS,configs,scripts}` and the two associated
tracking documents. Commit message:
`chore(analysis): import EV-Eye and EX-Gaze handover artifacts`.

### S3. Checkpoint hardware and legacy evidence

Stage the explicit `hardware/configs/xr_accel`, selected `hardware/docs` trees,
and `references/legacy-codebase/hardware/hgpipe_iccad24_docs`. Commit message:
`chore(hardware): import accelerator handover evidence`.

### S4. Build provenance and collision inventory

Pin all six source repositories, enumerate tracked code/config/docs/tests, and
classify caches/generated assets. Reconfirm identical and changed mappings.

### S5. Analyze software candidates

Compare HGTXR `software/src/hbtxr`, scripts, tests, and configs against current
`algorithm/hybrid`. Identify behavior missing from HBTXR, import the relevant
test contract into the recommendation, and reject whole-tree copies.

### S6. Analyze hardware and automation candidates

Compare HGTXR HLS/PYNQ and XR_Accel cyclic/ZCU104 automation to current
`hardware`. Record interface, pragma, config-schema, toolchain, golden-vector,
and board dependencies. ViT_Accel remains generic/reference unless a reusable
board-independent component is proven.

### S7. Apply license, artifact, and validation gates

Map LICENSE/NOTICE scope to every candidate. Mark unclear ERVT/TENNs-Eye/
TDTracker/BRAT material `license-blocked` for active promotion. Exclude generated
data and record manifest/checksum policy.

### S8. Produce recommendations

For each candidate, choose exactly one: `NO-OP`, `REFERENCE_ONLY`, `ADAPT`,
`ACTIVE_CANDIDATE`, `EXCLUDED`, or `LICENSE_BLOCKED`. Include priority, expected
benefit, source/target path, required work, tests, dependencies, and rollback.
This plan does not implement those recommendations.

## Slice Validation Plan

| Slice | Behavior | First failing proof | Write scope | Lane | Owner |
|---|---|---|---|---|---|
| S0 | Plan is durable and resumable | required-section checker | `.agents/plans`, root `docs` | L1 docs | Master |
| S1 | Archive is preserved without activation | staged-path allowlist | two archive directories | L1 static | Git worker |
| S2 | Analysis import is checkpointed | JSON/Python/shell checks | algorithm analysis and track docs | L2 package | Algorithm worker |
| S3 | Hardware evidence is checkpointed | JSON/shell checks | listed hardware/reference paths | L2 artifact | Hardware worker |
| S4 | Every source is pinned and classified | missing pin/count row | inventory document | L1 docs | Research worker |
| S5 | Software gaps are behavior-mapped | candidate without API/test mapping | software audit | L2 analysis | Software analyst |
| S6 | Hardware gaps are interface-mapped | candidate without tool/test mapping | hardware audit | L2 analysis | Hardware analyst |
| S7 | Unsafe candidates are blocked | candidate without license/artifact status | license audit | L2 compliance | Evaluator |
| S8 | Recommendations are actionable | incomplete matrix row | matrix and recommendations | Council | Master |

### Wave Validity

| Check | Status | Notes |
|---|---|---|
| Distinct write scopes | PASS | S4-S7 use separate analysis files |
| No shared generated file | PASS | generated artifacts excluded |
| Integration order declared | PASS | checkpoint precedes analysis |
| Owner per slice | PASS | see `docs/Sub-Plan.md` |
| Discard path | PASS | drop analysis commit or revert semantic commit |

**Wave decision:** run every slice sequentially in the user-approved order.

## File Dependency Matrix

| Consumer | Dependency |
|---|---|
| Software audit | inventory, license audit, current hybrid API map |
| Hardware audit | inventory, artifact rules, current HLS interface map |
| Integration matrix | all four audit documents |
| Recommendations | integration matrix and validation evidence |

## File Conflict Matrix

| Pair | Conflict | Resolution |
|---|---|---|
| S1/S2 | none | distinct algorithm subtrees |
| S2/S3 | none | algorithm vs hardware/reference |
| S4-S7 | none | separate documents, still executed sequentially |
| all/S8 | shared roll-up docs | S8 runs after all audits |

Cross-wave shared files: `docs/track/PROGRESS.md`, `docs/Execution.md`, and
`docs/Validation.md` are updated only by the Master after each wave.

## Conformance Checks

| Issue | Check Type | Check |
|---|---|---|
| S0 | files_exist | all planning/tracking files listed above |
| S0 | content_check | plan contains Baseline Audit, Gherkin, DAG, matrices |
| S1-S3 | command | `git diff --cached --check` and staged allowlist check |
| S2 | tests | parse 105 JSON, compile 81 Python, `bash -n` 26 shell files |
| S4 | content_check | six repository pins and exclusion rules |
| S5-S7 | content_check | every candidate includes evidence and classification |
| S8 | command | no matrix row lacks source, target, class, risk, or validation |

## Execution Order

```text
Wave 0: S0 plan checkpoint
  -> Wave 1: S1 archive commit
  -> Wave 2: S2 analysis import commit
  -> Wave 3: S3 hardware/reference commit
  -> Wave 4 sequential: S4 inventory -> S5 software -> S6 hardware -> S7 risk
  -> Wave 5: S8 recommendation synthesis and analysis-doc commit
  -> Approval gate: choose implementation candidates in a new plan
```

## Verification Commands

```bash
git rev-parse --show-toplevel
git status --short --branch
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
git log -1 --oneline
git status --short --branch
```

Analysis validation additionally uses `git ls-files`, blob hashes, normalized
path joins, `rg` symbol searches, JSON/YAML parsing, Python compile/import smoke,
`bash -n`, and targeted test collection. Long Vivado/Vitis/board runs are not
part of this analysis plan.

## Planning Rules Compliance

| Rule | Status | Justification |
|---|---|---|
| PR-001 Mechanical Enforcement | PASS | staged allowlists and matrix checks |
| PR-002 External Validation | PASS | source repos pinned by SHA |
| PR-003 Feedback Loops | PASS | per-wave gates and tracking docs |
| PR-004 Separation Over Layering | PASS | archive, analysis, hardware split |
| PR-005 Process Gates First | PASS | checkpoint/provenance/license precede promotion |
| PR-006 Cross-Layer Consistency | PASS | software, quantization, hardware mapped |
| PR-007 Phased Rollout | PASS | five waves and approval gate |
| PR-008 Symbol Verify | PASS | semantic/symbol comparison required |
| PR-009 Mechanical Count Verify | PASS | baseline counts recorded |
| PR-010 Small Batches | PASS | four checkpoint/recommendation commits |
| PR-011 Stakes-Matched Tests | PASS | static gates now; tool/board gates deferred |

Unchecked rules: 0.

## Spec-Kit Note

`specify check` passed, but this existing project has not been initialized with
Spec-Kit. `specify init --here` would add scaffolding to an already dirty
547-change baseline. Therefore `docs/Spec.md` is written manually now. After S0
and the existing checkpoint commits are clean, initialize Spec-Kit only if the
user approves the added project scaffolding, then use it to refine this spec.

## Next Steps

1. Execute S0-S3 sequentially without pushing.
2. Execute S4-S8 and present the concrete integration matrix.
3. Create a separate implementation plan only for approved candidates.
