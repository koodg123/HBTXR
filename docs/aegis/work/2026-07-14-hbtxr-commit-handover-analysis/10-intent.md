# Task Intent Draft

- Requested outcome: create reviewable local HBTXR checkpoint commits and a
  decision-ready HANDOVER integration analysis.
- Scope: `/HBTXR` only; C0-C4 from the approved canonical plan.
- Non-goals: push, sibling-repository writes, bulk HANDOVER copy/cherry-pick,
  active code promotion, long FPGA/training runs.
- Success evidence: five local commits with staged allowlists; clean HBTXR
  worktree; six complete analysis artifacts; validation and drift records.
- Stop conditions: done, blocked, needs-verification, or scope-exceeded.

## Baseline Read Set

- `.agents/plans/2026-07-14-hbtxr-commit-handover-analysis.md`
- `docs/Master-Plan.md`, `docs/Sub-Plan.md`, `docs/Spec.md`
- Git baseline `087912c2b27f`
- current `git status --short --branch`
- 2026-07-08 algorithm and hardware import manifests

## Impact Statement

The work changes local Git history and adds analysis documentation. It does not
change active product behavior or external state.

## Execution Readiness View

- Intent lock: checkpoint current imports, then analyze HANDOVER candidates.
- Scope fence: HBTXR only; no push; no active port.
- Baseline lock: `refactor/hbtxr-structure` at `087912c2b27f`.
- Owner constraint: one writer per slice; Master owns shared tracking records.
- Compatibility boundary: preserve current HBTXR APIs and hardware descendants.
- Retirement boundary: archives stay inactive; rejected candidates remain source evidence.
- Test obligations: staged allowlist, diff check, static/import checks, matrix completeness.
- Review gates: spec compliance, quality review, drift check per slice.
- Rewind rule: use semantic `git revert`, never destructive reset.
