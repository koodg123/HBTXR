> **작성** 2026-07-15 · **갱신** 2026-07-15
> **상태** active — governance
> **소유** repo

# Baseline Governance

## Authority order

1. Explicit current user direction.
2. Project instruction files when present.
3. `docs/Spec.md`, accepted ADRs, and an approved implementation plan.
4. Current HBTXR source and tests.
5. HANDOVER repositories as read-only provenance and behavioral evidence.

HANDOVER content never overrides current HBTXR merely because its path or symbol
matches. Generated artifacts and historic success reports are not runtime authority.

## Drift rule

Before each wave, compare branch, HEAD, worktree, relevant source paths,
dependencies/tools, source SHAs, license status, and approval state with the plan
baseline. Stop and revise the plan when a canonical owner, public contract,
source SHA, license, dataset split, toolchain, or board target changes.

## Completion rule

Completion requires focused verification, related regression checks, an exact
staged allowlist, rollback, and explicit uncovered-risk reporting. Training,
synthesis, board, external-storage, push, and release claims require separate
evidence and authorization.
