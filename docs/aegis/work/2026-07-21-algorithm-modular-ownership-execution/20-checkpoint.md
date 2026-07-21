# Todo Checkpoint

## TodoCheckpointDraft

- Active slice: AM-000 baseline and preservation lock.
- Completed: isolated worktree created at baseline SHA; clean checkout verified;
  8,044 preserved blobs and all 20 current-to-target mapping rows recorded;
  `EvEye`, `src`, CLI, config, notebook and dynamic-import consumers censused.
- Pending: AM-010, AM-020, AM-030, AM-040, AM-050, AM-060, AM-070,
  AM-080, AM-090 and AM-900.
- Evidence refs: `10-intent.md`, `90-evidence.md`,
  `docs/track/algorithm-modular-baseline.md`,
  `docs/provenance/algorithm-modular-paths.tsv`.
- Blocked on: awaiting user decision for pre-existing Hybrid baseline failures:
  4 collection errors remain; partial suite is 16 failed, 128 passed, 28
  subtests passed. AM-000 artifact generation itself is not blocked.
- Next: independent AM-000 review, then pause before AM-010 until the user
  chooses how to handle the failing exact-baseline suite.

## ResumeStateHint

Resume in this worktree on `refactor/algorithm-modular-ownership`. Read
`10-intent.md`, this checkpoint, the approved parent plan in the primary
worktree, then compare HEAD/status and every frozen zone before editing.

## DriftCheckDraft

- Intent: aligned.
- Scope: aligned; only AM-000 owned documentation/provenance records changed.
- Compatibility: unchanged.
- New owner/fallback: none.
- Retirement: explicit and unstarted.
- Evidence: exact HEAD, preserved path/blob manifest, no-drift/status gates,
  consumer census, baseline test defects and diff allowlist captured.
- Decision: AM-000 artifact generation complete; pause before AM-010 pending
  user decision on the failing baseline.
