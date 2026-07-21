# Todo Checkpoint

## TodoCheckpointDraft

- Active slice: AM-070 pools removal (AM-040, AM-050, AM-060 complete).
- Completed: AM-000 baseline/preservation lock, AM-010 package/import
  contract, AM-020 dataset migration, AM-030 utility migration, AM-040 owner split,
  AM-050 config/test/launcher flattening, and AM-060 EvEye retirement; isolated worktree created at baseline SHA; clean checkout verified;
  8,044 preserved blobs and all 20 current-to-target mapping rows recorded;
  `EvEye`, `src`, CLI, config, notebook and dynamic-import consumers censused.
- Pending: AM-070, AM-080, AM-090 and AM-900.
- Evidence refs: `10-intent.md`, `90-evidence.md`,
  `docs/track/algorithm-modular-baseline.md`,
  `docs/provenance/algorithm-modular-paths.tsv`.
- Blocked on: none. On 2026-07-21 the user directed execution to continue and
  requested testing against the refactored result.
- Baseline policy: the pre-existing 4 collection errors and partial result of
  16 failed, 128 passed and 28 subtests passed remain comparison evidence. They
  do not block the structural slices and must not be hidden or waived.
- AM-010 amendment: multi-root discovery alone was proven invalid. Explicit
  package-to-source mappings and exact/descendant discovery allowlists are the
  verified contract. Dual-case `EvEye`/`eveye` validation forces Linux
  `TMPDIR`/`TEMP`/`TMP`; no dual-payload migration checkpoint may be committed.
- AM-020 result: dataset implementation is canonical under `eveye.dataset`;
  explicit legacy wrappers retain identity, two demos moved to the engine tool,
  and forced-Linux-temp focused regression passed 20 tests. One inactive
  pre-existing EllipseMobileNet import is allowlisted for AM-040 disposition.
- Next: execute AM-070. AM-060 removed the legacy EvEye payload, so the
  dual-case constraint is lifted and migration commits are now permitted.
  AM-900 reports the complete post-refactor result and exact delta from AM-000.

## ResumeStateHint

Resume in `/tmp/hbtxr-algorithm-modular-ownership-exec` on
`refactor/algorithm-modular-ownership-exec`. Read
`10-intent.md`, this checkpoint, the approved parent plan in the primary
worktree, then compare HEAD/status and every frozen zone before editing.

## DriftCheckDraft

- Intent: aligned.
- Scope: aligned; only AM-000 owned documentation/provenance records changed.
- Compatibility: aligned; current `EvEye` origin remains unchanged and future
  `eveye.<owner>` paths are explicitly mapped.
- New owner/fallback: no implementation owner yet; packaging metadata only.
- Retirement: explicit and unstarted; case-only coexistence cannot cross a
  source-migration commit boundary.
- Evidence: AM-010 focused tests 3 passed; current and synthetic wheels built;
  synthetic six-owner clean install/import and lookalike/bare exclusion passed;
  preserved zones and diff allowlist passed; spec and quality reviews approved.
- Decision: continue to AM-040. Final regression remains an AM-900 obligation.
