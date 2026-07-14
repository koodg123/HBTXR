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
