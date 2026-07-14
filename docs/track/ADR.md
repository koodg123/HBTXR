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
