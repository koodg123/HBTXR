# Code/File Integration Intent — 2026-07-15

## Requested outcome

Integrate the code and durable files that can be adopted directly into HBTXR,
while excluding installation, reproduction, experiments, FPGA execution,
Spec-Kit initialization, and remote operations.

## Scope

- Selected: T-010, T-020, T-100, T-110, T-210.
- Excluded: T-000, T-120, T-200, T-220, T-300–T-630.
- Repository/commit boundary: this HBTXR repository only.
- HANDOVER boundary: read-only evidence; no bulk copy or direct runtime owner.

## Success evidence

- Each selected task has an atomic local commit.
- Fresh spec and quality review pass per task.
- Available focused/static and related regression checks pass.
- Missing runtime dependencies are reported, never installed or treated as pass.
- No experiment, generated artifact, external mutation, or push occurs.

## Baseline read set

- `docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md`
- `docs/aegis/baseline/2026-07-15-follow-up-baseline.md`
- `docs/Spec.md`, `docs/track/ADR.md`
- `docs/analysis/HANDOVER-INTEGRATION-MATRIX.md`
- current source/tests owned by each selected task

## Baseline usage

Required references are present and acknowledged. Python 3.12 and PyYAML are
available; NumPy, PyTorch, and pytest are unavailable. No missing reference
blocks planning, but runtime-dependent task completion may be partial.

## Impact statement

This work adds contracts, validators, and configuration schema tooling at their
current HBTXR owners. It does not activate HANDOVER trees, change
default runtime strategy, run experiments, or alter remote state.

## Execution Readiness View

- Intent lock: code/file integration only.
- Scope fence: T-010, T-020, T-100, T-110, T-210 and their owned paths.
- Baseline lock: `refactor/hbtxr-structure` plan checkpoint before worktree fork.
- Compatibility: current data/checkpoint/config/HLS default semantics retained.
- Retirement: conditional helper/adapter remains removable on duplication.
- Verification: stdlib/PyYAML checks only; no package installation or runtime
  experiment. T-100/T-110 use dependency-light contract tests.
- Reviews: implementer → spec reviewer → quality reviewer, sequentially.
- Drift: stop on new installation, reproduction, experiment, or owner boundary.
- Advisory boundary: verification evidence does not authorize push/release.
