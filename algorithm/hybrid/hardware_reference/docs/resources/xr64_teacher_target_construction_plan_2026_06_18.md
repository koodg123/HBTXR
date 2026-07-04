# XR-64 Leakage-Safe P10 Teacher-Target Construction Plan

## Goal

Turn XR-63 oracle headroom into a trainable branch without using test labels or test-derived choices for training.

XR-63 oracle upper bound:

| Diagnostic | Center px | P10 % | P5 % |
|---|---:|---:|---:|
| XR-63 oracle | 16.04023192701366 | 36.48596938775512 | 13.41751700680271 |

Active gates:

- center `<16.468481131962367`
- P10 `>35.02295998845781`
- P5 `>12.133503770828247`

## Prompt Pack

- Phase/domain: P1 / Research Workflow.
- Parent skill: `paper-idea-generator`.
- Objective: create a leakage-safe teacher-target branch from existing HGTXR leaders.
- Source artifacts: XR-63 JSON/MD, existing eval rows, manifest1 train/val/test manifests, current Stage2 loss/data pipeline.
- Assumption: sample-wise best teacher selection can be approximated by a supervised selector or pseudo-target loss using train/val only.
- Missing evidence: train/val eval rows for XR-62A, XR-39, XR-56B, and XR-58A are not yet generated.
- Acceptance: final full-test eval must beat all active gates using no test-derived pseudo labels.

## Required Implementation

1. Generate train and val eval rows for all teacher candidates:
   - XR-62A center leader.
   - XR-39 P10 leader.
   - XR-56B P5 leader.
   - XR-58A P10-teacher refresh.
2. Build pseudo-target files only for train/val:
   - key: `sample_id`.
   - fields: selected teacher name, selected `track_state`, center error vs ground truth, selection reason, source eval path.
   - selection rule: choose teacher with minimum center error on train/val target.
   - optional conservative rule: choose XR-62A unless another teacher improves center by at least `0.25 px` or flips P10/P5 threshold.
3. Add dataset support:
   - config key: `data.track_target_override_path`.
   - loader attaches `track_target_override_state`, `track_target_override_weight`, and `track_target_override_source` when sample id exists.
   - no override on test split.
4. Add loss support:
   - config key: `loss.track_target_override_weight`.
   - apply auxiliary loss from `track/state` and optional `track/state_aux` to override state.
   - keep original `cur_state` metric/loss active as center guard.
5. Add runner:
   - lane A: conservative override, XR-62A init, XR-39 teacher, head-only, LR `3e-7`.
   - lane B: stronger override, XR-56B init, XR-39 teacher, head-only, LR `5e-7`.

## Control Variables

- Same manifest1 train/val/test split.
- Same support-adaptive fixed255k event config as XR-62.
- Same eval script and active gate definitions.
- Test split never used for pseudo-target construction.
- Distillation teacher checkpoint remains explicit and logged.

## Ablation Matrix

| ID | Override rule | Init | Teacher | Train scope | Expected signal |
|---|---|---|---|---|---|
| XR-64A | conservative min-error with `0.25 px` margin | XR-62A | XR-39 | head-only | center safe, mild P10/P5 gain |
| XR-64B | threshold-flip priority for P10/P5 | XR-56B | XR-39 | head-only | P5 preserved, P10 recovery attempt |
| XR-64C | pure oracle min-error | XR-62A | XR-39 | head-only | upper-bound pressure, higher overfit risk |

## Validation

- `py_compile` for new scripts/modules.
- Unit test: override file maps only matching sample ids and leaves missing ids unchanged.
- Unit test: test split runner rejects `data.track_target_override_path` unless explicit diagnostic flag is set.
- Dry-run runner A/B.
- Train/val pseudo-target provenance JSON.
- Final full-test eval of best-P10 and best-P5 checkpoints.

## Risks

- XR-63 headroom is an oracle; selector may not learn the same routing.
- Train/val oracle could overfit subject/session artifacts.
- If override weight is too high, model may regress center despite oracle headroom.
- If override selection uses test split by mistake, result is invalid.

## Next Worker Task

Implement `scripts/external/build_xr64_teacher_target_overrides.py`, dataset override loading, loss override term, unit tests, and `scripts/external/run_xr64_teacher_target_construction.sh`.
