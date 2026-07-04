# XR-58 P10 Teacher Refresh Plan

Date: 2026-06-17

## Prompt Brief

- Goal: create a stronger P10-specialized teacher/anchor than XR-39 before another student-preservation attempt.
- Inputs: XR-39 P10 soup, XR-56/XR-57 closeouts, fixed255k support-adaptive event-count contract, manifest1 train/val/test split.
- Assumption: current P10 ceiling is teacher/selection limited because XR-57 residual refinement could not exceed XR-39 test P10.
- Constraint: do not claim improvement unless full-test P10 beats `35.02295998845781`; center/P5 may temporarily regress because this is a teacher-generation branch.
- Output: two-lane runner and full-test eval evidence for `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px` when present.
- Acceptance: train/eval exit `0`, full-test metric table, and a clear decision on whether a refreshed teacher is usable for later distillation.

## Rationale

XR-57's best P10 was Lane A `best_track_p5` at `34.849490649359566`, below the XR-39 P10 gate `35.02295998845781`. This means a small residual correction on top of the current student surface is not enough. XR-58 deliberately stops optimizing for unified center/P5 preservation and instead asks whether the model can create a better P10 teacher first.

## Lane Matrix

| Lane | GPU | Init | Teacher | Distill | LR | Scope | Intent |
|---|---:|---|---|---|---:|---|---|
| A | 0 | XR-39 P10 soup | XR-39 P10 soup | state distill `0.00010` | `1e-6` | heatmap, track head, event path, backbone stages 4/5, norms | anchored P10 teacher refresh |
| B | 1 | XR-39 P10 soup | XR-39 P10 soup | off | `5e-7` | same expanded scope | free P10 teacher refresh |

## Active Gates

- Center: `<16.4701875601496`
- P10: `>35.02295998845781`
- P5: `>12.133503770828247`

## Runner

- `scripts/external/run_xr58_p10_teacher_refresh.sh`

Default commands:

```bash
bash scripts/external/run_xr58_p10_teacher_refresh.sh a cuda:0
bash scripts/external/run_xr58_p10_teacher_refresh.sh b cuda:1
```

## Validation Checklist

- [x] `bash -n scripts/external/run_xr58_p10_teacher_refresh.sh`
- [x] `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh a cuda:0`
- [x] `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh b cuda:1`
- [x] actual lane A/B train/eval
- [x] full-test gate decision

## Results

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 4 | 16.503614359242576 | 34.90306201662336 | 11.51275544847761 | miss |
| A | `best_track_p5` | 3 | 16.501813726765768 | 34.68920147078378 | 11.259779255730765 | miss |
| B | `best_track_p10` | 4 | 16.506438190596445 | 34.84226275852748 | 11.501275873184204 | miss |
| B | `best_track_p5` | 3 | 16.49833288192749 | 34.540391949244906 | 11.062075165339879 | miss |

Full logs:

- `runs/_logs/xr58_p10_teacher_refresh_a_gpu0_20260617_082242.log`
- `runs/_logs/xr58_p10_teacher_refresh_b_gpu1_20260617_082242.log`

No active gate promoted. XR-58A best-P10 is the best teacher-refresh result so far at `34.90306201662336`, but it remains below XR-39 `35.02295998845781`.

## Decision Rule

If either lane beats P10 `35.02295998845781`, treat that checkpoint as a refreshed teacher candidate for a later center/P5-preserving student. If neither lane beats XR-39 P10, stop teacher-refresh via scalar P10 soft threshold and move to data/protocol diagnostics or a different P10 representation.

## Next Decision

Run one narrow XR-59 bracket around XR-58A before closing this family: anchored self-teacher only, slightly stronger LR/P10-soft settings, and no free no-distill lane unless used as a diagnostic.
