# XR-56 Soft-Threshold P10 Supervision Plan

Date: 2026-06-17

## Prompt Pack

- Objective: recover the active P10 gate `>35.02295998845781` without losing the XR-52 center/P5 gates.
- Source artifacts: XR-49 through XR-55 closeouts, `src/hbtxr/loss/bundles/track.py`, `scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh`.
- IDEA-Gen phase: P1
- Domain: Research Workflow
- Parent skill: `paper-idea-generator`
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker, fixed255k support-adaptive event window, manifest1 full train/val/test split.
- Constraints: preserve checkpoint compatibility, avoid architecture change in first XR-56 branch, use GPU0/GPU1 parallel lanes, compare against current mixed active gates.
- Assumptions: metric is hard center-error hit rate, so differentiable soft-threshold loss is closer to P10 than the previous band-weighted boundary loss.
- Required artifact: loss addition, runner, validation evidence, and progress tracking.
- Verification evidence: unit tests, py_compile, `bash -n`, lane A/B dry-run, runtime startup logs, final full-test eval summaries.
- Risks: stronger P10 pressure may regress center/P5; P5 soft guard is included but may be insufficient.

## Rationale

XR-55 closed the low-LR expanded-scope branch with no promotion. The failed P10 routes now include checkpoint soup, event-adapter continuation, band-weighted boundary loss, and final-block scope expansion. XR-56 changes the supervision surface directly by replacing the Gaussian band-limited P10 boundary pressure with a metric-aligned positive BCE:

```text
loss = softplus((center_error - margin_px) / temperature_px)
```

This gives gradient to all outside-threshold samples instead of only samples near the 10 px band.

## Change Set

- `src/hbtxr/loss/bundles/track.py`
  - Add `center_soft_threshold_loss`.
  - Add `track_p10_soft_threshold` and `track_p5_soft_threshold` loss terms.
  - Add default-zero aux variants for `track_state_aux`.
- `src/hbtxr/loss/bundles/__init__.py`
  - Export `center_soft_threshold_loss`.
- `tests/test_track_center_l2_loss.py`
  - Cover default-zero behavior, formula correctness, config weights, and stage2 log wiring.
- `scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
  - Two-lane train/eval runner.

## Lane Matrix

| lane | GPU | init | teacher | LR | P10 soft | P10 temp | P5 soft | trainable scope |
|---|---:|---|---|---:|---:|---:|---:|---|
| A | 0 | XR-39 P10 soup `c25p45f30` | XR-39 P10 soup | `7.5e-7` | `0.006` | `1.5` | `0.001` | heatmap head + event adapter + event projection |
| B | 1 | XR-52B best-P10/P5 leader | XR-39 P10 soup | `5e-7` | `0.008` | `1.25` | `0.002` | heatmap head + event adapter + event projection |

## Active Gates

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

## Validation Checklist

- [x] `chmod +x scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
- [x] `bash -n scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
- [x] `python3 -m py_compile src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py`
- [x] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`
- [x] `DRY_RUN=1 bash scripts/external/run_xr56_soft_threshold_p10_supervision.sh a cuda:0`
- [x] `DRY_RUN=1 bash scripts/external/run_xr56_soft_threshold_p10_supervision.sh b cuda:1`
- [x] actual lane A/B launched on GPU0/GPU1
- [x] lane A/B train exit `0`
- [x] full-test eval summaries for `best_track_p10`, `best_track_p5`, and available center checkpoint
- [x] gate promotion/no-promotion decision

## Results

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 8 | 16.49389898266111 | 34.32270488057818 | 11.699830286843437 | no promotion |
| A | `best_track_p5` | 3 | 16.4924229485648 | 34.62330012321472 | 11.304422106061663 | no promotion |
| B | `best_track_p10` | 1 | 16.477364584377835 | 34.37670146397182 | 12.044218049730574 | P5 only versus old gate |
| B | `best_track_p5` | 5 | 16.4701875601496 | 34.29294293948582 | 12.133503770828247 | center and P5 promoted |

`best_metric_track_center_px.pt` was not produced by either lane, so center-checkpoint eval was skipped.

## Decision

XR-56 did not recover the strict P10 gate. The best P10 observed was lane A `best_track_p5` at `34.62330012321472`, below the active XR-39 gate `35.02295998845781`.

XR-56B `best_track_p5` promoted center and P5:

- Center gate: `<16.4701875601496`
- P10 gate: `>35.02295998845781`
- P5 gate: `>12.133503770828247`

## Next Decision

Move to XR-57 architecture-side direct calibration only after preserving inference routing into `track/state`. Candidate XR-57 paths: dedicated P10 calibration/refinement head, explicit P10 teacher retraining from the best P10 anchor, or a hybrid teacher-refresh plus calibration probe. Do not spend the next P0 slot on another checkpoint soup or scalar loss-only branch.
