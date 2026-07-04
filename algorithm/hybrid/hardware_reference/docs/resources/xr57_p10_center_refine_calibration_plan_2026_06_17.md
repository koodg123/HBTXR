# XR-57 P10 Center-Refine Calibration Plan

Date: 2026-06-17

## Prompt Pack

- Objective: recover the strict P10 gate `>35.02295998845781` while preserving the XR-56 center/P5 gates.
- Source artifacts: XR-56 closeout, `src/hbtxr/models/heads.py`, `src/hbtxr/models/tracker/track_branch.py`, `src/hbtxr/loss/bundles/track.py`, and XR-56 runner.
- IDEA-Gen phase: P1
- Domain: Research Workflow
- Parent skill: `paper-idea-generator`
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker, fixed255k support-adaptive event window, manifest1 full train/val/test split.
- Constraints: preserve checkpoint compatibility, route calibration output into `track/state` so eval metrics can see it, keep residual bounded, use GPU0/GPU1 lanes.
- Facts: XR-56 improved center/P5 only; P10 remains XR-39-owned.
- Assumptions: a zero-initialized bounded residual center head can learn small P10-directed corrections without destroying the heatmap-state representation.
- Required artifact: head implementation, loss wiring, runner, validation evidence, and progress tracking.
- Verification evidence: py_compile, target pytest, model-build smoke, `bash -n`, lane A/B dry-run, final full-test eval summaries.
- Risks: residual may overfit validation threshold, state distillation can pull toward the P10 teacher but may regress P5, and new head adds parameters not present in prior checkpoints.

## Mechanism

XR-57 adds `TrackCenterRefineHead`, a zero-initialized bounded residual head:

```text
delta_xy = tanh(raw_xy) * max_delta_px * sigmoid(gate_logit)
track/state.xy = track/state.xy + blend * delta_xy
```

Default behavior is off, so old configs and checkpoints remain compatible. When enabled, checkpoint loading remains non-strict and missing refine-head parameters are initialized to identity residual.

## Lane Matrix

| Lane | GPU | Init | Teacher | LR | Max delta | Trainable scope | Intent |
|---|---:|---|---|---:|---:|---|---|
| A | 0 | XR-56B best-P5 center/P5 leader | XR-39 P10 soup | `1e-4` | `4.0` | refine head + heatmap head | recover P10 from current center/P5 leader |
| B | 1 | XR-39 P10 soup | XR-39 P10 soup | `7.5e-5` | `3.0` | refine head only | improve P10 anchor without large drift |

## Active Gates

- Center: `<16.4701875601496`
- P10: `>35.02295998845781`
- P5: `>12.133503770828247`

## Implemented Change Set

- `src/hbtxr/models/heads.py`: `TrackCenterRefineHead`.
- `src/hbtxr/models/tracker/head_factory.py`: optional head construction.
- `src/hbtxr/models/tracker/track_branch.py`: route residual into final `track/state`.
- `src/hbtxr/models/hybrid_tracker.py`: constructor and module wiring.
- `src/hbtxr/training/model_factory.py`: config key wiring.
- `src/hbtxr/loss/bundles/track.py`: optional refine delta regularization.
- `src/hbtxr/loss/stage2.py`: refine loss logs.
- `tests/test_track_center_l2_loss.py`: head identity, loss logs, state routing.
- `scripts/external/run_xr57_p10_center_refine_calibration.sh`: two-lane runner.

## Validation Checklist

- [x] `chmod +x scripts/external/run_xr57_p10_center_refine_calibration.sh`
- [x] `bash -n scripts/external/run_xr57_p10_center_refine_calibration.sh`
- [x] `python3 -m py_compile` on modified model/loss files
- [x] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`
- [x] `PYTHONPATH=src .venv/bin/python -c ... build_model(...)`
- [x] `DRY_RUN=1 bash scripts/external/run_xr57_p10_center_refine_calibration.sh a cuda:0`
- [x] `DRY_RUN=1 bash scripts/external/run_xr57_p10_center_refine_calibration.sh b cuda:1`
- [x] actual lane A/B train/eval
- [x] gate promotion/no-promotion decision

## Results

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 3 | 16.69362453562873 | 34.38307912690299 | 10.973214619500297 | miss |
| A | `best_track_p5` | 7 | 16.715463175092424 | 34.849490649359566 | 10.925595603670393 | miss |
| B | `best_track_p10` | 12 | 16.504537062985555 | 34.137755966186525 | 11.207483332497732 | miss |
| B | `best_track_p5` | 5 | 16.4997801729611 | 34.190476996558054 | 11.476615987505232 | miss |

Full logs:

- `runs/_logs/xr57_p10_center_refine_calibration_a_gpu0_20260617_075353.log`
- `runs/_logs/xr57_p10_center_refine_calibration_b_gpu1_20260617_075354.log`

No active gate promoted. Lane A best-P5 was the best XR-57 P10 at `34.849490649359566`, still below the XR-39 P10 gate `35.02295998845781`.

## Next Decision

Run XR-58 P10 teacher refresh. The next experiment should first create a stronger P10-specialized teacher/anchor instead of adding another residual or scalar-loss variant to the current student.
