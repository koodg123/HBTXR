# XR-27 Track-Heatmap Result Closeout - 2026-06-16

## Summary

XR-27 changed the coordinate representation instead of repeating scalar P10-boundary, hinge, LR-only, or optimizer polish. It added a track-center heatmap head, decoded the heatmap peak plus sub-cell offset into `track/state[:, :2]`, and trained only the new heatmap head from the XR-06C best-P10 checkpoint with XR-06C as teacher.

Result: XR-27A LR `1e-4` is the new active software leader and promotes all three active gates.

| Run | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| XR-27A heatmap-state LR `1e-4` | best-P10 | 17.274589475563594 | 31.915391901561193 | 10.289966331209456 | promotes center/P10/P5 |
| XR-27A heatmap-state LR `1e-4` | best-P5 | 17.6397743497576 | 29.80569808142526 | 10.210459525244577 | promotes center/P10/P5, weaker than best-P10 |
| XR-27B heatmap-state LR `3e-5` | best-P10 | 18.388037020819528 | 27.238521099090576 | 8.488520683561052 | promotes center/P10 only |
| XR-27B heatmap-state LR `3e-5` | best-P5 | 19.03528357914516 | 24.192602627617973 | 6.854166896002633 | promotes center only |

New active gates:

- Center: `<17.274589475563594`
- P10: `>31.915391901561193`
- P5: `>10.289966331209456`

Active leader checkpoint:

```text
runs/raw_mode1_stage2_count255000_adamw_lr1e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr06cp10_heatmapstate_nostatedistill_lr1e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_192010/train/best_track_p10.pt
```

## Commands

XR-27A:

```bash
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 1e-4 cuda:0 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr1e4
```

XR-27B:

```bash
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 3e-5 cuda:1 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr3e5
```

## Implementation Artifacts

- `src/hbtxr/models/heads.py`: added `TrackCenterHeatmapHead`.
- `src/hbtxr/models/tracker/head_factory.py`: registered optional `track_center_heatmap_head`.
- `src/hbtxr/models/tracker/track_branch.py`: added heatmap decode and optional heatmap-as-state replacement.
- `src/hbtxr/models/hybrid_tracker.py`: passed heatmap-head options into the track branch.
- `src/hbtxr/training/model_factory.py`: loaded `model.heads.track_center_heatmap*` options.
- `src/hbtxr/loss/bundles/track.py`: added grid CE and sub-cell offset loss.
- `src/hbtxr/loss/stage2.py`: wired heatmap losses into Stage2.
- `tests/test_track_center_l2_loss.py`: added heatmap decode/loss/Stage2 log coverage.
- `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`: XR-27 config.
- `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`: reproducible runner.

## Validation

- `bash -n scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` passed.
- Python compile passed for modified Python modules.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py` passed: `22 passed, 1 warning`.
- `git diff --check` passed for the relevant files before closeout.
- Config smoke confirmed:
  - `distillation.state_similarity: false`
  - `distillation.state_weight: 0.0`
  - trainable scope: `track_center_heatmap_head.*`
- Checkpoint smoke confirmed XR-06C load has no missing non-heatmap weights and only initializes the new heatmap head randomly.

## Important Risk

The first XR-27 launch had `distillation.state_similarity=true`; that would have distilled the student's heatmap-decoded `track/state` against a teacher with random heatmap-state output. Those runs were interrupted and are invalid.

Clean XR-27 runs explicitly disabled state distillation and used only feature/prediction distillation.

## Next Experiment

XR-28 should consolidate XR-27 rather than jump to another unrelated branch:

1. Compare XR-27 leader failure buckets against XR-06C, XR-20A, and XR-22.
2. Add or evaluate a best-center checkpoint path because XR-27 currently selects best-P10 only.
3. Run a narrow LR/weight refinement around LR `1e-4` only after diagnostics: candidate LRs `7e-5` and `1.5e-4`.
4. Consider 20-30 epoch extension if validation/test behavior remains stable.
