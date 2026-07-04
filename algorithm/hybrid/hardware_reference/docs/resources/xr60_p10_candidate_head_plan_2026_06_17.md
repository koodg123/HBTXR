# XR-60 P10 Candidate-Head Calibration Plan

Date: 2026-06-17

## Prompt Pack

- Objective: recover or exceed the active P10 gate while preserving the current center/P5 gates as much as possible.
- Source artifacts: XR-56 through XR-59 results, current HGTXR stage2 raw event-count pipeline, and new default-off `TrackCenterCandidateHead`.
- IDEA-Gen phase: P1
- Domain: Research Workflow
- Parent skill: `paper-idea-generator`
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracking on manifest1 full train/val/test split.
- Constraints: keep raw event-count contract fixed at `255000` with adaptive support `160000/384000`, reference `4000003 us`, scale power `0.5`; do not change data split; keep candidate head default-off outside this experiment.
- Assumptions: XR-59 scalar P10-soft continuation is closed because it regressed P10; P10 recovery now needs a representation/protocol change rather than another scalar-loss bracket.
- Required artifact: runner, ablation matrix, validation evidence, and promotion decision criteria.
- Verification evidence: syntax check, candidate model-build smoke, targeted unit tests, dry-run for both lanes, then full test eval of `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px` when present.
- Risks and missing inputs: full training result is not available yet; candidate selection uses max logit and may overfit validation without improving full-test P10.

## Current Gates

| Metric | Gate |
|---|---:|
| Center px | `<16.4701875601496` |
| P10 % | `>35.02295998845781` |
| P5 % | `>12.133503770828247` |

## Rationale

XR-56 improved center/P5 through direct metric-shaped loss, but did not recover P10. XR-57 bounded residual refinement degraded center/P5. XR-58 narrowed the P10 gap but still missed XR-39. XR-59 scalar P10-soft continuation regressed. XR-60 therefore changes the output representation: predict multiple bounded center candidates and train them with direct P10 membership and minimum soft-threshold losses.

## Mechanism

- Head: `TrackCenterCandidateHead`.
- Output: `K` bounded `delta_xy` candidates and candidate logits.
- Routing: selected candidate XY can replace final `track/state[:2]` through `model.heads.track_center_candidate_as_track_state=true`.
- Loss:
  - `loss.track_center_candidate_p10_bce_weight`: trains candidate logits to identify candidates within 10 px.
  - `loss.track_center_candidate_min_soft_threshold_weight`: pushes at least one candidate below the P10 margin.
  - `loss.track_center_candidate_delta_l2_weight`: bounds drift and protects center/P5.

## Ablation Matrix

| Lane | Init | Teacher | Trainable scope | K | Max delta | Blend | LR | Candidate BCE | Candidate min-soft | Delta L2 | P5 guard | Intent |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | XR-59B best-P10 | XR-39 P10 | `track_center_candidate_head.*` | 4 | 8 px | 0.50 | `1e-4` | `0.004` | `0.012` | `0.00025` | `0.0010` | safest candidate-only calibration |
| B | XR-59B best-P10 | XR-39 P10 | `track_center_candidate_head.*`, `track_center_heatmap_head.*` | 6 | 10 px | 0.75 | `7.5e-5` | `0.006` | `0.016` | `0.00020` | `0.0015` | stronger candidate+heatmap pressure |

Control variables:

- Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Event count: `255000`
- Adaptive support: min `160000`, max `384000`, reference `4000003`, scale power `0.5`
- Heatmap grid: `32`
- Heatmap/offset weights: `0.004/0.0015`
- Best metric: `metric_track_p10_pct`
- Optimizer: AdamW
- Distillation: XR-39 P10 teacher with state similarity enabled

## Runner

`scripts/external/run_xr60_p10_candidate_head.sh`

Dry run:

```bash
DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh b cuda:1
```

Full run:

```bash
bash scripts/external/run_xr60_p10_candidate_head.sh a cuda:0
bash scripts/external/run_xr60_p10_candidate_head.sh b cuda:1
```

## Validation Checklist

- [ ] `bash -n scripts/external/run_xr60_p10_candidate_head.sh`
- [ ] `python3 -m py_compile` for modified model/loss modules
- [ ] `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py`
- [ ] Candidate build smoke with `model.heads.track_center_candidate=true`
- [ ] Lane A/B dry-run
- [ ] Full train/eval logs captured in `runs/_logs`
- [ ] Promotion decision against current center/P10/P5 gates

## Decision Rule

- Promote any checkpoint that improves one active gate without unacceptable collapse in the other two metrics.
- If neither lane improves P10 and center/P5 remain below active gates, close candidate-head stage1 and move to teacher-target construction or failure-bucket data protocol changes.
