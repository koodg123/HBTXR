# XR-42C Explicit Tiny State-Distillation Recovery Plan

Date: 2026-06-17

## Goal

Recover or improve one of the active gates after XR-42 failed to promote center, P10, or P5.

Active gates before XR-42C:

- Center: `<16.491779099191938` from XR-39 `c60p25f15`
- P10: `>35.02295998845781` from XR-39 `c25p45f30`
- P5: `>11.868197652271816` from XR-41A best-P5

## Problem Found In XR-42

XR-42 loaded a teacher checkpoint but reused the XR-27 runner. That runner explicitly forces:

- `distillation.state_similarity=false`
- `distillation.state_weight=0.0`

With the heatmap-center head configured as track state, the useful teacher anchor is `track/state`, not the mostly frozen `track/fused` or `track/pupil` paths. XR-42 therefore had no effective heatmap-state teacher anchor. The remaining feature/prediction distillation losses were effectively tiny and did not promote any gate.

## Sub-Agent Audit

Task Card:

```yaml
task_card:
  task_id: XR42C-AUDIT
  sub_agent: gpt-5.5
  role: analyst
  objective: Audit Stage2 distillation keys and recommend a safe XR-42C lane.
  file_ownership: []
  assigned_skill: [caveman, computer-vision-expert, ablation-study-designer]
  inputs:
    - src/hbtxr/loss/distillation.py
    - src/hbtxr/training/model_factory.py
    - src/hbtxr/training/step_runner.py
    - scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh
    - configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml
  outputs:
    - Teacher-anchor key audit
    - XR-42C recommended lane defaults
  validation:
    - Confirm teacher model creation path
    - Confirm distillation loss addition path
    - Confirm useful trainable key for heatmap-state head
  dependencies: []
```

Audit conclusion:

- Teacher model exists only with `distillation.enabled=true` and `distillation.teacher_student=true`.
- Teacher outputs are used only during training.
- Effective track anchors are `track/fused`, `track/state`, and `track/pupil`.
- Since only `track_center_heatmap_head.*` is trainable, the safest useful anchor is `track/state`.
- Feature/prediction/KD/RKD anchors are disabled for XR-42C by default to avoid anchoring frozen or indirect paths.

## Experiment Lanes

### XR-42C-A: Center-Preserve With P5 Teacher State Anchor

- Script: `scripts/external/run_xr42c_explicit_distill_recovery.sh a cuda:0`
- Init: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c60p25f15.pt`
- Teacher: `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_weakdistill_trackonly_heatmapstate_g32_hm0_004_off0_0015_c0_0015_xr41a_p5init_xr38b_bestp5_xr39p10_ref_lossratio_lr1e5_p5select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_013052/train/best_track_p5.pt`
- LR: `1e-6`
- Best metric: `metric_track_center_px`
- Distillation:
  - `enabled=true`
  - `teacher_student=true`
  - `force_teacher_from_student=false`
  - `state_similarity=true`
  - `state_weight=0.0005`
  - `feature_similarity=false`
  - `feature_weight=0.0`
  - `prediction_similarity=false`
  - `prediction_weight=0.0`
  - `kd.enabled=false`
  - `rkd.enabled=false`

### XR-42C-B: P10 Self-Preserve Tiny State Anchor

- Script: `scripts/external/run_xr42c_explicit_distill_recovery.sh b cuda:1`
- Init: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`
- Teacher: same as init by default
- LR: `1e-6`
- Best metric: `metric_track_p10_pct`
- Distillation: same as XR-42C-A

## Shared Hyperparameters

- Raw event count: `255000`
- Adaptive count min/max: `160000` / `384000`
- Adaptive reference: `4000003 us`
- Adaptive scale power: `0.5`
- Heatmap grid: `32`
- Heatmap loss: `0.004`
- Offset loss: `0.0015`
- Center L2 loss: `0.0015`
- Trainable scope: `track_center_heatmap_head.*`
- Epochs: `10`
- Optimizer: AdamW

## Promotion Criteria

Promote only if a lane beats at least one active gate:

- Center `<16.491779099191938`
- P10 `>35.02295998845781`
- P5 `>11.868197652271816`

If XR-42C does not promote, pivot away from XR41 P5-teacher recovery and prioritize a new branch: teacher retraining or head/loss changes that directly address low-similarity and subject-39 residual buckets.
