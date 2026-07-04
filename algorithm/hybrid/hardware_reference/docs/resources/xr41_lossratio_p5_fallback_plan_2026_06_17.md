# XR-41 Loss-Ratio P5 Fallback Plan - 2026-06-17

## Prompt Pack

- Objective: recover P5 beyond XR-38B while limiting damage to XR-39 center/P10 leaders.
- IDEA-Gen phase/domain: P1 / Research Workflow.
- Parent skill: `paper-idea-generator`, using `ablation-study-designer`.
- Task family: event-based pupil tracking, trainable Stage2 fallback for heatmap-state coordinate head.
- Fixed controls: support-adaptive fixed255k, min/base/max `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap grid `32`, state distillation disabled, head-only heatmap trainable scope.
- Active gates before XR-41: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.

## Inputs

| Anchor | Checkpoint | Test center | Test P10 | Test P5 | Role |
|---|---|---:|---:|---:|---|
| XR-38B best-P5 | `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38b_xr37a050_init_xr36a_bestp5_ref_lr1p25e5_p10select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002120/train/best_track_p5.pt` | 16.513694180761064 | 33.93452457700457 | 11.843962955474854 | P5 init |
| XR-39 `c25p45f30` | `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt` | 16.503940873486656 | 35.02295998845781 | 11.460459525244577 | P10 init/reference |

## Ablation Matrix

| Lane | Device | Init | Reference | LR | Best metric | Loss ratio `hm/off/center` | Purpose |
|---|---|---|---|---:|---|---|---|
| XR-41A | GPU0 | XR-38B best-P5 | XR-39 `c25p45f30` | `1e-5` | `metric_track_p5_pct` | `0.004/0.0015/0.0015` | preserve P5, recover center/P10 softly |
| XR-41B | GPU1 | XR-39 `c25p45f30` | XR-38B best-P5 | `8e-6` | `metric_track_p10_pct` | `0.004/0.0015/0.0015` | preserve P10, test P5 recovery with low drift |

## Commands

```bash
bash scripts/external/run_xr41_lossratio_p5_fallback.sh a cuda:0
bash scripts/external/run_xr41_lossratio_p5_fallback.sh b cuda:1
```

## Promotion Rule

- Promote center only if test center `<16.491779099191938`.
- Promote P10 only if test P10 `>35.02295998845781`.
- Promote P5 only if test P5 `>11.843962955474854`.
- If both lanes fail, next experiment should be a real loss/head change rather than another soup or tiny LR replay.

## Validation

- Required checkpoint existence must pass.
- Static validation required: `bash -n scripts/external/run_xr41_lossratio_p5_fallback.sh`.
- Full-test eval summaries from `run_xr27_trackheatmap_p10teacher_probe.sh` must exist before promotion decision.

## Closeout

| Lane / checkpoint | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---|
| XR-41A best-P10 | 16.519287032740458 | 34.66199056080409 | 11.25637790134975 | no promotion |
| XR-41A best-P5 | 16.519514334201812 | 34.09821502821786 | 11.868197652271816 | P5 promoted |
| XR-41B best-P10 | 16.50909768513271 | 34.426021228517804 | 11.286139808382307 | no promotion |
| XR-41B best-P5 | 16.50978491306305 | 34.228742252077375 | 11.734694249289376 | no promotion |

- Decision: XR-41A best-P5 promotes P5 only.
- Active gates after XR-41: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Center/P10 remain XR-39-owned. P5 is now XR-41A-owned.
