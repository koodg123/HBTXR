# XR-40 XR39-Leader/P5 Soup Plan - 2026-06-17

## Prompt Pack

- Objective: move P5 beyond XR-38B while preserving XR-39 center/P10 gains.
- IDEA-Gen phase/domain: P1 / Research Workflow.
- Parent skill: `paper-idea-generator`, using `ablation-study-designer`.
- Task family: event-based pupil tracking, no-train checkpoint soup for heatmap-state coordinate head.
- Fixed controls: support-adaptive fixed255k, min/base/max `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap grid `32`, heatmap/offset/center loss ratio `0.005/0.001/0.001`, state distillation disabled.
- Active gates before XR-40: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.

## Inputs

| Anchor | Checkpoint | Test center | Test P10 | Test P5 | Role |
|---|---|---:|---:|---:|---|
| XR-39 `c60p25f15` | `runs/interpolated_checkpoints/xr39_mixedleader_soup_c60p25f15.pt` | 16.491779099191938 | 34.5306130204882 | 11.50000034059797 | center anchor |
| XR-39 `c25p45f30` | `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt` | 16.503940873486656 | 35.02295998845781 | 11.460459525244577 | P10 anchor |
| XR-38B best-P5 | `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38b_xr37a050_init_xr36a_bestp5_ref_lr1p25e5_p10select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002120/train/best_track_p5.pt` | 16.513694180761064 | 33.93452457700457 | 11.843962955474854 | P5 anchor |

## Ablation Matrix

Weights are `(center, P10, P5)`.

| Lane | Device | Weights | Tag | Purpose |
|---|---|---|---|---|
| A | GPU0 | `0.45,0.35,0.20` | `c45p35f20` | preserve center/P10, test small P5 pull |
| A | GPU0 | `0.40,0.40,0.20` | `c40p40f20` | balanced center/P10 with small P5 pull |
| A | GPU0 | `0.35,0.45,0.20` | `c35p45f20` | P10-preserving small P5 pull |
| B | GPU1 | `0.35,0.25,0.40` | `c35p25f40` | P5 recovery with center guard |
| B | GPU1 | `0.30,0.30,0.40` | `c30p30f40` | balanced P5 recovery |
| B | GPU1 | `0.25,0.35,0.40` | `c25p35f40` | P10/P5 recovery |
| B | GPU1 | `0.20,0.35,0.45` | `c20p35f45` | P5-heavy stress test |

## Commands

```bash
bash scripts/external/run_xr40_xr39leader_p5_soup_eval.sh a cuda:0
bash scripts/external/run_xr40_xr39leader_p5_soup_eval.sh b cuda:1
```

## Promotion Rule

- Promote center only if test center `<16.491779099191938`.
- Promote P10 only if test P10 `>35.02295998845781`.
- Promote P5 only if test P5 `>11.843962955474854`.
- If no gate promotes, trainable XR-38C loss-ratio fallback is preferred over more no-train soups.

## Validation

- Checkpoint existence passed for all three anchors.
- Model-key compatibility passed: all three checkpoints have `138` model keys and matching key sets.
- Static validation required: `bash -n scripts/external/run_xr40_xr39leader_p5_soup_eval.sh`.

## Closeout

| Tag | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---|
| `c45p35f20` | 16.493494159834725 | 34.77508579662868 | 11.255527530397687 | no promotion |
| `c40p40f20` | 16.494016419138227 | 34.7750858102526 | 11.249149976457868 | no promotion |
| `c35p45f20` | 16.49459844657353 | 34.730442953109744 | 11.249149976457868 | no promotion |
| `c35p25f40` | 16.49522715806961 | 34.64115719795227 | 11.529762240818568 | no promotion; best XR-40 P5 |
| `c30p30f40` | 16.495732394286563 | 34.59013679368155 | 11.37457515852792 | no promotion |
| `c25p35f40` | 16.496299374103547 | 34.575255850383215 | 11.380952712467739 | no promotion |
| `c20p35f45` | 16.49735657657896 | 34.575255850383215 | 11.354166998182023 | no promotion |

- Decision: XR-40 did not promote center, P10, or P5.
- Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.
- No-train soup follow-up is closed for now. Next P0 should be trainable loss-ratio fallback using a P5-preserving init/anchor.
