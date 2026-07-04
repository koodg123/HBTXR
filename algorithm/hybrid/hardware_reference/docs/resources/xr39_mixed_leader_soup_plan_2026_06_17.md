# XR-39 Mixed-Leader Soup Plan - 2026-06-17

## Prompt Pack

- Objective: recover P10 without losing XR-37 center and XR-38B P5 gains.
- IDEA-Gen phase/domain: P1 / Research Workflow.
- Parent skill: `paper-idea-generator`, using `ablation-study-designer` controls.
- Task family: event-based pupil tracking, no-train checkpoint soup for heatmap-state coordinate head.
- Fixed controls: support-adaptive fixed255k, min/base/max `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap grid `32`, heatmap/offset/center loss ratio `0.005/0.001/0.001`, state distillation disabled.
- Active gates before XR-39: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.

## Inputs

| Anchor | Checkpoint | Test center | Test P10 | Test P5 | Role |
|---|---|---:|---:|---:|---|
| XR-37 alpha `0.50` | `runs/interpolated_checkpoints/xr37_xr36b_center_xr36a_p5p10_alphaa0p50.pt` | 16.507612899371555 | 34.33205857958112 | 11.539116007941109 | center anchor |
| XR-36A best-P5 | `runs/raw_mode1_stage2_count255000_adamw_lr5e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr36a_p5anchor_xr34binit_xr29teacher_lr5e5_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_233614/train/best_track_p5.pt` | 16.576952314376832 | 34.74064704350063 | 11.415816688537598 | P10 anchor |
| XR-38B best-P5 | `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38b_xr37a050_init_xr36a_bestp5_ref_lr1p25e5_p10select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002120/train/best_track_p5.pt` | 16.513694180761064 | 33.93452457700457 | 11.843962955474854 | P5 anchor |

## Ablation Matrix

Weights are `(center, P10, P5)`.

| Lane | Device | Weights | Tag | Purpose |
|---|---|---|---|---|
| A | GPU0 | `0.85,0.05,0.10` | `c85p05f10` | minimal drift from center leader |
| A | GPU0 | `0.75,0.10,0.15` | `c75p10f15` | center-preserving P5 reinforcement |
| A | GPU0 | `0.70,0.20,0.10` | `c70p20f10` | light P10 recovery with center guard |
| B | GPU1 | `0.60,0.25,0.15` | `c60p25f15` | balanced recovery |
| B | GPU1 | `0.50,0.35,0.15` | `c50p35f15` | stronger P10 pull |
| B | GPU1 | `0.45,0.35,0.20` | `c45p35f20` | P10/P5 recovery stress test |
| C | GPU0 | `0.34,0.33,0.33` | `c34p33f33` | equal-ish sidecar-audited soup |
| C | GPU0 | `0.40,0.30,0.30` | `c40p30f30` | center-weighted equal-ish check |
| C | GPU0 | `0.30,0.40,0.30` | `c30p40f30` | P10-weighted equal-ish check |
| D | GPU1 | `0.30,0.30,0.40` | `c30p30f40` | P5-weighted equal-ish check |
| D | GPU1 | `0.25,0.45,0.30` | `c25p45f30` | stronger P10 sidecar check |
| D | GPU1 | `0.25,0.30,0.45` | `c25p30f45` | stronger P5 sidecar check |
| D | GPU1 | `0.20,0.40,0.40` | `c20p40f40` | P10/P5-heavy stress check |

## Commands

```bash
bash scripts/external/run_xr39_mixed_leader_soup_eval.sh a cuda:0
bash scripts/external/run_xr39_mixed_leader_soup_eval.sh b cuda:1
bash scripts/external/run_xr39_mixed_leader_soup_eval.sh c cuda:0
bash scripts/external/run_xr39_mixed_leader_soup_eval.sh d cuda:1
```

## Promotion Rule

- Promote center only if test center `<16.507612899371555`.
- Promote P10 only if test P10 `>34.74064704350063`.
- Promote P5 only if test P5 `>11.843962955474854`.
- If no gate promotes, prefer XR-38C loss-ratio training over another wide no-train soup sweep.

## Validation

- Checkpoint existence passed for all three anchors.
- Model-key compatibility passed: all three checkpoints have `138` model keys and matching key sets.
- Static validation required: `bash -n scripts/external/run_xr39_mixed_leader_soup_eval.sh`.

## Closeout

| Tag | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---|
| `c85p05f10` | 16.49972459588732 | 34.349065392357964 | 11.724064990452357 | center only |
| `c75p10f15` | 16.49540707213538 | 34.34268781798227 | 11.651786054883685 | center only |
| `c70p20f10` | 16.492875189440593 | 34.34566400391715 | 11.744473137174333 | center only; best XR-39 P5 |
| `c60p25f15` | 16.491779099191938 | 34.5306130204882 | 11.50000034059797 | center promoted |
| `c50p35f15` | 16.49391234772546 | 34.96981372152056 | 11.502126196452549 | P10 promoted |
| `c45p35f20` | 16.494208661147525 | 34.80824909891401 | 11.325680603299823 | P10 promoted |
| `c34p33f33` | 16.495203292369844 | 34.78146335738046 | 11.397959504808698 | P10 promoted |
| `c40p30f30` | 16.493308511802127 | 34.78146337100438 | 11.249149983269827 | P10 promoted |
| `c30p40f30` | 16.499327284949167 | 34.80187153816223 | 11.412840461730957 | P10 promoted |
| `c30p30f40` | 16.495379853248597 | 34.745323910032 | 11.412840461730957 | P10 promoted |
| `c25p45f30` | 16.503940873486656 | 35.02295998845781 | 11.460459525244577 | P10 promoted |
| `c25p30f45` | 16.496938497679576 | 34.74532390322004 | 11.463860872813633 | P10 promoted |
| `c20p40f40` | 16.502889422007968 | 34.77210963794163 | 11.386054754257202 | P10 promoted |

- Decision: XR-39 promotes center and P10.
- New center gate: `c60p25f15`, center `<16.491779099191938`.
- New P10 gate: `c25p45f30`, P10 `>35.02295998845781`.
- P5 gate remains XR-38B best-P5, P5 `>11.843962955474854`.
- Next bounded option: use XR-39 center/P10 leaders and XR-38B P5 leader as anchors for a narrower second soup or train XR-38C loss-ratio fallback if P5 must move next.
