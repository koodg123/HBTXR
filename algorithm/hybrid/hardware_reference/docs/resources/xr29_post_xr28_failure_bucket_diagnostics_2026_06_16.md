# XR-29 Post-XR-28 Failure-Bucket Diagnostics

Date: 2026-06-16

## Objective

Consolidate the XR-28 LR `1.5e-4` promotion before broadening the experiment space. The immediate question is whether the promoted heatmap-state checkpoint fixed the known XR-27 residual risks:

- high-similarity `similarity_target > 0.9`
- subject `39` P10/P5 weakness
- `session_201` heavy-error buckets

## Inputs

Promoted XR-28 eval rows:

```text
runs/eval_fixed255k_xr27_trackheatmap_xr28_heatmap_centerselect_lr1p5e4_adamw_lr1_5e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu1_w0_20260616_202311/eval/test/eval_rows.json
```

Generated diagnostic:

```text
runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json
```

Command:

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows runs/eval_fixed255k_xr27_trackheatmap_xr28_heatmap_centerselect_lr1p5e4_adamw_lr1_5e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu1_w0_20260616_202311/eval/test/eval_rows.json \
  --output runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --state-key track_state \
  --top-k 12 \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override training.num_workers=0
```

Validation: joined `2238` rows, missing predictions `0`.

## Overall Comparison

Weighted center/P10/P5:

| Model | Center px | P10 pct | P5 pct |
|---|---:|---:|---:|
| XR-28 LR `1.5e-4` | 16.9868 | 32.3454 | 10.5774 |
| XR-27A LR `1e-4` | 17.1708 | 32.1921 | 10.3219 |
| XR-20A center anchor | 19.9949 | 26.0092 | 8.5335 |
| XR-22 P5 anchor | 20.0528 | 26.6224 | 8.8912 |
| XR-06C P10 anchor | 20.2065 | 26.9801 | 8.5846 |

XR-28 LR `1.5e-4` improves the XR-27A weighted aggregate by center `-0.1840px`, P10 `+0.1533pp`, and P5 `+0.2555pp`.

## Similarity Buckets

XR-28 LR `1.5e-4` versus XR-27A deltas:

| Bucket | XR-28 center | XR-28 P10 | XR-28 P5 | Center delta | P10 delta | P5 delta |
|---|---:|---:|---:|---:|---:|---:|
| `similarity <=0.1` | 19.8681 | 26.60 | 7.36 | -0.2868 | -1.66 | 0.00 |
| `0.1..0.3` | 18.3364 | 31.99 | 10.29 | -0.0907 | +1.47 | +1.84 |
| `0.3..0.6` | 16.7592 | 30.51 | 12.20 | -0.1942 | +0.39 | +1.18 |
| `0.6..0.9` | 15.2350 | 36.12 | 11.53 | -0.1846 | +0.61 | -0.46 |
| `>0.9` | 13.7900 | 42.27 | 10.31 | +0.0587 | 0.00 | -3.09 |

Interpretation:

- LR `1.5e-4` improves center in low/mid similarity buckets.
- The weakest remaining aggregate is still `similarity <=0.1`.
- High-similarity P10 remains worse than older anchors: XR-20A/XR-22 scored P10 `48.45` and P5 `13.40` in `>0.9`, while XR-28 scores P10 `42.27`, P5 `10.31`.

## Subject Buckets

Worst weighted center subjects after XR-28:

| Subject | Center px | P10 pct | P5 pct | Delta vs XR-27A |
|---|---:|---:|---:|---|
| `42` | 21.7877 | 21.64 | 5.85 | center -0.2239, P10 -1.17, P5 0.00 |
| `45` | 20.2426 | 24.18 | 8.50 | center +0.0084, P10 +2.61, P5 +0.65 |
| `39` | 19.8477 | 18.90 | 3.15 | center -0.2960, P10 +1.57, P5 -2.36 |
| `41` | 18.4566 | 26.19 | 10.12 | center -0.1039, P10 +2.38, P5 +0.60 |
| `38` | 18.0787 | 30.54 | 5.99 | center -0.2432, P10 +0.60, P5 -1.80 |

Subject `39` remains a real P5 risk. It improves center and P10 over XR-27A, but P5 falls further and remains far below XR-20A/XR-22/XR-06C subject-39 P5.

## Worst Sessions

Worst weighted center sessions after XR-28:

| Session | Weighted count | Center px | P10 pct | P5 pct |
|---|---:|---:|---:|---:|
| `user45/right/session_201` | 29 | 25.5307 | 6.90 | 0.00 |
| `user42/left/session_201` | 41 | 25.0462 | 14.63 | 0.00 |
| `user42/right/session_201` | 39 | 24.1036 | 12.82 | 2.56 |
| `user39/left/session_102` | 10 | 23.8526 | 10.00 | 0.00 |
| `user42/right/session_202` | 22 | 22.7549 | 27.27 | 13.64 |
| `user39/left/session_202` | 17 | 22.0732 | 11.76 | 0.00 |
| `user39/right/session_102` | 20 | 21.8485 | 15.00 | 0.00 |
| `user41/left/session_201` | 37 | 21.7034 | 24.32 | 13.51 |
| `user45/right/session_202` | 22 | 21.4592 | 22.73 | 13.64 |
| `user39/right/session_201` | 36 | 21.3304 | 19.44 | 0.00 |

## Decision

XR-28 LR `1.5e-4` is a real promotion, but the residual failure is not a general optimizer problem. It is concentrated in low-similarity rows, subject `39`, subjects `42/45`, and specific session buckets.

Next execution:

1. Run bounded LR neighbors around the promoted LR because `7e-5` underfit and `1.5e-4` was still improving at epoch `10`.
   - GPU0: LR `1.25e-4`
   - GPU1: LR `1.75e-4`
2. Queue a longer-budget LR `1.5e-4` reproduction/extension after one GPU frees.
3. Defer new loss/teacher changes until the LR-neighbor results show whether high-similarity and subject-39 P5 can improve without changing the objective.

Launched XR-29 commands:

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 1.25e-4 cuda:0 \
  0.005 0.001 0.001 32 xr29_heatmap_centerselect_lr1p25e4
```

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 1.75e-4 cuda:1 \
  0.005 0.001 0.001 32 xr29_heatmap_centerselect_lr1p75e4
```

Logs:

- `runs/_logs/xr29_heatmap_centerselect_lr1p25e-4_gpu0_20260616.log`
- `runs/_logs/xr29_heatmap_centerselect_lr1p75e-4_gpu1_20260616.log`
