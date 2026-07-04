# XR-28 Track-Heatmap Consolidation Diagnostics - 2026-06-16

## Goal

Consolidate XR-27 before launching more training. The diagnostic question is whether the XR-27 track-heatmap representation only improved the aggregate metric or whether it fixed the previously dominant failure buckets.

## Inputs

All runs use the same test manifest:

```text
data/_internal/manifests/manifest1/test_manifest.jsonl
```

Generated diagnostics:

- `runs/diagnostics/xr28_xr27a_bestp10_failure_buckets_20260616.json`
- `runs/diagnostics/xr28_xr06c_bestp10_failure_buckets_20260616.json`
- `runs/diagnostics/xr28_xr20a_bestcenter_failure_buckets_20260616.json`
- `runs/diagnostics/xr28_xr22_c34p33f33_failure_buckets_20260616.json`

Comparison anchors:

| Anchor | Role | Eval rows |
|---|---|---|
| XR-27A best-P10 | new heatmap-state leader | `runs/eval_fixed255k_xr27_trackheatmap_xr06cp10_heatmapstate_nostatedistill_lr1e4_adamw_lr1e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu0_w0_20260616_193440/eval/test/eval_rows.json` |
| XR-06C best-P10 | previous P10 leader | `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_050034/eval/test/eval_rows.json` |
| XR-20A best-center | previous center leader | `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr2p5e7_adamw_lr2_5e_7_bestcenter_test_gpu0_w0_20260616_160441/eval/test/eval_rows.json` |
| XR-22 c34/p33/f33 | previous P5/balanced anchor | `runs/eval_fixed255k_xr22_trileader_soup_c34_p33_f33_gpu0_w0_20260616_171305/eval/test/eval_rows.json` |

## Overall Weighted Metrics

| Anchor | Center px | P10 pct | P5 pct |
|---|---:|---:|---:|
| XR-27A best-P10 | 17.1708 | 32.1921 | 10.3219 |
| XR-06C best-P10 | 20.2065 | 26.9801 | 8.5846 |
| XR-20A best-center | 19.9949 | 26.0092 | 8.5335 |
| XR-22 c34/p33/f33 | 20.0528 | 26.6224 | 8.8912 |

Decision: XR-27A improves center by about `2.8-3.0 px`, P10 by about `5.2-6.2 pp`, and P5 by about `1.4-1.8 pp` against the previous anchors.

## Similarity Buckets

Weighted center/P10/P5:

| Similarity bucket | XR-27A | XR-06C | XR-20A | XR-22 |
|---|---:|---:|---:|---:|
| `<=0.1` | 20.1549 / 28.2660 / 7.3634 | 26.9471 / 19.0024 / 4.9881 | 26.5436 / 17.8147 / 5.2257 | 26.5630 / 18.5273 / 4.9881 |
| `0.1..0.3` | 18.4272 / 30.5147 / 8.4559 | 21.7632 / 21.3235 / 5.1471 | 21.7671 / 19.8529 / 4.4118 | 21.8417 / 20.2206 / 5.1471 |
| `0.3..0.6` | 16.9534 / 30.1181 / 11.0236 | 19.9102 / 23.4252 / 7.0866 | 19.7939 / 23.4252 / 7.6772 | 19.8892 / 23.6220 / 7.6772 |
| `0.6..0.9` | 15.4196 / 35.5083 / 11.9879 | 16.5990 / 34.4461 / 12.7466 | 16.3371 / 32.4734 / 12.2914 | 16.3798 / 33.5357 / 13.2018 |
| `>0.9` | 13.7313 / 42.2680 / 13.4021 | 12.6461 / 45.3608 / 13.4021 | 12.5056 / 48.4536 / 13.4021 | 12.5925 / 48.4536 / 13.4021 |

Interpretation:

- XR-27A gives the largest gain in the historical weak region: `similarity_target <= 0.3`.
- The only clear regression is the easy/high-similarity `>0.9` bucket, where prior anchors keep lower center error and higher P10.
- P5 in `0.6..0.9` and `>0.9` is not improved versus XR-22, so XR-28 should avoid changes that further trade away easy-case precision.

## Subject Buckets

Weighted center/P10/P5:

| Subject | XR-27A | XR-06C | XR-20A | XR-22 |
|---|---:|---:|---:|---:|
| `42` | 22.0117 / 22.8070 / 5.8480 | 26.2106 / 19.2982 / 4.6784 | 26.3411 / 18.7135 / 4.6784 | 26.3269 / 19.8830 / 4.6784 |
| `45` | 20.2342 / 21.5686 / 7.8431 | 23.0746 / 19.6078 / 8.4967 | 22.9677 / 20.9150 / 8.4967 | 22.7956 / 20.2614 / 9.8039 |
| `39` | 20.1437 / 17.3228 / 5.5118 | 22.0049 / 25.1969 / 11.8110 | 21.2044 / 23.6220 / 12.5984 | 21.2684 / 25.1969 / 11.8110 |
| `44` | 14.0377 / 45.2128 / 15.4255 | 16.7015 / 37.2340 / 13.2979 | 16.6288 / 33.5106 / 10.1064 | 16.7106 / 36.7021 / 12.2340 |
| `47` | 14.8440 / 42.3280 / 12.6984 | 16.4604 / 32.2751 / 10.0529 | 16.3726 / 31.2169 / 9.5238 | 16.3754 / 31.2169 / 10.0529 |
| `40` | 14.9253 / 40.0000 / 14.0541 | 18.3006 / 28.6486 / 4.3243 | 18.3148 / 28.6486 / 5.4054 | 18.2551 / 28.6486 / 5.4054 |

Interpretation:

- XR-27A substantially improves subjects `42`, `44`, `47`, and `40`.
- Subject `39` is the important exception: center improves, but P10/P5 regress sharply versus XR-06C/XR-22. This is the main targeted XR-28 diagnostic bucket.
- Subject `45` improves center and P10 but loses P5 versus XR-22.

## Remaining Hard Sessions

Top weighted-center XR-27A sessions:

| Session | Count | Weighted center | Weighted P10 | Weighted P5 |
|---|---:|---:|---:|---:|
| `user39/left/session_102` | 14 | 26.6537 | 10.0000 | 0.0000 |
| `user42/left/session_201` | 46 | 25.2326 | 9.7561 | 2.4390 |
| `user45/right/session_201` | 31 | 24.9945 | 6.8966 | 3.4483 |
| `user42/right/session_201` | 46 | 24.1551 | 17.9487 | 2.5641 |
| `user42/right/session_202` | 25 | 22.8476 | 27.2727 | 13.6364 |
| `user39/right/session_102` | 21 | 22.4086 | 10.0000 | 0.0000 |
| `user39/left/session_202` | 20 | 22.2336 | 5.8824 | 0.0000 |
| `user45/right/session_202` | 24 | 21.7775 | 18.1818 | 13.6364 |

## XR-28 Decision

XR-27 is not just an aggregate win. It directly improves the low-similarity and mid-similarity failure surface that blocked prior branches. The next experiment should consolidate the heatmap representation, not restart old scalar losses or optimizer sweeps.

Recommended priority:

1. Add/evaluate best-center checkpoint selection for XR-27 because current XR-27 saves best-P10 and best-P5, not best-center.
2. Run two narrow heatmap refinements around the winner:
   - LR `7e-5`, same heatmap weights.
   - LR `1.5e-4`, same heatmap weights.
3. If subject `39` remains a P10/P5 regression, test a small high-similarity/subject-regression preservation term, not broad LR-only polishing.
4. Keep `distillation.state_similarity=false` unless a teacher with trained heatmap head exists.

Implementation update:

- `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` now accepts:
  - `XR27_BEST_METRIC_NAME`, default `metric_track_p10_pct`.
  - `XR27_SCHEDULER_METRIC_NAME`, default equal to `XR27_BEST_METRIC_NAME`.
- For the existing XR-27A LR `1e-4` run, validation best-center and validation best-P10 both occurred at epoch `10`, so the existing `best_track_p10.pt` is also the best validation-center epoch for that run.
- Future XR-28 lanes can produce `best_metric_track_center_px.pt` by setting `XR27_BEST_METRIC_NAME=metric_track_center_px`.

Concrete next commands:

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 7e-5 cuda:0 \
  0.005 0.001 0.001 32 \
  xr28_heatmap_centerselect_lr7e5
```

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 1.5e-4 cuda:1 \
  0.005 0.001 0.001 32 \
  xr28_heatmap_centerselect_lr1p5e4
```

Rejected next steps:

- Do not repeat P10-boundary/hinge scalar losses from XR-21/XR-26.
- Do not repeat broad optimizer substitution from XR-23/XR-25.
- Do not train from XR-22 as a primary branch unless XR-28 shows heatmap cannot recover high-similarity precision.
- Do not enable state distillation with a teacher lacking trained heatmap weights.
