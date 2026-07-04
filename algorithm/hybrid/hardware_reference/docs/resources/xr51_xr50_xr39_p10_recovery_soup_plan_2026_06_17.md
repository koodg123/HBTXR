# XR-51 XR50/XR39 P10 Recovery Soup Plan

## Prompt Pack

- Objective: recover the active P10 gate after XR-50 promoted center and P5.
- Source artifacts: XR-50B best-P10, XR-50B best-P5, XR-39 P10 leader.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full test.
- Constraints: no training, no subject-39 leakage, strict full-test gate comparison.
- Assumptions: XR-50B and XR-39 checkpoints have identical model keys and can be interpolated/souped.
- Required artifact: executable XR-51 no-train runner plus eval evidence.
- Verification evidence: `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, full-test eval summaries.
- Risks and missing inputs: interpolation may recover P10 only by giving back XR-50 center/P5 gains; no-train soup cannot fix representation mismatch if anchors are not linearly compatible.

## Active Gates

- Center: `<16.482515714849743`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

## Rationale

XR-50 validated event-path adaptation and promoted center/P5, but P10 remained below XR-39. XR-51 is a no-train diagnostic to test whether XR-50B and XR-39 lie on a usable checkpoint-space path.

Known anchors:

- XR-50B best-P10: `16.483026616913932/34.394133479254585/11.978316681725639`
- XR-50B best-P5: `16.482515714849743/34.48086814199175/11.81250035422189`
- XR-39 P10: `16.503940873486656/35.02295998845781/11.460459525244577`

## Ablation Matrix

| lane | mode | anchors | weights/alphas | intent |
|---|---|---|---|---|
| XR-51A | interpolation | XR-50B best-P10 -> XR-39 P10 | `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, `0.35` | find smallest P10 recovery weight |
| XR-51B | tri-soup | XR-50B best-P10, XR-50B best-P5, XR-39 P10 | `80/10/10`, `75/10/15`, `70/10/20`, `70/15/15`, `65/15/20`, `60/20/20` | stabilize center/P5 while injecting P10 |

## Control Variables

- Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Test manifest: `data/_internal/manifests/manifest1/test_manifest.jsonl`
- Event count: fixed `255000` with adaptive range `160000/384000`
- Heatmap-state head: grid `32`, heatmap as evaluated track state
- Eval-only, no optimizer updates
- CUDNN disabled for stable eval runtime

## Execution

```bash
bash scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh a cuda:0
bash scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh b cuda:1
```

## Decision Rule

Promote only if one strict gate improves:

- center `<16.482515714849743`
- P10 `>35.02295998845781`
- P5 `>11.978316681725639`

Preference order:

1. P10 promotion with center `<16.491429926667895` and P5 `>11.868197652271816`.
2. Any strict P10 promotion, even if center/P5 regress, as diagnostic for a later trainable recovery.
3. No promotion but monotonic P10 trend with bounded center/P5 loss, used to seed XR-52.

## Results

Both lanes completed with exit `0`.

| candidate | center | P10 | P5 | gate result |
|---|---:|---:|---:|---|
| XR-51B `t60c20p20` | 16.478984827655 | 34.561650446483 | 11.716837085996 | center promoted |
| XR-51B `t65c15p20` | 16.479341397967 | 34.561650446483 | 11.665816674914 | center promoted |
| XR-51B `t70c15p15` | 16.479410881656 | 34.621174260548 | 11.738095596858 | center promoted |
| XR-51B `t70c10p20` | 16.479780200550 | 34.621174260548 | 11.612670407976 | center promoted |
| XR-51B `t75c10p15` | 16.479912798745 | 34.621174260548 | 11.729592193876 | center promoted |
| XR-51B `t80c10p10` | 16.480233781678 | 34.576531403405 | 11.780612598147 | center promoted |
| XR-51A `a0p25` | 16.480831880229 | 34.710459961210 | 11.517007139751 | center promoted; best XR-51 P10 |
| XR-51A `a0p20` | 16.480896479743 | 34.650936160769 | 11.517007139751 | center promoted |
| XR-51A `a0p15` | 16.481148343427 | 34.517007589340 | 11.633928925650 | center promoted |
| XR-51A `a0p35` | 16.481268215179 | 34.504252474649 | 11.346939107350 | center promoted |
| XR-51A `a0p10` | 16.481587026800 | 34.472364732197 | 11.780612598147 | center promoted |
| XR-51A `a0p05` | 16.482212608201 | 34.442602831977 | 11.978316681726 | center promoted; P5 tie |

## Judgment

XR-51 promoted the center gate only.

- New center leader: XR-51B `t60c20p20`, `16.478984827655`.
- P10 leader remains XR-39 `c25p45f30`, `35.02295998845781`.
- P5 leader remains XR-50B best-P10 / XR-51A `a0p05` tie, `11.978316681725639` to printed precision.

Interpretation:

- XR-50B and XR-39 are checkpoint-space compatible enough to improve center through no-train soup.
- P10 injection improved XR-51 P10 up to `34.710459961210`, but still missed the active P10 gate by about `0.3125`.
- P5 degrades as XR-39 P10 weight increases, so pure no-train P10 recovery is closed except as diagnostic.

Next experiment should use trainable recovery from the XR-51 center leader or XR-50B best-P10, with XR-39 P10 as teacher/reference and explicit P5 preservation.
