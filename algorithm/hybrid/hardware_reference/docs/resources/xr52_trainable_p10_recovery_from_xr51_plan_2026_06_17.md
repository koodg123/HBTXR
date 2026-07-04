# XR-52 Trainable P10 Recovery From XR-51 Plan

## Prompt Pack

- Objective: recover P10 after XR-51 promoted center, without losing XR-50/XR-51 center and P5 gains.
- Source artifacts: XR-51B `t60c20p20`, XR-51A `a0p05`, XR-39 P10 leader, XR-50 event-adapter result.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full train/val/test.
- Constraints: no subject-39 leakage, full-test gate comparison, keep heatmap-state evaluation path, keep event-adapter trainable scope validated by XR-50.
- Assumptions: XR-51 no-train soup exposed a center-improving manifold, but P10 recovery requires gradient updates with XR-39 P10 teacher/reference.
- Required artifact: executable XR-52 runner and validation evidence.
- Verification evidence: `bash -n`, lane A/B `DRY_RUN=1`, checkpoint existence, raw event-count startup, train/eval exit status, full-test eval summaries.
- Risks and missing inputs: XR-39 teacher may pull P5 below the strict XR-50/XR-51 P5 gate; low LR may be too weak to move P10 above `35.02295998845781`.

## Active Gates

- Center: `<16.478984827655`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

## Rationale

XR-51 showed checkpoint-space compatibility:

- Best center: XR-51B `t60c20p20`, `16.478984827655/34.561650446483/11.716837085996`.
- Best XR-51 P10: XR-51A `a0p25`, `16.480831880229/34.710459961210/11.517007139751`.
- Best P5-preserving candidate: XR-51A `a0p05`, `16.482212608201/34.442602831977/11.978316681726`.

The no-train path did not reach XR-39 P10. XR-52 keeps the XR-50 event-path mechanism that promoted center/P5 and uses a very low LR continuation to test whether P10 can be pulled up without full representation drift.

## Ablation Matrix

| lane | init | teacher/reference | LR | best metric | center guard | P10 boundary | state distill | intent |
|---|---|---|---:|---|---:|---:|---:|---|
| XR-52A | XR-51B `t60c20p20` | XR-39 P10 | `5e-7` | P10 | `0.0060` | `0.008` | `0.00035` | start from best center and recover P10 |
| XR-52B | XR-51A `a0p05` | XR-39 P10 | `3e-7` | P10 | `0.0070` | `0.005` | `0.00050` | start from P5-preserve point and reduce drift |

## Control Variables

- Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Train/val/test manifests: `data/_internal/manifests/manifest1`
- Event count: fixed `255000`, adaptive min/max `160000/384000`, `reference_us=4000003`, `scale_power=0.5`
- Heatmap-state head: grid `32`, heatmap as evaluated track state
- Trainable scope:
  - `track_center_heatmap_head.*`
  - `event_adapter.*`
  - `patch_frontend.event_embed.proj.*`
- Optimizer: AdamW
- Boundary margin: P10 margin `10px`, band `4px`, temperature `1.0`
- CUDNN disabled for stable eval runtime

## Execution

```bash
bash scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh a cuda:0
bash scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh b cuda:1
```

## Decision Rule

Strict promotion if any metric beats the active gate:

- center `<16.478984827655`
- P10 `>35.02295998845781`
- P5 `>11.978316681725639`

Priority:

1. P10 promotion with P5 not below `11.9`.
2. Any strict P10 promotion, even with center/P5 regression, because it proves trainable recovery works from XR-51/XR-50 event path.
3. Center or P5 promotion with bounded P10 loss, used only as a secondary leader.
4. No promotion: pivot to explicit teacher retraining or dual-teacher loss; do not repeat low-LR event-adapter recovery without new supervision.

## Results

Both lanes completed with train exit `0`. Both lanes early-stopped at epoch `7/8`. `best_metric_track_center_px` was not emitted and was skipped by the runner.

| lane/checkpoint | center | P10 | P5 | gate result |
|---|---:|---:|---:|---|
| XR-52A best-P10 | 16.483039610726 | 34.517007589340 | 11.625425529480 | no promotion |
| XR-52A best-P5 | 16.470460832119 | 34.667942953110 | 11.840136425836 | center promoted |
| XR-52B best-P10 | 16.472449232851 | 34.456633458819 | 12.017432342257 | P5 promoted |
| XR-52B best-P5 | 16.479750164918 | 34.427721875054 | 11.929422126498 | no promotion |

## Judgment

XR-52 did not recover P10, but it promoted center and P5.

- Center promoted to XR-52A best-P5: `16.470460832119`.
- P5 promoted to XR-52B best-P10: `12.017432342257`.
- P10 remains XR-39 `c25p45f30`: `35.02295998845781`.

New active gates:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Interpretation:

- Event-adapter continuation is still productive for center/P5.
- P10 remained below XR-51's best no-train P10 `34.710459961210`, so low-LR P10 teacher continuation weakened P10 rather than recovering it.
- Next P10 attempt should not repeat this low-LR teacher setup. Prefer a checkpoint-space follow-up using XR-52 center/P5 leaders plus XR-39 P10, or a stronger P10-specific teacher/model branch.
