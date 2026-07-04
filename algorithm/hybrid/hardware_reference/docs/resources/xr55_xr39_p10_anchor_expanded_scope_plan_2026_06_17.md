# XR-55 XR39 P10 Anchor Expanded-Scope Plan

## Prompt Pack

- Objective: build a stronger P10 branch after XR-53 showed no usable P10 near-tie.
- Source artifacts: XR-39 P10 leader, XR-52B P5 leader, XR-53 no-promotion result.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full train/val/test.
- Constraints: no subject-39 train leakage, full-test gate comparison, preserve heatmap-state evaluation path.
- Assumptions: P10 recovery now needs capacity beyond checkpoint-space recombination and low-LR event-adapter-only tuning.
- Required artifact: executable XR-55 runner and validation evidence.
- Verification evidence: `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, trainable filter count, train/eval exit status, full-test eval summaries.
- Risks and missing inputs: expanded last-block training can improve P10 while regressing center/P5; any P10-only promotion should be retained as a teacher branch, not necessarily balanced deployment leader.

## Active Gates

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

## Rationale

XR-51, XR-53, and XR-52 tested three lower-capacity P10 recovery paths:

- XR-51 no-train XR50/XR39 recombination improved center but not P10.
- XR-52 trainable low-LR event-adapter continuation improved center/P5 but not P10.
- XR-53 no-train XR52/XR39 recombination produced best P10 only `34.623300109591`, far below XR-39.

XR-55 pivots to an XR39-anchored model branch. It starts from the current P10 leader and expands trainable scope to the heatmap head, event path, and final backbone block. This tests whether a P10-capable branch can be updated without immediately destroying the P10 behavior.

## Ablation Matrix

| lane | init | teacher | LR | trainable scope | center guard | P10 boundary | state distill | intent |
|---|---|---|---:|---|---:|---:|---:|---|
| XR-55A | XR-39 P10 | XR-39 P10 | `2e-7` | head + event path + block 5 | `0.0025` | `0.012` | `0.00025` | self-teacher P10 branch stabilization |
| XR-55B | XR-39 P10 | XR-52B P5 | `3e-7` | head + event path + block 5 | `0.0030` | `0.010` | `0.00035` | P10 branch with weak P5/center pull |

Trainable include:

```json
[
  "track_center_heatmap_head.*",
  "event_adapter.*",
  "patch_frontend.event_embed.proj.*",
  "backbone.attn_stages.5.*",
  "backbone.mlp_stages.5.*",
  "backbone.norm.*"
]
```

Expected trainable size from XR-39 checkpoint keys:

- head/event/patch path: `14` tensors, `1504320` params.
- final attention block: `10` tensors, `148608` params.
- final MLP block: `6` tensors, `296256` params.
- final backbone norm: `2` tensors, `384` params.
- total expected: `32` tensors, about `1,949,568` params.

## Control Variables

- Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Train/val/test manifests: `data/_internal/manifests/manifest1`
- Event count: fixed `255000`, adaptive min/max `160000/384000`, `reference_us=4000003`, `scale_power=0.5`
- Heatmap-state head: grid `32`, heatmap as evaluated track state
- Optimizer: AdamW
- Boundary margin: P10 margin `10px`, band `4px`, temperature `1.0`
- CUDNN disabled for stable runtime

## Execution

```bash
bash scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh a cuda:0
bash scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh b cuda:1
```

## Decision Rule

Promote if any strict gate improves:

- center `<16.470460832119`
- P10 `>35.02295998845781`
- P5 `>12.017432342257`

Priority:

1. Any P10 `>35.02295998845781`. Keep as P10 teacher branch even if center/P5 regress.
2. P10 tie with better center than XR-39 and P5 above `11.5`.
3. Center/P5 promotion only if P10 remains above `34.7`.
4. No promotion: pivot to explicit teacher retraining or a dedicated P10 head/loss change rather than more checkpoint-space recombination.

## Results

Both lanes completed with train/eval exit `0`. `best_metric_track_center_px` was not emitted and was skipped by the runner.

| lane/checkpoint | center | P10 | P5 | gate result |
|---|---:|---:|---:|---|
| XR-55A best-P10 | 16.501659829276 | 34.399235514232 | 11.324830266408 | no promotion |
| XR-55A best-P5 | 16.497514723028 | 34.793368141992 | 11.382228217806 | no promotion; best XR-55 P10 |
| XR-55B best-P10 | 16.497369331973 | 34.567177718026 | 11.202381290708 | no promotion |
| XR-55B best-P5 | 16.500960135460 | 34.399235521044 | 11.312925515856 | no promotion |

## Judgment

XR-55 did not promote any active gate.

- Best XR-55 P10 was XR-55A best-P5: `34.793368141992`, still below XR-39 P10 `35.02295998845781`.
- Center and P5 both regressed substantially versus XR-52 active leaders.
- Expanded last-block adaptation did not preserve XR39 P10 behavior under the current loss/teacher setup.

Active gates remain:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Decision: close XR39-anchored expanded last-block tuning as no-promotion. Next branch should change supervision, not only trainable scope: candidate is explicit P10 teacher retraining or a dedicated P10 classification/calibration head that optimizes the 10px threshold more directly.
