# XR-53 XR52/XR39 P10 Recombination Plan

## Prompt Pack

- Objective: recover P10 after XR-52 promoted center and P5.
- Source artifacts: XR-52A best-P5 center leader, XR-52B best-P10 P5 leader, XR-39 P10 leader.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full test.
- Constraints: no training, no subject-39 leakage, strict full-test gate comparison.
- Assumptions: XR-52 leaders and XR-39 P10 have compatible checkpoint keys and can be interpolated/souped.
- Required artifact: executable XR-53 no-train runner plus eval evidence.
- Verification evidence: `bash -n`, lane A/B `DRY_RUN=1`, checkpoint existence, full-test eval summaries.
- Risks and missing inputs: XR-52 center/P5 gains may be too far from XR-39 P10; P10 may only recover by giving back the new P5 gate.

## Active Gates

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

## Rationale

XR-52 produced new center/P5 leaders but did not recover P10:

- Center leader: XR-52A best-P5, `16.470460832119/34.667942953110/11.840136425836`.
- P5 leader: XR-52B best-P10, `16.472449232851/34.456633458819/12.017432342257`.
- P10 leader remains XR-39 `c25p45f30`, `16.503940873486656/35.02295998845781/11.460459525244577`.

XR-53 is a low-cost no-train diagnostic. It tests whether the newly improved center/P5 manifold can be recombined with XR-39 P10 better than the XR-51/XR50 recombination.

## Ablation Matrix

| lane | mode | anchors | weights/alphas | intent |
|---|---|---|---|---|
| XR-53A | interpolation | XR-52A best-P5 -> XR-39 P10 | `0.03`, `0.06`, `0.10`, `0.15`, `0.20`, `0.30` | find smallest P10 injection after XR-52 center promotion |
| XR-53B | tri-soup | XR-52A best-P5, XR-52B best-P10, XR-39 P10 | `50/35/15`, `45/35/20`, `40/35/25`, `35/35/30`, `30/40/30`, `25/35/40` | preserve center/P5 while pulling toward P10 leader |

## Control Variables

- Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Test manifest: `data/_internal/manifests/manifest1/test_manifest.jsonl`
- Event count: fixed `255000`, adaptive min/max `160000/384000`, `reference_us=4000003`, `scale_power=0.5`
- Heatmap-state head: grid `32`, heatmap as evaluated track state
- Eval-only, no optimizer updates
- CUDNN disabled for stable eval runtime

## Execution

```bash
bash scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh a cuda:0
bash scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh b cuda:1
```

## Decision Rule

Promote if any strict gate improves:

- center `<16.470460832119`
- P10 `>35.02295998845781`
- P5 `>12.017432342257`

Priority:

1. Any strict P10 promotion.
2. P10 near-tie with center `<16.48` and P5 `>11.9`, used as a trainable XR-54 init.
3. Center/P5 promotion only if P10 loss is bounded.
4. No promotion: pivot away from checkpoint-space P10 recovery and plan a stronger teacher/model branch.

## Results

Both lanes completed with exit `0`.

| candidate | center | P10 | P5 | gate result |
|---|---:|---:|---:|---|
| XR-53A `a0p03` | 16.470736992359 | 34.623300095967 | 11.840136425836 | no promotion |
| XR-53A `a0p06` | 16.471054373469 | 34.512755877631 | 11.789116014753 | no promotion |
| XR-53A `a0p10` | 16.471541745322 | 34.512755877631 | 11.738095603670 | no promotion |
| XR-53A `a0p15` | 16.472256399904 | 34.623300109591 | 11.525510549545 | no promotion |
| XR-53A `a0p20` | 16.473095047474 | 34.578657252448 | 11.370323467255 | no promotion |
| XR-53A `a0p30` | 16.475181637491 | 34.616922562463 | 11.310799653190 | no promotion |
| XR-53B `c50f35p15` | 16.472414144448 | 34.512755877631 | 11.627551378523 | no promotion |
| XR-53B `c45f35p20` | 16.473200055531 | 34.512755877631 | 11.355442510332 | no promotion |
| XR-53B `c40f35p25` | 16.474105545453 | 34.608419152669 | 11.310799653190 | no promotion |
| XR-53B `c35f35p30` | 16.475128199373 | 34.497874947957 | 11.310799653190 | no promotion |
| XR-53B `c30f40p30` | 16.475147432940 | 34.497874947957 | 11.310799653190 | no promotion |
| XR-53B `c25f35p40` | 16.477522144999 | 34.497874947957 | 11.266156796047 | no promotion |

## Judgment

XR-53 did not promote any active gate.

- Best center: XR-53A `a0p03`, `16.470736992359`, still above the active center gate `16.470460832119`.
- Best P10: XR-53A `a0p15`, `34.623300109591`, far below the active P10 gate `35.02295998845781`.
- Best P5: XR-53A `a0p03`, `11.840136425836`, far below the active P5 gate `12.017432342257`.

Active gates remain:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Decision: close checkpoint-space recombination for P10 recovery around XR-52/XR39. There is no usable P10 near-tie for XR-54. Next branch should be XR-55: XR39-anchored P10 model branch with expanded trainable scope.
