# XR-49 P5-Boundary Heatmap Calibration Plan

## Prompt Pack

- Objective: improve the active P5 gate after XR-48 proved P10-boundary head-only calibration insufficient.
- Source artifacts: XR-41A P5 leader, XR-43 center leader, XR-48 negative result, configurable boundary loss in `track.py`.
- IDEA-Gen phase: P1.
- Domain: Research Workflow.
- Parent skill: paper-idea-generator.
- Target model/dataset: HGTXR mode1 stage2 raw event-count tracker on manifest1 full train/val/test.
- Constraints: no subject-39 train/val slice; promote only on strict center/P10/P5 gate improvement.
- Assumptions: `track_p10_boundary_loss` can be reused as a P5-boundary objective by setting margin `5.0`.
- Required artifact: executable XR-49 runner plus validation record.
- Verification evidence: `bash -n`, dry-run for both lanes, required checkpoint checks, leakage check, train/eval logs.
- Risks: lane A may improve P5 while worsening center/P10; lane B may preserve center but underfit P5.

## Task Card

```yaml
task_card:
  task_id: T-XR49
  sub_agent: "gpt-5.5"
  role: "expert"
  objective: "Review explicit P5-boundary calibration after XR-48 failed."
  file_ownership: []
  assigned_skill: ["cv-dl-expert", "ablation-study-designer", "caveman"]
  inputs:
    - "docs/Validation.md"
    - "src/hbtxr/loss/bundles/track.py"
    - "scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh"
  outputs:
    - "XR-49 P5-boundary calibration recommendation"
  validation:
    - "No subject-39 train leakage"
    - "Strict gate comparison"
  dependencies: []
```

## Active Gates

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

## Ablation Matrix

| lane | init | teacher | LR | epochs | best metric | center L2 | boundary margin | boundary band | boundary weight | state distill | intent |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| XR-49A | XR-41A P5 | XR-41A P5 | `2e-6` | 8 | P5 | `0.0015` | `5.0` | `2.0` | `0.012` | `0.00050` | polish current P5 leader |
| XR-49B | XR-43 center | XR-41A P5 | `2e-6` | 8 | P5 | `0.0045` | `5.0` | `2.0` | `0.008` | `0.00025` | inject P5 pressure with center guard |

## Control Variables

- Full train/val/test manifests from `data/_internal/manifests/manifest1`
- Event count target `255000`, adaptive min/max `160000/384000`
- Heatmap-state head, grid `32`, heatmap as evaluated track state
- Trainable scope inherited from heatmap head-only config: `track_center_heatmap_head.*`
- Optimizer: AdamW

## Execution

```bash
bash scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh a cuda:0
bash scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh b cuda:1
```

## Decision Rule

Promote only if at least one strict gate improves:

- center `<16.491429926667895`
- P10 `>35.02295998845781`
- P5 `>11.868197652271816`

If neither lane promotes, stop boundary-only head calibration and move to non-head-only adapter update or teacher retraining.

## Results

Execution date: 2026-06-17.

Active gates before XR-49:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 | evidence |
|---|---:|---:|---:|---|
| XR-49A best-P10 | 16.52969342981066 | 34.083759260177615 | 11.41369082587106 | `runs/eval_fixed255k_xr49_a_manual_cudnnoff_best_track_p10_test_gpu0_w0_20260617_050821/eval/test/eval_summary.json` |
| XR-49A best-P5 | 16.532351425715856 | 33.93452455656869 | 11.70068063054766 | `runs/eval_fixed255k_xr49_a_manual_cudnnoff_best_track_p5_test_gpu0_w0_20260617_051400/eval/test/eval_summary.json` |
| XR-49B best-P10 | 16.496637644086565 | 34.431973586763654 | 11.417942530768258 | `runs/eval_fixed255k_xr49_b_xr49b_xr43_p5t_p5b_cg_adamw_lr2e_6_g32_hm0_004_off0_0015_c0_0045_p5b0_008_state0_00025_best_track_p10_test_gpu1_w0_20260617_050852/eval/test/eval_summary.json` |
| XR-49B best-P5 | 16.497121804101127 | 34.444728687831336 | 11.54124187060765 | `runs/eval_fixed255k_xr49_b_xr49b_xr43_p5t_p5b_cg_adamw_lr2e_6_g32_hm0_004_off0_0015_c0_0045_p5b0_008_state0_00025_best_track_p5_test_gpu1_w0_20260617_051153/eval/test/eval_summary.json` |

## Runtime Notes

- XR-49B first attempted with the original long experiment name and failed before training with `OSError: [Errno 36] File name too long`.
- The runner was shortened and XR-49B state distillation was set to `0.00025` before the successful B run.
- XR-49A used the pre-shortening experiment name. Training completed and produced checkpoints, but the shell wrapper hit a post-train eval-loop syntax error. Manual eval was then run with `HBTXR_DISABLE_CUDNN=1`.
- A separate manual eval without `HBTXR_DISABLE_CUDNN=1` failed with a cuDNN sublibrary version mismatch, so the CUDNN-off manual eval summaries are the accepted XR-49A evidence.

## Judgment

XR-49 did not promote any active gate.

The best XR-49 P5 was XR-49A best-P5 at `11.70068063054766`, below the active XR-41A P5 gate `11.868197652271816`. XR-49B preserved center better than XR-49A, but still missed the active center gate and did not recover P10/P5.

Decision: close boundary-only heatmap head calibration for now. The next experiment should change trainable mechanism, preferably a non-head-only lightweight adapter or track-event-adapter update with explicit center preservation.
