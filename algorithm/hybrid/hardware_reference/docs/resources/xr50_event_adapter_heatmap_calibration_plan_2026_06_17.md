# XR-50 Event-Adapter Heatmap Calibration Plan

## Prompt Brief

- Goal: improve one active center/P10/P5 gate after XR-48 and XR-49 showed boundary-only heatmap head calibration is saturated.
- Inputs: XR-43 center leader, XR-39 P10 leader, XR-41A P5 leader, XR-48/XR-49 negative results, historical XR-15D/E and XR-16A adapter probes.
- Assumption: current heatmap-state head is the correct output mechanism; the next useful change is to let event-path representation adapt while keeping frame/backbone frozen.
- Constraint: no subject-39 training; full manifest1 train/val/test contract; promote only on strict active gate improvement.
- Expected output: executable XR-50 runner with two bounded lanes and explicit eval evidence.
- Acceptance criteria: static validation passes, trainable scope is exactly heatmap head plus event path, and final eval summaries are compared against current gates.

## Task Card

```yaml
task_card:
  task_id: T-XR50
  sub_agent: "codex-gpt5.5"
  role: "analyst"
  objective: "Review adapter-based next experiment after XR-48/XR-49 boundary-only failures."
  file_ownership: []
  assigned_skill: []
  inputs:
    - "scripts/external/run_xr15d_trackadapters_probe.sh"
    - "scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh"
    - "docs/track/PROGRESS.md"
    - "docs/resources/xr49_p5_boundary_heatmap_calibration_plan_2026_06_17.md"
  outputs:
    - "XR-50 event-adapter heatmap calibration recommendation"
  validation:
    - "Do not edit files"
    - "Cite prior result evidence"
  dependencies: []
```

## Active Gates

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

## Rationale

XR-48 tried explicit P10-boundary loss with heatmap head-only training and failed to promote any active gate. XR-49 reused the same boundary mechanism at a 5px margin for P5 and also failed. The best XR-49 P5, `11.70068063054766`, remained below the active P5 gate `11.868197652271816`.

Historical adapter probes are also negative, but they used older coordinate-head branches. XR-15D/E both-adapter variants drifted, and XR-16A event-only adapter remained below its then-current gates. Therefore XR-50 keeps the current heatmap-state head and only broadens the trainable scope from `track_center_heatmap_head.*` to:

- `track_center_heatmap_head.*`
- `event_adapter.*`
- `patch_frontend.event_embed.proj.*`

This is a mechanism change without reopening the full backbone or frame path.

## Ablation Matrix

| lane | init | teacher | objective | LR | epochs | best metric | center L2 | boundary margin | boundary band | boundary weight | state distill | trainable scope |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| XR-50A | XR-43 center | XR-41A P5 | P5 recovery | `1e-6` | 8 | P5 | `0.0045` | `5.0` | `2.0` | `0.008` | `0.00025` | heatmap head + event path |
| XR-50B | XR-43 center | XR-39 P10 | P10 recovery | `1e-6` | 8 | P10 | `0.0050` | `10.0` | `4.0` | `0.006` | `0.00025` | heatmap head + event path |

## Execution

```bash
bash scripts/external/run_xr50_event_adapter_heatmap_calibration.sh a cuda:0
bash scripts/external/run_xr50_event_adapter_heatmap_calibration.sh b cuda:1
```

## Decision Rule

Promote only if at least one strict gate improves:

- center `<16.491429926667895`
- P10 `>35.02295998845781`
- P5 `>11.868197652271816`

If XR-50 fails, do not continue simple adapter LR sweeps. The next larger branch should be teacher retraining or a paper-backed head/loss change beyond current heatmap-state calibration.

## Results

Execution date: 2026-06-17.

| lane/checkpoint | center | P10 | P5 | evidence |
|---|---:|---:|---:|---|
| XR-50A best-P10 | 16.483389932768684 | 34.3227048942021 | 11.933673824582781 | `runs/eval_fixed255k_xr50_a_xr50a_evadapt_p5_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0045_b0_008_state0_00025_best_track_p10_test_gpu0_w0_20260617_053611/eval/test/eval_summary.json` |
| XR-50A best-P5 | 16.482811435631344 | 34.48086814199175 | 11.863520765304566 | `runs/eval_fixed255k_xr50_a_xr50a_evadapt_p5_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0045_b0_008_state0_00025_best_track_p5_test_gpu0_w0_20260617_053921/eval/test/eval_summary.json` |
| XR-50B best-P10 | 16.483026616913932 | 34.394133479254585 | 11.978316681725639 | `runs/eval_fixed255k_xr50_b_xr50b_evadapt_p10_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0050_b0_006_state0_00025_best_track_p10_test_gpu1_w0_20260617_053613/eval/test/eval_summary.json` |
| XR-50B best-P5 | 16.482515714849743 | 34.48086814199175 | 11.81250035422189 | `runs/eval_fixed255k_xr50_b_xr50b_evadapt_p10_adamw_lr1e_6_g32_hm0_004_off0_0015_c0_0050_b0_006_state0_00025_best_track_p5_test_gpu1_w0_20260617_053922/eval/test/eval_summary.json` |

## Judgment

XR-50 promoted two active gates.

- Center gate promoted from XR-43 `16.491429926667895` to XR-50B best-P5 `16.482515714849743`.
- P5 gate promoted from XR-41A `11.868197652271816` to XR-50B best-P10 `11.978316681725639`.
- P10 gate remains XR-39 `35.02295998845781`.

New active gates after XR-50:

- Center: `<16.482515714849743`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

Decision: event-path adaptation is now validated as a useful mechanism change. The next experiment should try to recover P10 without losing the new XR-50 center/P5 gains, using XR-50B best-P10 or a no-train soup between XR-50B best-P10 and XR-39 P10.
