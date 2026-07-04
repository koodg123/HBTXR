# XR-42 P5-Preserve Low-Drift Plan

Date: 2026-06-17

## Prompt Pack

- Goal: preserve XR-39 center/P10 leaders while using XR-41A best-P5 as a bounded P5 teacher/reference.
- Task family: event-based pupil tracking, Stage2 track-center heatmap head-only adaptation.
- Source evidence: XR-39 promoted center/P10; XR-40 no-train soup failed; XR-41A promoted P5 but regressed center/P10.
- Success gate: promote at least one active metric without undocumented contract change.
- Active gates before launch:
  - center `<16.491779099191938`
  - P10 `>35.02295998845781`
  - P5 `>11.868197652271816`
- Constraints: fixed-count `255000`, adaptive count `160000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap-state track head, head-only trainable scope.

## Task Card

```yaml
task_card:
  task_id: T-095
  sub_agent: "gpt5.5 evaluator + main codex implementer"
  role: "experiment implementer / evaluator"
  objective: "Run low-drift P5-preserve training from XR-39 center/P10 leaders using XR-41A best-P5 as teacher/reference."
  file_ownership:
    - "scripts/external/run_xr42_p5_preserve_lowdrift.sh"
    - "docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md"
    - "docs/Master-Plan.md"
    - "docs/Paper-Backed-Experiment-Plan.md"
    - "docs/Spec.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/CHANGELOG.md"
    - "docs/track/log.md"
  assigned_skill:
    - "computer-vision-expert"
    - "ablation-study-designer"
    - "caveman"
  inputs:
    - "runs/interpolated_checkpoints/xr39_mixedleader_soup_c60p25f15.pt"
    - "runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt"
    - "runs/raw_mode1_stage2_count255000_adamw_lr1e_5_weakdistill_trackonly_heatmapstate_g32_hm0_004_off0_0015_c0_0015_xr41a_p5init_xr38b_bestp5_xr39p10_ref_lossratio_lr1e5_p5select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_013052/train/best_track_p5.pt"
  outputs:
    - "XR-42 train/eval logs"
    - "best_track_p10.pt / best_track_p5.pt / optional best_metric_track_center_px.pt eval summaries"
  validation:
    - "bash -n runner"
    - "checkpoint existence checks"
    - "raw event-count contract check through run_prepare_and_train.sh"
    - "test split eval summaries against active gates"
  dependencies:
    - "XR-39"
    - "XR-41"
```

## Ablation Matrix

| Lane | Init | Teacher/reference | LR | Best metric | Loss ratio | Hypothesis |
|---|---|---|---:|---|---|---|
| XR-42A | XR-39 center `c60p25f15` | XR-41A best-P5 | `3e-6` | `metric_track_center_px` | heatmap/off/center `0.0035/0.00125/0.0020` | retain center gate while absorbing small P5 correction |
| XR-42B | XR-39 P10 `c25p45f30` | XR-41A best-P5 | `3e-6` | `metric_track_p10_pct` | heatmap/off/center `0.0035/0.00125/0.0020` | retain P10 gate while testing whether P5 teacher transfers without XR-41B drift |

Control variables:

- Same raw event-count support-adaptive contract as XR-39/XR-41.
- Same heatmap-state track center representation.
- Same head-only trainable filter: `track_center_heatmap_head.*`.
- Same AdamW optimizer family.
- Same test split and canonical root.

Rejected alternatives:

- More no-train soup: XR-40 already failed to recover P5 with XR-39 center/P10 + P5 anchor.
- Higher LR continuation: XR-41A improved P5 but regressed center/P10; XR-41B did not recover gates.
- Full trunk training: high drift risk and not justified before exhausting head-only low-drift branch.

## Execution

Commands:

```bash
bash scripts/external/run_xr42_p5_preserve_lowdrift.sh a cuda:0
bash scripts/external/run_xr42_p5_preserve_lowdrift.sh b cuda:1
```

Expected evidence:

- `XR42_EXIT:0` in both logs.
- `eval_summary.json` for best-P10 and best-P5, plus best-center for XR-42A.
- Promotion decision against current active gates.

## Initial Status

- Runner created: `scripts/external/run_xr42_p5_preserve_lowdrift.sh`.
- Static validation passed: `bash -n`, required checkpoint existence checks, and `git diff --check`.
- GPT5.5 read-only evaluator reviewed XR42 and agreed with XR39-anchored trainable repair as the primary path.
- Evaluator risk note: because `distillation.state_similarity=false` and `state_weight=0.0`, the teacher/reference checkpoint may not provide a strong explicit anchor. Treat XR-42A/B as low-drift fine-tuning; if they fail, design XR-42C with deliberate tiny distillation or regularization instead of assuming teacher pull.
- Launch status: XR-42A on GPU0 and XR-42B on GPU1 passed startup validation and entered epoch `1/10`.

## Closeout

- XR-42A and XR-42B both completed with train/eval exit `0`.
- Both lanes early-stopped at epoch `7/10`.
- XR-42A best-P10: `16.49802110535758 / 34.48299399103437 / 11.366922119685581`.
- XR-42A best-P5: `16.498888087272643 / 34.30739874839783 / 11.347789451054163`.
- XR-42A best-center: `16.498888087272643 / 34.30739874839783 / 11.347789451054163`.
- XR-42B best-P10: `16.505801352432798 / 34.3171777180263 / 11.278911903926305`.
- XR-42B best-P5: `16.508924693720683 / 34.29506881577628 / 11.323554761069161`.
- Decision: no center/P10/P5 gate promoted.
- Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Next P0: if continuing this line, use XR-42C only as an explicit tiny-distillation/regularization fallback; XR-42A/B show teacher-as-reference without explicit anchor is insufficient.
