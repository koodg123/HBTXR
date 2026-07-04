# XR-15 Support-Adaptive Event-Window Plan

Date: 2026-06-16

## Prompt Pack

Objective: improve HGTXR second-goal accuracy beyond XR-06C by changing event evidence density while preserving the proven track-only weak-distill/direct-aux contract.

Source artifacts:

- `docs/Paper-Backed-Experiment-Plan.md`
- `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`
- `anlaysis/xr-eye-tracking/experiment_integration.md`
- `runs/diagnostics/xr14a_*_state_failure_buckets_20260616.json`

IDEA-Gen phase: P1

Domain: Research Workflow

Parent skill: `paper-idea-generator`

Target model, algorithm, hardware, or dataset: HGTXR mode1 Stage2 fixed255k track-only raw event-count training on EV-Eye canonical1/manifest1.

Constraints:

- Active gates after XR-15B: center `<20.215672533852715`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Do not repeat pure low-sim loss weighting, hard subset training, sampler-only balancing, SimDR-only replay, or XR-14A search/event fallback.
- Preserve XR-06C init/teacher, weak distillation, AdamW `5e-7`, direct track-state auxiliary weights `0.0005/0.0125/0.005`, full width, and track-only inference.

Assumptions:

- Low `similarity_target`, `session_201`, and subject `42/45/39` failures are partly caused by weak or stale event evidence, not only label noise.
- Support-adaptive fixed-count windows can change event evidence without drifting into a new model family.
- The raw fixed-count manifest contract remains valid when runtime overrides enable adaptive event count.

Required artifact:

- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`
- Runner: `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`
- Full-test eval summaries and failure buckets after execution.

Verification evidence:

- `bash -n` runner syntax.
- raw event-count contract pass on base config.
- dataset smoke showing `adaptive_count_resolved` changes selected target count.
- train/eval exits `0`.
- comparison against active XR-06C gates.
- post-eval low-sim/session/subject bucket summary.

Risks and missing inputs:

- Adaptive count may collapse to near-fixed255k for most rows.
- Larger windows can preserve stale events and hurt P5.
- Low-sim rows may reflect annotation or subject-specific domain shift rather than event support.

## Ablation Matrix

| ID | Change | Control Variables | Expected Evidence | Promotion Signal |
|---|---|---|---|---|
| XR-15A | support-adaptive fixed count, min `192k`, base `255k`, max `320k`, reference `4000003us`, power `0.5` | XR-06C init/teacher, weak distill, direct aux, AdamW `5e-7`, track-only | eval center/P10/P5, low-sim/session/subject buckets | completed; center promoted to `20.225680075372967` |
| XR-15B | narrower support-adaptive range, min `224k`, base `255k`, max `288k` | same as XR-15A | checks whether broad range over-regularizes | lower center without P10/P5 regression |
| XR-15C | wider support-adaptive range, min `160k`, base `255k`, max `384k` | same as XR-15A | checks whether high-gap rows need more support | completed; center promoted to `20.19088832650866`, P10/P5 did not promote |

## XR-15B Closeout

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center_xr15b | best-center | 20.215672533852715 | 26.575255823135375 | 8.627551317214966 | new center leader |
| p10_xr15b | best-center | 20.21979672227587 | 26.324405458995273 | 8.28741525241307 | center promotion vs XR-15A, not leader |

Artifact gaps:

- Center-lane best-P10 and P10-lane best-P10 eval runs created `hypers/*` only and did not produce `eval/test/eval_summary.json` before the XR-15B tmux sessions were cleaned up.
- XR-15B diagnostic generation for the center-lane best-center checkpoint was interrupted after a long no-output run; retry later only if XR-15C does not resolve P10/P5.

Decision:

- XR-15B validates that a narrower adaptive range preserves and slightly improves center accuracy.
- P10/P5 still do not beat XR-06C/XR-05A gates.
- Continue to XR-15C wider support range before leaving the event-support branch.

## XR-15C Launch

- Center lane started in tmux session `hgtxr_xr15c_supportadaptive_center_gpu0_20260616`; log `runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819`.
- P10 lane started in tmux session `hgtxr_xr15c_supportadaptive_p10_gpu1_20260616`; log `runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816`.
- Both lanes passed startup validation: raw event-count contract, intended checkpoint load, resolved CUDA device, train/val counts `5929/844`, and epoch `1/12` train steps.

## XR-15C Closeout

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center_xr15c | best-center | 20.19088832650866 | 26.009779623576573 | 8.436224787575858 | new center leader |
| p10_xr15c | best-center | 20.205999997683932 | 26.303146975381033 | 8.397959463936942 | center promotion vs XR-15B, not leader |

Decision:

- XR-15C validates that wider support can further reduce center error, but it loses P10/P5 compared with XR-15B and does not beat the XR-06C/XR-05A P10/P5 gates.
- Active gates are now center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- The event-support range branch should pause here. Next 2차 목표 experiment should move to XR-15D bounded track-adapter/coordinate-head training.

## XR-15A Closeout

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center | best-center | 20.230558776855467 | 26.303146975381033 | 8.573554713385446 | center improved vs XR-06C, not leader |
| center | best-P10 | 20.259641446386066 | 25.812925890513828 | 8.451105751310076 | center improved vs XR-06C, not leader |
| p10 | best-center | 20.225680075372967 | 26.50085105895996 | 8.304422058377947 | new center leader |
| p10 | best-P10 | 20.259886418070113 | 26.12457557405744 | 8.357993486949375 | center improved vs XR-06C, not leader |

Diagnostics:

- `runs/diagnostics/xr15_p10_bestcenter_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr15_center_bestcenter_track_state_failure_buckets_20260616.json`

Decision:

- XR-15A validates adaptive event support as a center-error improvement axis.
- P10/P5 did not promote.
- Low-similarity/session_201/subject failure structure remains.
- Best-P5 helper eval did not produce summary artifacts under sandbox/logged timeout attempts; keep as an artifact gap.

## XR-15B Recommended Launch

Center lane:

```bash
bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh \
  255000 224000 288000 4000003 0.5 5e-7 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205/train/best_metric_track_center_px.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205/train/best_metric_track_center_px.pt \
  center_xr15b
```

P10 lane:

```bash
bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh \
  255000 224000 288000 4000003 0.5 5e-7 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  p10_xr15b
```

Launch status:

- Center lane launched in tmux session `hgtxr_xr15b_supportadaptive_center_gpu0_20260616`; log `runs/_logs/xr15b_supportadaptive_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512`.
- P10 lane launched in tmux session `hgtxr_xr15b_supportadaptive_p10_gpu1_20260616`; log `runs/_logs/xr15b_supportadaptive_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111515`.
- Both lanes passed startup validation: raw event-count contract, intended checkpoint load, resolved CUDA device, train/val `5929/844`, and epoch `1/12` train steps.

## Recommended Launch

Center-preserve lane:

```bash
bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh \
  255000 192000 320000 4000003 0.5 5e-7 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
  center
```

P10-preserve lane:

```bash
bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh \
  255000 192000 320000 4000003 0.5 5e-7 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  p10
```

## Rejected Alternatives

- Low-sim hard subset plus SimDR head-only: rejected for next P0 because hard-subset training and SimDR variants have already failed promotion.
- Search/event calibrated fallback from XR-14A: rejected because `search_state`/`event_state` scored P10/P5 `0.0`.
- Previous-state/blend fallback: rejected because it worsened center error.
