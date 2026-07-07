# HBTXR v3.0 Past Experiment Results

Date: 2026-06-12

Source root:

- `/home/kjm26/project/PRJXR/XR-VIT/HBTXR_v3_0/docs`

Current integration target:

- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`

## Purpose

This document imports the useful past experiment knowledge from `HBTXR_v3_0`
into the current `HGTXR/software` tracking surface. It is not a verbatim copy of
the legacy docs. It is a decision-oriented summary for current raw EV-Eye
event-count training.

## Source Inventory

Primary legacy sources reviewed:

- `docs/exps/20260329_022334_12_mode0_stage1_experiment_results_analysis.md`
- `docs/exps/20260401_042845_14_mode0_stage1_ciou_100ep_results_analysis.md`
- `docs/exps/20260402_010646_15_mode0_stage1_mask_guidance_ablation_analysis.md`
- `docs/exps/20260408_105839_16_mode0_stage1_expA_vs_expD_qualitative_analysis.md`
- `docs/exps/20260408_105839_17_mode0_stage1_expE_bbox_head_experiments.md`
- `docs/exps/20260401_042845_13_mode0_stage2_experiment_results_analysis.md`
- `docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md`
- `docs/exps/20260330_170419_15_all48_v2_baseline_failure_review.md`
- `docs/exps/20260330_170419_17_all48_sampled_dataset_construction_experiment.md`
- `docs/exps/20260330_170419_13_v3_real_ev_eye_mode2_optimizer_pool_cpu_pilot.md`
- `docs/exps/progress/20260408_100403_04_tsgss_dataset_refinement_and_guided_event_feature_progress.md`
- `docs/others/update/20260410_170000_timelens_xl_and_v2e_experiment_status_report.md`
- `docs/prj/20260408_214500_prj_start_here_operational_reading_order.md`

Legacy `docs/exps` plus `docs/exps/progress` contains 17 markdown files and
about 3710 lines of experiment notes. The highest-value evidence is concentrated
in Stage1, Stage2, Grounded-SAM label construction, TSGSS event-feature work,
and TimeLens/v2e status documents.

## Executive Summary

Past v3.0 experiments show the same failure pattern currently visible in
`HGTXR/software`:

1. Long training is not automatically better.
2. Stage2 often peaks at the first or early validation point.
3. `loss_total` can improve while `track_p10` or test center gets worse.
4. Spatial/data contracts dominate result quality.
5. Stable Stage2 requires preserving the useful Stage1 prior instead of letting
   event/track optimization destroy it.

The current 200/200 stage-split result in `HGTXR/software` is therefore
consistent with legacy evidence: Stage2 LR `1e-4` plus fresh Stage1 did not
reproduce the AdamW alpha-init leader, and the best useful checkpoint appeared
early.

## Legacy Stage1 Results

### Early Mode0 Stage1 Debugging

| Run / family | Key setup | Best / outcome | Current lesson |
| --- | --- | --- | --- |
| `mode0_stage1_20260328_213706` | full Stage1, `facet_square_direct`, distillation + pruning | best search P10 `29.1592` at epoch 58; center `15.9964` at epoch 55; later severe auxiliary/mask blow-up | Can reach useful search accuracy, but KD/pruning recipe was unsafe. |
| `mode0_stage1_20260328_235641` | full Stage1, no distillation/pruning | stable; best search P10 `14.6155`; center `24.1905` | Stability alone is insufficient if transform/target is weak. |
| `mode0_stage1_20260329_004755` | metadata-control rerun | epoch-1 metrics matched prior run within about `1e-6` | Manifest metadata rewrite alone did not change learning behavior. |
| `mode0_stage1_eye_only_letterbox30_20260329_013549` | `sensor_full_letterbox`, eye-only | best eye loss epoch 10; random-8 sensor IoU mean `0.6638` | `sensor_full_letterbox` made eye ROI a real localization task. |

Decision carried forward:

- Prefer full-sensor/aspect-preserving contracts.
- Do not rely on near-degenerate eye loss under square/crop contracts.
- Reintroduce distillation/pruning only after a stable baseline is proven.

### CIoU 100-Epoch Stable Stage1

Run:

- `mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411`

Key setup:

- `sensor_full_letterbox`
- dense eye + search + mask
- `search_xy_weight=0.25`
- `search_ciou_weight=0.25`
- `search_geo_weight=0.0`
- cosine scheduler
- early stopping disabled

Best validation:

| Metric | Best epoch | Value |
| --- | ---: | ---: |
| `metric_search_p10_pct` | 71 | `22.29` |
| `metric_search_p5_pct` | 55 | `6.42` |
| `metric_search_center_px` | 73 | `17.28` |
| `metric_eye_iou` | 39 | `0.7446` |
| `metric_eye_center_px` | 41 | `13.15` |

Comparison to previous tuned Stage1:

- search P10 improved from `11.46` to `22.29`
- search center improved from `27.41` to `17.28`
- eye IoU improved from `0.7281` to `0.7446`

Current lesson:

- Long Stage1 can help when the recipe is stable and the target contract is
  meaningful.
- The remaining Stage1 bottleneck was pupil `xy` and `ab` generalization, not
  eye/mask convergence.

### Mask-Guidance Ablation

Reference:

- `mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411`

Follow-up variants:

| Variant | Mechanism | Best P10 | Best P5 | Best center | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Baseline | dense eye + search + mask + CIoU | `22.29` | `6.42` | `17.28` | stable reference |
| Exp-A | hard `xy_from_mask_centroid` | `99.63` | `98.89` | `1.82` | practical Stage2 initializer |
| Exp-B | soft mask-center consistency | `27.25` | `8.86` | `16.58` | conservative improvement |
| Exp-C | soft center + mask-axis consistency | `20.26` | `5.76` | `19.85` | do not continue as-is |
| Exp-D | mask cascade with residual state | `98.75` | `88.98` | `3.02` | promising but `ab` unstable |

Stage2 initialization recommendation from legacy docs:

- Primary: Exp-A checkpoint
  `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109/train/best_search_p10.pt`
- Required contract: keep `model.search.xy_from_mask_centroid=true`
- Fallback: Exp-B for interpretability
- Do not default to Exp-D until `ab` / full ellipse quality is controlled.

Current lesson:

- Center shortcuts can produce excellent center metrics, but downstream Stage2
  must preserve the routing contract.
- Do not judge only center. Track `ab`, mask quality, P10/P5, and test transfer.

### Exp-E BBox Cascade

Purpose:

- Replace standard pupil-state head with two-stage bbox cascade.

Results:

| Variant | Eye IoU | Eye center | Pupil bbox IoU | Pupil bbox center | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| ROI Align cascade | `0.0` | `183.00` | `0.05995` | `32.07` | failed eye stage |
| No-ROIAlign cascade | `0.0` | `132.90` | `0.05906` | `32.03` | still failed eye stage |
| Dense-eye guided cascade | `0.7311` | `14.36` | `0.10013` | `27.18` | first usable variant |

Current lesson:

- Weak first-stage eye provider collapses the cascade.
- Dense eye fixes upstream ROI quality, but pupil bbox refinement remained weak.
- Keep bbox cascade exploratory; do not replace the mask-guided Stage1 path.

## Legacy Stage2 Results

### Stage2 Failure Pattern

Main legacy Stage2 sequence:

- `mode0_stage2_multigpu_fixcheck_20260331_210333`
- `mode0_stage2_singlegpu_geo0_sanity_20260331_210333`
- `mode0_stage2_multigpu_geo0_subset64_20260331_222906`
- `mode0_stage2_50ep_nw4_geo0_denseeye_20260331_223948`
- `mode0_stage2_from_expA_50ep_gpu0_nw4_20260402_023825`

Key findings:

1. Previous DataParallel crash was fixed.
2. Geometry loss terms dominated the objective when enabled.
3. Setting search/event/track geo weights to `0.0` made Stage2 numerically sane.
4. Even stable Stage2 did not improve tracking over the first validation point.
5. `loss_total` was misleading: it could fall while `track_p10` degraded.

### Representative Stage2 Runs

| Run | Setup | Best / outcome | Lesson |
| --- | --- | --- | --- |
| `mode0_stage2_multigpu_fixcheck_20260331_210333` | multi-GPU, geo on | epoch-1 loss `2795.5292`; geo terms dominated | crash fixed, recipe not useful |
| `mode0_stage2_singlegpu_geo0_sanity_20260331_210333` | single GPU, geo=0 | stable, much lower loss | geo=0 is stabilization step |
| `mode0_stage2_50ep_nw4_geo0_denseeye_20260331_223948` | multi-GPU, full manifests, geo=0 | stopped at epoch 15; best track P10 epoch 1 `9.1779` | long run did not help track |
| `mode0_stage2_from_expA_50ep_gpu0_nw4_20260402_023825` | Exp-A init, `sensor_full_letterbox`, geo=0 | completed 50 epochs; best track P10 epoch 1 `27.8998`, P5 `11.1905`; search collapsed by epoch 50 | Stage2 destroyed useful Stage1 search prior |

Current lesson:

- The current `HGTXR/software` 200/200 result repeats this old pattern.
- Stage2 should use short gates, early stopping, and best-checkpoint evals.
- If a Stage2 run peaks at epoch 1-5, more epochs are not a substitute for
  objective redesign.

## Label And Dataset Construction Results

### Grounded-SAM ROI-Crop Prompted Path

Main conclusion:

- Tuned `v2` operating point is the best legacy default for eye ROI + crop-local
  pupil prompting.

Important settings:

- stronger eye prompt
- strict prompted crop-local pupil stage
- `crop_mask_area_ratio_max=0.30`
- `crop_bbox_fill_max=0.85`
- fallback disabled by default

All-48 / 4-session v2 baseline:

| Metric | Value |
| --- | ---: |
| sampled frames | `1536` |
| completed | `1506` |
| eye_failed | `0` |
| pupil_failed | `30` |
| fail_like | `28` |
| csv_positive | `12` |
| no_raw_csv_supervision | `1524` |
| blink_labeled | `1143` |

Interpretation:

- Stage-1 eye ROI did not broadly collapse.
- Remaining failures were mostly conservative no-detection on blink/near-close
  frames or geometry violations.
- Some `pupil_failed` cases were hidden tiny-eye-ROI failures, so minimum eye ROI
  sanity gates are still useful.

### All-48 Sampled Dataset Construction

Scope:

- 48 users
- 4 official sessions
- 2 eyes
- 8 sampled frames per eye-session
- total sampled frames: `3072`

Aggregate:

| Metric | Value |
| --- | ---: |
| sessions | `384` |
| sampled frames | `3072` |
| eye_roi_present | `3071` |
| eye_roi_missing | `1` |
| pupil_geometry_complete | `16` |
| pupil_state_only | `2288` |
| pupil_related_missing | `768` |

CSV status:

- `csv_positive=16`
- `csv_region_zero=2288`
- `csv_file_missing=768`

Current lesson:

- Eye ROI coverage is almost solved for the sampled slice.
- Pupil geometry coverage is dominated by raw CSV supervision availability.
- Current software experiments should separate:
  - complete geometry rows
  - state-only rows
  - eye-only / missing-geometry rows
  - blink / close-eye policy rows

## Event / Interpolation / Mode2 Results

### TSGSS Event-Feature Progress

Sample construction:

- sessions: `388`
- sampled frames: `3104`
- 8 consecutive frames per session

Event alignment:

- intervals: `5432`
- target event count: `1000`
- raw event count mean: `738.2852`
- mode distribution:
  - `interpolated_dense=4532`
  - `subsampled=865`
  - `interp_single=5`
  - `empty_pad=30`

Critical finding:

- Raw `x/y/t` tuple interpolation can create radial or linear synthetic-trace
  artifacts.
- The project direction moved from raw tuple synthesis to derived event-feature
  interpolation.

Guided event-feature interpolation:

| Artifact | Observed feature | Interpolated feature | Key note |
| --- | ---: | ---: | --- |
| full-sensor guided feature | `2982` | `2450` | uses count maps and frame guidance |
| ROI-first guided feature | `2996` | `2436` | focuses event features around eye structure |

Current lesson:

- Current raw event-count training should not reintroduce raw tuple interpolation
  as a default.
- If interpolation returns, prefer derived features or ROI-first feature paths.

### Real EV-Eye Mode2 Optimizer-Pool CPU Pilot

Scope:

- real EV-Eye bounded subset
- 2 sessions
- target FPS `100`
- CPU-only
- Stage1/Stage2 optimizer sweep

Stage1 pilot:

- `metric_search_p10_pct=0.0` for all optimizers
- `adam_mini` had lowest `val.loss_total=423.8818`

Stage2 pilot:

- `metric_track_p10_pct=100.0` for 10/11 optimizers
- all candidates had `train_loss=NaN`
- `soap` collapsed to NaN validation

Current lesson:

- Treat this as a stability/infrastructure screen, not final optimizer ranking.
- Current CUDA-based AdamW result is stronger evidence than this old CPU pilot.

### TimeLens-XL And v2e

TimeLens-XL:

- export, native launcher, checkpoint generation, and reconnect path worked.
- real-mini result example:
  - `train PSNR=30.2466`
  - `val l1_loss=0.0025`
  - `val PSNR=40.4634`
  - `val SSIM=0.9960`
- quality was stable but did not clearly beat `linear_blend`.

v2e:

- backend connection worked.
- absolute timestamp issue was mitigated by relative timestamp normalization.
- threshold finding:
  - `0.50` under-generates
  - `0.45` can already over-generate
  - next useful range is around `0.47-0.49`

Current lesson:

- TimeLens/v2e are available research axes, but neither should displace the
  current raw event-count leader without downstream training evidence.

## Current HGTXR/Software Mapping

Current `HGTXR/software` leader before the manual 200/200 stage-split retry:

- run:
  `raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215`
- eval:
  `eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913`
- test center: `27.0897`
- test P10: `15.7109`
- test P5: `4.6301`

Manual 200/200 stage-split retry:

- Stage1:
  `raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452`
- Stage2:
  `raw_mode1_stage2_count255000_adamw_lr8e_6_200ep_2gpu_20260611_222452`
- actual Stage2 LR override: `1e-4`
- best-center test: center `37.0958`, P10 `11.1722`, P5 `3.4787`
- best-P10 test: center `41.9219`, P10 `11.2389`, P5 `3.6050`

Decision:

- Do not promote the 200/200 stage-split run.
- Keep the AdamW `8e-6` fixed255k alpha-init run as the current reference.
- Interpret the 200/200 failure as confirmation of the legacy Stage2 pattern:
  early peak, later plateau/degradation, and loss/epoch count not aligned with
  test quality.

## Actionable Experiment Rules

Use these rules for the next current-software experiments.

1. Use short Stage2 gates before long runs.
   - If best P10/center appears at epoch 1-5 and later epochs degrade, stop.
2. Promote only by test split, not validation alone.
   - Validation center and P10 have repeatedly failed to transfer cleanly.
3. Keep run names honest.
   - The 200/200 directory contained `lr8e_6`, but actual override used
     `STAGE2_LR=1e-4`. Future summaries must record actual resolved LR.
4. Preserve initialization contracts.
   - Alpha-init or mask-centroid-like assumptions must remain enabled in
     downstream configs.
5. Avoid raw tuple interpolation as default.
   - Legacy TSGSS showed artifacts; prefer raw fixed-count, derived features, or
     ROI-first guided features.
6. Treat local-crop / previous-pupil anchor paths as risky until target mismatch
   is diagnosed.
   - Current `prev_pupil_anchor` evals had center around `65`, indicating a
     distribution/target mismatch.
7. Use `center px`, P10, and P5 together.
   - Center-only and P10-only checkpoint choices can diverge.

## Recommended Next Experiments

High priority:

1. Warm-start from the AdamW fixed255k leader.
   - LR bracket: `2e-6`, `4e-6`, `6e-6`
   - epochs: `30-50`
   - early stopping: enabled
   - evaluate best-center and best-P10 on test.
2. Count/LR bracket around fixed255k.
   - counts: `250k`, `255k`, `260k`
   - optimizer: AdamW
   - LR: `6e-6`, `8e-6`, `1e-5`
3. Stage2 preservation probe.
   - freeze or partially freeze search/backbone parts
   - constrain drift from the loaded alpha-init leader
   - monitor search degradation separately from track metrics

Medium priority:

1. Conservative Stage1 rebuild.
   - full-sensor contract
   - no distillation/pruning initially
   - compare mask-centroid routing vs learned search path
2. Derived event-feature or ROI-first feature branch.
   - only after raw fixed-count count/LR plateau is confirmed
3. v2e threshold sweep.
   - `0.47`, `0.48`, `0.49`
   - require downstream train/eval, not just event-count similarity

Low priority / paused:

1. Fresh Stage1 200 epoch followed by high-LR Stage2.
2. Exp-E-style bbox cascade as a replacement path.
3. Raw tuple interpolation.
4. `prev_pupil_anchor` until coordinate/target contract mismatch is diagnosed.

## Verification Notes

This document was built by reading the legacy docs directly from
`/home/kjm26/project/PRJXR/XR-VIT/HBTXR_v3_0/docs` and cross-checking against
current `docs/track/PROGRESS.md` and `docs/track/RAW_EVENT_COUNT_TRAINING.md`.

Known limitation:

- Some legacy artifact paths point to ignored or machine-local workspaces. They
  are preserved as provenance, but not all images/JSON roots are guaranteed to
  exist in the current `HGTXR/software` checkout.
