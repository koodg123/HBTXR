# Stage1 Frame-Search Escalation Plan - 2026-06-22

## Purpose

Define the next Stage1 frame-based Search experiments after the active promoted-polish bracket reaches the 50-epoch gate. This plan keeps the current GPU training intact and prevents launching duplicate or stale capacity branches before the gate is reached.

## Current Authority

Active Stage1 baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Search P10: `28.27717937613433`
- Search P5: `10.153863744915656`
- Search center: `17.257583906065744`

Closed promoted-polish bracket:

- Lane C:
  `runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916`
- Lane D:
  `runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916`
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430_report.txt`
- Gate result: no promotion. Both lanes reached the minimum-epoch gate and stayed below the active baseline by more than `0.2 pp` P10 with worse center.

Final direct reporter snapshot before stop:

- Lane D: epoch `53/120`, P10 `27.788634912023003`, P5 `9.55076399389303`, center `17.328227780899912`, delta P10 `-0.4885444641113281`.
- Lane C: epoch `55/120`, P10 `27.423630372533257`, P5 `9.480009258918042`, center `17.351748246066975`, delta P10 `-0.8535490036010742`.
- State: `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.
- Action: stopped Lane C/D PIDs `2526793` and `2526794`, plus watcher PID `2556516`, to free both GPUs for the next branch.

Closed S1-K/S1-L follow-up bracket:

- Lane S1-K:
  `runs/NON_XR/raw/stage1_s1k_geometry_nodistill_lr1e6_xy2_ab0p75_trig1p25_20260622_111904`
- Lane S1-L:
  `runs/NON_XR/raw/stage1_s1l_low_lr_ema_selfdistill_lr7p5e7_xy2_ema9997_20260622_111904`
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1k_s1l_min50_watch_20260622_host.log`
- Watcher PID: `2758111`
- Gate: completed. Both lanes reached at least `50` epochs.
- Startup validation:
  - Both lanes loaded the baseline checkpoint with `loaded_count=142 partial=0 skipped=0`.
  - Both lanes entered training and reached epoch `6/120`.
  - Host GPU check showed both RTX 5080 devices active with low but nonzero training utilization.

Final gate reporter snapshot:

- S1-K: epoch `53/120`, P10 `27.693172202920014`, P5 `9.173405485333136`, center `17.315994528104675`, delta P10 `-0.5840071732143173`.
- S1-L: epoch `51/120`, P10 `27.670710329739553`, P5 `9.227313833416632`, center `17.30932722451552`, delta P10 `-0.6064690463947784`.
- State: `keep_baseline_unless_later_improves`; no promotion.
- Gate decision helper:
  `scripts/external/decide_stage1_s1kl_gate_next.py`
- Final helper output: `action=stop_s1kl_and_run_s1m_mask_probe`, `all_min_ready=True`, `promote_checkpoint=None`.
- Action: stopped S1-K PID `2728912`, S1-L PID `2728913`, and watcher PID `2758111`.

Active S1-M/S1-N follow-up bracket:

- S1-M probe completed:
  `runs/NON_XR/raw/stage1_s1m_mask_eye_active_all_probe_stage1_s1m_mask_probe_after_s1kl_20260622_115634`
- S1-M probe result:
  - `loss_eye`, `loss_mask`, and `loss_search_*` were nonzero.
  - Best P10 matched baseline at `28.27717937613433`.
  - Final epoch P10 was `27.90655940433718`.
  - Conclusion: mask/eye path works, but `eye_weight=1.0` dominates the Search objective.
- Active S1-MA:
  `runs/NON_XR/raw/stage1_s1ma_loweye_maskassist_50ep_after_s1kl_20260622_120007`
  - PID `2969847`, GPU0.
  - `eye_weight=0.02`, `mask_weight=0.05`, `max_train_batches=256`, `max_val_batches=106`, `epochs=50`.
  - Best P10 `28.27717937613433`; latest epoch `12/50` P10 `27.575247674618126`, P5 `9.247529443704858`, center `17.35772168861245`.
- Active S1-MB:
  `runs/NON_XR/raw/stage1_s1mb_maskonly_searchpreserve_50ep_after_s1n_20260622_120546`
  - PID `3012910`, GPU1.
  - `eye_weight=0.0`, `mask_weight=0.025`, `search_xy_weight=2.25`, `search_geo_weight=0.25`, LR `3e-7`, `epochs=50`.
  - Startup validation: `loaded_count=126 partial=0 skipped=0`.
  - Early epoch `2/50` P10 `28.294025727038115`, P5 `8.87690948990156`, center `17.308569012947803`.
  - This is an early P10 improvement over baseline but not promotable before epoch `50`, especially with center worse than baseline.
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1ma_s1mb_min50_watch_20260622.log`
  - PID `3026345`
  - Poll interval `180` seconds.
- Closed S1-N lane H:
  `runs/NON_XR/raw/stage1_s1n_reduced224_d6_teacher_distill_lr5e7_stage1_s1n_reduced224_teacher_laneh_50ep_after_s1m_20260622_120110`
  - PID `2978876`, GPU1, stopped.
  - Reduced capacity `224x6`, weak teacher distillation, LR `5e-7`.
  - Startup warning: `loaded_count=7 partial=0 skipped=135`.
  - Early metrics failed with P10 `0.0`, P5 `0.0`, center around `173 px`.
  - Decision: architecture/warm-start mismatch; do not continue this branch.

## Evidence From Paper And Codebase Analysis

DistillGaze-style training strategy:

- Relevant analysis: `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- Useful idea: adapt a stronger teacher and distill into the deployable model using representation guidance and EMA/self-training consistency.
- HGTXR implication: self-distillation and teacher-guided Stage1 are valid, but only if the teacher does not overwrite the current frame Search anchor.

RITnet / mask-teacher strategy:

- Relevant analysis: `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- Useful idea: compact segmentation teacher with boundary-aware losses can improve pupil/iris masks.
- HGTXR implication: use mask or ellipse pseudo-labels only after label-quality checks. Current `active_head=search` zeroes mask/eye losses, so pseudo-mask training requires a separate mask-enabled Stage1 branch.

PAPER_REF Stage1 teacher route:

- Relevant analysis: `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`
- Useful idea: Stage1 frame Search teacher should keep robust frame anchors, then Stage2 hybrid student can initialize from it.
- HGTXR implication: do not run blind long reruns; improve the teacher with lower LR, geometry losses, and explicit promotion gates.

Negative evidence:

- Prior large capacity branch with `embed_dim=256`, depth `8`, LR `1e-4` to `2e-4` collapsed:
  - P10 only `3.8704581961247593` to `9.044234036269346`.
  - Center `28.80148663001038` to `44.76474802188964`.
- Therefore, any large/teacher branch must use a smaller capacity jump or much lower LR, and must be gated before full training.

## Gate Logic

At the 50-epoch gate:

1. Promote immediately only if a lane has:
   - P10 `> 28.27717937613433`
   - center `<= 17.257583906065744`
   - `best_search_p10.pt` exists
2. If P10 improves but center regresses, mark as diagnostic only and do not replace the baseline.
3. If neither lane improves P10 but one preserves center and improves P5 materially, continue that lane to `120` epochs but do not promote yet.
4. If both lanes remain below baseline by more than `0.2 pp` P10 and center is worse, stop opening additional polish variants and move to the next escalation branch.

## Accuracy Plateau Diagnosis - 2026-06-22 20:10 KST

The current Stage1 issue is not that Search P10 never improves. S1-MT improved P10 from the strict baseline `28.27717937613433` to `28.482705062290407`, a `+0.2055256861560757 pp` gain. The blocker is that the same checkpoint regressed center from `17.257583906065744` to `17.358525235697908` and P5 from `10.153863744915656` to `9.32389961098725`.

Current failure mode:

- P10-oriented fine-tuning moves more samples inside the 10 px radius.
- Fine localization still degrades, so P5 and mean center error fail the strict baseline gate.
- Weighted sampler branches did not fix this; they mainly reduced P5 and kept center slightly worse.
- Eye-head variants alone are not an immediate lever for the active Stage1 Search setup because the active runner disables `model.heads.eye` and uses `active=search`.

Therefore the next experiments must target the actual Stage1 Search path:

- `PupilSearchHead` capacity and residual correction.
- Search bbox/OBB auxiliary losses.
- Mask-derived center guidance with low weight.
- Conservative depth/capacity probes with strict center-best checkpointing.

Promotion remains strict:

- P10 must exceed `28.27717937613433`.
- Center must be `<= 17.257583906065744`.
- P5 must be tracked and should recover toward `10.153863744915656`.
- Minimum gate is `50/50` epochs before promotion.

## Structure Ablation Queue - 2026-06-22 20:10 KST

Implemented code/config support:

- `src/hbtxr/models/heads.py`: `PupilSearchHead` now supports `legacy`, `residual_mlp`, and `deep_residual_mlp`.
- `src/hbtxr/models/tracker/head_factory.py`: passes the search head variant into the search head.
- `src/hbtxr/models/hybrid_tracker.py`: exposes search head variant and residual hidden size.
- `src/hbtxr/training/model_factory.py`: reads `model.heads.search_variant` and `model.heads.search_residual_hidden_dim`.
- `configs/external/base.yaml`: documents the new config surface.
- `scripts/external/run_stage1_frame_search_ensemble_teacher.sh`: accepts lane-specific extra overrides.
- `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`: launches two-lane architecture ablations.
- `scripts/external/run_stage1_frame_search_arch_ablation_when_gpu_free.sh`: waits for current GPU jobs to finish, then launches architecture ablations.

Experiment queue:

1. S1-MU/S1-MV center-direct refinement from S1-MT.
   - Status: active on GPU0/GPU1.
   - Purpose: recover center/P5 while preserving the high-P10 S1-MT seed.
2. S1-MW residual Search Head.
   - `model.heads.search_variant=residual_mlp`
   - `model.heads.search_residual_hidden_dim=192`
   - Purpose: add zero-init residual correction without changing initial checkpoint behavior.
3. S1-MX residual Search Head + search bbox/OBB aux.
   - S1-MW settings plus `search_bbox_aux=true`, `search_obb_aux=true`.
   - Aux loss weights: `search_bbox_aux_weight=0.0005`, `search_obb_aux_weight=0.0005`, `search_obb_aux_angle_weight=0.05`.
   - Purpose: test whether pupil geometry auxiliary supervision improves P5/center.
4. S1-MY depth-8 center-best probe.
   - `model.depth=8`
   - `distillation.enabled=false`
   - `training.best_metric_name=metric_search_center_px`
   - Purpose: test conservative extra depth while explicitly preserving the center-best checkpoint.
5. S1-MZ residual Search Head + low-weight mask cascade.
   - `search_variant=residual_mlp`
   - `model.heads.mask=true`
   - `model.search.mask_cascade.enabled=true`
   - `use_xy=true`, `use_ab=false`, `xy_residual_scale=0.25`.
   - Purpose: use mask-derived center guidance without letting mask training dominate the Search objective.

Execution commands:

```bash
# Dry-run the first structure suite.
DRY_RUN=1 RUN_TAG=stage1_arch_ablation_dryrun \
  bash scripts/external/run_stage1_frame_search_arch_ablation_queue.sh

# Dry-run the depth/mask suite.
DRY_RUN=1 SUITE=depth_mask RUN_TAG=stage1_arch_depth_mask_dryrun \
  bash scripts/external/run_stage1_frame_search_arch_ablation_queue.sh

# Wait for current GPU jobs, then run the first suite.
WAIT_PIDS="2368700 2368701" SUITE=head_aux RUN_TAG=stage1_arch_after_s1mu_s1mv_20260622 \
  bash scripts/external/run_stage1_frame_search_arch_ablation_when_gpu_free.sh
```

Validation completed:

- `python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py`
- `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh`
- `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`
- `bash -n scripts/external/run_stage1_frame_search_arch_ablation_when_gpu_free.sh`
- Dry-run command materialization for both `head_aux` and `depth_mask` suites.

## Latest Result Addendum - 2026-06-22 16:13 KST

No-train checkpoint interpolation did not produce a strict replacement baseline.

- Eval summary:
  `runs/NON_XR/shared/eval/stage1_baseline_s1mj_interp_stage1_baseline_s1mj_interp_20260622_155014/summary.json`
- Same-evaluator current baseline `best_search_p10.pt`:
  - P10 `28.27717937613433`
  - P5 `8.334456677706736`
  - center `17.292674766396576`
- S1-MJ / alpha `1.0`:
  - P10 `28.482705062290407`
  - P5 `9.088050554383475`
  - center `17.419374744847136`
- Decision:
  - S1-MJ improves P10/P5 but regresses center by `0.12669997845056002 px`.
  - Midpoint interpolation does not preserve the P10 gain.
  - S1-MJ is a seed, not a strict baseline replacement.

Checkpoint variant matrix confirms that no single checkpoint dominates all Stage1 Search metrics.

- Eval summary:
  `runs/NON_XR/shared/eval/stage1_ckpt_matrix_20260622_160523/summary.json`
- `baseline_best_p5`: P10 `27.623540590394217`, P5 `10.153863744915656`, center `17.398822599986815`.
- `baseline_best_center`: P10 `27.294475033598125`, P5 `9.241913993403596`, center `17.25758159385537`.
- `s1mj_best_p5`: P10 `28.190701412704755`, P5 `9.205975082685363`, center `17.40067045193798`.

Active branch:

- S1-MM:
  `runs/NON_XR/raw/stage1_s1mm_s1mj_p10seed_baselinep10teacher_lr1e7_20260622_161345`
  - Start from S1-MJ `best_search_p10.pt`.
  - Teacher: current baseline `best_search_p10.pt`.
  - Goal: preserve S1-MJ P10/P5 gain while pulling toward the current P10 baseline distribution.
- S1-MN:
  `runs/NON_XR/raw/stage1_s1mn_s1mj_p10seed_baselinecenterteacher_lr7p5e8_20260622_161345`
  - Start from S1-MJ `best_search_p10.pt`.
  - Teacher: current baseline `best_metric_search_center_px.pt`.
  - Goal: recover center while retaining enough of the S1-MJ P10/P5 gain.
- Both lanes:
  - `50` epoch gate.
  - `loaded_count=132 partial=0 skipped=0`.
  - CUDA devices resolved: S1-MM `cuda:0`, S1-MN `cuda:1`.

## Next Stage1 Escalation Branches

### S1-K: Conservative Geometry Teacher Polish

Purpose: test whether stronger geometry supervision can recover P10 without the capacity-collapse failure.

Use when:

- Active polish lanes fail P10 at the 50-epoch gate, or only improve P5.
- Current status: launched after promoted-polish Lane C/D failed the 50-epoch gate.

Proposed settings:

- Seed: current promoted Stage1 baseline checkpoint.
- Architecture: keep base `embed_dim=192`, `depth=6`, `num_heads=3`.
- Stage: `stage1`.
- Active head: `search`.
- LR: `1e-6`.
- Epochs: `80` minimum, `120` target.
- Loss weights:
  - `search_xy_weight=2.0`
  - `search_ab_weight=0.75`
  - `search_trig_weight=1.25`
  - `search_geo_weight=0.75`
  - `search_conf_weight=0.1`
- Distillation: disabled.

Reason:

- FACET-style geometry argues center-only prediction underuses pupil shape.
- This branch avoids new capacity and tests only a geometry-weighted refinement of the existing teacher.

Runner readiness:

- `scripts/external/run_stage1_frame_search_warmstart_refine.sh` now exposes per-lane geometry knobs:
  - `LANE_C_SEARCH_AB_WEIGHT`
  - `LANE_C_SEARCH_TRIG_WEIGHT`
  - `LANE_D_SEARCH_AB_WEIGHT`
  - `LANE_D_SEARCH_TRIG_WEIGHT`
- The runner passes these through as `loss.search_ab_weight` and `loss.search_trig_weight` overrides.
- Validation completed:
  - `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh`
  - S1-K dry run confirmed both lane commands include the new overrides.
- S1-K launched on 2026-06-22 after active Lane C/D failed the `50` epoch gate.

### S1-L: Low-LR EMA Self-Distill Anchor

Purpose: apply DistillGaze-style teacher/self-training more conservatively than prior collapsed capacity-distill runs.

Use when:

- S1-K fails or active Lane D shows better trend than Lane C but remains below baseline.
- Current status: launched in parallel with S1-K because both GPUs were free and S1-L tests a lower-risk self-distill anchor.

Proposed settings:

- Seed: current promoted Stage1 baseline checkpoint.
- Teacher: same checkpoint, EMA teacher enabled.
- Architecture: same as base, no capacity jump.
- LR: `7.5e-7`.
- Epochs: `80` minimum, `120` target.
- Distillation:
  - `ema_decay=0.9997`
  - `feature_weight=0.002`
  - `state_weight=0.002`
  - `prediction_weight=0.001`
  - `mask_weight=0.0` unless mask head is enabled and label quality is proven.
- Loss:
  - `search_xy_weight=2.0`
  - `search_geo_weight=0.75`

Reason:

- The active Lane D is already the safer branch compared with Lane C.
- This branch reduces LR and distillation weights further to prevent center drift.

### S1-M: Mask-Enabled Pseudo-Label Preparation Branch

Purpose: open RITnet/EllSeg/Grounded-SAM-backed Stage1 supervision only after label quality is checked.

Use when:

- Search-only branches plateau and there is evidence that mask/ellipse labels can improve pupil geometry.

Required preflight:

- Verify which samples have reliable mask/ellipse targets.
- Produce a split-safe pseudo-label manifest or quality mask.
- Confirm `active_head=all` or mask-enabled training actually contributes nonzero `loss_mask`.
- Prepared runner:
  `scripts/external/run_stage1_frame_search_mask_probe.sh`
- Runner default is `DRY_RUN=1`, so it is safe while S1-K/S1-L are still using both GPUs.
- Dry-run validation completed and confirmed the intended overrides:
  - `model.heads.active=all`
  - `model.heads.eye=true`
  - `model.heads.search=true`
  - `model.heads.mask=true`
  - `model.heads.event=false`
  - `model.heads.track=false`
  - `training.max_train_batches=128`
  - `training.max_val_batches=106`
  - `loss.mask_weight=0.1`
  - `loss.mask_coarse_weight=0.025`

Proposed settings:

- Do not launch while S1-K/S1-L are below the 50-epoch gate and actively using both GPUs.
- First run the diagnostic for `3` epochs or less:

```bash
DRY_RUN=0 \
DEVICE=cuda:0 \
RUN_TAG=stage1_s1m_mask_probe_after_s1kl \
bash scripts/external/run_stage1_frame_search_mask_probe.sh
```

- Accept the probe only if the log shows nonzero `loss_mask`, `loss_eye`, and `loss_search_*`, and Search P10 does not immediately collapse relative to the active baseline.
- Only then run a 50-epoch branch.

Reason:

- In current code, `active_head=search` disables mask/eye losses.
- Mask-teacher work without this preflight would look active but not affect training.

### S1-N: Reduced Capacity Distill Retry

Purpose: retest capacity only after conservative branches fail, avoiding the earlier 256x8 collapse.

Use when:

- S1-K/S1-L fail and GPU time is available.

Proposed settings:

- Prepared wrapper:
  `scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh`
- Wrapper default is `DRY_RUN=1`, so it cannot accidentally compete with active GPU training.
- Student architecture:
  - `embed_dim=224`
  - `depth=6`
  - `num_heads=4`
- Teacher architecture:
  - `embed_dim=192`
  - `depth=6`
  - `num_heads=3`
- LR:
  - no-distill lane: `7.5e-7`
  - teacher-distill lane: `5e-7`
- Batch size: `4`.
- Epochs: `50`.
- Distillation: use current baseline as teacher, EMA `0.9997`, feature/state `0.002`, prediction `0.001`, mask `0.0`.
- Gate at `10` and `50` epochs.

Reason:

- Prior capacity jump was too large and LR too high.
- A smaller width-only probe can test capacity without repeating the known bad configuration.
- Dry-run validation completed and confirmed the wrapper overrides the older 256x8/high-LR defaults.

## Execution Order

1. Completed: wait for promoted-polish Lane C/D to reach `50` epochs.
2. Completed: run reporter:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --baseline runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 \
  --min-epochs 50 \
  --target-epochs 120 \
  --followup runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916 \
  --followup runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916 \
  --format summary
```

3. Completed: no Lane C/D promotion; baseline remains the promoted Lane J checkpoint.
4. Completed: launch S1-K on GPU0.
5. Completed: launch S1-L on GPU1 in parallel because both GPUs were free and the branch is low-risk.
6. Active: monitor S1-K/S1-L to the `50` epoch gate.
7. Defer S1-M and S1-N until after conservative branches fail.

Gate decision helper:

```bash
.venv/bin/python scripts/external/decide_stage1_s1kl_gate_next.py --format summary
```

Expected behavior:

- Before both lanes reach `50` epochs: `action=wait_min_epochs`.
- If a lane is promotable: `action=promote_stage1_baseline`.
- If both lanes are clearly below baseline after the gate: `action=stop_s1kl_and_run_s1m_mask_probe`.

## Validation

- `bash -n` for any new runner before launch.
- Raw event-count contract check.
- Torch CUDA smoke on two GPUs.
- Minimum 50 epochs before promotion judgment.
- Promotion only through `best_search_p10.pt` and reporter gate.

## 2026-06-22 12:41 KST Runtime Update

### Closed S1-MA/S1-MB Gate

- S1-MA completed `50/50` epochs with no promotion.
  - Best P10 matched the active baseline at `28.27717937613433`.
  - Best P5 was `9.436208688987875`, below the baseline P5 `10.153863744915656`.
  - Best center was `17.29856945883553`, worse than the baseline center `17.257583906065744`.
- S1-MB completed `50/50` epochs with no promotion.
  - Best P10 reached `28.294025727038115`, a small `+0.016846350903783502 pp` improvement over baseline.
  - Best P5 was `9.301437791788354`, below the baseline P5.
  - Best center was `17.295753600462426`, worse than the baseline center.
- Decision: S1-MB is diagnostic only. Do not promote it because the P10 gain is tiny and center/P5 regress.

### Active S1-MC/S1-MD Center-Preservation Follow-Up

- Seed checkpoint:
  `runs/NON_XR/raw/stage1_s1mb_maskonly_searchpreserve_50ep_after_s1n_20260622_120546/train/best_search_p10.pt`
- S1-MC active run:
  `runs/NON_XR/raw/stage1_s1mc_s1mb_centerpolish_maskweak_50ep_20260622_123430`
  - GPU0.
  - Weak mask assist: `mask_weight=0.01`, `mask_coarse_weight=0.0`.
  - Search preserve: `search_xy_weight=2.25`, `search_geo_weight=0.5`.
  - Center constraint: `constraint_center_weight=0.2`, `constraint_center_radius=8.0`.
  - LR `1e-7`, min LR `5e-8`, epochs `50`.
- S1-MD active run:
  `runs/NON_XR/raw/stage1_s1md_s1mb_centerpolish_searchonly_50ep_20260622_123440`
  - GPU1.
  - Search-only center polish: `mask_weight=0.0`, `mask_coarse_weight=0.0`.
  - Same search and center-constraint weights as S1-MC.
  - LR `1e-7`, min LR `5e-8`, epochs `50`.
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1mc_s1md_min50_watch_20260622.log`
- Runtime snapshot at `2026-06-22 12:42 KST`:
  - Reporter: S1-MC `12/50`, S1-MD `12/50`, `status=wait_min_epochs`.
  - Watcher and train logs show both runs progressing normally.
  - Current best P10 for both is `28.176101198736227`, below baseline by `-0.10107817739810443 pp`.
  - Current best P5 for both is `9.183513263486466`, below baseline by `-0.9703504814291897 pp`.
  - Current best center remains worse than baseline by about `0.0546 px`.
- Decision: keep both active until at least `50` epochs. Do not promote or stop before the minimum gate unless a hard runtime failure appears.
- Validation at this checkpoint passed:
  - `bash -n scripts/external/run_stage1_frame_search_mask_probe.sh`
  - `bash -n scripts/external/run_stage1_frame_search_capacity_distill.sh`
  - `bash -n scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh`
  - `bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`
  - `bash -n scripts/external/run_stage1_frame_search_post_followup_queue.sh`
  - `python3 -m py_compile scripts/external/decide_stage1_s1kl_gate_next.py scripts/external/report_stage1_frame_search_followup.py`
  - `git diff --check`
- Queue guard fix:
  - `scripts/external/run_stage1_frame_search_post_followup_queue.sh` now accepts and forwards `BASELINE_RUN` and `FOLLOWUP_RUNS` to the reporter.
  - Dry-run with `DETACH=0 DRY_RUN=1 POLL_SEC=1 MAX_WAIT_SEC=1 START_FALLBACK_WATCHER=0` confirmed the queue evaluates S1-MC/S1-MD, not stale default followups.

### Closed S1-MC/S1-MD Gate

- Final gate at `2026-06-22 13:00 KST`:
  - S1-MC `50/50`, P10 `28.176101198736227`, P5 `9.183513263486466`, center `17.31221647532481`.
  - S1-MD `50/50`, P10 `28.176101198736227`, P5 `9.183513263486466`, center `17.312216362863218`.
  - Baseline remains P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Decision: no promotion. Both lanes are worse than baseline on P10, P5, and center.

### Active S1-ME/S1-MF Baseline Rescue Bracket

- Launch time: `2026-06-22 13:04 KST`.
- Launch mode: host GPU execution was required because sandbox Torch CUDA smoke returned `torch.cuda.is_available() is false`; host preflight passed with `torch_cuda_device_count=2`.
- Shared seed:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- S1-ME:
  `runs/NON_XR/raw/stage1_s1me_baseline_geometry_micro_lr3e7_xy2p25_ab0p75_trig1p25_20260622_130436`
  - GPU0.
  - LR `3e-7`, `search_xy_weight=2.25`, `search_ab_weight=0.75`, `search_trig_weight=1.25`, `search_geo_weight=0.75`.
  - Distillation disabled.
- S1-MF:
  `runs/NON_XR/raw/stage1_s1mf_baseline_ultraweak_selfdistill_lr2e7_xy2_ema9999_20260622_130436`
  - GPU1.
  - LR `2e-7`, `search_xy_weight=2.0`, `search_ab_weight=0.5`, `search_trig_weight=1.0`, `search_geo_weight=0.75`.
  - Ultra-weak self-distillation: EMA `0.9999`, feature/state `0.0005`, prediction `0.00025`, mask `0.0`.
- Both lanes loaded the baseline checkpoint with `loaded_count=142 partial=0 skipped=0`.
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1me_s1mf_min50_watch_20260622.log`
- Note: this runner uses the full train manifest, `742` train steps per epoch, so it will be slower than the previous `max_train_batches=256` mask-probe runs.

### S1-ME/S1-MF Runtime Refresh at 2026-06-22 13:12 KST

- Status: active and still below the `50` epoch decision gate.
- Reporter snapshot:
  - S1-ME `10/50`, `min_ready=False`, best P10 `28.024483914645213`, best P5 `9.112758546505335`, best center `17.306600507700217`.
  - S1-MF `10/50`, `min_ready=False`, best P10 `28.277179340146624`, best P5 `9.264375812602493`, best center `17.299979461813873`.
  - `status=wait_min_epochs`, `promote_checkpoint=None`.
- Log evidence after the reporter snapshot:
  - S1-ME had advanced to epoch `11/50` validation.
  - S1-MF had advanced to epoch `11/50` training.
- Error scan: no `Traceback`, `ERROR`, `RuntimeError`, `CUDA out of memory`, `Killed`, `failed`, `exited`, or `returncode` signatures in the active lane logs or watcher log.
- Interpretation: experiments are progressing; perceived slowness comes from full-manifest training (`742` train steps and `106` validation steps per epoch), not from a stalled run.
- Post-document reporter verification advanced to S1-ME `13/50` and S1-MF `12/50`; status remains `wait_min_epochs`.

### S1-ME/S1-MF Closeout and S1-MG/S1-MH Threshold-Loss Branch

- S1-ME/S1-MF completed `50/50` and did not promote.
- Closeout decision:
  - S1-ME: P10 `28.024483914645213`, P5 `9.112758546505335`, center `17.306600507700217`.
  - S1-MF: P10 `28.277179340146624`, P5 `9.264375812602493`, center `17.299979461813873`.
  - Baseline remains P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- New mechanism added after the repeated no-promotion polish branches:
  - Stage1 Search soft-threshold loss for P10/P5 was added in `src/hbtxr/loss/stage1.py`.
  - This reuses the existing Stage2 idea of directly penalizing center error above metric thresholds, but applies it to frame Search.
  - Default weights are zero, so historical configs keep previous behavior.
- S1-MG/S1-MH launched from the active baseline:
  - S1-MG: `runs/NON_XR/raw/stage1_s1mg_p10soft_lr2e7_p10w0p05_p5w0p01_20260622_134341`, GPU0, no distillation, P10 threshold emphasis.
  - S1-MH: `runs/NON_XR/raw/stage1_s1mh_p10p5soft_selfdistill_lr1p5e7_p10w0p03_p5w0p03_20260622_134341`, GPU1, P10/P5 threshold emphasis plus ultra-weak self-distillation.
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1mg_s1mh_min50_watch_20260622.log`
- Gate:
  - Minimum `50` epochs before any result decision.
  - Promotion requires P10 `>28.27717937613433` and center no worse than the active baseline.

### S1-MG/S1-MH Closeout at 2026-06-22 14:22 KST

- S1-MG/S1-MH completed `50/50` and did not promote.
- Final reporter status: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Baseline remains P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- S1-MG:
  - Run: `runs/NON_XR/raw/stage1_s1mg_p10soft_lr2e7_p10w0p05_p5w0p01_20260622_134341`.
  - P10 `28.159254811844736`, P5 `9.129604933396825`, center `17.303716394136536`.
  - P10 delta `-0.11792456428959497`; center regressed by `0.04613248807079273 px`.
- S1-MH:
  - Run: `runs/NON_XR/raw/stage1_s1mh_p10p5soft_selfdistill_lr1p5e7_p10w0p03_p5w0p03_20260622_134341`.
  - P10 `28.142408460940956`, P5 `9.247529461698711`, center `17.297416754488676`.
  - P10 delta `-0.13477091519337492`; center regressed by `0.03983284842293244 px`.
- Interpretation:
  - Direct threshold pressure did not recover P10/P5 and also moved center away from the active baseline.
  - Continue treating the active baseline as the durable Stage1 frame-search baseline until a larger-axis experiment beats it after the minimum epoch gate.

### S1-MI/S1-MJ Full-Manifest Mask-Guard Branch

- Runner:
  `scripts/external/run_stage1_frame_search_mask_guard_full.sh`
- Purpose:
  - Revisit the only branch that showed any P10 upside, S1-MB mask-assisted Search, but remove the bounded-batch instability and add stronger center guards.
  - Avoid the S1-N capacity failure mode by keeping the same model shape as the active baseline.
- Shared setup:
  - Baseline checkpoint:
    `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
  - Full manifest, `50` epochs, batch `8`, workers `4`.
  - Active heads: Search + mask; eye/event/track/aux disabled.
  - Distillation disabled.
  - Promotion gate remains P10 `>28.27717937613433` and center `<=17.257583906065744`.
- S1-MI:
  - Run: `runs/NON_XR/raw/stage1_s1mi_fullmask_searchguard_lr2e7_mask0125_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`.
  - GPU0, LR `2e-7`, mask `0.0125`, coarse mask `0.003125`, center guard `0.20`, radius `16 px`.
- S1-MJ:
  - Run: `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`.
  - GPU1, LR `1.5e-7`, mask `0.00625`, coarse mask `0.0015625`, center guard `0.30`, radius `12 px`.
- Startup:
  - Both lanes loaded shape-compatible checkpoint weights: `loaded_count=126 partial=0 skipped=0`.
  - Initial epoch-2 watcher snapshot is below baseline, so no early promotion. Continue to the `50` epoch gate.

### S1-MI/S1-MJ Closeout and Follow-Up Decision

- S1-MI/S1-MJ completed `50/50`.
- S1-MJ produced the strongest Stage1 frame-search P10 seen in this branch:
  - P10 `28.482705062290407`, `+0.2055256861560757` over baseline.
- S1-MJ did not promote because:
  - P5 `9.205975082685363`, `-0.9478886622302927` below baseline.
  - center `17.31833925787008`, `0.060755351804335334 px` worse than baseline.
- S1-MI did not promote:
  - P10 `28.19070143069861`.
  - P5 `9.32389961098725`.
  - center `17.316034978290773`.
- Next experiment:
  - Start from S1-MJ `best_search_p10.pt`, not from the old baseline.
  - Run a center/P5 recovery polish with very low LR and stronger center constraint.
  - Promotion still requires P10 above `28.27717937613433` and center no worse than `17.257583906065744`.

### S1-MK/S1-ML Center/P5 Recovery Branch

- Seed:
  `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658/train/best_search_p10.pt`
- Runner:
  `scripts/external/run_stage1_frame_search_mask_guard_full.sh`
- Runner update:
  - Added lane-level P10/P5 soft-threshold loss weights.
- Shared setup:
  - Full manifest, `50` epochs, batch `8`, workers `4`.
  - Best metric `metric_search_center_px`.
  - Mask/eye loss disabled; Search remains active.
  - P10/P5 soft losses are enabled to keep the S1-MJ P10 gain from collapsing while trying to recover center/P5.
- S1-MK:
  - Run: `runs/NON_XR/raw/stage1_s1mk_s1mj_p10seed_centerrecover_lr1e7_p5soft02_20260622_150603`.
  - LR `1e-7`, center constraint `0.50`, radius `10 px`, P10/P5 soft `0.01/0.02`.
- S1-ML:
  - Run: `runs/NON_XR/raw/stage1_s1ml_s1mj_p10seed_strongcenter_lr7p5e8_p5soft04_20260622_150603`.
  - LR `7.5e-8`, center constraint `0.75`, radius `8 px`, P10/P5 soft `0.01/0.04`.
- Startup:
  - Both lanes loaded shape-compatible checkpoint weights: `loaded_count=132 partial=0 skipped=0`.
  - Continue to the required `50` epoch gate before judging promotion.

### S1-MK/S1-ML Closeout

- S1-MK/S1-ML completed `50/50` and did not promote.
- S1-MK retained the S1-MJ P10 signal and improved P5 versus S1-MJ:
  - P10 `28.482705062290407`.
  - P5 `9.632749593482828`.
  - center `17.43487750809148`.
- S1-ML was weaker:
  - P10 `27.994160562191368`.
  - P5 `9.514825083174795`.
  - center `17.44643774122562`.
- Interpretation:
  - S1-MJ's P10-rich checkpoint is not recoverable through low-LR center/P5 training under the current objective.
  - Next branch should avoid more full training and evaluate checkpoint interpolation between the promoted baseline and S1-MJ best-P10.

### S1-MM/S1-MN Teacher-Distill Closeout

- S1-MM/S1-MN completed the required `50/50` gate.
- Seed:
  `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658/train/best_search_p10.pt`
- S1-MM:
  - Teacher: current strict baseline `best_search_p10.pt`.
  - Result: P10 `28.482705062290407`, P5 `9.088050554383475`, center `17.416398777152008`.
- S1-MN:
  - Teacher: current strict baseline `best_metric_search_center_px.pt`.
  - Result: P10 `28.482705062290407`, P5 `9.088050554383475`, center `17.42239284515381`.
- Decision:
  - No strict baseline promotion.
  - Teacher distillation preserved the S1-MJ P10 gain but did not recover P5 or center.
  - Avoid launching another long full-manifest branch until a no-train checkpoint-composition check is complete.

### Immediate Next Check: Baseline-Internal Soup

- Evaluate weighted soups/interpolations among the current baseline run's own checkpoint variants:
  - `best_search_p10.pt`
  - `best_search_p5.pt`
  - `best_metric_search_center_px.pt`
- Rationale:
  - These checkpoints share model keys and represent the observed P10/P5/center tradeoff inside the strongest strict baseline run.
  - This is cheaper than another 50-epoch GPU branch and directly tests whether the strict baseline can be improved by checkpoint composition.
- Promotion gate:
  - P10 must stay above the current strict baseline P10 `28.27717937613433`.
  - Center must be no worse than the current strict baseline center `17.257583906065744`.

### Baseline-Internal Soup Result

- Summary:
  `runs/NON_XR/shared/eval/stage1_baseline_internal_soup_20260622_1710/summary.json`
- Result:
  - No weighted checkpoint soup is promotable.
  - `ref_p10` remains the highest-P10 candidate at P10 `28.27717937613433`.
  - The closest non-reference soup, `soup_p10_090_p5_005_center_005`, has P10 `28.159254847832447`, P5 `8.452381206008623`, and center `17.291879788884575`.
  - P5-heavy soups recover P5 only by sacrificing P10 and center.
- Decision:
  - Do not spend another long branch on weight-space interpolation.
  - Run output-space ensemble evaluation next. If output-space averaging improves P10/center, use it as a teacher or pseudo-label target; if not, the next training branch must change target construction or sampling, not just LR/loss weights.

### Output-Space Ensemble Result

- Summary:
  `runs/NON_XR/shared/eval/stage1_output_ensemble_20260622_1720/summary.json`
- Key result:
  - `out_p10_090_p5_050_center_050` ties the current P10 reference at `28.27717937613433`.
  - It improves same-evaluator P5 from `8.334456677706736` to `8.452381206008623`.
  - It improves same-evaluator center from `17.292674766396576` to `17.288852102351637`.
- Decision:
  - Not a strict baseline promotion, but useful as a low-risk teacher target.
  - Convert it into a fixed ensemble teacher rather than averaging checkpoint weights.

### S1-MO/S1-MP Ensemble-Teacher Branch

- Implementation:
  - `src/hbtxr/training/ensemble_teacher.py`
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh`
  - `tests/test_ensemble_teacher.py`
- Teacher:
  - `best_search_p10.pt`: `0.90`
  - `best_search_p5.pt`: `0.05`
  - `best_metric_search_center_px.pt`: `0.05`
- S1-MO:
  - GPU0.
  - LR `1e-7`.
  - State distillation `0.02`, prediction distillation `0.005`, center constraint `0.20`, P5 soft threshold `0.01`.
- S1-MP:
  - GPU1.
  - LR `7.5e-8`.
  - State distillation `0.03`, prediction distillation `0.005`, center constraint `0.30`, P5 soft threshold `0.02`.
- Launch evidence:
  - Host launch tag: `stage1_s1mo_s1mp_ensemble_teacher_20260622_1725`.
  - S1-MO run:
    `runs/NON_XR/raw/stage1_s1mo_baseline_ensembleteacher_lr1e7_stage1_s1mo_s1mp_ensemble_teacher_20260622_1725_20260622_172100`.
  - S1-MP run:
    `runs/NON_XR/raw/stage1_s1mp_baseline_ensembleteacher_lr7p5e8_stage1_s1mo_s1mp_ensemble_teacher_20260622_1725_20260622_172100`.
  - Both lanes reached epoch `3/50` startup territory after completing epoch `2/50` validation.
- Gate:
  - Wait until at least `50/50` before promotion judgment.
  - Promotion requires P10 `>28.27717937613433` and center `<=17.257583906065744`.
- Current watcher:
  - Refreshed host PID `1188026`.
  - Log: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800.log`.
  - Summary: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800_report.txt`.
  - JSON: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800_report.json`.
  - First report: S1-MO `6/50`, S1-MP `6/50`, `status=wait_min_epochs`, no promotion decision yet.
  - Latest report at `2026-06-22 17:31 KST`: S1-MO `16/50`, S1-MP `16/50`, both still below the `50/50` decision gate. Both lanes tie baseline P10 at `28.27717937613433`; P5 and center remain below the strict baseline, so the only valid action is to continue training. Log tail confirms both lanes have entered epoch `17/50` training.
  - Latest direct reporter at `2026-06-22 17:37 KST`: S1-MO `24/50`, S1-MP `24/50`, `status=wait_min_epochs`, `promote_checkpoint=None`.
  - Latest direct reporter at `2026-06-22 17:42 KST`: S1-MO `29/50`, S1-MP `29/50`, still below the decision gate.

### S1-MQ/S1-MR Weighted Ensemble-Teacher Queue

- Status: prepared, not launched.
- Trigger: launch only if S1-MO/S1-MP reach `50/50` and fail strict promotion.
- Script:
  - `scripts/external/run_stage1_frame_search_weighted_ensemble_teacher.sh`
  - Defaults to `DRY_RUN=1` for queue safety.
- Runner support:
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh` now accepts optional weighted sampler overrides.
  - Default `SAMPLER_ENABLED=false`, preserving existing behavior.
- Experiment idea:
  - Keep full train manifest visible.
  - Oversample difficult rows instead of repeating hard-subset training.
  - Keep fixed output ensemble teacher from current baseline P10/P5/center checkpoints.
- Sampler:
  - `similarity_target <= 0.1`, multiplier `3.0`.
  - `session_201`, multiplier `1.5`.
  - `max_multiplier=6.0`.
  - Train manifest check produced finite weights with unique values `{1.0, 1.5, 3.0, 4.5}`.
- Lanes:
  - S1-MQ: LR `1.5e-7`, state `0.02`, prediction `0.005`, center `0.25`, P5 soft `0.03`.
  - S1-MR: LR `1e-7`, state `0.03`, prediction `0.005`, center `0.35`, P5 soft `0.02`.
- Validation:
  - `bash -n` passed for both scripts.
  - Dry-run confirmed both lane commands include `training.sampler.*` overrides.
- Launch update:
  - S1-MO/S1-MP failed the `50/50` promotion gate, so S1-MQ/S1-MR launched at `2026-06-22 17:57 KST`.
  - S1-MQ run: `runs/NON_XR/raw/stage1_s1mq_lowsim_weighted_ensemble_lr1p5e7_stage1_s1mq_s1mr_weighted_ensemble_teacher_20260622_1758_20260622_175755`.
  - S1-MR run: `runs/NON_XR/raw/stage1_s1mr_lowsim_weighted_centerhold_lr1e7_stage1_s1mq_s1mr_weighted_ensemble_teacher_20260622_1758_20260622_175755`.
  - Watcher: `runs/NON_XR/shared/_logs/stage1_s1mq_s1mr_min50_watch_20260622_1800_report.txt`.
  - Early reporter at `2026-06-22 18:03 KST`: S1-MR `7/50` P10 `28.27717937613433`, P5 `8.721923000407669`, center `17.280365035219013`; S1-MQ `7/50` P10 `28.142408460940956`, P5 `8.806154790914283`, center `17.283770327298146`.

### S1-MQ/S1-MR Closeout

- Final reporter at `50/50`:
  - S1-MQ: P10 `28.142408460940956`, P5 `8.806154790914283`, center `17.27892105084545`.
  - S1-MR: P10 `28.27717937613433`, P5 `8.839847528709555`, center `17.26318852856474`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Decision:
  - No strict baseline promotion.
  - Weighted sampling did not recover the P5/center gap.
  - Do not repeat the same sampler-heavy full-manifest approach as the next branch.

### S1-MS/S1-MT P10-Seed Recovery Branch

- Status: launched at `2026-06-22 18:40 KST`.
- Script:
  - `scripts/external/run_stage1_frame_search_p10seed_recovery_ensemble_teacher.sh`
- Runner extension:
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh` now supports separate `PRETRAIN_CKPT`.
  - Teacher weights are configurable with `ENSEMBLE_P10_WEIGHT`, `ENSEMBLE_P5_WEIGHT`, and `ENSEMBLE_CENTER_WEIGHT`.
- Seed:
  - `runs/NON_XR/raw/stage1_s1mm_s1mj_p10seed_baselinep10teacher_lr1e7_20260622_161345/train/best_search_p10.pt`.
- Teacher:
  - Current strict baseline P10/P5/center checkpoints.
  - Weights: P10 `0.75`, P5 `0.10`, center `0.15`.
- Lanes:
  - S1-MS: GPU0, LR `5e-8`, state distill `0.04`, prediction distill `0.01`, center constraint `0.75`, P5 soft threshold `0.05`.
  - S1-MT: GPU1, LR `4e-8`, state distill `0.05`, prediction distill `0.015`, center constraint `1.00`, P5 soft threshold `0.08`.
- Sampling:
  - Disabled.
  - Rationale: S1-MQ/S1-MR showed weighted sampling preserved/tied P10 at best but worsened P5/center, so the next lever is high-P10 seed recovery, not more sampler pressure.
- Runs:
  - S1-MS:
    `runs/NON_XR/raw/stage1_s1ms_p10seed_centerrecover_lr5e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824`.
  - S1-MT:
    `runs/NON_XR/raw/stage1_s1mt_p10seed_p5centerrecover_lr4e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824`.
- Watcher:
  - `runs/NON_XR/shared/_logs/stage1_s1ms_s1mt_min50_watch_20260622_1842_report.txt`.
- Gate:
  - Wait for `50/50`.
  - Promote only if P10 is above `28.27717937613433` and center is no worse than `17.257583906065744`.

### S1-MS/S1-MT Closeout

- Final reporter at `50/50`:
  - S1-MS: P10 `28.347934165090884`, P5 `9.32389961098725`, center `17.358882256273954`.
  - S1-MT: P10 `28.482705062290407`, P5 `9.32389961098725`, center `17.358525235697908`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Result:
  - P10-only Search accuracy improved.
  - Strict baseline promotion fails because center remains `0.10094132963216396 px` worse than baseline and P5 remains `0.8299641339284065 pp` lower.
- Diagnosis:
  - The bottleneck is no longer P10; it is preserving the center/P5 precision while holding the P10 gain.
  - S1-MT should be treated as the high-P10 seed for a center-direct refinement branch, not as the strict baseline.

### Next Branch: Center-Direct Refinement From S1-MT

- Seed:
  - `runs/NON_XR/raw/stage1_s1mt_p10seed_p5centerrecover_lr4e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824/train/best_search_p10.pt`.
- Required change:
  - Increase center checkpoint teacher weight.
  - Increase center and P5 losses.
  - Keep LR at or below `2e-8` so the high-P10 basin is not destroyed.
- Promotion target:
  - P10 `>28.27717937613433`.
  - Center `<=17.257583906065744`.
  - Prefer P5 recovery toward `10.153863744915656`, but P5 is secondary to strict P10+center gate.

### 2026-06-23 Center-Constraint Normalization Fix

- Finding:
  - `src/hbtxr/models/pruning.py` normalized `loss.constraint_center_weight` to `0.0` whenever `model.heads.active != track`.
  - Stage1 Search runners set `model.heads.active=search`, so center-direct runs silently disabled the intended center constraint.
  - This explains why prior "center-direct" runs had `loss_constraint_center=0.0` despite nonzero command-line overrides.
- Fix:
  - Search and Track now preserve center constraint: `active in {"search", "track"}`.
  - Non-search/non-track active heads still zero center constraint.
- Validation:
  - `python3 -m py_compile src/hbtxr/models/pruning.py`.
  - Config-normalization smoke:
    - `model.heads.active=search`, `loss.constraint_center_weight=1.5` -> normalized value `1.5`.
    - `model.heads.active=eye`, `loss.constraint_center_weight=1.5` -> normalized value `0.0`.
- Invalidated/limited interpretation:
  - Original S1-MU/S1-MV completed `50/50`, but all epochs show `loss_constraint_center=0.0`.
  - Best report against strict baseline:
    - S1-MU: P10 `28.347934165090884`, P5 `9.32389961098725`, center `17.369157035395784`, no promotion.
    - S1-MV: P10 `28.347934165090884`, P5 `9.32389961098725`, center `17.38588412302845`, no promotion.
  - Treat those runs as non-promoted P10/P5-soft/distill trials, not valid center-direct evidence.
- Relaunch:
  - Corrected S1-MU/S1-MV launched with `RUN_TAG=stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622`.
  - PIDs: S1-MU `2597105`, S1-MV `2597106`.
  - New run roots:
    - `runs/NON_XR/raw/stage1_s1mu_s1mt_centerheavy_lr2e8_stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622_20260623_091716`.
    - `runs/NON_XR/raw/stage1_s1mv_s1mt_centermax_lr1e8_stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622_20260623_091716`.
  - Epoch-1 history confirms center constraint is active:
    - S1-MU `loss_constraint_center`: train `98.21118527024261`, val `95.75954721558769`.
    - S1-MV `loss_constraint_center`: train `163.8419010028685`, val `159.91095028283462`.
  - Final `50/50` report:
    - S1-MU: P10 `28.482705062290407`, P5 `9.514825083174795`, center `17.427942365970253`, no promotion.
    - S1-MV: P10 `28.482705062290407`, P5 `9.514825083174795`, center `17.418982910660077`, no promotion.
  - Interpretation: center constraint is now active, but the strong center term did not recover the strict center gate; it preserved the P10-only S1-MT level and improved P5 relative to S1-MT, but center still regressed.

### Expanded Stage1 Search Ablation Queue

- Suite chain, queued after corrected S1-MU/S1-MV:
  - Chain order: `head_aux -> opt_sched_loss -> augmentation -> deit_preload -> head_factory -> depth_mask`.
  - Chain watcher: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_chain_after_fixed_center_20260623.log`.
  - Chain watcher PID: `3219476`.
- Extension chain:
  - Existing chain does not include later-added `opt_sched_loss_ext` and `augmentation_ext`.
  - Host-visible extension watcher PID `3546751` waits for PID `3219476`, then runs `opt_sched_loss_ext -> augmentation_ext`.
  - Log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_ext_after_existing_chain_20260623_host.log`.
- Head/Aux suite:
  - S1-MW: residual Search Head (`model.heads.search_variant=residual_mlp`).
  - S1-MX: residual Search Head plus search bbox/OBB auxiliary geometry.
  - Status: closed at `50/50` with no promotion.
  - S1-MW final bests: P10 `28.232255755730396`, P5 `9.24528328877575`, center `17.31714263502157`.
  - S1-MX final bests: P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.317125887241005`.
  - Interpretation: residual head and auxiliary geometry did not recover the strict P10/P5/center gate.
- Depth/Mask suite:
  - S1-MY: backbone depth `8`, center-best probe.
  - S1-MZ: residual Search Head plus mask cascade.
  - Override fix: S1-MZ now sets `model.heads.active=all` and explicitly enables `search=true`, `mask=true`; otherwise the Search-only normalizer would disable mask.
- HeadFactory/ROI suite:
  - S1-NA: `eye_variant=sot_center` with ROI bbox auxiliary guidance.
  - S1-NB: `eye_variant=yolo_detect` with ROI bbox auxiliary guidance.
  - Override fix: these lanes now set `model.heads.active=all`, `eye=true`, and `search=true`, otherwise the external-eye ROI path would not instantiate the eye head.
- DeiT preload suite:
  - S1-NC: DeiT-Tiny pretrained backbone, AdamW + cosine schedule.
  - S1-ND: DeiT-Tiny pretrained backbone plus residual Search Head.
  - Candidate checkpoint: `/home/kjm26/project/PRJXR/model_zoo/deit_tiny_distilled_patch16_224.pth`.
  - Preload smoke with `source=deit`, `strict_shape=false`: `loaded_count=98`, `skipped_count=0`.
- Optimizer/loss/scheduler suite:
  - S1-NE: Lion optimizer, plateau scheduler, stronger P5/center guard.
  - S1-NF: Adopt optimizer, step scheduler, stronger P5/center guard.
  - Status: launched by existing suite chain after Head/Aux closeout.
  - Closeout at `50/50`: no promotion.
  - S1-NE: P10 `28.192947567633862`, P5 `9.380054185975272`, center `17.319932807166623`.
  - S1-NF: P10 `28.35018028403228`, P5 `9.363207799083781`, center `17.306676203349852`.
  - Interpretation: S1-NF improves P10 by `+0.07300090789794922 pp`, but P5 regresses by `-0.7906559458318743 pp` and center is `0.049092297284108355 px` worse. S1-NE underperforms all strict gates. Keep current baseline.
- Optimizer/loss/scheduler extension suite:
  - S1-NI: AdamW schedule-free + cautious modifier, no external scheduler, residual Search Head, P5/center guard.
  - S1-NJ: MuSGD optimizer, cosine scheduler, residual Search Head, conservative Muon/SGD mix, P5/center guard.
- Optimizer/loss/scheduler plus suite:
  - S1-NO: AdEMAMix optimizer, cosine scheduler, residual Search Head, P5/center guard.
  - S1-NP: MARS optimizer, plateau scheduler on `metric_search_center_px`, residual Search Head, stronger center/P5 guard.
  - Rationale: S1-NE/S1-NF already cover Lion+plateau and Adopt+step; S1-NI/S1-NJ cover schedule-free AdamW and MuSGD. S1-NO/S1-NP add a mixed-momentum Adam-family optimizer and a gradient-correction AdamW variant without changing geometry or data split.
  - Validation: `bash -n` passed for queue and chain scripts; `DRY_RUN=1` generated both commands; CPU optimizer construction smoke passed for `adema_mix` and `mars`.
  - Reservation: host-visible watcher PID `3645713` waits for extension chain PID `3546751`, then runs `opt_sched_loss_plus`.
  - Reservation log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_plus_after_ext_20260623_host2.log`.
- Distillation extension suite:
  - S1-NQ: EMA self-distillation from the current baseline seed, ensemble disabled, teacher initialized from the student, `ema_decay=0.999`, residual Search Head, feature/state/prediction distillation, KD, RKD, and P5/center guard.
  - S1-NR: S1-MT high-P10 single teacher checkpoint with ensemble disabled, residual Search Head, feature/state/prediction distillation, KD, RKD, plateau scheduler on `metric_search_center_px`, and stronger P5/center guard.
  - S1-MT teacher checkpoint:
    `runs/NON_XR/raw/stage1_s1mt_p10seed_p5centerrecover_lr4e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824/train/best_search_p10.pt`.
  - Rationale: prior branches improved P10 but failed center/P5 preservation. This suite avoids another fixed ensemble-only lane by pairing EMA self-distillation with an explicit high-P10 single-teacher lane.
  - Validation: `bash -n` passed for queue and chain scripts; `DRY_RUN=1 SUITE=distill_ext` generated both commands; CPU teacher creation smoke passed for EMA self-distill and single-teacher modes.
  - Reservation: host-visible watcher PID `3681125` waits for plus watcher PID `3645713`, then runs `distill_ext`.
  - Reservation log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_distill_after_plus_20260623_host.log`.
- Loss-only extension suite:
  - S1-NS: AdamW + cosine kept fixed; tighter center radius, stronger center/P5 soft-threshold loss, and moderate XY/geometry weights.
  - S1-NT: AdamW + cosine kept fixed; geometry-balanced Search loss with higher AB/trig/GWD weights, lower confidence weight, and sharper P10 soft-threshold.
  - Rationale: S1-NE/S1-NF and later optimizer suites change optimizer, scheduler, and loss together. S1-NS/S1-NT isolate the loss axis so any improvement can be attributed to Search objective weighting rather than optimizer state dynamics.
- Scheduler-only extension suite:
  - S1-NU: AdamW + fixed residual Search Head/loss weights with plateau scheduler on `metric_search_center_px`.
  - S1-NV: AdamW + the same fixed residual Search Head/loss weights with step scheduler.
  - Rationale: existing optimizer suites cover `cosine`, `plateau`, `step`, and `none`, but each scheduler is tied to optimizer/loss changes. S1-NU/S1-NV isolate the scheduler axis under one fixed optimizer/objective family.
- P10 recovery extension suite:
  - S1-NW: Adopt + step from the S1-NF P10 signal, but with residual Search Head, tighter center radius, and stronger center/P5 guard.
  - S1-NX: HeadFactory `sot_center` ROI aux from the S1-NA P10 signal, but with reduced aux/eye weights and stronger center/P5 guard.
  - Rationale: S1-NF and S1-NA/S1-NB produced small P10-only gains while failing center/P5. This suite tests whether those P10-lift mechanisms can be kept after explicit center/P5 recovery pressure.
- Augmentation suite:
  - S1-NG: frame gain/bias/noise augmentation plus residual Search Head.
  - S1-NH: frame+event gain/dropout/noise augmentation plus residual Search Head.
  - Augmentation is train-only; validation disables augmentation through loader `training=bool(shuffle)`.
- Augmentation extension suite:
  - S1-NK: mild frame-only gain/bias/noise augmentation to test whether lighter regularization preserves center/P5 better than S1-NG.
  - S1-NL: mild frame augmentation plus stronger event gain/dropout/noise to test event robustness separately from geometry/model changes.
- Combined optimizer/loss/scheduler/micro-augmentation guard suite:
  - S1-OA: Adopt optimizer, plateau scheduler on `metric_search_center_px`, tight center/P5 loss guard, residual Search Head, and micro frame augmentation.
  - S1-OB: AdamW schedule-free/cautious optimizer, no external scheduler, tight center/P5 loss guard, residual Search Head, and micro frame/event augmentation.
  - Rationale: isolated optimizer, loss, scheduler, and augmentation suites did not pass the strict P10/center gate. The next useful test is not stronger augmentation, but a conservative joint perturbation that keeps optimizer dynamics and objective pressure aligned while limiting validation-distribution drift.

### 2026-06-23 Completed Main-Chain Results

- Completed `augmentation`, `deit_preload`, and `head_factory` at the required `50/50` gate.
- Strict baseline remains:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- `augmentation`:
  - S1-NG/S1-NH both reached P10 `28.176101198736227`, P5 `9.278976026571021`, center `17.296725754467946`.
  - Decision: no promotion.
- `deit_preload`:
  - S1-NC: P10 `5.795148381647074`, P5 `2.6448787653221273`, center `43.2115170460827`.
  - S1-ND: P10 `5.20552577612535`, P5 `2.015947935716161`, center `44.50417672463183`.
  - Decision: no promotion; direct DeiT-Tiny preload is currently incompatible with the fine-tuned Stage1 Search target without a dedicated adaptation phase.
- `head_factory`:
  - S1-NA: P10 `28.32771844683953`, P5 `9.24528328877575`, center `17.31330567036035`.
  - S1-NB: P10 `28.32771844683953`, P5 `9.24528328877575`, center `17.314067854071563`.
  - Decision: no promotion; HeadFactory ROI aux gives a small P10-only lift but worsens center/P5.
- Current main chain state:
  - Completed `depth_mask`; no promotion.
  - S1-MY depth-8 center-best reached P10 `28.310872077941895`, P5 `9.127358760473863`, center `17.354738820273923`.
  - S1-MZ mask-cascade residual reached P10 `5.133647990676592`, P5 `1.769991046977493`, center `44.22043274933437`.
  - Interpretation: S1-MY is a P10-only diagnostic signal and S1-MZ is a hard failure. Keep the strict baseline.
- Queued follow-up state:
  - Completed `opt_sched_loss_ext` at `50/50` with no promotion:
    - S1-NI P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.318899788946474`.
    - S1-NJ P10 `28.27717937613433`, P5 `8.77583133049731`, center `17.292718172073364`.
    - Decision: keep baseline because S1-NJ only ties P10 and regresses P5/center; S1-NI underperforms all gates.
  - Completed `augmentation_ext` at `50/50` with no promotion:
    - S1-NK mild frame augmentation P10 `28.04133031953056`, P5 `9.396900554872909`, center `17.312655894261486`.
    - S1-NL event dropout/noise augmentation P10 `28.05817667043434`, P5 `9.380054185975272`, center `17.31234776298955`.
    - Decision: keep baseline because both augmentation variants underperform P10 and regress P5/center.
  - Auto-chain reconciled after the extension wait and launched active `opt_sched_loss_plus`:
    - S1-NO AdEMAMix + cosine P5-guard lane on GPU0, PID `1020984`, run `stage1_s1no_ademamix_cosine_p5guard_lr8e8_stage1_arch_plus_after_ext_20260623_host2_opt_sched_loss_plus_20260623_154020`.
    - S1-NP MARS + plateau center-guard lane on GPU1, PID `1020986`, run `stage1_s1np_mars_plateau_centerguard_lr5e8_stage1_arch_plus_after_ext_20260623_host2_opt_sched_loss_plus_20260623_154020`.
    - Final reporter: both lanes reached epoch `50/50`, status `keep_baseline_unless_later_improves`; S1-NO P10 `28.04133031953056`, P5 `9.262129657673386`, center `17.31566175424828`; S1-NP P10 `28.249102106634176`, P5 `9.24528328877575`, center `17.311815576733284`; no promotion.
    - Runtime evidence: no error signature was found.
    - `distill_ext` completed `50/50` with no promotion: S1-NQ and S1-NR matched baseline P10 `28.27717937613433`, but P5 and center were worse.
    - `loss_ext` completed `50/50` with no promotion: S1-NS and S1-NT matched baseline P10 `28.27717937613433`, but P5 and center were worse.
    - `scheduler_ext` completed `50/50` with no promotion: S1-NU and S1-NV best P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.31507064711373`.
    - Active `p10_recover_ext`: S1-NW is at epoch `40/50` with best P10 `28.232255755730396`, P5 `9.380054185975272`, center `17.308769406012768`; S1-NX is at epoch `36/50` with best P10 `28.27717937613433`, P5 `9.380054185975272`, center `17.29898632697339`.
    - Promotion decision for `p10_recover_ext` is blocked until both lanes reach `50/50`.

### 2026-06-23 Requested Axis Coverage Verification

The current queue covers all user-requested Stage1 Search experiment axes:

- Optimizer:
  - `opt_sched_loss`: Lion and Adopt.
  - `opt_sched_loss_ext`: AdamW schedule-free/cautious and MuSGD.
  - `opt_sched_loss_plus`: AdEMAMix and MARS.
- Loss:
  - `loss_ext`: center/P5 loss-only and geometry-balanced loss-only probes under fixed AdamW + cosine.
- Scheduler:
  - `scheduler_ext`: plateau and step schedulers under fixed AdamW/head/loss settings.
- Data augmentation:
  - `augmentation`: frame-only and frame+event augmentation.
  - `augmentation_ext`: mild frame-only augmentation and event robustness stress.
  - `opt_loss_sched_aug_guard`: micro frame/event augmentation combined with optimizer/loss/scheduler guard settings.

Execution policy:

- Do not start another fresh default suite chain while these host-visible chains are alive.
- Continue monitoring the active/queued chain order and evaluate each lane only after the `50/50` gate.

### 2026-06-23 P10-Recover And Interpolation Closeout

- `p10_recover_ext` reached `50/50`; no promotion:
  - S1-NW: P10 `28.232255755730396`, P5 `9.380054185975272`, center `17.308769406012768`.
  - S1-NX: P10 `28.27717937613433`, P5 `9.380054185975272`, center `17.29898632697339`.
- P10-anchor checkpoint interpolation was evaluated with `scripts/external/run_stage1_frame_search_p10_anchor_interp_eval.sh`.
- Summary file: `runs/NON_XR/shared/eval/stage1_p10_anchor_interp_after_recover_20260623/summary.json`.
- Decision: no interpolated checkpoint met promotion criteria. P10-only branches remain diagnostic signals, not baseline replacements.
- Next planned suite: `opt_loss_sched_aug_guard`, then output-level gated ensemble/calibration if this suite also fails.

### 2026-06-23 Combined Guard Suite Closeout

- `opt_loss_sched_aug_guard` reached `50/50`; no promotion:
  - S1-OA: P10 `28.27717937613433`, P5 `9.24528328877575`, center `17.302655732856607`.
  - S1-OB: P10 `28.23225573773654`, P5 `9.380054185975272`, center `17.3148428808968`.
- Decision: keep current baseline. This closes the requested optimizer/loss/scheduler/augmentation axis sweep under the current fine-tune recipe.
- Next mechanism: output-level ensemble/calibration. Do not launch another broad hyperparameter chain unless a new diagnostic explains how it will avoid the repeated center/P5 regression pattern.

### 2026-06-23 Learned Output Gate Closeout

- Output-level static ensemble:
  - Summary: `runs/NON_XR/shared/eval/stage1_output_ensemble_p10_anchors_after_guard_20260623/summary.json`.
  - Result: no static weighted ensemble exceeded the validation P10 gate.
- Oracle upper bound:
  - Summary: `runs/NON_XR/shared/eval/stage1_output_ensemble_p10_anchors_oracle_after_guard_20260623/summary.json`.
  - Oracle validation: P10 `30.908581067930978`, P5 `12.384321968510466`, center `16.39156784201568`.
  - Interpretation: checkpoint outputs are complementary at sample level.
- Learned gate:
  - Script: `scripts/external/train_stage1_search_output_gate.py`.
  - Summary: `runs/NON_XR/shared/gate/stage1_output_gate_p10_anchors_50ep_20260623/summary.json`.
  - Validation hard gate: P10 `27.83692722371966`, P5 `10.222371967654992`, center `17.400613902092832`.
  - Validation soft gate: P10 `28.039083557951468`, P5 `9.67430368373765`, center `17.31651116715608`.
  - Reference validation base: P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
  - Reference validation S1-NF: P10 `28.3501796945193`, P5 `8.835354896675652`, center `17.394714167304436`.
  - Split audit: train/val/test overlap count `0`.
- Decision:
  - No promotion. Shallow output-state-only gate fails to recover the oracle headroom.
  - Next experiments should add richer gate evidence rather than repeat broad optimizer/loss/scheduler/augmentation sweeps:
    - confidence/context gate using image/event features, residual error proxies, and mask/quality metadata;
    - model-integrated calibration head trained end-to-end with supervised oracle labels;
    - temperature/top-k mixture gate with validation-selected regularization to avoid over-selecting P5/center anchors.

### 2026-06-24 Confidence/Context Gate Closeout

- Implemented richer post-hoc gate features in `scripts/external/train_stage1_search_output_gate.py`.
  - `output_context`: runtime frame/event summary statistics plus output-state features.
  - `output_conf`: main Search confidence, aux bbox/OBB confidence/agreement, pooled summary statistics plus output-state features.
  - `output_conf_context`: combined feature set.
  - `softmin_center`: train-only soft target distribution from center error.
- Full 50-epoch result:
  - Summary: `runs/NON_XR/shared/gate/stage1_output_conf_context_softmin_gate_50ep_20260624/summary.json`.
  - Feature dim `208`, feature mode `output_conf_context`, target mode `softmin_center`.
- Validation:
  - base reference: P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
  - S1-NF reference: P10 `28.3501796945193`, P5 `8.835354896675652`, center `17.394714167304436`.
  - hard gate: P10 `26.69923629829289`, P5 `9.935983827493263`, center `17.358085787071374`.
  - soft gate: P10 `28.18171608265946`, P5 `8.876909254267746`, center `17.321360889128062`.
- Decision:
  - No promotion. Richer post-hoc features and soft labels still do not clear the validation P10/center gate.
- Active/planned next experiments:
  - `S1-OC`: model-integrated calibration/head-selection branch trained end-to-end.
  - `S1-OD`: regularized top-k mixture gate with entropy and selection-balance terms.
  - `S1-OE`: confidence-only ablation to isolate whether input context hurts validation P10.
  - `XR-64A/B/C`: keep as the broader second-goal teacher-target construction queue, but it is a separate Stage2/target-override path rather than the immediate Stage1 frame-search baseline path.

### 2026-06-24 S1-OE/S1-OD Gate Family Closeout

- S1-OE confidence-only ablation:
  - `output_conf + softmin_center`: `runs/NON_XR/shared/gate/stage1_output_conf_softmin_gate_50ep_20260624/summary.json`.
    - Validation soft: P10 `28.06379155435757`, P5 `8.876909254267746`, center `17.320374500826468`.
  - `output_conf + oracle_ce`: `runs/NON_XR/shared/gate/stage1_output_conf_oraclece_gate_50ep_20260624/summary.json`.
    - Validation soft: P10 `27.74707996406108`, P5 `9.67430368373765`, center `17.3249394708995`.
- S1-OD regularized top-k gate:
  - Script support added in `scripts/external/train_stage1_search_output_gate.py`: `--topk`, `--entropy-weight`, and `--balance-weight`.
  - `output_conf + softmin_center + topk2`: `runs/NON_XR/shared/gate/stage1_output_conf_topk2_reg_gate_50ep_20260624/summary.json`.
    - Validation hard: P10 `27.14285714285713`, P5 `9.548517520215634`, center `17.332235361280897`.
    - Validation soft: P10 `28.18171608265946`, P5 `8.876909254267746`, center `17.318211609591046`.
    - Validation top-k: P10 `27.109164420485165`, P5 `9.138589398023361`, center `17.344294708651248`.
  - `output_conf_context + softmin_center + topk2`: `runs/NON_XR/shared/gate/stage1_output_conf_context_topk2_reg_gate_50ep_20260624/summary.json`.
    - Validation hard: P10 `27.305705300988304`, P5 `8.767969451931716`, center `17.390973981692788`.
    - Validation soft: P10 `28.06379155435757`, P5 `8.876909254267746`, center `17.322542203361674`.
    - Validation top-k: P10 `27.85040431266845`, P5 `9.168912848158133`, center `17.362701884669868`.
- Decision:
  - No promotion. Baseline validation reference remains P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
  - Frozen-output post-hoc gate experiments are now closed. The oracle remains a diagnostic upper bound, but learned post-hoc gates did not convert it into a validation baseline.
  - Next high-value work should be either:
    - `S1-OC`: integrated calibration/head-selection branch trained end-to-end inside the Stage1 model; or
    - `XR-64A/B`: leakage-safe teacher-target construction in the broader second-goal plan.

### 2026-06-25 S1-OC Implementation And Documentation Status

- Current progress has been consolidated in:
  `docs/resources/current_stage1_work_progress_2026_06_25.md`.
- S1-OC is now implemented as an integrated Stage1 Search candidate branch, not as a frozen-output post-hoc gate.
- Implemented surfaces:
  - `src/hbtxr/models/heads.py`: `SearchCenterCandidateHead`.
  - `src/hbtxr/models/tracker/head_factory.py`: optional Search candidate head construction.
  - `src/hbtxr/models/hybrid_tracker.py`: Search candidate config passthrough.
  - `src/hbtxr/models/tracker/search_branch.py`: candidate xy/logit/weight outputs and soft mixture routing into `search/state`.
  - `src/hbtxr/loss/bundles/search_event.py`: `pupil_center_candidate_losses`.
  - `src/hbtxr/loss/stage1.py`: Stage1 candidate loss integration and Search soft-threshold losses.
  - `configs/external/base.yaml`: default-disabled Search residual/candidate config keys.
  - `scripts/external/run_stage1_s1oc_search_candidate.sh`: two-lane S1-OC launcher.
- S1-OC lane plan:
  - S1-OC-A: train only `search_center_candidate_head.*`, LR `5e-5`, candidate blend `1.0`.
  - S1-OC-B: train `search_center_candidate_head.*` plus `search_head.residual.*`, LR `2e-5`, candidate blend `0.75`.
  - Both lanes run for `50` epochs and use the current baseline checkpoint plus P10/P5/center teacher ensemble.
- Validation completed:
  - Python compile for touched model/loss files.
  - Shell syntax checks for `run_stage1_s1oc_search_candidate.sh` and the shared ensemble-teacher launcher.
  - Dry-run command materialization.
  - Dummy CPU forward/loss smoke verifying candidate outputs and finite `loss_total`.
- Execution status:
  - No S1-OC run directory exists under `runs/NON_XR/raw`.
  - No S1-OC log exists under `runs/NON_XR/shared/_logs`.
  - The 50-epoch GPU launch has not started because the escalated launch attempt was aborted.
- Next action:
  - Launch S1-OC A/B with host GPU access and hold promotion judgment until both lanes reach `50/50`.
