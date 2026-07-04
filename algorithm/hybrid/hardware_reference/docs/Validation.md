# HGTXR-SW Validation Plan

Date: 2026-06-10

## 2026-06-27 Documentation And Handover Validation

Documentation artifacts created:

- `docs/resources/current_project_synthesis_2026_06_27.md`
- `docs/HANDOVER_2026_06_27.md`
- `HANDOVER.md`

Read-only S1-OC artifact checks:

```bash
find runs/NON_XR/raw -maxdepth 1 -type d -name '*s1oc*'
find runs/NON_XR/shared/_logs -maxdepth 1 -type f -name '*s1oc*'
```

Observed state during this pass:

```text
no S1-OC run directory found
no S1-OC log file found
```

Quality gate for the handover:

- Baseline run/checkpoint and P10/P5/center gates are present.
- S1-OC state is split into implemented, validated, and executed.
- The handover does not claim S1-OC was trained.
- Continuation command and mandatory pre-checks are included.
- Stage1 Search gates are not mixed with Stage2/XR gates.
- Closed post-hoc gate family is marked closed unless a new mechanism is introduced.

## 2026-06-22 Stage1 Promoted Baseline Validation

Stage1 frame-search promotion checks:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --baseline runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 \
  --min-epochs 50 \
  --target-epochs 80 \
  --followup runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 \
  --followup runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 \
  --format summary
.venv/bin/python scripts/external/write_stage1_frame_search_baseline_manifest.py --format summary
python3 -m json.tool docs/resources/current_stage1_frame_search_baseline_manifest.json
```

Observed promotion:

```text
state=promoted_followup
active_source=followup
active_checkpoint=runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt
best_search_p10=28.27717937613433
best_search_p5=10.153863744915656
best_search_center_px=17.257583906065744
```

XR-64 downstream readiness after prep and A/B/C execution:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary
.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary
```

Observed state:

```text
resume_status=ready_to_train
ready_to_train=true
missing_eval_rows=0
missing_overrides=0
candidate_count=6
required_candidate_matrix_complete=true
```

Latest runtime refresh at `2026-06-22 03:18:35 KST`:

```text
lane_i_no_distill=80/80
lane_j_self_distill_reporter=80/80
lane_j_self_distill_manifest=80/80
xr64_ready_to_train=true
xr64_missing_eval_rows=0
xr64_missing_overrides=0
gpu0_stage1_active=false
gpu1_stage1_active=false
```

XR-65 P10/P5 recovery runner checks:

```bash
bash -n scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh b cuda:1
.venv/bin/python -m pytest -q tests/test_xr65_runner_contract.py
```

Expected contract:

```text
XR65_DRY_RUN:1
checkpoint kinds: best_track_p10, best_track_p5, best_metric_track_center_px
test eval target override: null
test eval allow_test_target_override: false
test eval target override loss: 0.0
```

Observed XR-65A/B execution result:

```text
XR65_TRAIN_EXIT:0 for lane a and lane b
XR65_DONE eval_count=3 for lane a and lane b
best XR-65 center=16.464880844524927
best XR-65 P10=34.31505176680429
best XR-65 P5=12.08248336655753
promotion=false
```

Closeout artifact:

```text
docs/resources/xr65_p10p5_recovery_results_2026_06_22.md
```

XR-66 no-train diagnostic artifacts:

```text
runs/diagnostics/xr66_failure_buckets_xr64c_bestp10_20260622.json
runs/diagnostics/xr66_failure_buckets_xr65a_bestp5_20260622.json
docs/resources/xr66_no_train_failure_bucket_diagnostic_2026_06_22.md
```

Observed XR-66 diagnostic interpretation:

```text
low-similarity bucket remains hard
worst subjects: 42, 39, 45
worst sessions include user45/right/session_201 and user42/session_201
left eye remains weaker than right eye
generic confidence/local-update training is not justified before targeted bucket comparison
```

XR-66 targeted manifest artifacts:

```text
data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/train_manifest.jsonl: 1624 rows
data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/val_manifest.jsonl: 254 rows
data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl: 493 rows
data/_internal/manifests/manifest1/xr66_failure_buckets/user45_right_session201_testonly/test_manifest.jsonl: 31 rows
data/_internal/manifests/manifest1/xr66_failure_buckets/user42_left_session201_testonly/test_manifest.jsonl: 46 rows
data/_internal/manifests/manifest1/xr66_failure_buckets/user42_right_session201_testonly/test_manifest.jsonl: 46 rows
```

XR-66 manifest caveat:

```text
worst-session buckets are test-only in the current split
only low_similarity_le0p1 is currently usable as a train/val/test targeted subset
next action is targeted no-train comparison before any XR-66 training launch
```

XR-66 targeted comparison and low-similarity recovery validation:

```bash
python3 -m py_compile scripts/external/compare_xr66_targeted_buckets.py
bash -n scripts/external/run_xr66_lowsim_targeted_recovery.sh
DRY_RUN=1 bash scripts/external/run_xr66_lowsim_targeted_recovery.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr66_lowsim_targeted_recovery.sh b cuda:1
.venv/bin/python scripts/external/compare_xr66_targeted_buckets.py --format summary
bash scripts/external/run_xr66_lowsim_targeted_recovery.sh a cuda:0
bash scripts/external/run_xr66_lowsim_targeted_recovery.sh b cuda:1
```

Observed XR-66A/B execution result:

```text
XR66_TRAIN_EXIT:0 for lane a and lane b
XR66_DONE eval_count=6 for lane a and lane b
epochs=50/50 for lane a and lane b
best XR-66 full-test center=16.498384244101388
best XR-66 full-test P10=34.58631029129028
best XR-66 full-test P5=11.964286068507603
best XR-66 low-sim P10=36.36136802550285
promotion=false
```

XR-66 closeout artifacts:

```text
runs/diagnostics/xr66_targeted_bucket_comparison_20260622.json
docs/resources/xr66_targeted_bucket_comparison_2026_06_22.md
docs/resources/xr66_lowsim_targeted_recovery_results_2026_06_22.md
scripts/external/compare_xr66_targeted_buckets.py
scripts/external/run_xr66_lowsim_targeted_recovery.sh
```

## 2026-06-21 PAPER_REF Experiment Trace

This validation is read-only and must not launch train/eval jobs.

```bash
python3 -m json.tool docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json
python3 -m py_compile scripts/external/check_second_goal_paper_ref_experiment_trace.py
.venv/bin/python scripts/external/check_second_goal_paper_ref_experiment_trace.py --format summary
.venv/bin/python scripts/external/check_paper_ref_analysis_coverage.py --format summary
.venv/bin/python -m pytest -q tests/test_second_goal_paper_ref_experiment_trace.py
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_paper_ref_experiment_trace.py
```

Historical expected state at artifact creation:

```text
execution_state: paused_by_user_directive
experiment_count: 8
paper_group_count: 6
xr64_ready_to_train: false
missing_eval_rows: 8
missing_overrides: 6
completion_allowed: false
direct_submission_comparison_allowed: false
authority_links_missing: 0
```

## Static Validation

```bash
bash -n scripts/external/run_accuracy_experiment_matrix.sh
.venv/bin/python -m py_compile scripts/external/summarize_training_history.py
.venv/bin/python -m py_compile scripts/external/check_event_count_override_sanity.py
```

## 2026-06-18 Pause-State Documentation Validation

Experiments are paused by user directive. These checks are read-only and do not
launch train/eval jobs.

```bash
python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py
python3 -m py_compile scripts/external/report_second_goal_status.py
.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
```

Observed status summary:

```text
execution_state: paused_by_user_directive
active_goal_complete: false
authority_links_missing: 0
xr64_resume_status: generated_incomplete
xr64_ready_to_train: false
xr64_can_run_lane: false
xr64_missing_eval_rows: 8
xr64_missing_overrides: 6
xr64_leakage_risk: none
```

Interpretation: current documentation and preconditions are coherent, but the
second goal remains incomplete until XR-64 train/val eval rows, override JSONs,
and follow-up training/evaluation are produced after explicit user resume.

## 2026-06-18 Software Cleanup Inventory Validation

Cleanup/refactor preparation is read-only in this pass. No files were deleted or
moved.

```bash
python3 -m py_compile scripts/external/maintenance/inventory_software_tree.py
python3 -m py_compile scripts/external/maintenance/audit_path_references.py
.venv/bin/python scripts/external/maintenance/inventory_software_tree.py
.venv/bin/python scripts/external/maintenance/audit_path_references.py
```

Observed inventory:

```text
runs=849
software_dirty_entries=236
runs_size=44.5 GiB
path_reference_findings=1166
```

Cleanup decision:

- `runs/` requires preserve/archive manifests before deletion.
- `scripts/external` and `configs/external` should use wrapper/path policy before physical moves.
- `anlaysis` path spelling must be preserved until links are migrated.

## 2026-06-18 Runs Catalog Validation

`runs/` is ignored by root gitignore and was kept as local experiment output.
Original run directories were not moved. The catalog is a symlink view only.

```bash
python3 -m py_compile scripts/external/maintenance/build_runs_catalog.py
.venv/bin/python scripts/external/maintenance/build_runs_catalog.py --dry-run
.venv/bin/python scripts/external/maintenance/build_runs_catalog.py
find runs/_catalog -maxdepth 2 -type l | wc -l
```

Observed result:

```text
source_run_dirs=849
catalog_symlinks=2078
categories=13
```

Catalog categories include:

- `runs/_catalog/00_current_authority`
- `runs/_catalog/01_xr64_resume`
- `runs/_catalog/02_leaders_teachers_review`
- `runs/_catalog/03_shared_support`
- `runs/_catalog/30_eval_runs`
- `runs/_catalog/31_raw_stage_runs`
- `runs/_catalog/32_raw_stage2_runs`
- `runs/_catalog/33_raw_stage1_runs`
- `runs/_catalog/80_smoke_cleanup_candidates`
- `runs/_catalog/81_duplicate_review_candidates`
- `runs/_catalog/90_archive_candidates`
- `runs/_catalog/91_smoke_candidates`
- `runs/_catalog/99_preserve_or_review`

## 2026-06-18 Runs Experiment Reorganization Validation

User requested physical organization of run directories by experiment name while
keeping `runs/` as local experiment output. This pass moved run contents and left
compatibility symlinks at original paths.

```bash
python3 -m py_compile scripts/external/maintenance/reorganize_runs_by_experiment.py
.venv/bin/python scripts/external/maintenance/reorganize_runs_by_experiment.py --dry-run
.venv/bin/python scripts/external/maintenance/reorganize_runs_by_experiment.py
find runs -maxdepth 1 -mindepth 1 -type l | wc -l
find runs -xtype l | head -20
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
```

Observed result:

```text
source_dirs_considered=849
moved_with_compat_symlink=849
top_level_compat_symlinks=849
broken_symlinks=0
xr64_resume_status=generated_incomplete
```

Post-review fix:

- `scripts/external/run_prepare_and_train.sh` now includes top-level compatibility
  symlinks in `latest_run_root()` discovery by checking `-type d -o -type l`.
- `scripts/external/maintenance/reorganize_runs_by_experiment.py --verify` writes a
  dedicated verification report and checks broken/bad-target symlinks.

Verification report:

- `docs/resources/runs_experiment_reorganization_verify_2026_06_18.md`
- `docs/resources/runs_experiment_reorganization_verify_2026_06_18.json`

Observed verification:

```text
verify ok=true
top_level_compat_symlinks=849
organized_run_dirs=849
broken=0
bad_targets=0
```

Top-level run organization now uses:

- `runs/XR-<n>/<run_id>` for runs with an `xrNN` token.
- `runs/NON_XR/eval/<run_id>` for non-XR eval runs.
- `runs/NON_XR/raw/<run_id>` for non-XR raw stage runs.
- `runs/NON_XR/shared/<name>` for `_logs`, `diagnostics`, and `interpolated_checkpoints`.
- `runs/NON_XR/other/<run_id>` for remaining non-XR runs.

`runs/_catalog` was treated as generated metadata and was not moved.

## Config Validation

```bash
PYTHONPATH=src .venv/bin/python -c "from scripts.v3._config import load_config; [print(p, load_config(p)['experiment']['name']) for p in ['configs/external/mode1_stage2_raw_event_count_lr2e-4_nodistill_fullwidth.yaml','configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_fullwidth.yaml','configs/external/mode1_stage2_raw_event_count_da_roi_lr2e-4_nodistill_fullwidth.yaml']]"
```

Contract validation for 5000-count raw configs:

```bash
.venv/bin/python scripts/external/check_raw_event_count_contract.py \
  --stage2-config configs/external/mode1_stage2_raw_event_count_lr2e-4_nodistill_fullwidth.yaml
```

## Run Validation

After any full Stage2 ablation:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_result.py \
  --stage1-run runs/raw_mode1_stage1_event_count_20260610_192838 \
  --stage2-experiment-name <experiment_name>
```

Event-count override sanity:

```bash
PYTHONPATH=src .venv/bin/python scripts/external/check_event_count_override_sanity.py \
  --config runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_205627/hypers/resolved_config.json \
  --manifest data/_internal/manifests/manifest1/train_manifest.jsonl \
  --counts 1000 2500 5000 \
  --index 100
```

Metric coordinate-frame sanity:

```bash
PYTHONPATH=src .venv/bin/python scripts/external/check_metric_coordinate_sanity.py \
  --config runs/raw_mode1_stage2_event_count_20260610_193719/hypers/resolved_config.json \
  --limit 10000
```

## Decision Rule

Promote an experiment only if it improves both:

- validation `metric_track_p10_pct` over `7.2653`
- validation `metric_track_center_px` below `43.9917`

If P10 improves but center error worsens, inspect qualitative outputs before promotion.

## Experiment Tracking Validation Rules

Imported reference:

- `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md`

Current promotion gates supersede the early baseline thresholds above once a
newer leader is established. For each experiment, record and validate:

1. Resolved config, not only run-directory name.
   - Include actual LR, optimizer, event count, adaptive/fixed-count state,
     checkpoint init, active heads, and trainable filters.
2. Validation and test leaders separately.
   - Do not promote from validation alone.
   - Evaluate best-center and best-P10 checkpoints when they differ.
3. Center, P10, and P5 separately.
   - Center leader and P10 leader can diverge. Track both.
4. First-best epoch versus final epoch.
   - If best tracking appears at epoch 1-5 and later epochs degrade, mark the
     run as early-peak and do not extend blindly.
5. Failure bucket or stop reason.
   - Examples: LR too high, stale queue threshold, crop/target mismatch, raw
     event tuple artifact risk, coordinate-frame mismatch, geometry-loss
     domination, no test transfer.
6. Stale threshold guard.
   - Queue scripts must compute current leader thresholds from available eval
     summaries or be updated when a leader changes.
7. Coordinate contract check.
   - Any crop, resize, previous-pupil anchor, mask-centroid routing, or
     interpolation change must document target coordinate frame and transform
     contract before training is promoted.

## Validation Results

Baseline reference:

- `runs/raw_mode1_stage2_event_count_20260610_193719`: best val `metric_track_p10_pct=7.2653`, final val `metric_track_center_px=43.9917`.

Completed ablations:

- `raw_mode1_stage2_lr2e-4_nodistill_fullwidth`: result checker `ok=true`; best val `metric_track_p10_pct=8.8926`, best val `metric_track_center_px=44.2296`.
- `raw_mode1_stage2_lr1e-4_weakdistill_fullwidth`: result checker `ok=true`; best val `metric_track_p10_pct=13.8646`, best val `metric_track_center_px=44.8683`; pre-fix reference only because event tensors were zero before `src/hbtxr/data/event_builder.py` was fixed.

Sampled and stopped:

- `raw_mode1_stage2_da_roi_lr2e-4_nodistill_fullwidth`: stopped at epoch 7; best val `metric_track_p10_pct=4.6597`, best val `metric_track_center_px=53.6560`.

Metric sanity:

- `metric_track_center_px` is measured in post-transform input coordinates, not direct sensor pixels.
- For current 256x256 ROI-direct runs, 10 input px corresponds roughly to median `8.24` sensor px on x and `5.16` sensor px on y for checked val samples.
- Direct comparison with the submission `0.1812 px` target is invalid until evaluation protocol/coordinate frame is matched.

Event sanity after fix:

- Root cause fixed: `accumulation_weights()` now uses float64 timestamp arithmetic, and fixed-count windows drop manifest `start_timestamp_us`.
- Validation command passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_event_builder.py tests/test_raw_event_count_training_result.py tests/test_raw_event_count_train_wrapper.py`
- `check_event_count_override_sanity.py` confirmed nonzero event tensors and count-dependent event input for train index 100.

2026-06-16 XR-04B/XR-04C closeout validation:

- Training logs:
  - `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log`
  - `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log`
- Both logs show raw event-count contract pass, XR-03D best-P10 checkpoint load, intended CUDA device resolution, epoch `1/12` entry, clean train exit, and test eval exit code `0`.
- Parsed 4 eval summaries with gate comparison against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`; all `promote=False`.
- Best observed focused branch: XR-04B best-center `20.484137114456722/25.600340850012643/8.277636350904192`, still worse than all active gates.
- XR-04D/XR-04E manifest-subset validation:
  - `python3 -m py_compile scripts/external/build_failure_bucket_manifest.py` passed.
  - `bash -n scripts/external/run_xr04d_failbucket_subset_probe.sh` passed.
  - Subset generation produced train `1186` and val `182` rows for `similarity_target <= 0.1` and `session_201`.
  - XR-04D/XR-04E launch logs show train_samples `1186`, val_samples `182`, intended CUDA devices, and epoch `1/12` entry.

Corrected event-count sweep:

- `runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_210856`: result checker `ok=true`; best val `metric_track_p10_pct=14.1543`, best val `metric_track_center_px=44.8353`; best P10 among corrected count sweep.
- `runs/raw_mode1_stage2_count2500_lr1e-4_weakdistill_fullwidth_20260610_210855`: result checker `ok=true`; best val `metric_track_p10_pct=13.6961`, best val `metric_track_center_px=44.6096`; better center than count1000, lower P10.
- `runs/raw_mode1_stage2_lr1e-4_weakdistill_fullwidth_20260610_212149`: result checker `ok=true`; expected count `5000`; best val `metric_track_p10_pct=13.7298`, best val `metric_track_center_px=44.5356`; better center than count1000/count2500, lower P10.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_fullwidth_20260610_212150`: result checker `ok=true`; expected count `10000`; best val `metric_track_p10_pct=13.8702`, best val `metric_track_center_px=43.3363`; best center among corrected count sweep, lower P10 than count1000.

Center-aware count10000 experiments:

- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707`: result checker `ok=true`; best val `metric_track_p10_pct=13.8702`, best val `metric_track_center_px=39.5347`; best corrected center-error result so far.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709`: result checker `ok=true`; best val `metric_track_p10_pct=14.3430`, best val `metric_track_center_px=39.8471`; best corrected P10 result so far, but weaker center error than the center-checkpoint control.

Test split validation:

- Added eval-side `HBTXR_DISABLE_CUDNN=1` handling in `scripts/external/eval_hbtxr.py`; validated with `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_train_hbtxr_runtime_env.py`.
- `runs/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803`: checkpoint epoch `28`; test `metric_track_center_px=36.9280`, test `metric_track_p10_pct=9.9209`, test `metric_track_p5_pct=2.9307`.
- `runs/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812`: checkpoint epoch `1`; test `metric_track_center_px=42.3723`, test `metric_track_p10_pct=10.9864`, test `metric_track_p5_pct=3.2819`.
- `runs/eval_centerloss_bestcenter_test_gpu1_20260610_230859`: checkpoint epoch `23`; test `metric_track_center_px=36.5813`, test `metric_track_p10_pct=10.2759`, test `metric_track_p5_pct=3.2551`. This is the current best test-center result and also improves P10 over the center-checkpoint best-center test.

Active validation watch:

- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth_20260610_222436`: `causal_weight_power=2.0` recency-sharpening run was stopped after epoch 10 validation. Best val center was `44.6194`, worse than the count10000 center baseline `43.3363` and the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243` was stopped after epoch 10 validation. Best val center was `44.2607`, still worse than the count10000 center baseline `43.3363` and the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120` has no active process now. Last observed log reached epoch 13 train; best val center was `43.3086`, still not competitive with the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355` has no active process now. Best useful val center was `44.5171`; not promoted.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302` was stopped as noncompetitive. Epoch 8 best val center was `44.6362`, still above the epoch-8 stop threshold `42.0`; epoch 9 briefly reached `43.9921` but remained worse than the plain center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230041` was stopped after epoch 8/9 gate. Best val center was `44.9089`, so center-checkpointing did not recover the count1000 center error.
- `runs/raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218` was stopped after epoch 8 gate. Best val center was `44.8672`, so the intermediate count did not approach the center leader.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_strong_fullwidth_20260610_231233` was stopped after epoch-8 gate. Best val center was `44.1654`, so stronger center-loss remained worse than the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_schedulefree_fullwidth_20260610_232114` was stopped after epoch-8 gate. Best val center was `45.3146`, so schedule-free AdamW did not beat the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_231556` was stopped after epoch-9 observation. Best val center was `44.3874`, so the count5000 centerloss branch remained worse than the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr3e-5_weakdistill_centerckpt_lion_fullwidth_20260610_232431` was stopped after epoch-8 gate. Best val center was `43.4448`, so Lion at LR `3e-5` did not beat the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_teachercenter_fullwidth_20260610_233212` was stopped after epoch-8 gate. Best val center was `44.0106`, so teacher initialization from the center-checkpoint leader did not beat the center-checkpoint leader `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233558` was aborted during epoch 1 because search/event OBB aux weights `0.20/0.30` pushed early loss to about `19k`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233657` was stopped after epoch-8 gate. Best val center was `44.4728`, above the stop threshold `42.0`; light FACET-style OBB auxiliary supervision did not beat the center-checkpoint leader.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_selfreg_fullwidth_20260610_234009` was stopped after epoch-9 gate. Best val center was `43.5076`, above the stop threshold `42.0`; local-global self-regularization did not beat the center-checkpoint leader.
- `runs/raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649` completed by early stop at epoch 9. Best val center was `40.0042`, so it did not beat the centerloss leader `39.8471`; no test promotion.
- `runs/raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952` completed by early stop at epoch 9. Best val center was `39.9555`, so it did not beat the centerloss leader `39.8471`; no test promotion.
- `runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth_20260610_235630` completed by early stop at epoch 9. Best val center was `39.9155`, so it did not beat the centerloss leader `39.8471`; no test promotion.
- `runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth_20260611_000139` completed by early stop at epoch 9. Best val center was `39.9662`, so it did not beat the centerloss leader `39.8471`; no test promotion.
- `runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerckptinit_fullwidth_20260611_000343` completed by early stop at epoch 8. Best val center was `39.7478`, which is the current best validation-center result. Test eval `runs/eval_trackonly_centerckptinit_bestcenter_test_gpu1_w0_20260611_000907` produced test center `37.2887`, P10 `10.5884`, P5 `2.6990`; this improves P10 versus the current test-center leader but does not beat test center `36.5813`.
- `runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerckptinit_fullwidth_20260611_000343` best-P10 checkpoint test eval `runs/eval_trackonly_centerckptinit_bestp10_test_gpu0_w0_20260611_002015` produced test center `37.1481`, P10 `10.9864`, P5 `3.4881`; it improves P10/P5 but not test center.
- `runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerckptinit_fullwidth_20260611_001034` completed by early stop at epoch 8. Best val center was `39.5315`, which is the current best validation-center result. Best-center test eval produced center `37.0766`, P10 `10.2194`, P5 `2.8533`; best-P10 test eval produced center `37.2173`, P10 `10.2755`, P5 `3.5285`. Neither beats test center `36.5813`.
- Interpolation artifacts:
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.25.pt`: test center `36.7546`, P10 `9.0094`, P5 `2.9928`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.50.pt`: test center `38.6547`, P10 `5.4537`, P5 `1.5480`.
- `runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth_20260611_004917` early-stopped at epoch 8 with best val center `39.5598`; not promoted versus validation leader `39.5315`.
- `runs/raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943` early-stopped at epoch 8 with best val center `39.5780`; not promoted versus validation leader `39.5315`.
- Centerloss/center-checkpoint interpolation artifacts:
  - `runs/interpolated_checkpoints/centerloss_bestcenter__centerckpt_bestcenter_alpha0.25.pt`: test center `36.7547`, P10 `8.9222`, P5 `2.9418`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__centerckpt_bestcenter_alpha0.50.pt`: test center `38.6246`, P10 `5.8129`, P5 `1.4673`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__centerckpt_bestcenter_alpha0.75.pt`: test center `38.1375`, P10 `6.4111`, P5 `1.5455`.
- `runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerloss_fullwidth_20260611_010010` was stopped after epoch-10 gate. Best val center was `44.6844`, above stop threshold `42.0`; no promotion.
- Added `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_centerloss_finetune_fullwidth.yaml`. Config load resolved experiment `raw_mode1_stage2_lr2e-6_nodistill_centerloss_finetune_fullwidth`, LR `2e-06`, distillation disabled, all heads active. Raw event-count contract passed.
- `runs/raw_mode1_stage2_count10000_lr2e-6_nodistill_centerloss_finetune_fullwidth_20260611_011131` early-stopped at epoch 6. Best val center was `39.8800`, so it was not promoted.
- Added `training.trainable.include/exclude` support in `src/hbtxr/training/trainer.py`; `tests/test_trainable_filter.py` and `tests/test_train_hbtxr_runtime_env.py` passed.
- Added `configs/external/mode1_stage2_raw_event_count_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth.yaml`. Config load resolved LR `1e-05`, active head `track`, trainable include `track_head.*`. Raw event-count contract passed.
- `runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth_20260611_011606` early-stopped at epoch 6. Best val center was `39.9387`, so it was not promoted.
- Added `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`. Config load resolved LR `5e-06`, active head `track`, trainable include `track_head.*`, `event_adapter.*`, `patch_frontend.event_embed.proj.*`. Raw event-count contract and wrapper dry-run passed.
- `runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth_20260611_012329` early-stopped at epoch 8. Best val center was `39.7622`, so it was not promoted versus validation leader `39.5315`.
- Added `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth.yaml`. Config load resolved LR `2e-06`, active head `track`, trainable include `track_head.*`, `backbone.attn_stages.5.*`, `backbone.mlp_stages.5.*`, `backbone.norm.*`. Raw event-count contract and wrapper dry-run passed.
- `runs/raw_mode1_stage2_count10000_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth_20260611_012629` early-stopped at epoch 6. Best val center was `39.8832`, so it was not promoted versus validation leader `39.5315`.
- Added `configs/external/mode1_stage2_raw_event_count_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth.yaml`. Config load resolved LR `3e-06`, active head `track`, trainable include `track_head.*`, `frame_adapter.*`, `event_adapter.*`, `patch_frontend.frame_embed.proj.*`, `patch_frontend.event_embed.proj.*`. Raw event-count contract and wrapper dry-run passed.
- `runs/raw_mode1_stage2_count10000_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth_20260611_012935` early-stopped at epoch 8. Best val center was `39.7822`, so it was not promoted versus validation leader `39.5315`.
- Small/fine interpolation artifacts from centerloss best-center to track-only centerckpt-init LR `5e-6` best-center:
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.05.pt`: test center `36.5083`, P10 `10.1156`, P5 `3.1424`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.075.pt`: test center `36.4907`, P10 `9.9179`, P5 `3.1807`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.10.pt`: test center `36.4873`, P10 `10.0880`, P5 `2.9213`. This is the current test-center leader.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.125.pt`: test center `36.4968`, P10 `10.1679`, P5 `2.7619`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.15.pt`: test center `36.5190`, P10 `9.9320`, P5 `2.8312`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.175.pt`: test center `36.5556`, P10 `9.5344`, P5 `2.8950`.
- Added reusable eval runner `scripts/external/run_interp_eval_series.sh`; `bash -n scripts/external/run_interp_eval_series.sh` passed.
- Micro interpolation around alpha `0.10`:
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.090.pt`: test center `36.4872`, P10 `9.9775`, P5 `3.0702`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.095.pt`: test center `36.4870`, P10 `9.9775`, P5 `2.9660`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.105.pt`: test center `36.4882`, P10 `10.1327`, P5 `2.7747`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.110.pt`: test center `36.4896`, P10 `10.1412`, P5 `2.7236`.
- Second micro interpolation around alpha `0.095`:
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.093.pt`: test center `36.4870`, P10 `9.9775`, P5 `2.9596`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.094.pt`: test center `36.486977`, P10 `9.9775`, P5 `2.9660`. This is the current test-center leader.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.096.pt`: test center `36.4870`, P10 `9.9775`, P5 `2.9660`.
  - `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.097.pt`: test center `36.4871`, P10 `10.0285`, P5 `2.9660`.
- Paper-driven follow-up evidence:
  - EyeTrAES argues that fixed time/event windows are weak under wide eye-motion rate variation and motivates adaptive slicing.
  - FACET motivates direct ellipse prediction with center/offset heads and ellipse-aware loss instead of mask-first fitting.
  - Local-global distillation motivates local expert teachers and denoising distillation into a global student.

Interpretation: pre-fix event-count and weak-distillation results remain useful for pipeline debugging only. Accuracy decisions should use corrected post-fix runs. The best current test checkpoint is now the alpha `0.094` interpolation between centerloss best-center and track-only centerckpt-init LR `5e-6` best-center. Center-aware loss no longer looks like a pure center regression when the best-center checkpoint is used; checkpoint-space interpolation currently transfers better to the test split than direct validation-center fine-tuning. Further alpha-only search has diminishing returns.

2026-06-11 continuation:

- `configs/external/mode1_stage2_raw_event_count_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth.yaml`: config load passed; trainable include `track_head.*`, `event_adapter.*`, `patch_frontend.event_embed.proj.*`; raw event-count contract passed.
- `configs/external/mode1_stage2_raw_event_count_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth.yaml`: config load passed; trainable include `track_head.*`, `backbone.attn_stages.5.*`, `backbone.mlp_stages.5.*`, `backbone.norm.*`; raw event-count contract passed.
- `bash -n scripts/external/run_prepare_and_train.sh` passed.
- `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackheadonly_centerloss_alphainit_finetune_fullwidth_20260611_020317`: early-stopped at epoch 5; best val center `39.8494`; no test promotion.
- `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackadapters_centerloss_alphainit_finetune_fullwidth_20260611_020333`: early-stopped at epoch 7; best val center `39.8169`; no test promotion.
- Active validation watch:
  - GPU1 `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`
  - GPU0 `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`
- Added adaptive-count event slicing support in `src/hbtxr/data/event_builder.py`; disabled by default unless config includes `adaptive_count.enabled: true`.
- Added `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`; config load resolves fixed-count base `5000` plus adaptive-count settings. Use `--override=data.mode1.event_builder.event_count_target=10000` for the planned count10000 run.
- Targeted validation passed: `tests/test_event_builder.py tests/test_raw_event_count_contract.py` -> `6 passed`; raw event-count contract passed for the adaptive-count config.
- `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`: early-stopped at epoch 7; best val center `39.8169`; no test promotion.
- `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`: early-stopped at epoch 5; best val center `39.8546`; no test promotion.
- `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`: config load and raw event-count contract passed.
- Active validation watch:
  - GPU0 `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`
  - GPU1 `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`
- `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`: completed by early stop at epoch 7. Best val center was `38.6105`, a new validation-center leader. Test split eval `runs/eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233` produced test center `35.7182`, P10 `11.3461`, P5 `3.7279`, replacing the previous alpha `0.094` interpolation test leader `36.486977`.
- `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`: completed by early stop at epoch 7. Best val center was `38.9294`; no immediate test priority because the LR `5e-6` adaptive-count sibling is stronger on validation center.
- Started GPU1 parallel adaptive-count sqrt-scaling probe: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_sqrt_alphainit_fullwidth_20260611_022233`, with `adaptive_count.scale_power=0.5`; raw event-count contract passed before training.
- Started GPU0 adaptive-count power-1.25 probe: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_pow125_alphainit_fullwidth_20260611_022703`, with `adaptive_count.scale_power=1.25`; raw event-count contract passed before training.
- Manifest delta validation showed `reference_us=10000` saturates adaptive-count to only `5000` or `20000` events for train samples regardless of `scale_power=0.5/1.0/1.25`; the power-1.25 probe was stopped as redundant.
- `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_sqrt_alphainit_fullwidth_20260611_022233`: completed by early stop at epoch 7. Best val center tied the leader at `38.6105`; best val P10 was `12.7920`. Best-P10 test eval `runs/eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812` produced test center `35.6666`, P10 `10.9566`, P5 `3.6947`, improving test center versus `35.7182` but not improving P10.
- Started fixed-count 20000 control on GPU1: `raw_mode1_stage2_count20000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023021`.
- Started non-saturated adaptive-count ref4m control on GPU0: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_ref4m_alphainit_fullwidth_20260611_023021`.

Current promotion gate: new validation candidates must beat val center `38.6105`; new center candidates must beat test center `35.6666`; P10/P5 should be tracked as secondary accuracy axes because the best-center and best-P10 checkpoints differ.

## 2026-06-16 PAPER_REF/XR-03 Validation

- Added `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` from local PAPER_REF extracted text, existing inventory summary, submission draft claims, and current second-goal experiment results.
- Added `scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh`.
- Added `scripts/external/compare_xr01_adamw_bracket.py`.
- Validation passed:

```bash
chmod +x scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh
bash -n scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh
python3 -m py_compile scripts/external/compare_xr01_adamw_bracket.py
test -f runs/raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215/train/best_metric_track_center_px.pt
```

- XR-01 fixed250k/fixed260k AdamW LR `8e-6` completed and produced all four best-center/best-P10 test summaries. Scanner result: no candidate promoted against the current leader.
- Completed XR-01 count-bracket metrics:
  - fixed250k best-center: center `27.14032256262643`, P10 `15.30442224230085`, P5 `4.363520533697946`.
  - fixed250k best-P10: center `28.13574755532401`, P10 `14.668792915344238`, P5 `4.3401361874171664`.
  - fixed260k best-center: center `27.208868653433665`, P10 `15.642857592446463`, P5 `4.374149778911046`.
  - fixed260k best-P10: center `32.203995956693376`, P10 `11.9897962978908`, P5 `3.186224603652954`.
- Extended `scripts/external/compare_xr01_adamw_bracket.py` to also scan the fixed255k LR `6e-6` and `1e-5` branch. Validation passed:

```bash
python3 -m py_compile scripts/external/compare_xr01_adamw_bracket.py
scripts/external/compare_xr01_adamw_bracket.py
```

- Started and completed fixed255k LR bracket in tmux on both GPUs. Runtime logs:

```text
hgtxr_xr01_lr6e6_255k_gpu0_20260616  -> runs/_logs/xr01_adamw_count255k_lr6e-6_gpu0_20260616.log
hgtxr_xr01_lr1e5_255k_gpu1_20260616  -> runs/_logs/xr01_adamw_count255k_lr1e-5_gpu1_20260616.log
```

- Final scanner result:

```bash
scripts/external/compare_xr01_adamw_bracket.py
```

- fixed255k LR `6e-6` best-center/best-P10: center `28.610539082118443`, P10 `14.511054842812674`, P5 `4.037415075302124`; no promotion.
- fixed255k LR `1e-5` best-center: center `26.174878706250873`, P10 `16.502126346315656`, P5 `5.099915143421718`; promoted over prior leader center `27.089664377485004`, P10 `15.710884816305978`, P5 `4.630102171216692`.
- fixed255k LR `1e-5` best-P10: center `26.50779542582376`, P10 `16.803571953092302`, P5 `4.259779044560024`; promoted by the P10 gate but not selected as primary leader because center and P5 are weaker than best-center.
- XR-03 should now use the promoted best-center checkpoint as its warm start:
  `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt`.

## 2026-06-16 XR-03 Launch Validation

- Launched `hgtxr_xr03_ellipsestate_lr1e5_gpu0_20260616` in tmux on GPU0 and parallel stability branch `hgtxr_xr03_ellipsestate_lr6e6_gpu1_20260616` in tmux on GPU1.
- Command:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.025 0.01 1e-5 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt
```

- GPU0 log/run: `runs/_logs/xr03_ellipsestate_lr1e-5_gpu0_20260616.log`; `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016`.
- GPU1 log/run: `runs/_logs/xr03_ellipsestate_lr6e-6_gpu1_20260616.log`; `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012145`.
- Startup checks passed: raw event-count contract passed, checkpoint exists, GPU0/GPU1 processes active, both branches entered epoch 1/12.

## 2026-06-16 XR-04 Failure-Bucket Validation

- Added `scripts/external/summarize_eval_failure_buckets.py`.
- Validation passed:

```bash
python3 -m py_compile scripts/external/summarize_eval_failure_buckets.py
chmod +x scripts/external/summarize_eval_failure_buckets.py
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py ... --limit 64 --override training.num_workers=0
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py ... --output runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json --override training.num_workers=0
```

- Full diagnostic joined `2238/2238` eval rows with zero missing predictions.
- Global recomputed weighted metrics from row-level targets are center `26.9068`, P10 `15.7895`, P5 `4.7522`. This differs slightly from `eval_hbtxr.py` summary because this diagnostic recomputes global row-weighted metrics, while the eval summary stores averaged batch metrics.
- Main failure buckets:
  - `similarity_target <= 0.1`: weighted center `42.8642`, P10 `7.1259`, P5 `1.4252`.
  - `similarity_target 0.1..0.3`: weighted center `30.1790`, P10 `7.7206`, P5 `1.4706`.
  - Worst subject buckets: `subject_id:42` weighted center `31.4714`, `subject_id:41` weighted center `30.9859`.
  - Worst session bucket: `user41/right/session_201` weighted center `42.4245`.
- XR-04 implication: next local/temporal/geometry change should target low-similarity/high-motion rows first; blink rows are mostly zero-weighted and should not drive the immediate promotion metric unless the metric contract changes.

## 2026-06-16 XR-03 Closeout Validation

- All four XR-03 test summaries were found under `runs/eval_fixed255k_xr03_ellipsestate_*_20260616_*/eval/test/eval_summary.json`.
- Test metric fields used for promotion judgment: `metric_track_center_px`, `metric_track_p10_pct`, `metric_track_p5_pct`. The `*_value` fields are validation/checkpoint-selection metrics and must not be substituted for test metrics.
- Promotion gate before XR-03: center `26.174878706250873`, P10 `16.502126346315656`, P5 `5.099915143421718`.

Validated XR-03 test results:

```text
LR 1e-5 best-center: center 21.682174137660436, P10 21.820153658730643, P5 7.044643061501639
LR 1e-5 best-P10:    center 21.95296255179814,  P10 21.74532369886126,  P5 6.735119233812605
LR 6e-6 best-center: center 22.879396969931467, P10 20.383504002434865, P5 5.965986571993146
LR 6e-6 best-P10:    center 23.06402723789215,  P10 20.23724546432495,  P5 6.090986551557268
```

Decision:

- Primary leader: LR `1e-5` best-center, because it improves center, P10, and P5 over the prior leader and over the LR `6e-6` branch.
- Secondary: LR `1e-5` best-P10. It promotes versus the prior leader but is weaker than best-center on all tracked test metrics.
- Next validation requirement: XR-03A/B stronger axis/angle sweep must compare against the new gate center `21.682174137660436`, P10 `21.820153658730643`, P5 `7.044643061501639`.

## 2026-06-16 XR-03A/B Launch Validation

- Runner syntax check passed: `bash -n scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh`.
- Init checkpoint exists: `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016/train/best_metric_track_center_px.pt`.
- GPU availability before launch: GPU0/GPU1 each reported about `15827 MiB` free and `0%` utilization.
- Launched active branches:
  - GPU0 XR-03A: axis `0.05`, angle `0.02`, LR `1e-5`, log `runs/_logs/xr03a_axis0p05_angle0p02_lr1e-5_gpu0_20260616.log`.
  - GPU1 XR-03B: axis `0.05`, angle `0.02`, LR `6e-6`, log `runs/_logs/xr03b_axis0p05_angle0p02_lr6e-6_gpu1_20260616.log`.
- Startup logs confirm raw event-count contract pass, promoted XR-03 checkpoint load, correct CUDA device resolution, and epoch `1/12` entry for both branches.

## 2026-06-16 XR-03A/B Closeout Validation

- Training logs reported clean exits for both branches:
  - XR-03A: `XR03_ELLIPSESTATE_TRAIN_EXIT:0:2026-06-16T02:05:06+09:00`.
  - XR-03B: `XR03_ELLIPSESTATE_TRAIN_EXIT:0:2026-06-16T02:00:01+09:00`.
- Test summaries were found under:
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_020507/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_020747/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_020002/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p05_angle0p02_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_020310/eval/test/eval_summary.json`.
- Promotion gate before XR-03A/B: center `21.682174137660436`, P10 `21.820153658730643`, P5 `7.044643061501639`.

Validated XR-03A/B test results:

```text
XR-03A LR 1e-5 best-center: center 20.453865163666862, P10 25.86862314088004,  P5 8.310374430247716
XR-03A LR 1e-5 best-P10:    center 20.453865163666862, P10 25.86862314088004,  P5 8.310374430247716
XR-03B LR 6e-6 best-center: center 21.3059159551348,   P10 23.510204751150948, P5 7.570153277260917
XR-03B LR 6e-6 best-P10:    center 21.5007520709719,   P10 23.427296597617012, P5 8.20748325756618
```

Decision:

- Primary leader: XR-03A LR `1e-5`, axis `0.05`, angle `0.02`, best-center checkpoint. It improves center, P10, and P5 over the previous XR-03 leader.
- Secondary: XR-03B LR `6e-6` best-center. It improves over previous XR-03 but is weaker than XR-03A.
- Next validation gate: any follow-up must beat center `20.453865163666862`, P10 `25.86862314088004`, P5 `8.310374430247716`, or provide a clearly separated P5/tail tradeoff worth keeping as secondary.

## 2026-06-16 XR-03C/D Launch Validation

- Pre-launch checks:
  - GPU0/GPU1 were idle before launch.
  - No existing run/log collisions were found for axis `0.075`, angle `0.03`, LR `1e-5` or `6e-6`.
  - XR-03A best-center checkpoint exists: `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt`.
  - Runner syntax passed: `bash -n scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh`.
- Active branches:
  - XR-03C GPU0: axis `0.075`, angle `0.03`, LR `1e-5`, log `runs/_logs/xr03c_axis0p075_angle0p03_lr1e-5_gpu0_20260616.log`.
  - XR-03D GPU1: axis `0.075`, angle `0.03`, LR `6e-6`, log `runs/_logs/xr03d_axis0p075_angle0p03_lr6e-6_gpu1_20260616.log`.
- Startup logs confirm raw event-count contract pass, XR-03A checkpoint load, correct CUDA device resolution, and epoch `1/12` entry for both branches.
- Next validation requirement: compare XR-03C/D best-center and best-P10 test evals against XR-03A gate center `20.453865163666862`, P10 `25.86862314088004`, P5 `8.310374430247716`.

## 2026-06-16 XR-03C/D Closeout Validation

- Training logs reported clean Stage2 exits:
  - XR-03C LR `1e-5`: `XR03_ELLIPSESTATE_TRAIN_EXIT:0:2026-06-16T02:26:33+09:00`.
  - XR-03D LR `6e-6`: `XR03_ELLIPSESTATE_TRAIN_EXIT:0:2026-06-16T02:27:49+09:00`.
- Test summaries were found under:
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestcenter_test_gpu0_w0_20260616_022634/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr1e_5_bestp10_test_gpu0_w0_20260616_022942/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr6e_6_bestcenter_test_gpu1_w0_20260616_022750/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr03_ellipsestate_axis0p075_angle0p03_adamw_lr6e_6_bestp10_test_gpu1_w0_20260616_023057/eval/test/eval_summary.json`.
- Promotion gate before XR-03C/D: XR-03A center `20.453865163666862`, P10 `25.86862314088004`, P5 `8.310374430247716`.

Validated XR-03C/D test results:

```text
XR-03C LR 1e-5 best-center: center 20.474989349501474, P10 26.320153747286117, P5 7.537415218353272
XR-03C LR 1e-5 best-P10:    center 20.474989349501474, P10 26.320153747286117, P5 7.537415218353272
XR-03D LR 6e-6 best-center: center 20.49403738464628,  P10 25.965136766433716, P5 8.044643136433193
XR-03D LR 6e-6 best-P10:    center 20.433587510245186, P10 25.78656539235796,  P5 7.749149915150234
```

Decision:

- New test-center leader: XR-03D LR `6e-6`, axis/angle `0.075/0.03`, best-P10 checkpoint. It improves center by `0.020277653421676 px` over XR-03A.
- Secondary: XR-03A LR `1e-5`, axis/angle `0.05/0.02`, best-center checkpoint. It remains stronger on P10/P5.
- Next validation gate for center-first experiments: beat center `20.433587510245186`. For balanced experiments, preserve or exceed XR-03A P10 `25.86862314088004` and P5 `8.310374430247716`.

## 2026-06-16 XR-04 Low-Similarity Loss Validation And Launch

- Init checkpoint exists and was used for both active branches: `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641/train/best_track_p10.pt`.
- Runner syntax previously passed: `bash -n scripts/external/run_xr04_lowsim_ellipsestate_probe.sh`.
- Active branches:
  - GPU0 XR-04 low-sim LR `6e-6`, threshold `0.3`, scale `1.0`, log `runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr6e-6_gpu0_20260616.log`.
  - GPU1 XR-04 low-sim LR `1e-5`, threshold `0.3`, scale `1.0`, log `runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr1e-5_gpu1_20260616.log`.
- Startup logs confirm raw event-count contract pass, XR-03D checkpoint load, correct CUDA device resolution, and epoch `1/12` entry for both branches.

- Unit tests passed:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py
# 6 passed; known parent .pytest_cache read-only warning only
```

- Compile/syntax checks passed:

```bash
python3 -m py_compile src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py
bash -n scripts/external/run_xr04_lowsim_ellipsestate_probe.sh
```

- Raw event-count contract passed for the new config:

```bash
PYTHONPATH=src .venv/bin/python scripts/external/check_raw_event_count_contract.py \
  --manifest-root data/_internal/manifests/manifest1 \
  --stage2-config configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml
```

- Note: `scripts/external/run_xr04_lowsim_ellipsestate_probe.sh` is a bash runner (`set -euo pipefail`), not a POSIX `sh` script.

## 2026-06-16 XR-04 Closeout And XR-05A Launch Validation

- XR-04 produced all four best-center/best-P10 test summaries:
  - `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_adamw_lr1e_5_bestcenter_test_gpu1_w0_20260616_024507/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_adamw_lr1e_5_bestp10_test_gpu1_w0_20260616_024816/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_024603/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_024905/eval/test/eval_summary.json`.
- XR-04 failed promotion:

```text
LR 1e-5 best-center: center 20.648689174652098, P10 25.145834023611886, P5 7.927296195711408
LR 1e-5 best-P10:    center 20.739680664879934, P10 25.440902028765,    P5 8.661990063531059
LR 6e-6 best-center: center 20.43466943332127,  P10 25.784439495631627, P5 8.428571728297642
LR 6e-6 best-P10:    center 20.639369245937893, P10 25.642432710102625, P5 7.786139719826835
```

- Current center-first gate remains XR-03D center `20.433587510245186`.
- Current balanced gate remains XR-03A P10 `25.86862314088004`, P5 `8.310374430247716`.
- XR-05A direct track-state auxiliary head validation passed:
  - `bash -n scripts/external/run_xr05a_trackstateaux_probe.sh`.
  - `python3 -m py_compile` on modified model/loss files.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` -> `8 passed`; only known parent `.pytest_cache` read-only warning.
  - Raw event-count readiness for `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_finetune_fullwidth.yaml` returned `ready=true`, contract `ok=true`.
  - Model build smoke created `TrackStateAuxHead` with `150918` parameters.
- XR-05A launched:
  - GPU0 LR `6e-6`, log `runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr6e-6_gpu0_20260616.log`.
  - GPU1 LR `3e-6`, log `runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr3e-6_gpu1_20260616.log`.
- Startup logs confirm raw event-count contract pass, XR-03D checkpoint load, correct CUDA device resolution, and epoch `1/12` entry for both XR-05A branches.

## 2026-06-16 XR-05A Closeout And XR-05B/C Launch Validation

- XR-05A produced all four test summaries:
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_030404/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_030712/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_030750/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_031047/eval/test/eval_summary.json`.
- XR-05A test metrics:

```text
GPU0 LR 6e-6 best-center: center 20.46567486694881,  P10 24.83716058731079,  P5 8.1322281564985
GPU0 LR 6e-6 best-P10:    center 20.316973662376405, P10 25.863946315220424, P5 8.239796202523367
GPU1 LR 3e-6 best-center: center 20.38325309753418,  P10 26.210459920338224, P5 8.69557854788644
GPU1 LR 3e-6 best-P10:    center 20.321967782293047, P10 26.57610618046352,  P5 8.033588709150042
```

- Promotion decision:
  - Center-first leader: GPU0 LR `6e-6` best-P10, center `20.316973662376405`.
  - P10 leader: GPU1 LR `3e-6` best-P10, P10 `26.57610618046352`.
  - P5/balanced secondary: GPU1 LR `3e-6` best-center, P5 `8.69557854788644` with center `20.38325309753418` and P10 `26.210459920338224`.
- XR-05B launch validation:
  - tmux session `hgtxr_xr05b_trackstateaux_light_lr2e6_gpu1_20260616`.
  - Log `runs/_logs/xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_lr2e-6_gpu1_20260616.log`.
  - Startup log confirms raw event-count contract pass, init checkpoint load from XR-05A LR `3e-6` best-P10, CUDA `cuda:1`, and epoch `1/12` entry.
- XR-05C launch validation:
  - tmux session `hgtxr_xr05c_trackstateaux_mid_lr45e7_gpu0_20260616`.
  - Log `runs/_logs/xr05c_trackstateaux_c0p001_axis0p025_angle0p01_lr4p5e-6_gpu0_20260616.log`.
  - Startup log confirms raw event-count contract pass, init checkpoint load from XR-03D best-P10, CUDA `cuda:0`, and epoch `1/12` entry.

## 2026-06-16 XR-05B/C Closeout And XR-05D/E Launch Validation

- XR-05B produced two repaired test summaries:
  - `runs/eval_fixed255k_xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2e_6_bestcenter_test_gpu1_w0_20260616_032948/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2e_6_bestp10_test_gpu1_w0_20260616_033310/eval/test/eval_summary.json`.
- XR-05C produced two runner-generated summaries:
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr4_5e_6_bestcenter_test_gpu0_w0_20260616_033026/eval/test/eval_summary.json`.
  - `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr4_5e_6_bestp10_test_gpu0_w0_20260616_033327/eval/test/eval_summary.json`.
- XR-05B/XR-05C metrics:

```text
XR-05B best-center: center 20.34794154507773,  P10 26.231718444824217, P5 8.644558129991804
XR-05B best-P10:    center 20.29270099571773,  P10 26.04464359964643,  P5 8.07823155948094
XR-05C best-center: center 20.596219372749328, P10 25.269133343015397, P5 7.835034254619053
XR-05C best-P10:    center 20.32359070096697,  P10 26.16709260940552,  P5 7.850765575681414
```

- Promotion decision:
  - XR-05B best-P10 is the new center-first leader: center `20.29270099571773`.
  - XR-05B best-center is near-balanced but does not beat P10 `26.57610618046352` or P5 `8.69557854788644`.
  - XR-05C does not promote.
- Manual eval recovery note:
  - First XR-05B manual eval attempt without `HBTXR_DISABLE_CUDNN=1` failed with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
  - Recovered evals used `HBTXR_DISABLE_CUDNN=1`, matching the training runners.
- XR-05D launch validation:
  - tmux session `hgtxr_xr05d_trackstateaux_ultralight_lr1e6_gpu1_20260616`.
  - Log `runs/_logs/xr05d_trackstateaux_c0p00025_axis0p00625_angle0p0025_lr1e-6_gpu1_20260616.log`.
  - Startup log confirms raw event-count contract pass, init checkpoint load from XR-05B best-P10, CUDA `cuda:1`, and epoch `1/12` entry.
- XR-05E launch validation:
  - tmux session `hgtxr_xr05e_trackstateaux_midlight_lr15e7_gpu0_20260616`.
  - Log `runs/_logs/xr05e_trackstateaux_c0p00075_axis0p01875_angle0p0075_lr1p5e-6_gpu0_20260616.log`.
  - Startup log confirms raw event-count contract pass, init checkpoint load from XR-05A LR `3e-6` best-center, CUDA `cuda:0`, and epoch `1/12` entry.

## 2026-06-16 XR-05D/E Closeout And XR-07 Launch Validation

- XR-05D and XR-05E train/eval runners completed cleanly.
- XR-05D best-center: center `20.329813245364598`, P10 `26.293368080684118`, P5 `8.298044504438128`; no promotion.
- XR-05D best-P10: center `20.29814441204071`, P10 `25.887330681937083`, P5 `7.81760230745588`; no promotion.
- XR-05E best-center: center `20.36729155949184`, P10 `26.653912333079745`, P5 `8.482993486949375`; promoted P10 gate.
- XR-05E best-P10: center `20.307639326368058`, P10 `25.878827265330724`, P5 `8.103741775240216`; no promotion.
- XR-06 fallback validation:
  - `bash -n scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh` passed.
  - `PYTHONPATH=src .venv/bin/python -c ... load_config(...)` confirmed `track_state_aux=true`, `distillation.enabled=true`, `teacher_student=true`, `feature_weight=0.0025`, and AdamW inheritance.
  - Script mode is executable: `775`.
- XR-07 validation:
  - `bash -n scripts/external/run_xr07_trackstateaux_centerhinge_probe.sh` passed.
  - XR-07A GPU1 startup log confirms raw event-count contract pass, XR-05B best-P10 init checkpoint, linear hinge `margin=10.0`, `weight=0.01`, CUDA `cuda:1`, and epoch `1/12` entry.
  - XR-07B GPU0 startup log confirms raw event-count contract pass, XR-05B best-P10 init checkpoint, squared hinge `margin=10.0`, `weight=0.001`, CUDA `cuda:0`, and epoch `1/12` entry.

## 2026-06-16 XR-07 Closeout Validation

- XR-07A training log reached `[early-stop] stopping at epoch=5` and `[DONE] stage2_run_root=...linearhinge...`.
- XR-07B training log reached `[early-stop] stopping at epoch=5` and `[DONE] stage2_run_root=...squaredhinge...`.
- Parsed validation metrics from `train/history.jsonl`:

```text
XR-07A linear best val center:  23.626078749602694 at epoch 1
XR-07A linear best val P10:     21.898023947229927 at epoch 2
XR-07A linear best val P5:      7.794250002447164 at epoch 1
XR-07B squared best val center: 23.628568100479413 at epoch 1
XR-07B squared best val P10:    21.898023947229927 at epoch 2
XR-07B squared best val P5:     7.794250002447164 at epoch 1
```

- Active gates remain center `< 20.29270099571773`, P10 `> 26.653912333079745`, P5 `> 8.69557854788644`.
- XR-07 did not meet any gate and is not promoted.
- Test eval risk: runner-created XR-07 eval directories contain hypers/run-contract files but no `eval_summary.json`; manual full eval was interrupted after extended runtime with no output. This leaves no XR-07 test metric, but validation evidence is sufficient for no-promotion.
- XR-02 dense-gap validation: `data/_internal/manifests/manifest1/test_manifest.jsonl` has 2238 rows, 72 groups, median adjacent timestamp gap `4000003us`, minimum `360000us`, maximum `24000019us`. With `--max-gap-us 50000`, current test rows are not valid dense trajectories for smoothing.
- XR-06 launch validation:
  - tmux session `hgtxr_xr06_weakdistill_centerleader_gpu1_20260616` launched.
  - Log `runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log` shows raw event-count contract pass, XR-05B best-P10 checkpoint used as init and teacher, `resolved_device=cuda:1`, and epoch `1/12` train entry.
- XR-06B launch validation:
  - tmux session `hgtxr_xr06b_weakdistill_p10leader_gpu0_20260616` launched.
  - Log `runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log` shows raw event-count contract pass, XR-05E best-center checkpoint used as init and teacher, `resolved_device=cuda:0`, and epoch `1/12` train entry.

## 2026-06-16 XR-06A Closeout And XR-05F Launch Validation

- XR-06A training log reached `[early-stop] stopping at epoch=10` and `XR06_WEAKDISTILL_TRACKSTATEAUX_TRAIN_EXIT:0`.
- XR-06A best-center eval summary:
  - Path: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestcenter_test_gpu1_w0_20260616_042332/eval/test/eval_summary.json`.
  - Test center `20.283847980839866`, P10 `26.277211591175625`, P5 `8.472364248548235`.
  - Decision: promotes center gate over XR-05B best-P10 center `20.29270099571773`.
- XR-06A best-P10 eval summary:
  - Path: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestp10_test_gpu1_w0_20260616_042640/eval/test/eval_summary.json`.
  - Test center `20.368657435689652`, P10 `26.00425246102469`, P5 `8.164115946633475`.
  - Decision: no promotion.
- XR-06B validation status:
  - Training log reached `XR06_WEAKDISTILL_TRACKSTATEAUX_TRAIN_EXIT:0`.
  - Best-center eval summary:
    - Path: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p00075_axis0p01875_angle0p0075_adamw_lr1e_6_bestcenter_test_gpu0_w0_20260616_043100/eval/test/eval_summary.json`.
    - Test center `20.295043339048114`, P10 `26.217687790734427`, P5 `8.519132954733712`.
  - Best-P10 eval summary:
    - Path: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p00075_axis0p01875_angle0p0075_adamw_lr1e_6_bestp10_test_gpu0_w0_20260616_043343/eval/test/eval_summary.json`.
    - Test center `20.46945102555411`, P10 `25.931973491396224`, P5 `8.630952685219901`.
  - Decision: no promotion.
- XR-05F no-distill control launch validation:
  - `bash -n scripts/external/run_xr05a_trackstateaux_probe.sh` passed before launch.
  - tmux session `hgtxr_xr05f_nodistill_xr05e_lightaux_gpu1_20260616` launched.
  - Log `runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log` shows raw event-count contract pass, XR-05E best-center checkpoint load, `resolved_device=cuda:1`, and epoch `1/12` train entry.
  - Best-center eval summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestcenter_test_gpu1_w0_20260616_043850/eval/test/eval_summary.json`, center `20.356191604478017`, P10 `26.432823869160245`, P5 `8.298044504438128`.
  - Best-P10 eval summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestp10_test_gpu1_w0_20260616_044115/eval/test/eval_summary.json`, center `20.31945925780705`, P10 `26.150936106273107`, P5 `8.142857428959438`.
  - Decision: no promotion.
- XR-06C/XR-05G launch validation:
  - `bash -n scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh` passed before launch.
  - `bash -n scripts/external/run_xr05a_trackstateaux_probe.sh` passed before launch.
  - XR-06C tmux session `hgtxr_xr06c_weakdistill_xr06a_lr5e7_gpu0_20260616` launched; log `runs/_logs/xr06c_weakdistill_xr06a_lr5e-7_gpu0_20260616.log` shows raw event-count contract pass, XR-06A best-center used as init and teacher, `resolved_device=cuda:0`, and epoch `1/12` train entry.
  - XR-05G tmux session `hgtxr_xr05g_nodistill_xr06a_lr5e7_gpu1_20260616` launched; log `runs/_logs/xr05g_nodistill_xr06a_lr5e-7_gpu1_20260616.log` shows raw event-count contract pass, XR-06A best-center used as init, `resolved_device=cuda:1`, and epoch `1/12` train entry.

## 2026-06-16 XR-06C/XR-05G Closeout And XR-06D/XR-06E Launch Validation

- XR-05G training log reached `[early-stop] stopping at epoch=5` and `XR05A_TRACKSTATEAUX_TRAIN_EXIT:0`.
- XR-05G best-center eval summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_045158/eval/test/eval_summary.json`, center `20.291977088791985`, P10 `26.442177615846905`, P5 `8.456632941109794`.
- XR-05G best-P10 eval summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_045506/eval/test/eval_summary.json`, center `20.375956610270908`, P10 `26.27423542567662`, P5 `8.281462873731341`.
- XR-05G decision: no promotion.
- XR-06C training log reached `[early-stop] stopping at epoch=10` and `XR06_WEAKDISTILL_TRACKSTATEAUX_TRAIN_EXIT:0`.
- XR-06C best-center eval summary: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_045809/eval/test/eval_summary.json`, center `20.283125744547164`, P10 `26.363521112714494`, P5 `8.43664994921003`.
- XR-06C best-P10 eval summary: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_050034/eval/test/eval_summary.json`, center `20.382882516724724`, P10 `26.74489871433803`, P5 `8.526786014011927`.
- XR-06C decision: promotes center and P10 gates.
- New active gates: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-06D/XR-06E launch validation:
  - XR-06D tmux session `hgtxr_xr06d_nodistill_xr06c_center_gpu0_20260616` launched with log `runs/_logs/xr06d_nodistill_xr06c_center_lr2p5e-7_gpu0_20260616.log`.
  - XR-06D log shows raw event-count contract pass, XR-06C best-center checkpoint load, `resolved_device=cuda:0`, and epoch `1/12` train entry.
  - XR-06E tmux session `hgtxr_xr06e_weakdistill_xr06c_p10_gpu1_20260616` launched with log `runs/_logs/xr06e_weakdistill_xr06c_p10_lr2p5e-7_gpu1_20260616.log`.
  - XR-06E log shows raw event-count contract pass, XR-06C best-P10 checkpoint used as init and teacher, `resolved_device=cuda:1`, and epoch `1/12` train entry.
- XR-06D/XR-06E closeout validation:
  - XR-06D training reached `XR05A_TRACKSTATEAUX_TRAIN_EXIT:0`; best-center and best-P10 eval summaries both produced center `20.295614736420767`, P10 `26.158589158739364`, P5 `8.504677152633667`.
  - XR-06E training reached `XR06_WEAKDISTILL_TRACKSTATEAUX_TRAIN_EXIT:0`; best-center eval produced center `20.29422003201076`, P10 `26.280612965992518`, P5 `8.538690771375384`.
  - XR-06E best-P10 eval produced center `20.31491228171757`, P10 `26.13392930030823`, P5 `8.681122759410313`.
  - Decision: no promotion against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-08 checkpoint interpolation validation:
  - `python3 -m py_compile scripts/external/interpolate_hbtxr_checkpoints.py` passed.
  - `bash -n scripts/external/run_xr08_checkpoint_interp_eval.sh` passed.
  - Alpha `0.50` payload creation reported 132 total tensors and 132 floating tensors interpolated.
  - Successful eval summary: `runs/eval_fixed255k_xr08_xr06c_center_p10_interp_alpha0p50r_gpu0_w0_20260616_053156/eval/test/eval_summary.json`, center `20.32587662594659`, P10 `26.49830005509513`, P5 `8.386479888643537`.
  - Decision: no promotion.
- XR-05I/XR-06F launch validation:
  - `bash -n scripts/external/run_xr05a_trackstateaux_probe.sh` passed before launch.
  - `bash -n scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh` passed before launch.
  - XR-05I log `runs/_logs/xr05i_p5preserve_nodistill_xr05a_lr5e-7_gpu0_20260616.log` shows raw event-count contract pass, XR-05A P5/balanced checkpoint load, `resolved_device=cuda:0`, and epoch `1/12` train entry.
  - XR-06F log `runs/_logs/xr06f_p5preserve_weakdistill_xr05a_lr5e-7_gpu1_20260616.log` shows raw event-count contract pass, XR-05A P5/balanced checkpoint used as init and teacher, `resolved_device=cuda:1`, and epoch `1/12` train entry.
  - Sidecar evaluator confirmed both command shapes match runner parameters and warned not to relaunch identical conditions while active because log names and latest-run resolution could confuse provenance.
- XR-05I/XR-06F closeout validation:
  - XR-05I train log reached `XR05A_TRACKSTATEAUX_TRAIN_EXIT:0`; best-center and best-P10 evals reached `XR05A_TRACKSTATEAUX_CENTER_EXIT:0` / `XR05A_TRACKSTATEAUX_P10_EXIT:0`.
  - XR-06F train log reached `XR06_WEAKDISTILL_TRACKSTATEAUX_TRAIN_EXIT:0`; best-center and best-P10 evals reached `XR06_WEAKDISTILL_TRACKSTATEAUX_CENTER_EXIT:0` / `XR06_WEAKDISTILL_TRACKSTATEAUX_P10_EXIT:0`.
  - Parsed summaries show no gate promotion versus center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
  - XR-05I best-center/best-P10: `20.3510/26.1501/8.4290` and `20.3939/26.1701/8.4503`.
  - XR-06F best-center/best-P10: `20.3222/26.2007/8.6259` and `20.4218/26.3946/8.6301`.
  - Manual duplicate XR-06F center eval was started after an apparent delayed auto summary and was interrupted; the completed auto summaries above are the authoritative evidence.
- XR-04B/XR-04C targeted failure-bucket launch validation:
  - Existing runner `scripts/external/run_xr04_lowsim_ellipsestate_probe.sh` supports threshold/scale/LR/init checkpoint overrides; no new runner was needed.
  - XR-04B log `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log` shows raw event-count contract pass, XR-03D best-P10 checkpoint load, `resolved_device=cuda:0`, and epoch `1/12` train entry.
  - XR-04C log `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log` shows raw event-count contract pass, XR-03D best-P10 checkpoint load, `resolved_device=cuda:1`, and epoch `1/12` train entry.
  - `nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader` showed both GPUs active after launch.
- XR-04B/XR-04C targeted failure-bucket closeout validation:
  - Both branches reached `early-stop` at epoch `8/12` and train exit code `0`.
  - Best-center and best-P10 eval exits were `0` for XR-04B and XR-04C.
  - Parsed summaries:
    - XR-04B best-center `20.4841/25.6003/8.2776`; no promotion.
    - XR-04B best-P10 `20.6738/25.4014/7.8754`; no promotion.
    - XR-04C best-center `20.5883/25.6582/7.8499`; no promotion.
    - XR-04C best-P10 `20.5883/25.6582/7.8499`; no promotion.
  - GPU memory returned to idle after eval: `15 MiB` on both GPUs.
- XR-04D/XR-04E manifest-subset closeout validation:
  - Both branches reached `early-stop` at epoch `10/12` and train exit code `0`.
  - Best-center and best-P10 eval exits were `0` for LR `6e-6` and LR `3e-6`.
  - Parsed summaries:
    - XR-04D LR `6e-6` best-center `24.5081/16.7526/4.5179`; no promotion.
    - XR-04D LR `6e-6` best-P10 `25.7039/15.3074/4.2381`; no promotion.
    - XR-04E LR `3e-6` best-center `23.5577/18.1594/5.1063`; no promotion.
    - XR-04E LR `3e-6` best-P10 `23.4280/18.2555/5.0310`; no promotion.
  - GPU memory returned to idle after eval: `15 MiB` on both GPUs.
- XR-09 weighted-sampler setup validation:
  - `python3 -m py_compile src/hbtxr/data/loader.py scripts/external/train_hbtxr.py` passed.
  - `bash -n scripts/external/run_xr09_weightedsampler_trackstateaux_probe.sh` passed.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_weighted_sampler.py` passed: `3 passed`; known parent `.pytest_cache` read-only warning only.
  - Dataloader smoke with canonical root override passed on train manifest and returned valid mode1 batch tensors.
  - Weight distribution sanity for low-sim `3x`, session_201 `2x`, cap `6x`: rows `5929`, max `6.0`, mean `2.4322820037105752`, counts `6x=1186`, `3x=438`, `2x=1686`, `1x=2619`.
  - XR-09A/XR-09B launch validation: both logs show raw event-count contract pass, intended checkpoint load, resolved CUDA devices `cuda:0`/`cuda:1`, full train/val counts `5929/844`, and epoch `1/12` entry.
  - XR-09 closeout validation: both training runs exited `0`, all four full-test eval summaries exist, and no training/eval process remains active.
  - Parsed XR-09 summaries:
    - XR-09A best-center `20.5793/25.6327/8.0102`; no promotion.
    - XR-09A best-P10 `20.6414/24.7428/8.0412`; no promotion.
    - XR-09B best-center `20.4154/26.1964/8.3057`; no promotion.
    - XR-09B best-P10 `20.5593/25.6824/7.9328`; no promotion.
- XR-10 loss-weight setup validation:
  - `python3 -m py_compile src/hbtxr/data/components.py src/hbtxr/loss/stage_common.py src/hbtxr/loss/stage2.py scripts/external/train_hbtxr.py scripts/external/eval_hbtxr.py` passed.
  - `bash -n scripts/external/run_xr10_lossweight_trackstateaux_probe.sh` passed.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_loss_sample_weighting.py tests/test_weighted_sampler.py tests/test_track_center_l2_loss.py` passed: `14 passed`; known parent `.pytest_cache` read-only warning only.
  - Dataloader smoke with canonical root override passed on train manifest and confirmed `meta.session_key` is preserved.
  - Weight distribution sanity for low-sim `2x`, session_201 `1.5x`, cap `3x`: rows `5929`, max `3.0`, mean `1.6161241356046552`, counts `3x=1186`, `2x=438`, `1.5x=1686`, `1x=2619`.
  - XR-10A/XR-10B launch validation: both logs show raw event-count contract pass, intended XR-06C checkpoint load, resolved CUDA devices `cuda:0`/`cuda:1`, full train/val counts `5929/844`, and epoch `1/12` entry.
  - XR-10 closeout validation: both training runs exited `0`, all four full-test eval summaries exist, no training/eval process remains active, and GPU memory returned to idle.
  - Parsed XR-10 summaries:
    - XR-10A best-center `20.3488/26.2062/8.1280`; no promotion.
    - XR-10A best-P10 `20.4297/26.0480/7.9996`; no promotion.
    - XR-10B best-center `20.3266/25.9702/8.1335`; no promotion.
    - XR-10B best-P10 `20.3897/26.1871/7.9945`; no promotion.

## 2026-06-16 XR-11 SimDR Setup Validation

- `bash -n scripts/external/run_xr11_trackstate_simdr_probe.sh` passed.
- `python3 -m py_compile` passed for:
  - `src/hbtxr/models/heads.py`
  - `src/hbtxr/models/tracker/head_factory.py`
  - `src/hbtxr/models/tracker/track_branch.py`
  - `src/hbtxr/models/hybrid_tracker.py`
  - `src/hbtxr/training/model_factory.py`
  - `src/hbtxr/loss/bundles/track.py`
  - `src/hbtxr/loss/bundles/__init__.py`
  - `src/hbtxr/loss/stage2.py`
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_loss_sample_weighting.py tests/test_weighted_sampler.py` passed: `19 passed`; only known parent `.pytest_cache` read-only warning.
- Config/model smoke passed:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_simdr_fullwidth.yaml`
  - Output: `track/state_simdr` shape `torch.Size([2, 2, 64])`
  - Model head bins: `64`
- `git diff --check` passed over XR-11 touched files.
- XR-11 launch validation:
  - GPU0 center lane log `runs/_logs/xr11_simdr_center_lr5e-7_gpu0_20260616.log` shows raw contract pass, XR-06C best-center checkpoint load, `resolved_device=cuda:0`, and epoch `1/12` train entry.
  - GPU1 P10 lane log `runs/_logs/xr11_simdr_p10_lr5e-7_gpu1_20260616.log` shows raw contract pass, XR-06C best-P10 checkpoint load, `resolved_device=cuda:1`, and epoch `1/12` train entry.
  - `nvidia-smi` after launch showed both RTX 5080 GPUs allocated by the training jobs.
- XR-11 closeout validation:
  - GPU0 center lane train exit marker: `XR11_TRACKSTATE_SIMDR_TRAIN_EXIT:0:2026-06-16T07:40:28+09:00`.
  - GPU1 P10 lane train exit marker: `XR11_TRACKSTATE_SIMDR_TRAIN_EXIT:0:2026-06-16T07:40:30+09:00`.
  - All four full-test eval summaries exist and were parsed.
  - Center-init best-center: `20.3388/26.1607/8.3014`; no promotion.
  - Center-init best-P10: `20.3662/26.5455/8.5740`; no promotion.
  - P10-init best-center: `20.3430/25.9545/8.2355`; no promotion.
  - P10-init best-P10: `20.2893/26.3253/8.3099`; no promotion.
  - Active gates remain center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.

## 2026-06-16 XR-12 Weak-Distill SimDR Validation

- `bash -n scripts/external/run_xr12_weakdistill_simdr_probe.sh` passed.
- `git diff --check` passed over XR-12 config/runner and touched planning docs.
- Config/model smoke passed:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_fullwidth.yaml`
  - `distillation.enabled=True`
  - `model.heads.track_state_simdr=True`
  - `loss.track_state_simdr_weight=0.0001`
  - Model build has `track_state_simdr_head.bins=64`
- Checkpoint existence checks passed for XR-06C best-center and best-P10 checkpoints.
- XR-12 launch validation:
  - GPU0 center lane log `runs/_logs/xr12_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log` shows raw event-count contract pass, XR-06C best-center init/teacher checkpoint, `resolved_device=cuda:0`, train/val counts `5929/844`, and epoch `1/12` entry.
  - GPU1 P10 lane log `runs/_logs/xr12_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log` shows raw event-count contract pass, XR-06C best-P10 init/teacher checkpoint, `resolved_device=cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.
  - `nvidia-smi` after launch showed both RTX 5080 GPUs allocated by training jobs.
- XR-12 closeout validation:
  - Train exit markers exist: center lane `XR12_WEAKDISTILL_SIMDR_TRAIN_EXIT:0:2026-06-16T08:02:45+09:00`; P10 lane `XR12_WEAKDISTILL_SIMDR_TRAIN_EXIT:0:2026-06-16T08:02:57+09:00`.
  - Eval exit markers exist: center lane best-center `XR12_WEAKDISTILL_SIMDR_CENTER_EXIT:0:2026-06-16T08:05:51+09:00`, center lane best-P10 `XR12_WEAKDISTILL_SIMDR_P10_EXIT:0:2026-06-16T08:08:57+09:00`, P10 lane best-center `XR12_WEAKDISTILL_SIMDR_CENTER_EXIT:0:2026-06-16T08:06:03+09:00`, P10 lane best-P10 `XR12_WEAKDISTILL_SIMDR_P10_EXIT:0:2026-06-16T08:09:07+09:00`.
  - Four full-test eval summaries were parsed with active gates center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
  - Center-init best-center: `20.3049/26.1514/8.2207`; no promotion.
  - Center-init best-P10: `20.3599/26.0821/8.7917`; P5-only promotion, rejected as leader because center/P10 degrade.
  - P10-init best-center: `20.3070/26.1514/8.1696`; no promotion.
  - P10-init best-P10: `20.3070/26.1514/8.1696`; no promotion.

## 2026-06-16 XR-13 Head-Only Weak-Distill SimDR Validation

- Sub-agent T-047B GPT5.5 review completed and recommended the same P0: head-limited weak-distill SimDR with `track_head.*`, `track_state_aux_head.*`, and `track_state_simdr_head.*` trainable. It rejected SimDR-head-only because the SimDR auxiliary is not directly on the inference path.
- Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_headonly_fullwidth.yaml`.
- Added runner `scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh`.
- Static validation passed:
  - `chmod +x scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh`
  - `bash -n scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh`
  - XR-06C best-center and best-P10 checkpoint existence checks.
- Config/model/trainable smoke passed:
  - `distillation.enabled=True`
  - `model.heads.track_state_simdr=True`
  - `loss.track_state_simdr_weight=0.0001`
  - `training.trainable.include=['track_head.*', 'track_state_aux_head.*', 'track_state_simdr_head.*']`
  - matched trainable tensors `18`, trainable params `500494`, total params `3505306`.
- XR-13 launch validation:
  - GPU0 center lane log `runs/_logs/xr13_headonly_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log` shows raw event-count contract pass, XR-06C best-center init/teacher checkpoint, trainable-filter `18` tensors and `500494/3505306` trainable params, `resolved_device=cuda:0`, train/val counts `5929/844`, and epoch `1/12` entry.
  - GPU1 P10 lane log `runs/_logs/xr13_headonly_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log` shows raw event-count contract pass, XR-06C best-P10 init/teacher checkpoint, trainable-filter `18` tensors and `500494/3505306` trainable params, `resolved_device=cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.
- XR-13 closeout validation:
  - Train exit markers exist: center lane `XR13_HEADONLY_WEAKDISTILL_SIMDR_TRAIN_EXIT:0:2026-06-16T08:22:38+09:00`; P10 lane `XR13_HEADONLY_WEAKDISTILL_SIMDR_TRAIN_EXIT:0:2026-06-16T08:31:18+09:00`.
  - Eval exit markers exist in logs for all four best-center/best-P10 evaluations.
  - Four full-test eval summaries were parsed with active gates center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
  - Center-init best-center: `20.2916/26.2976/8.4694`; no promotion.
  - Center-init best-P10: `20.2897/26.2679/8.5204`; no promotion.
  - P10-init best-center: `20.3555/26.4622/8.4906`; no promotion.
  - P10-init best-P10: `20.3696/26.5514/8.5332`; no promotion.

## 2026-06-16 XR-02 Dense Trajectory Availability Validation

- `python3 -m py_compile scripts/external/check_xr02_dense_trajectory_availability.py` passed.
- `chmod +x scripts/external/check_xr02_dense_trajectory_availability.py` completed.
- Diagnostic run completed against `data/_internal/manifests/manifest1/test_manifest.jsonl` and `/home/kjm26/project/dataset/EV_Eye/canonical`.
- Output artifact: `runs/diagnostics/xr02_dense_trajectory_availability_20260616_084506.json`.
- Result: manifest and canonical annotations both have `dense_pair_count=0`, `longest_dense_segment=1`, min adjacent gap `360000us`, median adjacent gap `4000003us`.
- Decision: current labeled data is not usable for EyeLoRiN-style M2F metric promotion under `max_gap_us=50000`.

## 2026-06-16 XR-04 Diagnostics Refresh Validation

- `scripts/external/summarize_eval_failure_buckets.py` was patched to default `training.num_workers=0` when no explicit override is supplied. This avoids sandbox multiprocessing socket failures during diagnostics.
- `python3 -m py_compile scripts/external/summarize_eval_failure_buckets.py scripts/external/check_xr02_dense_trajectory_availability.py scripts/external/infer_hbtxr.py scripts/external/summarize_track_confidence_buckets.py` passed.
- Full diagnostic artifacts were generated for the XR-06C active leaders:
  - `runs/diagnostics/xr04_refresh_xr06c_bestcenter_failure_buckets_20260616.json`
  - `runs/diagnostics/xr04_refresh_xr06c_bestp10_failure_buckets_20260616.json`
- Both diagnostics joined `2238/2238` eval rows with `0` missing predictions.
- Best-center failure concentration:
  - `similarity_target <= 0.1`: count `493`, weighted center `27.1247`, P10 `19.2399`, P5 `4.7506`.
  - Worst session: `user42/left/session_201`, weighted center `30.9384`.
  - Worst subject: `subject_id:42`, weighted center `26.2739`.
- Best-P10 failure concentration:
  - `similarity_target <= 0.1`: count `493`, weighted center `26.9471`, P10 `19.0024`, P5 `4.9881`.
  - Worst session: `user42/left/session_201`, weighted center `30.4950`.
  - Worst subject: `subject_id:42`, weighted center `26.2106`.
- Search/event branch-state diagnostic attempts produced `joined_count=0`:
  - `runs/diagnostics/xr04_refresh_xr06c_bestcenter_searchstate_failure_buckets_20260616.json`
  - `runs/diagnostics/xr04_refresh_xr06c_bestcenter_eventstate_failure_buckets_20260616.json`
- Cause confirmed by normalized config: `model.heads.active=track` disables `eye/search/event/aux` heads for this track-only leader.
- Attempted full `infer_hbtxr.py` run for `track_pred` confidence collection was stopped after no output/no GPU activity. `scripts/external/infer_hbtxr.py` now supports `--limit` for bounded confidence-diagnostic smoke runs.
- Bounded smoke passed with `--limit 2`, output `runs/diagnostics/xr04_confidence_xr06c_bestcenter_infer_limit2_20260616`, and `num_rows=2`.
- Smoke rows confirm `track_pred` is available while `search_state` and `event_state` are null under the active track-only leader config.
- Added `scripts/external/summarize_track_confidence_buckets.py`; limit-2 smoke joined `2/2` rows and wrote `runs/diagnostics/xr04_confidence_xr06c_bestcenter_limit2_buckets_20260616.json`.
- Validation decision: XR-04 refresh is complete. Use it to drive a no-train confidence/relocalization diagnostic before any new training branch.

## 2026-06-16 XR-04 Confidence/Quality Probe Validation

- Added `scripts/external/build_confidence_probe_manifest.py`; `python3 -m py_compile` passed.
- Built balanced diagnostic manifest `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/test_manifest.jsonl`.
- Manifest summary:
  - source rows `2238`
  - available low-sim rows `493`
  - available high-sim rows `891`
  - selected low/high rows `128/128`
- XR-06C best-center inference completed on CPU with `num_rows=256`.
- XR-06C best-P10 inference completed on CPU with `num_rows=256`.
- Confidence summaries joined `256/256` rows with `0` missing predictions:
  - `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestcenter_buckets_20260616.json`
  - `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestp10_buckets_20260616.json`
- Center leader low-sim bucket: weighted center `24.5408`, P10 `21.5686`, P5 `2.9412`, mean confidence `0.9999997`, mean quality `0.9999998`.
- Center leader high-sim buckets are easier but slightly less confident: `0.6..0.9` weighted center `15.5990`, confidence `0.9999987`; `>=0.9` weighted center `11.1933`, confidence `0.9999980`.
- P10 leader shows the same pattern: low-sim weighted center `24.5044`, confidence `0.9999997`; high-sim `0.6..0.9` weighted center `15.8637`, confidence `0.9999981`; high-sim `>=0.9` weighted center `11.2084`, confidence `0.9999978`.
- Root-cause check: `src/hbtxr/data/components.py` sets track confidence target from `valid_track` and track quality target from `annotation_quality`; `src/hbtxr/loss/bundles/track.py` trains both with BCE. These labels are mostly positive on weighted valid rows, so saturation is expected.
- Validation decision: current `track_pred` confidence/quality is not calibrated for relocalization gating. No-train confidence gate is rejected.

## 2026-06-16 XR-14 Fallback And All-Head Setup Validation

- Added `scripts/external/eval_similarity_fallback.py`.
- Re-ran full-test XR-14 similarity-gated fallback diagnostic after adding official-like batchmean metrics.
- Artifact: `runs/diagnostics/xr14_similarity_prevstate_fallback_xr06c_bestcenter_20260616.json`.
- Joined rows: `2238/2238`.
- Raw batchmean metrics reproduce the official XR-06C best-center gate closely: center `20.283125752718494`, P10 `26.363520408163275`, P5 `8.436649659863942`.
- Best fallback candidate by batchmean center is `blend(threshold=0.05, alpha=0.75)`, but it degrades center to `22.57394233260353`, P10 to `22.974064625850357`, and P5 to `7.577380952380952`.
- Added `track_state_aux` and `track_state_simdr` fields to `scripts/external/infer_hbtxr.py`.
- Added `--state-key` to `scripts/external/summarize_track_confidence_buckets.py`.
- Aux-state probe artifacts:
  - `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_infer_20260616`
  - `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_trackstate_buckets_20260616.json`
  - `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_auxstate_buckets_20260616.json`
- Aux-state result: `track_state_aux` joined `256/256` but scored center `158.7825`, P10 `0.0`, P5 `0.0`, so it is not a usable fallback branch.
- Added XR-14A config and runner:
  - `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml`
  - `scripts/external/run_xr14_allhead_relocalize_probe.sh`
- Static validation passed:
  - `chmod +x scripts/external/run_xr14_allhead_relocalize_probe.sh`
  - `bash -n scripts/external/run_xr14_allhead_relocalize_probe.sh`
  - XR-06C default init checkpoint exists.
  - `python3 -m py_compile scripts/external/infer_hbtxr.py scripts/external/summarize_track_confidence_buckets.py scripts/external/eval_similarity_fallback.py`
- Config/model smoke passed with `PYTHONPATH=src:scripts/external`: search/event/track heads enabled, mask disabled, and distillation targets restricted to `track/fused`, `track/state`, and `track/pupil`.
- Runner validation update: `scripts/external/run_xr14_allhead_relocalize_probe.sh` now accepts a sixth `LANE_TAG`/`XR14_LANE_TAG` value and includes it in `EXPERIMENT`, preventing parallel center/P10 lane run-root collision.
- v2 startup validation passed:
  - Center lane log `runs/_logs/xr14a_allhead_relocalize_center_lr5e-7_gpu0_20260616_v2.log` shows lane `center`, raw contract pass, XR-06C best-center checkpoint, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_center_trackpreserve_fullwidth_20260616_094218`, `resolved_device=cuda:0`, train/val `5929/844`, and epoch `1/12` steps.
  - P10 lane log `runs/_logs/xr14a_allhead_relocalize_p10_lr5e-7_gpu1_20260616_v2.log` shows lane `p10`, raw contract pass, XR-06C best-P10 checkpoint, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_p10_trackpreserve_fullwidth_20260616_094220`, `resolved_device=cuda:1`, train/val `5929/844`, and epoch `1/12` steps.
- v2 closeout validation passed:
  - Center lane early-stopped at epoch `8/12`, train exit `0`, eval exits `0` for best-center and best-P10.
  - P10 lane early-stopped at epoch `8/12`, train exit `0`, eval exits `0` for best-center and best-P10.
  - Eval summaries exist for all four test evals under `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_*_test_gpu*_w0_*/eval/test/eval_summary.json`.
  - Diagnostic bucket artifacts exist for all 12 lane/checkpoint/state combinations under `runs/diagnostics/xr14a_*_state_failure_buckets_20260616.json`.
- XR-14A result validation:
  - Center lane best-center/best-P10 both scored `20.342467624800545 / 25.644133370263237 / 8.11947306905474`.
  - P10 lane best-center/best-P10 both scored `20.348247524670192 / 25.703657184328353 / 7.97278938974653`.
  - No XR-14A checkpoint beats the active XR-06C gates.
  - `search_state` and `event_state` bucket summaries joined `2238/2238` rows but scored P10/P5 `0.0`, so they are invalid fallback candidates.

## 2026-06-16 XR-15 Preflight

- Added:
  - `docs/XR15_SUPPORT_ADAPTIVE_PLAN.md`
  - `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`
  - `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`
  - `scripts/external/eval_xr15_p5_checkpoint.sh`
- Validation passed:
  - `chmod +x scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`
  - `chmod +x scripts/external/eval_xr15_p5_checkpoint.sh`
  - `bash -n scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`
  - `bash -n scripts/external/eval_xr15_p5_checkpoint.sh`
  - `.venv/bin/python scripts/external/check_raw_event_count_contract.py --manifest-root data/_internal/manifests/manifest1 --stage2-config configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`
  - `python3 -m py_compile scripts/external/build_confidence_probe_manifest.py scripts/external/summarize_eval_failure_buckets.py scripts/external/eval_similarity_fallback.py`
- Dataset smoke with runtime overrides passed. First test batch resolved adaptive target counts `192000/255000/255000/255000` and selected counts `281/84532/248055/254300`, proving the override path changes event support while preserving fixed-count policy.

## 2026-06-16 XR-15 Launch Validation

- Center lane:
  - tmux session `hgtxr_xr15_supportadaptive_center_gpu0_20260616`
  - log `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148`
  - startup log confirms raw event-count contract pass, XR-06C best-center checkpoint load as init/teacher, `resolved_device=cuda:0`, train/val counts `5929/844`, and epoch `1/12` entry.
- P10 lane:
  - tmux session `hgtxr_xr15_supportadaptive_p10_gpu1_20260616`
  - log `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205`
  - startup log confirms raw event-count contract pass, XR-06C best-P10 checkpoint load as init/teacher, `resolved_device=cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.
- GPU check after launch showed both RTX 5080 devices active with about `923MiB` used each and nonzero utilization.
- Sub-agent T-055 found no blocker in the XR-15 config/runner/plan. Warnings were accepted as follow-up items: the runner relies on runtime overrides for fixed255k adaptive count, and P5 checkpoint eval is handled by the added helper script after training.

## 2026-06-16 XR-15 Closeout Validation

- Train closeout:
  - Center lane log `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T10:49:02+09:00`.
  - P10 lane log `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T10:49:18+09:00`.
- Full-test eval summaries parsed:
  - `runs/eval_fixed255k_xr15_supportadaptive_center_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_104903/eval/test/eval_summary.json`: `20.230558776855467 / 26.303146975381033 / 8.573554713385446`.
  - `runs/eval_fixed255k_xr15_supportadaptive_center_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_105215/eval/test/eval_summary.json`: `20.259641446386066 / 25.812925890513828 / 8.451105751310076`.
  - `runs/eval_fixed255k_xr15_supportadaptive_p10_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_104918/eval/test/eval_summary.json`: `20.225680075372967 / 26.50085105895996 / 8.304422058377947`.
  - `runs/eval_fixed255k_xr15_supportadaptive_p10_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_105227/eval/test/eval_summary.json`: `20.259886418070113 / 26.12457557405744 / 8.357993486949375`.
- Gate decision:
  - Center promoted from XR-06C `20.283125744547164` to XR-15A `20.225680075372967`.
  - P10 remains XR-06C `26.74489871433803`.
  - P5 remains XR-05A `8.69557854788644`.
- Failure-bucket validation:
  - `runs/diagnostics/xr15_p10_bestcenter_track_state_failure_buckets_20260616.json` joined `2238/2238` rows, zero missing predictions.
  - `runs/diagnostics/xr15_center_bestcenter_track_state_failure_buckets_20260616.json` joined `2238/2238` rows, zero missing predictions.
  - P10-lane best-center low-sim bucket `similarity_target <= 0.1`: weighted center `26.8628818711799`, P10 `18.28978622327791`, P5 `4.275534441805226`.
  - Worst repeated session buckets remain `user42/left/session_201`, `user42/right/session_201`, and `user45/right/session_201`.
- Best-P5 artifact gap:
  - `scripts/external/eval_xr15_p5_checkpoint.sh` passed `bash -n`, and both P5 checkpoints exist.
  - Interrupted and log+timeout helper attempts created only `hypers/*` under `runs/eval_fixed255k_xr15_supportadaptive_*_bestp5_test_*` and no `eval/test/eval_summary.json`.
  - Sub-agent T-057 confirmed the helper writes no standalone trace before `eval_hbtxr.py` completes; future P5 reruns should use explicit log capture and timeout.

## 2026-06-16 XR-15B Launch Validation

- Static checks passed before launch:
  - `bash -n scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh scripts/external/eval_xr15_p5_checkpoint.sh`
  - XR-15A center-leader checkpoint exists.
  - XR-06C P10-leader checkpoint exists.
- Center lane launched:
  - session `hgtxr_xr15b_supportadaptive_center_gpu0_20260616`
  - log `runs/_logs/xr15b_supportadaptive_center_lr5e-7_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512`
  - startup log confirms raw event-count contract pass, XR-15A best-center init/teacher checkpoint, `resolved_device=cuda:0`, train/val counts `5929/844`, and epoch `1/12` train steps.
- P10 lane launched:
  - session `hgtxr_xr15b_supportadaptive_p10_gpu1_20260616`
  - log `runs/_logs/xr15b_supportadaptive_p10_lr5e-7_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111515`
  - startup log confirms raw event-count contract pass, XR-06C best-P10 init/teacher checkpoint, `resolved_device=cuda:1`, train/val counts `5929/844`, and epoch `1/12` train steps.
- Runtime note: sandbox `nvidia-smi` failed after launch with driver communication error, but train logs are the authoritative startup evidence for this launch.

## 2026-06-16 XR-15B Closeout And XR-15C Launch Validation

- XR-15B train closeout:
  - Center lane log `runs/_logs/xr15b_supportadaptive_center_lr5e-7_gpu0_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T11:32:15+09:00`.
  - P10 lane log `runs/_logs/xr15b_supportadaptive_p10_lr5e-7_gpu1_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T11:32:31+09:00`.
- XR-15B best-center eval summaries parsed:
  - `runs/eval_fixed255k_xr15_supportadaptive_center_xr15b_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_113216/eval/test/eval_summary.json`: `20.215672533852715 / 26.575255823135375 / 8.627551317214966`.
  - `runs/eval_fixed255k_xr15_supportadaptive_p10_xr15b_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_113231/eval/test/eval_summary.json`: `20.21979672227587 / 26.324405458995273 / 8.28741525241307`.
- Gate decision:
  - Center promoted from XR-15A `20.225680075372967` to XR-15B `20.215672533852715`.
  - P10 remains XR-06C `26.74489871433803`.
  - P5 remains XR-05A `8.69557854788644`.
- Artifact gaps:
  - XR-15B best-P10 eval runs created `hypers/*` but no `eval/test/eval_summary.json`.
  - XR-15B center best-center failure-bucket diagnostic was interrupted after a long no-output run.
- XR-15C launch validation:
  - Center lane session `hgtxr_xr15c_supportadaptive_center_gpu0_20260616`, log `runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819`.
  - P10 lane session `hgtxr_xr15c_supportadaptive_p10_gpu1_20260616`, log `runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816`.
  - Both startup logs confirm raw event-count contract pass, intended checkpoint load, CUDA device resolution, train/val `5929/844`, and epoch `1/12` entry.

## 2026-06-16 XR-15C Closeout Validation

- XR-15C train closeout:
  - Center lane log `runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T11:55:59+09:00`.
  - P10 lane log `runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log` contains `XR15_SUPPORT_ADAPTIVE_TRAIN_EXIT:0:2026-06-16T11:55:57+09:00`.
- XR-15C best-center eval summaries parsed:
  - `runs/eval_fixed255k_xr15_supportadaptive_center_xr15c_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_115600/eval/test/eval_summary.json`: `20.19088832650866 / 26.009779623576573 / 8.436224787575858`.
  - `runs/eval_fixed255k_xr15_supportadaptive_p10_xr15c_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_115558/eval/test/eval_summary.json`: `20.205999997683932 / 26.303146975381033 / 8.397959463936942`.
- Gate decision:
  - Center promoted from XR-15B `20.215672533852715` to XR-15C `20.19088832650866`.
  - P10 remains XR-06C `26.74489871433803`.
  - P5 remains XR-05A `8.69557854788644`.
- Next-experiment validation:
  - Because XR-15C improves center while regressing P10/P5, further support-range sweeps have lower priority than a bounded track-adapter/coordinate-head branch.
  - XR-15D should preserve the fixed255k test contract and compare against active gates center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.

## 2026-06-16 XR-15D Launch Validation

- Static validation:
  - `bash -n scripts/external/run_xr15d_trackadapters_probe.sh scripts/external/run_prepare_and_train.sh`
  - raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
  - `PYTHONPATH=src` model-build smoke passed with active head `track` and trainable include list `track_head.*`, `frame_adapter.*`, `event_adapter.*`, `patch_frontend.frame_embed.proj.*`, `patch_frontend.event_embed.proj.*`.
- Sub-agent T-062 launch-risk review:
  - Confirmed config matches track-adapter/coordinate-head scope.
  - Confirmed current blocker was lack of a dedicated XR-15D runner.
  - Recommended reusing the XR-15/XR-07 train/eval pattern with explicit fixed255k overrides and post-run artifact completeness checks.
- Runtime launch:
  - Center lane session `hgtxr_xr15d_trackadapters_center_gpu0_20260616`, log `runs/_logs/xr15d_trackadapters_center_lr3e-6_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackadapters_centerloss_center_xr15d_fullwidth_20260616_120804`.
  - P10 lane session `hgtxr_xr15d_trackadapters_p10_gpu1_20260616`, log `runs/_logs/xr15d_trackadapters_p10_lr3e-6_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackadapters_centerloss_p10_xr15d_fullwidth_20260616_120843`.
  - Both startup logs confirm raw event-count contract pass, intended checkpoint load, trainable filter `22` tensors / `448520` params, CUDA device resolution, train/val `5929/844`, and epoch `1/16` entry.
- Note: sandbox `nvidia-smi` returned a driver communication error after launch, but training logs provide direct CUDA device evidence.

## 2026-06-16 XR-15D Closeout And XR-15E Launch Validation

- XR-15D train closeout:
  - Center lane early-stopped at epoch `11/16`, train exit `0`, best validation center `23.44891523865034`.
  - P10 lane early-stopped at epoch `10/16`, train exit `0`, best validation center `23.523320144077516`.
- XR-15D test eval summaries parsed:
  - `runs/eval_fixed255k_xr15d_trackadapters_center_xr15d_adamw_lr3e_6_bestcenter_test_gpu0_w0_20260616_122330/eval/test/eval_summary.json`: `20.333278461865017 / 25.986395263671874 / 8.095238372257777`.
  - `runs/eval_fixed255k_xr15d_trackadapters_center_xr15d_adamw_lr3e_6_bestp10_test_gpu0_w0_20260616_122636/eval/test/eval_summary.json`: `20.36335334096636 / 25.821854482378278 / 8.246173749651227`.
  - `runs/eval_fixed255k_xr15d_trackadapters_p10_xr15d_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_122245/eval/test/eval_summary.json`: `20.362164442879813 / 26.141582359586444 / 8.562500286102296`.
  - `runs/eval_fixed255k_xr15d_trackadapters_p10_xr15d_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_122555/eval/test/eval_summary.json`: `20.410626077651976 / 26.1883510862078 / 8.385629544939313`.
- Gate decision:
  - No center/P10/P5 promotion versus center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-15E static validation:
  - `bash -n scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh`
  - raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
  - `PYTHONPATH=src` model+teacher smoke passed with active head `track` and same adapter trainable scope.
- T-064 note:
  - Requested sub-agent review failed because GPT-5.3-Codex-Spark quota was reached; Codex-native fallback performed validation and launch.
- XR-15E startup validation:
  - Center lane session `hgtxr_xr15e_weakdistill_trackadapters_center_gpu0_20260616`, log `runs/_logs/xr15e_weakdistill_trackadapters_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_center_xr15e_fullwidth_20260616_123419`.
  - P10 lane session `hgtxr_xr15e_weakdistill_trackadapters_p10_gpu1_20260616`, log `runs/_logs/xr15e_weakdistill_trackadapters_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_p10_xr15e_fullwidth_20260616_123431`.
  - Both startup logs confirm raw event-count contract pass, intended checkpoint load as init/teacher, trainable filter `22` tensors / `448520` params, CUDA device resolution, train/val `5929/844`, and epoch `1/12` entry.

## 2026-06-16 XR-15E Closeout And XR-16A Launch Validation

- XR-15E train/eval closeout:
  - Center and P10 lanes early-stopped at epoch `8/12` with train exit `0`.
  - Four test eval summaries exist under `runs/eval_fixed255k_xr15e_weakdistill_trackadapters_*`.
- XR-15E gate comparison:
  - Best center was `20.281941563742503`, above the required `<20.19088832650866`.
  - Best P10 was `26.333759232929776`, below the required `>26.74489871433803`.
  - Best P5 was `8.633928898402623`, below the required `>8.69557854788644`.
  - Decision: no promotion.
- XR-16A static validation:
  - `bash -n scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh`
  - raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`.
  - trainable-filter smoke reports `14` tensors / `324680` params: `patch_frontend.event_embed.proj.*`, `event_adapter.*`, and `track_head.*`.
  - `git diff --check` passed for the new XR-16A config and runner.
- XR-16A startup validation:
  - Center lane session `hgtxr_xr16a_eventadapter_center_gpu0_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_center_xr16a_fullwidth_20260616_125515`.
  - P10 lane session `hgtxr_xr16a_eventadapter_p10_gpu1_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_p10_xr16a_fullwidth_20260616_125515`.
  - Both startup logs confirm raw event-count contract pass, intended checkpoint load as init/teacher, trainable filter `14` tensors / `324680` params, CUDA device resolution, train/val `5929/844`, and epoch `1/12` entry.

## 2026-06-16 XR-16A/XR-16B Closeout And XR-15C P5 Validation

- XR-16A train/eval closeout:
  - Center and P10 lanes early-stopped at epoch `8/12` with train exit `0`.
  - Four test eval summaries exist under `runs/eval_fixed255k_xr16_weakdistill_trackeventadapter_*`.
- XR-16A gate comparison:
  - Best center was `20.281941199302672`, above the required `<20.19088832650866`.
  - Best P10 was `26.333759232929776`, below the required `>26.74489871433803`.
  - Best P5 was `8.633928898402623`, below the then-required `>8.69557854788644`.
  - Decision: no promotion.
- XR-16B static/runtime validation:
  - `bash -n scripts/external/run_xr16b_checkpoint_interp_eval.sh` passed.
  - Endpoint checkpoints existed before interpolation.
  - Alpha `0.125` eval summary exists at `runs/eval_fixed255k_xr16b_xr15c_center_xr06c_p10_interp_alpha0p125_gpu0_w0_20260616_131603/eval/test/eval_summary.json`.
- XR-16B gate comparison:
  - Result `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.
  - Decision: no promotion.
- XR-15C best-P5 artifact validation:
  - Center-lane best-P5 summary exists at `runs/eval_fixed255k_xr15_supportadaptive_center_xr15c_adamw_lr5e_7_bestp5_test_gpu0_w0_20260616_131952/eval/test/eval_summary.json`.
  - P10-lane best-P5 summary exists at `runs/eval_fixed255k_xr15_supportadaptive_p10_xr15c_adamw_lr5e_7_bestp5_test_gpu1_w0_20260616_131952/eval/test/eval_summary.json`.
  - P10-lane best-P5 result `20.245368467058455 / 26.487245675495693 / 8.703231593540737` improves P5 over XR-05A `8.69557854788644`.
  - Updated active gates: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.703231593540737`.

## 2026-06-16 XR-17A Interpolation Validation

- Static validation:
  - `bash -n scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh` passed.
  - `git diff --check -- scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh` passed.
  - Endpoint checkpoints existed before launch.
- Runtime validation:
  - Five interpolated checkpoints exist under `runs/interpolated_checkpoints/xr17a_xr15c_center_xr15c_p5_alpha*.pt`.
  - Five test eval summaries exist under `runs/eval_fixed255k_xr17a_xr15c_center_xr15c_p5_interp_alpha*/eval/test/eval_summary.json`.
  - No active XR-17A eval process remains after sweep completion.
- Gate comparison:
  - Alpha `0.25`: `20.199484479427337 / 26.20960956301008 / 8.625425474984306`.
  - Alpha `0.50`: `20.211353632381986 / 26.35204153742109 / 8.679847247259957`.
  - Alpha `0.625`: `20.218607200895036 / 26.449830661501203 / 8.703231607164655`.
  - Alpha `0.75`: `20.22669484274728 / 26.34566399029323 / 8.732993507385254`.
  - Alpha `0.875`: `20.235614109039307 / 26.527636807305473 / 8.673469693320138`.
  - Decision: alpha `0.75` promotes P5 over `8.703231593540737`; center/P10 gates remain XR-15C/XR-06C.
  - Updated active gates: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.732993507385254`.

## 2026-06-16 XR-17A Failure-Bucket Diagnostic Validation

- Inputs existed:
  - XR-15C center eval rows: `runs/eval_fixed255k_xr15_supportadaptive_center_xr15c_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_115600/eval/test/eval_rows.json`.
  - XR-17A alpha `0.75` eval rows: `runs/eval_fixed255k_xr17a_xr15c_center_xr15c_p5_interp_alpha0p75_gpu1_w0_20260616_134416/eval/test/eval_rows.json`.
- Diagnostic outputs exist:
  - `runs/diagnostics/xr17a_compare_xr15c_center_failure_buckets_20260616.json`.
  - `runs/diagnostics/xr17a_compare_xr17a_alpha0p75_failure_buckets_20260616.json`.
- The diagnostic joined the fixed test manifest and recomputed row-weighted metrics for both checkpoints.
- Result: XR-17A alpha `0.75` improves weighted P10/P5 but regresses weighted center slightly, so the validated next step is bounded P5-direction use, not direct center-seed promotion.

## 2026-06-16 XR-17B P5-Anchor Validation

- Static validation:
  - `bash -n scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh` passed.
  - `git diff --check` passed for the new XR-17B config and runner.
  - Config load smoke reported epochs `8`, LR `2.5e-7`, distillation enabled, `track_state_aux` enabled.
  - Raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml`.
  - Model build smoke succeeded with `HBTXRTracker` and `3307418` params.
- Runtime validation:
  - Initial sandboxed GPU launch failed because Torch saw no CUDA devices; escalated GPU0 launch succeeded and resolved `cuda:0`.
  - Train completed epoch `8/8` with exit `0`.
  - Best checkpoint eval summaries exist for best-center, best-P10, and best-P5.
- Gate comparison:
  - Best-center summary `runs/eval_fixed255k_xr17b_p5anchor_centerinit_p5teacher_adamw_lr2_5e_7_bestcenter_test_gpu0_w0_20260616_144255/eval/test/eval_summary.json`: `20.182338142395018 / 26.113946315220424 / 8.474490090778895`; center promotion.
  - Best-P10 summary `runs/eval_fixed255k_xr17b_p5anchor_centerinit_p5teacher_adamw_lr2_5e_7_bestp10_test_gpu0_w0_20260616_144521/eval/test/eval_summary.json`: `20.22993471963065 / 26.166242245265416 / 8.603316634041922`; no promotion.
  - Best-P5 summary `runs/eval_fixed255k_xr17b_p5anchor_centerinit_p5teacher_adamw_lr2_5e_7_bestp5_test_gpu0_w0_20260616_144749/eval/test/eval_summary.json`: `20.234592584201266 / 26.09183748109 / 8.844813244683403`; P5 promotion.
- Failure-bucket diagnostics:
  - `runs/diagnostics/xr17b_p5anchor_bestcenter_failure_buckets_20260616.json`: weighted `20.0015 / 26.2647 / 8.5335`.
  - `runs/diagnostics/xr17b_p5anchor_bestp5_failure_buckets_20260616.json`: weighted `20.0492 / 26.2136 / 8.8401`.
- Updated active gates: center `<20.182338142395018`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.

## 2026-06-16 XR-18A Interpolation Validation

- Static validation:
  - `bash -n scripts/external/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh` passed.
  - `git diff --check` passed for the runner.
  - Endpoint checkpoints existed before launch.
- Runtime validation:
  - Initial sandboxed evals hung with idle GPUs, matching the earlier sandbox CUDA issue; they were interrupted.
  - Escalated GPU0/GPU1 evals completed for alpha `0.025/0.05/0.075/0.10`.
  - Four eval summaries exist under `runs/eval_fixed255k_xr18a_xr17b_center_xr06c_p10_interp_alpha*`.
- Gate comparison:
  - Best P10 in sweep: alpha `0.10`, `20.193220179421562 / 26.20748372077942 / 8.629677173069545`.
  - No active gate promoted. Center and P5 remain XR-17B; P10 remains XR-06C.

## 2026-06-16 XR-19A P10-Recovery Micro-Polish Validation

- Static validation:
  - `bash -n scripts/external/run_xr19a_p10recovery_micro_polish.sh` passed.
  - `git diff --check` passed for the XR-19A config, runner, and updated experiment plan.
  - Config load smoke reported epochs `8`, LR `1.25e-7`, AdamW, optimizer LR `1.25e-7`.
  - Raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml`.
  - Default init and teacher checkpoints existed before launch.
- Runtime validation:
  - Escalated GPU0 launch resolved `cuda:0`.
  - Train completed epoch `8/8` with exit `0`.
  - Best checkpoint eval summaries exist for best-center, best-P10, and best-P5.
  - No active XR-19A train/eval process remains after completion; GPUs returned idle.
- Gate comparison:
  - Best-center summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_micro_adamw_lr1_25e_7_bestcenter_test_gpu0_w0_20260616_153556/eval/test/eval_summary.json`: `20.182335831437793 / 25.973640203475952 / 8.588435670307705`.
  - Best-P10 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_micro_adamw_lr1_25e_7_bestp10_test_gpu0_w0_20260616_153826/eval/test/eval_summary.json`: `20.20784169435501 / 26.232143613270352 / 8.609694181169782`.
  - Best-P5 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_micro_adamw_lr1_25e_7_bestp5_test_gpu0_w0_20260616_154053/eval/test/eval_summary.json`: `20.181213889803207 / 25.928997346333094 / 8.508503695896694`.
  - Center gate promoted to `<20.181213889803207`; P10/P5 gates remain XR-06C/XR-17B.
- Failure-bucket diagnostic:
  - `runs/diagnostics/xr19a_p10recovery_bestp10_failure_buckets_20260616.json` exists.
  - Weighted aggregate: `20.0236 / 26.3669 / 8.6357`.
  - `similarity_target < 0.1` remains weak: weighted `26.5449 / 17.5772 / 5.2257`.

## 2026-06-16 XR-20A/XR-20B P10-Recovery LR Ladder Validation

- Runtime validation:
  - XR-20A LR `2.5e-7` train completed epoch `8/8` with exit `0`.
  - XR-20B LR `5e-7` early-stopped at epoch `6` with exit `0`.
  - Six eval summaries exist for best-center, best-P10, and best-P5 checkpoints across both branches.
  - `pgrep -af "train_hbtxr.py|eval_hbtxr.py|run_xr19a_p10recovery"` returned no active process after closeout.
  - `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader` showed both RTX 5080 GPUs idle with `15 MiB` used and `0 %` utilization.
- Gate comparison:
  - XR-20A best-center summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr2p5e7_adamw_lr2_5e_7_bestcenter_test_gpu0_w0_20260616_160441/eval/test/eval_summary.json`: `20.175542894431523 / 25.884354482378278 / 8.486394848142352`.
  - XR-20A best-P10 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr2p5e7_adamw_lr2_5e_7_bestp10_test_gpu0_w0_20260616_160753/eval/test/eval_summary.json`: `20.202199470996856 / 26.031038168498448 / 8.74914996964591`.
  - XR-20A best-P5 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr2p5e7_adamw_lr2_5e_7_bestp5_test_gpu0_w0_20260616_161100/eval/test/eval_summary.json`: `20.211376798152923 / 26.00552796636309 / 8.71088467325483`.
  - XR-20B best-center summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr5e7_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_160526/eval/test/eval_summary.json`: `20.257083107743945 / 25.838861281531198 / 8.380102341515677`.
  - XR-20B best-P10 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr5e7_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_160835/eval/test/eval_summary.json`: `20.183283712182725 / 25.8312082358769 / 8.423894848142352`.
  - XR-20B best-P5 summary `runs/eval_fixed255k_xr19a_p10recovery_p5init_xr06cp10teacher_lr5e7_adamw_lr5e_7_bestp5_test_gpu1_w0_20260616_161146/eval/test/eval_summary.json`: `20.218086302280426 / 25.805698026929583 / 8.588435690743582`.
  - Center gate promoted to `<20.175542894431523`; P10/P5 gates remain XR-06C/XR-17B.
- Failure-bucket diagnostic:
  - `runs/diagnostics/xr20a_lr2p5e7_bestcenter_failure_buckets_20260616.json` exists.
  - Joined `2238/2238` rows with no missing predictions.
  - Weighted aggregate: `19.9949 / 26.0092 / 8.5335`.
  - `similarity_target < 0.1` remains weak: weighted `26.5436 / 17.8147 / 5.2257`.

## 2026-06-16 XR-21 P10-Margin Validation

- Static validation:
  - `bash -n scripts/external/run_xr21_p10margin_supportadaptive_probe.sh` passed.
  - `bash -n scripts/external/run_xr21_p10leader_margin_probe.sh` passed.
  - Raw event-count contract passed for both XR-21 configs.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` passed with `13 passed`.
  - `git diff --check` passed for the XR-21 config/runner set.
- Runtime validation:
  - Secondary `w=0.0015` train completed epoch `8/8` with exit `0`.
  - Secondary `w=0.003` early-stopped at epoch `6` with exit `0`.
  - Primary P10-leader train completed epoch `8/8` with exit `0`.
  - Eight eval summaries exist across secondary best-center/best-P10/best-P5 and primary best-P10/best-P5 checkpoints.
- Gate comparison:
  - Best secondary P10/P5 scalar came from `w=0.003` best-P10: `20.187303059441703 / 26.126701450347902 / 8.791666977746146`.
  - Primary P10-leader best-P10 and best-P5 both reached `20.26367484842028 / 26.20110617365156 / 8.696003689084733`.
  - No active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.

## 2026-06-16 XR-22 Tri-Leader Soup Validation

- Static validation:
  - `chmod +x scripts/external/average_hbtxr_checkpoints.py scripts/external/run_xr22_trileader_soup_eval.sh` applied.
  - `python3 -m py_compile scripts/external/average_hbtxr_checkpoints.py` passed.
  - `bash -n scripts/external/run_xr22_trileader_soup_eval.sh` passed.
  - `git diff --check` passed for the XR-22 script set.
- Runtime validation:
  - Six test eval summaries exist for the XR-22 soup grid.
  - `pgrep -af "train_hbtxr.py|eval_hbtxr.py|run_xr22_trileader|average_hbtxr"` returned no active process after closeout.
  - `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader` showed both RTX 5080 GPUs idle with `15 MiB` used and `0 %` utilization.
- Gate comparison:
  - Best center among XR-22: `c50_p25_f25`, `20.21947033064706 / 26.349915708814347 / 8.785289430618286`; no center promotion.
  - Best P10 among XR-22: `c34_p33_f33`, `20.236038860252926 / 26.4604599407741 / 8.869047941480364`; no P10 promotion.
  - Best P5 among XR-22: `c34_p33_f33`, `20.236038860252926 / 26.4604599407741 / 8.869047941480364`; P5 promotion.
  - Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

## 2026-06-16 XR-23 P10-Preserving Optimizer Probe Validation

- Static validation:
  - `bash -n scripts/external/run_xr23_p10_optimizer_probe.sh` passed.
  - Config load smoke reported epochs `8`, best metric `metric_track_p10_pct`, optimizer base `adamw`, and hinge weight `0.0`.
  - Optimizer registry/build smoke confirmed `adopt` and `lion` are constructible.
  - Raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml`.
  - `git diff --check` passed for the XR-23 config and runner before launch.
- Runtime validation:
  - ADOPT/GPU0 completed epoch `8/8` with train exit `0`.
  - Lion/GPU1 early-stopped at epoch `6/8` with train exit `0`.
  - Four eval summaries exist: ADOPT best-P10/best-P5 and Lion best-P10/best-P5.
- Gate comparison:
  - ADOPT best-P10 summary: `20.261837770257678 / 26.318027945927213 / 8.681122745786395`.
  - ADOPT best-P5 summary: `20.219128920350755 / 25.956633363451278 / 8.469388042177473`.
  - Lion best-P10 summary: `20.22320341382708 / 26.311650391987392 / 8.513180569240026`.
  - Lion best-P5 summary: `20.271730688640048 / 26.129677595411028 / 8.532313203811645`.
  - No center/P10/P5 gate promoted; active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

## 2026-06-16 XR-24 XR-22-Anchor P10-Recovery Validation

- Static validation:
  - `bash -n scripts/external/run_xr24_xr22_anchor_p10recovery.sh` passed.
  - Config load smoke reported epochs `8`, best metric `metric_track_p10_pct`, optimizer `adamw`, LR `1.25e-7`, hinge weight `0.0`, and teacher enabled.
  - Raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml`.
  - Input checkpoints existed: `runs/interpolated_checkpoints/xr22_trileader_soup_c34_p33_f33.pt` and XR-06C best-P10.
  - `git diff --check` passed for the XR-24 config and runner before launch.
- Runtime validation:
  - AdamW LR `1.25e-7` on GPU0 completed epoch `8/8` with train exit `0`.
  - AdamW LR `2.5e-7` on GPU1 completed epoch `8/8` with train exit `0`.
  - Four eval summaries exist: LR `1.25e-7` best-P10/best-P5 and LR `2.5e-7` best-P10/best-P5.
- Gate comparison:
  - LR `1.25e-7` best-P10 summary: `20.217043702942984 / 26.254252440588814 / 8.764881270272392`.
  - LR `1.25e-7` best-P5 summary: `20.223851100036075 / 26.415817070007325 / 8.722364262172155`.
  - LR `2.5e-7` best-P10 summary: `20.23367166178567 / 26.121599388122558 / 8.588435677119664`.
  - LR `2.5e-7` best-P5 summary: `20.238911376680647 / 26.07695653779166 / 8.791666984558105`.
  - No center/P10/P5 gate promoted; active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

## 2026-06-16 XR-25 XR-22-Anchor ADOPT-Defaults Validation

- Static validation:
  - `bash -n scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh` passed.
  - Config load smoke resolved optimizer `adopt`, LR `1.25e-7`, betas `(0.9, 0.9999)`, eps `1e-6`, and hinge weight `0.0`.
  - Optimizer build smoke reported `adopt [0.9, 0.9999] 1e-06 True`.
  - Raw event-count contract passed for `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml`.
  - Input checkpoints existed: `runs/interpolated_checkpoints/xr22_trileader_soup_c34_p33_f33.pt` and XR-06C best-P10.
  - `git diff --check` passed for the XR-25 config and runner before launch.
- Runtime validation:
  - ADOPT-defaults LR `1.25e-7` on GPU0 early-stopped at epoch `7/8` with train exit `0`.
  - ADOPT-defaults LR `2.5e-7` on GPU1 completed epoch `8/8` with train exit `0`.
  - Four eval summaries exist: LR `1.25e-7` best-P10/best-P5 and LR `2.5e-7` best-P10/best-P5.
- Gate comparison:
  - LR `1.25e-7` best-P10 summary: `20.21177260194506 / 26.46045993396214 / 8.728741809300013`.
  - LR `1.25e-7` best-P5 summary: `20.22005627495902 / 26.181123181751797 / 8.707483305249895`.
  - LR `2.5e-7` best-P10 summary: `20.23533037390028 / 26.108844266619002 / 8.64583364214216`.
  - LR `2.5e-7` best-P5 summary: `20.23760941709791 / 26.187500749315536 / 8.86607174192156`.
  - No center/P10/P5 gate promoted; active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

## 2026-06-16 XR-26 P10-Boundary Aux Head-Only Validation

- Static validation:
  - `bash -n scripts/external/run_xr26_p10boundary_aux_probe.sh` passed.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py` passed: `18 passed, 1 warning`.
  - Config load smoke resolved epochs `8`, best metric `metric_track_p10_pct`, trainable include `['track_head.*', 'track_state_aux_head.*']`, and boundary weights `0.02/0.01`.
  - Input checkpoint existed: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`.
  - `git diff --check` passed for the XR-26 runner/config/loss/test files before launch.
- Runtime validation:
  - Non-escalated launch failed with `cuda_is_available=False cuda_device_count=0`, consistent with sandbox CUDA isolation rather than driver failure.
  - Escalated GPU0 default boundary lane `0.02/0.01` early-stopped at epoch `6/8` with train exit `0`.
  - Escalated GPU1 light boundary lane `0.01/0.005` early-stopped at epoch `6/8` with train exit `0`.
  - Four eval summaries exist: default best-P10/best-P5 and light best-P10/best-P5.
- Gate comparison:
  - Default best-P10 summary: `20.32213627440589 / 26.45790890966143 / 8.535289403370449`.
  - Default best-P5 summary: `20.318065077917918 / 26.42814700944083 / 8.541666957310268`.
  - Light best-P10 summary: `20.32214218207768 / 26.45790890966143 / 8.535289403370449`.
  - Light best-P5 summary: `20.318073788711004 / 26.42814700944083 / 8.541666957310268`.
  - No center/P10/P5 gate promoted; active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

## 2026-06-16 XR-27 Track-Heatmap Validation

- Static validation:
  - `bash -n scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` passed.
  - Python compile passed for modified model/loss/training modules.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py` passed: `22 passed, 1 warning`.
  - `git diff --check` passed for the relevant XR-27 files.
  - Config smoke confirmed `distillation.state_similarity=false`, `distillation.state_weight=0.0`, and trainable include `['track_center_heatmap_head.*']`.
  - Checkpoint load smoke reported missing weights only for the new heatmap head; all existing XR-06C weights loaded without unexpected keys.
- Runtime validation:
  - The first XR-27 launch was interrupted and invalidated because `state_similarity=true` would have distilled against teacher heatmap-state outputs with random heatmap-head weights.
  - Clean GPU0 LR `1e-4` lane completed epoch `10/10` with train exit `0`.
  - Clean GPU1 LR `3e-5` lane completed epoch `10/10` with train exit `0`.
  - Four eval summaries exist: LR `1e-4` best-P10/best-P5 and LR `3e-5` best-P10/best-P5.
- Gate comparison:
  - LR `1e-4` best-P10 summary: `17.274589475563594 / 31.915391901561193 / 10.289966331209456`.
  - LR `1e-4` best-P5 summary: `17.6397743497576 / 29.80569808142526 / 10.210459525244577`.
  - LR `3e-5` best-P10 summary: `18.388037020819528 / 27.238521099090576 / 8.488520683561052`.
  - LR `3e-5` best-P5 summary: `19.03528357914516 / 24.192602627617973 / 6.854166896002633`.
  - XR-27A best-P10 promotes all active gates. New gates: center `<17.274589475563594`, P10 `>31.915391901561193`, P5 `>10.289966331209456`.

## 2026-06-16 XR-28 Consolidation Diagnostic Validation

- Generated failure-bucket JSON files:
  - `runs/diagnostics/xr28_xr27a_bestp10_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr06c_bestp10_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr20a_bestcenter_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr22_c34p33f33_failure_buckets_20260616.json`
- Each diagnostic joined `2238` test rows with `0` missing predictions.
- Overall weighted comparison:
  - XR-27A: `17.1708 / 32.1921 / 10.3219`.
  - XR-06C: `20.2065 / 26.9801 / 8.5846`.
  - XR-20A: `19.9949 / 26.0092 / 8.5335`.
  - XR-22: `20.0528 / 26.6224 / 8.8912`.
- Diagnostic conclusion: XR-27A is a real failure-bucket improvement, especially for `similarity_target <= 0.3`. Remaining risk is high-similarity `>0.9` P10 and subject `39` P10/P5.
- Runner validation:
  - `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` now supports `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME` for XR-28 best-center selection.
  - Existing XR-27A validation history has best center and best P10 both at epoch `10`; no separate current-run best-center checkpoint is needed to reinterpret XR-27A.

## 2026-06-16 XR-28 Heatmap LR Refinement Validation

- Static validation:
  - `bash -n scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` passed after adding center-selected metric controls.
  - `git diff --check` passed for the modified XR-27 runner and docs before launching the LR refinement.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py` passed: `22 passed, 1 warning`.
- Runtime validation:
  - LR `7e-5` lane on GPU0 completed epoch `10/10` with train exit `0`.
  - LR `1.5e-4` lane on GPU1 completed epoch `10/10` with train exit `0`.
  - Six eval summaries exist under `runs/eval_fixed255k_xr27_trackheatmap_xr28_heatmap_centerselect*/eval/test/eval_summary.json`.
  - `pgrep -af "train_hbtxr.py|eval_hbtxr.py|run_xr27_trackheatmap|hgtxr_xr28"` returned no live process after eval closeout.
- Gate comparison:
  - LR `7e-5` best-P10 summary: `17.935810264519283 / 28.53273880141122 / 10.105442489896502`.
  - LR `7e-5` best-P5 summary: `19.03988778250558 / 23.74872510773795 / 7.972789362498692`.
  - LR `7e-5` best-center summary: `17.55661221402032 / 31.095664044788904 / 9.788265630177088`.
  - LR `1.5e-4` best-P10 summary: `17.08485197339739 / 32.081208263124736 / 10.502551344462804`.
  - LR `1.5e-4` best-P5 summary: `17.08485197339739 / 32.081208263124736 / 10.502551344462804`.
  - LR `1.5e-4` best-center summary: `17.08485197339739 / 32.081208263124736 / 10.502551344462804`.
  - XR-28 LR `1.5e-4` promotes all active gates. New gates: center `<17.08485197339739`, P10 `>32.081208263124736`, P5 `>10.502551344462804`.

## 2026-06-16 XR-29 Post-XR-28 Diagnostic Validation

- Generated failure-bucket artifact:
  - `runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json`
- Diagnostic validation:
  - Joined `2238` test rows.
  - Missing predictions: `0`.
  - Overall weighted result: `16.9868 / 32.3454 / 10.5774`.
- Key residual buckets:
  - `similarity <=0.1`: `19.8681 / 26.60 / 7.36`.
  - `similarity >0.9`: `13.7900 / 42.27 / 10.31`, still below XR-20A/XR-22 P10/P5 in this bucket.
  - subject `39`: `19.8477 / 18.90 / 3.15`.
  - subject `42`: `21.7877 / 21.64 / 5.85`.
  - worst sessions include `user45/right/session_201`, `user42/left/session_201`, `user42/right/session_201`, and multiple subject-39 sessions with P5 `0.0`.
- Startup validation for launched LR-neighbor jobs:
  - LR `1.25e-4` GPU0 log reached raw event-count contract pass, intended XR-06C checkpoint load, `resolved_device=cuda:0`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
  - LR `1.75e-4` GPU1 log reached raw event-count contract pass, intended XR-06C checkpoint load, `resolved_device=cuda:1`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.

## 2026-06-16 XR-29 LR-Neighbor Closeout Validation

- Process closeout:
  - `pgrep -af "train_hbtxr.py|eval_hbtxr.py|run_xr27_trackheatmap|hgtxr_xr29"` returned no running XR-29 train/eval processes after closeout.
  - Six full-test eval summaries were generated: best-P10, best-P5, and best-center for each LR lane.
- LR `1.25e-4`:
  - Train exit: `0`.
  - Best-P10/best-P5/best-center test summary: `17.179786903517588 / 32.04761978558132 / 10.53443912097386`.
  - Decision: P5 improves over XR-28, but center and P10 miss the XR-28 active gates.
- LR `1.75e-4`:
  - Train exit: `0`.
  - Best-P10/best-P5/best-center test summary: `17.04961508342198 / 32.56462665285383 / 11.50467722075326`.
  - Delta versus XR-28 LR `1.5e-4`: center `-0.035236889975411856`, P10 `+0.48341838972909557`, P5 `+1.0021258762904566`.
  - Decision: promotes all active gates.
- New active gates:
  - Center `<17.04961508342198`.
  - P10 `>32.56462665285383`.
  - P5 `>11.50467722075326`.

## 2026-06-16 XR-30 LR Micro-Bracket Validation

- Startup validation:
  - LR `1.625e-4` GPU0 log reached raw event-count contract pass, intended XR-06C checkpoint load, `resolved_device=cuda:0`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
  - LR `1.875e-4` GPU1 log reached raw event-count contract pass, intended XR-06C checkpoint load, `resolved_device=cuda:1`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- Process closeout:
  - Both train lanes completed epoch `10/10` with train exit `0`.
  - `pgrep -af "eval_hbtxr.py|train_hbtxr.py|run_xr27_trackheatmap|xr30_heatmap"` returned no running XR-30 train/eval processes after closeout.
  - Six full-test eval summaries were generated: best-P10, best-P5, and best-center for each LR lane.
- LR `1.625e-4`:
  - Best-P10/best-P5/best-center test summary: `17.067689692974092 / 32.420068802152365 / 10.866496937615532`.
  - Decision: no active gate promoted.
- LR `1.875e-4`:
  - Best-P10/best-P5/best-center test summary: `17.035534060001375 / 32.42474567549569 / 11.266581957680838`.
  - Delta versus XR-29 LR `1.75e-4`: center `-0.014081023420605021`, P10 `-0.1398809773581391`, P5 `-0.23809526307242201`.
  - Decision: center gate only.
- Active gates after XR-30:
  - Center `<17.035534060001375`.
  - P10 `>32.56462665285383`.
  - P5 `>11.50467722075326`.

## 2026-06-16 XR-31 XR-29/XR-30 Interpolation Validation

- Static validation:
  - `bash -n scripts/external/run_xr31_xr29_xr30_heatmap_interp_eval.sh` passed.
  - XR-29 and XR-30 checkpoints both loaded as HBTXR checkpoints with `138` matching model tensors.
- Runtime validation:
  - Alpha `0.125`, `0.1875`, `0.25`, `0.50`, and `0.75` full-test eval summaries were generated.
  - `HBTXR_DISABLE_CUDNN=1` was active during eval.
- Test summaries:
  - Alpha `0.125`: `17.04816469124385 / 32.57313004221235 / 11.488945926938738`.
  - Alpha `0.1875`: `17.047394837651932 / 32.528487185069494 / 11.503826883860997`.
  - Alpha `0.25`: `17.04659355367933 / 32.57950758934021 / 11.503826883860997`.
  - Alpha `0.50`: `17.043099636690958 / 32.50170144353594 / 11.37500034059797`.
  - Alpha `0.75`: `17.039319237640925 / 32.5080789906638 / 11.362245225906372`.
- Decision:
  - Alpha `0.25` promotes P10 only.
  - No interpolation candidate preserves the strict XR-29 P5 gate.
  - Active gates after XR-31: center `<17.035534060001375`, P10 `>32.57950758934021`, P5 `>11.50467722075326`.
- Artifact: `docs/resources/xr31_xr29_xr30_interpolation_results_2026_06_16.md`.

## 2026-06-16 XR-32/XR-33 LR Midpoint Validation

- Startup/runtime validation:
  - XR-32 LR `1.8125e-4` ran on GPU0, completed epoch `10/10`, and exited train with code `0`.
  - XR-33 LR `1.84375e-4` ran on GPU1, completed epoch `10/10`, and exited train with code `0`.
  - Best-P10, best-P5, and best-center full-test eval summaries were generated for both lanes.
- Test summaries:
  - XR-32 LR `1.8125e-4`: `17.041867678506033 / 32.5463443006788 / 11.37500034059797`.
  - XR-33 LR `1.84375e-4`: `17.039179919447218 / 32.62074908529009 / 11.362245225906372`.
- Decision:
  - XR-32 does not promote any active gate.
  - XR-33 promotes P10 only.
  - Active gates after XR-33: center `<17.035534060001375`, P10 `>32.62074908529009`, P5 `>11.50467722075326`.
- Artifact: `docs/resources/xr32_xr33_lr_midpoint_results_2026_06_16.md`.

## 2026-06-16 XR-34 Heatmap Loss-Ratio Validation

- Startup/runtime validation:
  - XR-34A ran on GPU0 with XR-29 init/teacher, LR `1.75e-4`, and heatmap/offset/center weights `0.004/0.0015/0.0015`.
  - XR-34B ran on GPU1 with XR-33 init/teacher, LR `1.84375e-4`, and heatmap/offset/center weights `0.006/0.001/0.001`.
  - Both lanes passed raw event-count contract, loaded the intended checkpoint, resolved CUDA device, used trainable filter `6` tensors / `1,331,328` params, completed training with exit `0`, and produced best-P10, best-P5, and best-center full-test summaries.
- Test summaries:
  - XR-34A best-P10/best-P5/best-center: `17.026217068944657 / 32.430698088237214 / 10.911139822006225`.
  - XR-34B best-P10: `16.53321223940168 / 33.77168447630746 / 11.276786088943481`.
  - XR-34B best-P5/best-center: `17.0139569742339 / 32.49957566261291 / 11.226190853118897`.
- Decision:
  - XR-34A promotes center only.
  - XR-34B best-P10 promotes center and P10.
  - P5 remains below the XR-29 gate, so XR-29 remains the strict P5/unified anchor.
  - Active gates after XR-34: center `<16.53321223940168`, P10 `>33.77168447630746`, P5 `>11.50467722075326`.
- Artifacts:
  - `docs/resources/xr34_heatmap_lossratio_plan_2026_06_16.md`
  - `scripts/external/run_xr34_heatmap_lossratio_refinement.sh`
  - `runs/_logs/xr34a_heatmap_lossratio_gpu0_20260616.log`
  - `runs/_logs/xr34b_heatmap_lossratio_gpu1_20260616.log`

## 2026-06-16 XR-35 Preparation Validation

- Static validation:
  - `bash -n scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh` passed.
  - XR-29 source checkpoint exists.
  - XR-34B source checkpoint exists.
- Runtime validation:
  - GPU1 no-train interpolation sweep completed for alpha `0.03125/0.0625/0.09375/0.125`.
  - All four eval summaries exist under `runs/eval_fixed255k_xr35_xr29_xr34b_heatmap_interp_alpha*/eval/test/eval_summary.json`.
- Test summaries:
  - Alpha `0.03125`: `17.016330581051964 / 32.47534093856812 / 11.334609195164271`.
  - Alpha `0.0625`: `16.983979083810535 / 32.468963384628296 / 11.391156809670585`.
  - Alpha `0.09375`: `16.952843945366997 / 32.42432054110936 / 11.420918709891183`.
  - Alpha `0.125`: `16.923010180677686 / 32.352041605540684 / 11.255102368763515`.
- Decision:
  - No active gate promoted.
  - Active gates remain center `<16.53321223940168`, P10 `>33.77168447630746`, P5 `>11.50467722075326`.

## 2026-06-16 XR-36 Preparation Validation

- Static validation:
  - `bash -n scripts/external/run_xr36_p5_anchor_continuation.sh` passed.
  - XR-34B best-P10 source checkpoint exists.
  - XR-35 alpha `0.09375` source checkpoint exists.
  - XR-29 P5-anchor source checkpoint exists.
- Launch status:
  - Launched XR-36A on GPU0 and XR-36B on GPU1.
  - Both lanes passed raw event-count contract, loaded intended checkpoint, resolved CUDA device, and used trainable filter `6` tensors / `1,331,328` params.
  - Both lanes completed train with exit `0`, early-stopping at epoch `7/10`.
- Test summaries:
  - XR-36A best-P10: `16.59279990025929 / 34.39710958344596 / 11.738095617294311`.
  - XR-36A best-P5: `16.576952314376832 / 34.74064704350063 / 11.415816688537598`.
  - XR-36A best-center: `16.599328325475966 / 33.54209257534572 / 10.998724787575858`.
  - XR-36B best-P10: `16.53305721793856 / 34.19387831687927 / 11.502551344462804`.
  - XR-36B best-P5/best-center: `16.844227249281747 / 32.883078956604 / 11.231718029294695`.
- Decision:
  - Active gates after XR-36: center `<16.53305721793856`, P10 `>34.74064704350063`, P5 `>11.738095617294311`.
  - XR-36A best-P10 is practical P10/P5 leader; XR-36B best-P10 is strict center leader.

## 2026-06-17 XR-37 Interpolation Validation

- Static validation:
  - `bash -n scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh` passed before launch.
  - XR-36B best-P10 and XR-36A best-P10 source checkpoints existed.
- Runtime validation:
  - Ran `bash scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh cuda:1 0.10:a0p10 0.20:a0p20 0.35:a0p35 0.50:a0p50`.
  - All four eval summaries exist under `runs/eval_fixed255k_xr37_xr36b_xr36a_interp_alpha*/eval/test/eval_summary.json`.
- Test summaries:
  - Alpha `0.10`: `16.52253861086709 / 34.062075574057445 / 11.574830266407558`.
  - Alpha `0.20`: `16.51475806917463 / 34.28826605933053 / 11.44387788772583`.
  - Alpha `0.35`: `16.50803507396153 / 34.23086808749608 / 11.292517362322126`.
  - Alpha `0.50`: `16.507612899371555 / 34.33205857958112 / 11.539116007941109`.
- Decision:
  - XR-37 alpha `0.50` promotes center only.
  - Active gates after XR-37: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.738095617294311`.

## 2026-06-17 XR-38 Preparation Validation

- Static validation:
  - `bash -n scripts/external/run_xr38_center_preserve_p10p5_recovery.sh` passed.
  - XR-37 alpha `0.50` source checkpoint exists.
  - XR-36A best-P10 and best-P5 source checkpoints exist.
- Planned lanes:
  - XR-38A: GPU0, XR-37 alpha `0.50` init, XR-36A best-P10 reference, LR `2.5e-5`, best metric `metric_track_center_px`.
  - XR-38B: GPU1, XR-37 alpha `0.50` init, XR-36A best-P5 reference, LR `1.25e-5`, best metric `metric_track_p10_pct`.
  - Both lanes keep fixed255k support-adaptive contract and heatmap/offset/center loss ratio `0.005/0.001/0.001`.
- Launch validation:
  - Sandboxed launch failed for both lanes with `cuda_is_available=False`; unsandboxed launch was required for CUDA/NVML access.
  - XR-38A run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38a_xr37a050_init_xr36a_bestp10_ref_lr2p5e5_centerselect_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002109`.
  - XR-38B run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38b_xr37a050_init_xr36a_bestp5_ref_lr1p25e5_p10select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002120`.
  - Both lanes resolved to intended CUDA devices and trainable filter `6` tensors / `1,331,328` params.
- Closeout:
  - Both lanes completed with train exit `0` and early-stopped at epoch `7/10`.
  - XR-38A best-center: `16.556500560896737 / 33.96471167291914 / 11.519983339309693`.
  - XR-38A best-P10: `16.57439456837518 / 33.92984774453299 / 11.455357497079032`.
  - XR-38A best-P5: `16.513289058208464 / 34.1815484387534 / 11.56462620326451`.
  - XR-38B best-P10: `16.510086681161606 / 34.707483761651176 / 11.223214626312256`.
  - XR-38B best-P5: `16.513694180761064 / 33.93452457700457 / 11.843962955474854`.
  - Decision: XR-38B best-P5 promotes P5 only. Active gates after XR-38: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.

## 2026-06-17 XR-39 Mixed-Leader Soup Validation

- Static/input validation:
  - `bash -n scripts/external/run_xr39_mixed_leader_soup_eval.sh` passed.
  - `git diff --check` passed for the runner and XR-39 plan.
  - Source checkpoints existed: XR-37 alpha `0.50`, XR-36A best-P5, XR-38B best-P5.
  - Model-key compatibility passed: all three checkpoints have `138` model keys and matching key sets.
- Runtime validation:
  - Ran lanes A/C on GPU0 and lanes B/D on GPU1.
  - 13 full-test eval summaries were generated.
  - No `eval_hbtxr.py` or XR-39 runner process remained after closeout; GPUs returned to `15 MiB` and `0%` utilization.
- Test summaries:
  - `c60p25f15`: `16.491779099191938 / 34.5306130204882 / 11.50000034059797`.
  - `c25p45f30`: `16.503940873486656 / 35.02295998845781 / 11.460459525244577`.
  - `c70p20f10`: `16.492875189440593 / 34.34566400391715 / 11.744473137174333`.
- Decision:
  - `c60p25f15` promotes center.
  - `c25p45f30` promotes P10.
  - No XR-39 soup promotes P5; XR-38B best-P5 remains P5 leader.
  - Active gates after XR-39: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.

## 2026-06-17 XR-40 XR39-Leader/P5 Soup Validation

- Static/input validation:
  - `bash -n scripts/external/run_xr40_xr39leader_p5_soup_eval.sh` passed.
  - `git diff --check` passed for the runner and XR-40 plan.
  - Source checkpoints existed: XR-39 `c60p25f15`, XR-39 `c25p45f30`, XR-38B best-P5.
  - Model-key compatibility passed: all three checkpoints have `138` model keys and matching key sets.
- Runtime validation:
  - Ran lane A on GPU0 and lane B on GPU1.
  - 7 full-test eval summaries were generated.
  - No `eval_hbtxr.py`, `train_hbtxr.py`, or XR-40 runner process remained after closeout.
- Test summaries:
  - Best center lane `c45p35f20`: `16.493494159834725 / 34.77508579662868 / 11.255527530397687`.
  - Best P10 lane `c40p40f20`: `16.494016419138227 / 34.7750858102526 / 11.249149976457868`.
  - Best P5 lane `c35p25f40`: `16.49522715806961 / 34.64115719795227 / 11.529762240818568`.
- Decision:
  - No center/P10/P5 gate promoted.
  - Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.
  - Prefer trainable loss-ratio fallback over more no-train soup expansion.

## 2026-06-17 XR-41 Loss-Ratio P5 Fallback Validation

- Static/input validation:
  - `bash -n scripts/external/run_xr41_lossratio_p5_fallback.sh` passed.
  - `git diff --check` passed for the runner and XR-41 plan.
  - Required source checkpoints existed: XR-38B best-P5 and XR-39 `c25p45f30`.
- Runtime validation:
  - XR-41A ran on GPU0 with XR-38B best-P5 init, LR `1e-5`, best metric `metric_track_p5_pct`.
  - XR-41B ran on GPU1 with XR-39 `c25p45f30` init, LR `8e-6`, best metric `metric_track_p10_pct`.
  - Both lanes completed train/eval with exit `0` and early-stopped at epoch `7/10`.
  - No `eval_hbtxr.py`, `train_hbtxr.py`, or XR-41 runner process remained after closeout; GPUs returned to `15 MiB` and `0%` utilization.
- Test summaries:
  - XR-41A best-P10: `16.519287032740458 / 34.66199056080409 / 11.25637790134975`.
  - XR-41A best-P5: `16.519514334201812 / 34.09821502821786 / 11.868197652271816`.
  - XR-41B best-P10: `16.50909768513271 / 34.426021228517804 / 11.286139808382307`.
  - XR-41B best-P5: `16.50978491306305 / 34.228742252077375 / 11.734694249289376`.
- Decision:
  - XR-41A best-P5 promotes P5 only.
  - Active gates after XR-41: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.

## 2026-06-17 XR-42 P5-Preserve Low-Drift Validation

- Static/input validation:
  - `bash -n scripts/external/run_xr42_p5_preserve_lowdrift.sh` passed.
  - `git diff --check` passed for the runner and XR-42 plan.
  - Required source checkpoints existed: XR-39 `c60p25f15`, XR-39 `c25p45f30`, and XR-41A best-P5.
  - GPT5.5 read-only evaluator reviewed the lane design and recommended XR39-anchored trainable repair over another no-train soup.
- Runtime validation:
  - XR-42A ran on GPU0 with XR-39 center init, XR-41A best-P5 teacher/reference, LR `3e-6`, best metric `metric_track_center_px`.
  - XR-42B ran on GPU1 with XR-39 P10 init, XR-41A best-P5 teacher/reference, LR `3e-6`, best metric `metric_track_p10_pct`.
  - Both lanes passed raw event-count contract and trainable filter validation: `6` tensors / `1,331,328` trainable params.
  - Both lanes completed train/eval with exit `0` and early-stopped at epoch `7/10`.
  - No `eval_hbtxr.py`, `train_hbtxr.py`, or XR-42 runner process remained after closeout; GPUs returned to `15 MiB` and `0%` utilization.
- Test summaries:
  - XR-42A best-P10: `16.49802110535758 / 34.48299399103437 / 11.366922119685581`.
  - XR-42A best-P5: `16.498888087272643 / 34.30739874839783 / 11.347789451054163`.
  - XR-42A best-center: `16.498888087272643 / 34.30739874839783 / 11.347789451054163`.
  - XR-42B best-P10: `16.505801352432798 / 34.3171777180263 / 11.278911903926305`.
  - XR-42B best-P5: `16.508924693720683 / 34.29506881577628 / 11.323554761069161`.
- Decision:
  - No center/P10/P5 gate promoted.
  - Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
# 2026-06-17 XR-42C Validation

## Execution

- Script: `scripts/external/run_xr42c_explicit_distill_recovery.sh`
- Plan: `docs/resources/xr42c_explicit_distill_recovery_plan_2026_06_17.md`
- Static validation: `bash -n scripts/external/run_xr42c_explicit_distill_recovery.sh`
- Initial sandbox run failed because CUDA/NVML was unavailable inside sandbox:
  - `cuda_is_available=False`
  - `cuda_device_count=0`
- Approved GPU runtime succeeded.

## Runtime Evidence

- XR-42C-A log: `runs/_logs/xr42c_explicit_distill_a_gpu0_20260617_022713.log`
- XR-42C-B log: `runs/_logs/xr42c_explicit_distill_b_gpu1_20260617_022714.log`
- GPU state after completion: both GPUs idle, `15 MiB`, `0%`.

## Results

Active gates before and after XR-42C:

- Center: `<16.491779099191938`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

XR-42C-A:

- Train: early-stopped at epoch `7/10`, exit `0`
- best-P10: `16.4945193869727 / 34.34566405841282 / 11.454081984928676`
- best-P5: `16.4945193869727 / 34.34566405841282 / 11.454081984928676`
- best-center: `16.491862688745773 / 34.712585769380844 / 11.472364275796073`

XR-42C-B:

- Train: completed epoch `10/10`, exit `0`
- best-P10: `16.502160484450204 / 34.39710965156555 / 11.22236428941999`
- best-P5: `16.503420085566386 / 34.509779705320085 / 11.221513932091849`
- best-center: skipped because best metric was `metric_track_p10_pct`

## Judgment

XR-42C did not promote any active gate.

The closest miss was XR-42C-A best-center: `16.491862688745773`, which is `0.000083589553835` worse than the center gate. P10 and P5 remained materially below the active leaders.

Decision: stop the XR41 P5-teacher recovery line unless a no-train interpolation is used only as a cheap diagnostic. The next trainable branch should directly target low-similarity and subject-39 residual buckets.

# 2026-06-17 XR-43 Validation

## Execution

- Script: `scripts/external/run_xr43_xr39center_xr42ca_interp_eval.sh`
- Type: no-train interpolation/eval
- Inputs:
  - A: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c60p25f15.pt`
  - B: `runs/raw_mode1_stage2_count255000_adamw_lr1e_6_weakdistill_trackonly_heatmapstate_g32_hm0_004_off0_0015_c0_0015_xr42ca_xr39center_init_xr41ap5_teacher_tinystate_state0_0005_pred0_0_feat0_0_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_022713/train/best_metric_track_center_px.pt`
- Log: `runs/_logs/xr43_xr39center_xr42ca_interp_gpu0_20260617_025024.log`
- Static validation: `bash -n scripts/external/run_xr43_xr39center_xr42ca_interp_eval.sh`
- Runtime: GPU0 approved CUDA eval, exit `0`

## Results

Active gates before XR-43:

- Center: `<16.491779099191938`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

XR-43 interpolation results:

| alpha | center | P10 | P5 |
|---:|---:|---:|---:|
| 0.125 | 16.49161465849195 | 34.44132730620248 | 11.389456115450178 |
| 0.25 | 16.491504199164254 | 34.33078307424273 | 11.478741829735892 |
| 0.5 | 16.491429926667895 | 34.512755850383215 | 11.531888089861189 |
| 0.75 | 16.491550181593215 | 34.60204156466893 | 11.576530947004045 |

## Judgment

XR-43 alpha `0.5` promoted the center gate from `16.491779099191938` to `16.491429926667895`.

P10 and P5 gates remain unchanged:

- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

Decision: keep XR-43 alpha `0.5` as the active center leader. Do not treat this as evidence that more XR41-teacher training should continue; it was a cheap interpolation rescue after XR-42C nearly matched the center leader.

# 2026-06-17 XR-44 Validation

## Execution

- Script: `scripts/external/run_xr44_updated_trileader_soup_eval.sh`
- Type: no-train tri-leader soup/eval
- Inputs:
  - Center: `runs/interpolated_checkpoints/xr43_xr39center_xr42ca_center_interp_a0_5.pt`
  - P10: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`
  - P5: `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_weakdistill_trackonly_heatmapstate_g32_hm0_004_off0_0015_c0_0015_xr41a_p5init_xr38b_bestp5_xr39p10_ref_lossratio_lr1e5_p5select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_013052/train/best_track_p5.pt`
- Runtime: GPU0/GPU1 parallel eval, exit `0`

## Results

| soup | center | P10 | P5 |
|---|---:|---:|---:|
| c70p20f10 | 16.49202539069312 | 34.685800075531006 | 11.264030940192086 |
| c60p25f15 | 16.492678804056986 | 34.64753478595188 | 11.255527537209646 |
| c55p30f15 | 16.493043976170675 | 34.692177643094745 | 11.306547941480364 |
| c50p35f15 | 16.493474864959715 | 34.78784090450832 | 11.306547941480364 |
| c45p40f15 | 16.493977418967656 | 34.7963443006788 | 11.255527530397687 |
| c40p45f15 | 16.49455840757915 | 34.7368205002376 | 11.20450711931501 |
| c35p45f20 | 16.49523995944432 | 34.692177643094745 | 11.21088467325483 |
| c30p45f25 | 16.49610698393413 | 34.63265381540571 | 11.315051344462804 |

## Judgment

No active gate promoted. XR-44 worsened center relative to XR-43 and did not recover P10/P5 enough to beat XR-39/XR-41.

# 2026-06-17 XR-45 Validation

## Execution

- Script: `scripts/external/run_xr45_lowsim_heatmap_refresh.sh`
- Lane A: low-similarity `<=0.3` specialist manifest, GPU0, XR-43 center init/teacher, LR `1e-5`, best metric `metric_track_center_px`
- Lane B: full manifest refresh, GPU1, XR-43 center init, XR-39 P10 teacher, LR `1e-5`, best metric `metric_track_p10_pct`
- Leak check: subject `39` exists only in `test`, not in `train`/`val`; direct subject-39 training was rejected.
- Runtime: A completed `10/10`; B early-stopped at epoch `7/10`; train/eval exits `0`.
- Logs:
  - `runs/_logs/xr45_lowsim_heatmap_refresh_a_gpu0_20260617_031649.log`
  - `runs/_logs/xr45_lowsim_heatmap_refresh_b_gpu1_20260617_031651.log`

## Results

Active gates before XR-45:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-45A best-P10 | 17.03435743876866 | 32.714711679731096 | 11.164966344833374 |
| XR-45A best-P5 | 16.849295384543282 | 33.63605521747044 | 11.585034343174526 |
| XR-45A best-center | 17.808542433806828 | 30.72661645753043 | 9.841411910738264 |
| XR-45B best-P10 | 16.50801784992218 | 34.5969395501273 | 11.428571782793318 |
| XR-45B best-P5 | 16.50964238813945 | 33.96726266316005 | 11.820578595570156 |

## Judgment

XR-45 did not promote any active gate.

The full-manifest lane was useful as a negative result: it stayed much closer to the active leaders than the low-sim subset lane, but still missed center by about `0.0166`, P10 by about `0.4260`, and P5 by about `0.0476`. Subset-only heatmap specialization should not be continued as-is.

# 2026-06-17 XR-46 Validation

## Execution

- Script: `scripts/external/run_xr46_center_p10_compat_soup_eval.sh`
- Plan: `docs/resources/xr46_center_p10_compat_soup_plan_2026_06_17.md`
- Additional sub-agent recommendation: run XR43-manifold micro-alpha sweep around alpha `0.55/0.60/0.65/0.70/0.80/0.875`.
- Static validation:
  - `bash -n scripts/external/run_xr46_center_p10_compat_soup_eval.sh`
  - Required source checkpoints exist.
- Runtime: GPU0/GPU1 full-test eval, all exits `0`.

## Results

Active gates before XR-46:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

XR-46 A/B compatibility results:

| tag | center | P10 | P5 |
|---|---:|---:|---:|
| a0p125 | 16.491489429133278 | 34.563776268277849 | 11.517007132938931 |
| a0p25 | 16.491949367523194 | 34.659439550127303 | 11.412840468542916 |
| a0p50 | 16.494027202469962 | 34.793368135179790 | 11.317177200317383 |
| a0p75 | 16.498019000462122 | 34.852891956056865 | 11.370323453630720 |
| a0p875 | 16.500783746583121 | 35.022959988457814 | 11.421343871525355 |
| c70p20x05f05 | 16.491824064935958 | 34.590136814117429 | 11.264030940192086 |
| c60p25x10f05 | 16.492247397559030 | 34.647534785951883 | 11.204507126126970 |
| c55p30x10f05 | 16.492606668812886 | 34.736820500237599 | 11.255527530397687 |
| c50p35x10f05 | 16.493030984061104 | 34.787840904508322 | 11.306547941480364 |
| c45p40x10f05 | 16.493524081366402 | 34.867772872107366 | 11.306547941480364 |

XR-46C XR43-manifold micro-alpha results:

| alpha | center | P10 | P5 |
|---:|---:|---:|---:|
| 0.55 | 16.491439385073527 | 34.512755850383215 | 11.531888089861189 |
| 0.60 | 16.491455343791415 | 34.512755850383215 | 11.531888089861189 |
| 0.65 | 16.491479851518360 | 34.512755850383215 | 11.576530947004045 |
| 0.70 | 16.491511028153557 | 34.557398707526069 | 11.576530947004045 |
| 0.80 | 16.491597735881804 | 34.602041564668930 | 11.576530947004045 |
| 0.875 | 16.491683168070658 | 34.661565365110128 | 11.576530947004045 |

## Judgment

XR-46 did not promote any active gate.

The `a0p875` compatibility interpolation numerically ties the existing P10 gate within floating precision, but center and P5 regress and it is not a meaningful new leader. The sub-agent-recommended XR43-manifold micro sweep confirms there is a stable center-near plateau, but it does not recover enough P10/P5 to beat XR39/XR41 leaders.

Decision: stop no-train checkpoint-space sweeps around XR39/XR42C/XR45B anchors. Next experiment should be trainable full-manifest calibration with an explicit center-preservation term or teacher anchor from XR-43.

# 2026-06-17 XR-47 Validation

## Execution

- Script: `scripts/external/run_xr47_center_preserve_full_calibration.sh`
- Plan: `docs/resources/xr47_center_preserve_full_calibration_plan_2026_06_17.md`
- Type: trainable full-manifest calibration
- Init/teacher: `runs/interpolated_checkpoints/xr43_xr39center_xr42ca_center_interp_a0_5.pt`
- Shared settings: LR `2e-6`, epochs `8`, AdamW, heatmap/offset/center `0.004/0.0015/0.0025`, state distillation enabled with weight `0.0005`
- Lane A: best metric `metric_track_p10_pct`, GPU0
- Lane B: best metric `metric_track_p5_pct`, GPU1
- Static validation:
  - `bash -n scripts/external/run_xr47_center_preserve_full_calibration.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr47_center_preserve_full_calibration.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr47_center_preserve_full_calibration.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr47_center_preserve_full_calibration.sh docs/resources/xr47_center_preserve_full_calibration_plan_2026_06_17.md`
- Runtime: train/eval exits `0`.

## Results

Active gates before XR-47:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-47A best-P10 | 16.496871314729962 | 34.372449772698538 | 11.298894902638027 |
| XR-47A best-P5 | 16.495964876243047 | 34.196854530061991 | 11.303146593911308 |
| XR-47B best-P10 | 16.496871314729962 | 34.372449772698538 | 11.298894902638027 |
| XR-47B best-P5 | 16.497679948806763 | 34.444728687831336 | 11.592262281690324 |

## Judgment

XR-47 did not promote any active gate.

The lower LR and XR-43 self-teacher state distillation were not enough to preserve center while improving P10/P5. This is a negative result for same-head full-manifest calibration. Further progress likely needs a mechanism change rather than another LR sweep around this configuration.

# 2026-06-17 XR-48 Validation

## Execution

- Script: `scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh`
- Plan: `docs/resources/xr48_p10_boundary_heatmap_calibration_plan_2026_06_17.md`
- Type: trainable full-manifest heatmap-state calibration with metric-aligned P10 boundary loss
- Init: `runs/interpolated_checkpoints/xr43_xr39center_xr42ca_center_interp_a0_5.pt`
- Teacher: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`
- Shared settings: LR `2e-6`, epochs `8`, AdamW, heatmap/offset `0.004/0.0015`, P10 boundary margin/band/temperature `10.0/4.0/1.0`, trainable scope `track_center_heatmap_head.*`
- Lane A: center L2 `0.0030`, P10 boundary `0.010`, state distill `0.00025`, GPU0
- Lane B: center L2 `0.0040`, P10 boundary `0.006`, state distill `0.00040`, GPU1
- Static validation:
  - `bash -n scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh docs/resources/xr48_p10_boundary_heatmap_calibration_plan_2026_06_17.md`
- Leakage check: subject `39` count was train `0`, val `0`, test `144`.
- Runtime: sandboxed CUDA failed as expected with `cuda_is_available=False`; approved GPU runtime succeeded. Both train/eval exits were `0`.

## Results

Active gates before XR-48:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-48A best-P10 | 16.49682899543217 | 34.431973586763654 | 11.298894902638027 |
| XR-48A best-P5 | 16.495972565242223 | 34.19685453006199 | 11.303146593911308 |
| XR-48B best-P10 | 16.49668344429561 | 34.431973586763654 | 11.417942530768258 |
| XR-48B best-P5 | 16.50094030073711 | 34.45663345200675 | 11.52423505101885 |

## Judgment

XR-48 did not promote any active gate.

P10-boundary heatmap head-only calibration was not enough. The best XR-48 P10 remained below XR-39's `35.02295998845781`, the best XR-48 P5 remained below XR-41A's `11.868197652271816`, and all center values drifted above XR-43's `16.491429926667895`.

Decision: close this branch as a negative result. Next experiment should use a larger mechanism change: explicit P5-boundary objective, teacher retraining, or non-head-only adapter update.

# 2026-06-17 XR-49 Validation

## Execution

- Script: `scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh`
- Plan: `docs/resources/xr49_p5_boundary_heatmap_calibration_plan_2026_06_17.md`
- Type: trainable full-manifest heatmap-state calibration with explicit 5px boundary objective
- Shared settings: LR `2e-6`, epochs `8`, AdamW, heatmap/offset `0.004/0.0015`, boundary margin/band/temperature `5.0/2.0/0.75`, trainable scope `track_center_heatmap_head.*`
- Lane A: XR-41A best-P5 init/teacher, center L2 `0.0015`, boundary `0.012`, state distill `0.00050`, best metric `metric_track_p5_pct`, GPU0
- Lane B: XR-43 center init, XR-41A best-P5 teacher, center L2 `0.0045`, boundary `0.008`, state distill `0.00025`, best metric `metric_track_p5_pct`, GPU1
- Static validation:
  - `bash -n scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh docs/resources/xr49_p5_boundary_heatmap_calibration_plan_2026_06_17.md`
- Runtime notes:
  - XR-49B initial launch failed before training with `OSError: [Errno 36] File name too long`; the experiment-name template was shortened and rerun.
  - XR-49A training completed, but the old shell wrapper failed during automated eval. Manual eval with `HBTXR_DISABLE_CUDNN=1` completed and produced accepted summaries.
  - XR-49B train/eval completed with exit `0`; `best_metric_track_center_px` checkpoint was missing and skipped.

## Results

Active gates before XR-49:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-49A best-P10 | 16.52969342981066 | 34.083759260177615 | 11.41369082587106 |
| XR-49A best-P5 | 16.532351425715856 | 33.93452455656869 | 11.70068063054766 |
| XR-49B best-P10 | 16.496637644086565 | 34.431973586763654 | 11.417942530768258 |
| XR-49B best-P5 | 16.497121804101127 | 34.444728687831336 | 11.54124187060765 |

## Judgment

XR-49 did not promote any active gate.

The best XR-49 P5 was `11.70068063054766`, still below XR-41A's active P5 gate `11.868197652271816`. XR-49B had the least center drift, but `16.496637644086565` remains worse than XR-43's active center gate `16.491429926667895`, and its P10/P5 values are below the current leaders.

Decision: stop boundary-only heatmap head calibration. XR-50 should use a larger mechanism change such as adapter or track-event-adapter fine-tuning with center preservation.

# 2026-06-17 XR-50 Validation

## Execution

- Script: `scripts/external/run_xr50_event_adapter_heatmap_calibration.sh`
- Plan: `docs/resources/xr50_event_adapter_heatmap_calibration_plan_2026_06_17.md`
- Type: trainable full-manifest heatmap-state calibration with event-path adaptation
- Trainable scope:
  - `track_center_heatmap_head.*`
  - `event_adapter.*`
  - `patch_frontend.event_embed.proj.*`
- Shared settings: LR `1e-6`, epochs `8`, AdamW, heatmap/offset `0.004/0.0015`, state distill `0.00025`, adaptive count `160k/255k/384k`
- Lane A: XR-43 center init, XR-41A P5 teacher, best metric P5, center L2 `0.0045`, 5px boundary weight `0.008`, GPU0
- Lane B: XR-43 center init, XR-39 P10 teacher, best metric P10, center L2 `0.0050`, 10px boundary weight `0.006`, GPU1
- Static validation:
  - `bash -n scripts/external/run_xr50_event_adapter_heatmap_calibration.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr50_event_adapter_heatmap_calibration.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr50_event_adapter_heatmap_calibration.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr50_event_adapter_heatmap_calibration.sh docs/resources/xr50_event_adapter_heatmap_calibration_plan_2026_06_17.md`
- Runtime validation:
  - Both lanes passed raw event-count contract.
  - Both lanes loaded XR-43 init checkpoint.
  - Both lanes reported `trainable_tensors=14` and `trainable_params=1504320/4638746`.
  - Both lanes completed train/eval with exit `0`.
  - `best_metric_track_center_px` checkpoint was absent and skipped; center promotion is from evaluated best-P5/best-P10 checkpoints.

## Results

Active gates before XR-50:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-50A best-P10 | 16.483389932768684 | 34.3227048942021 | 11.933673824582781 |
| XR-50A best-P5 | 16.482811435631344 | 34.48086814199175 | 11.863520765304566 |
| XR-50B best-P10 | 16.483026616913932 | 34.394133479254585 | 11.978316681725639 |
| XR-50B best-P5 | 16.482515714849743 | 34.48086814199175 | 11.81250035422189 |

## Judgment

XR-50 promoted two active gates.

- Center promoted to XR-50B best-P5: `16.482515714849743`.
- P5 promoted to XR-50B best-P10: `11.978316681725639`.
- P10 remains XR-39: `35.02295998845781`.

New active gates:

- Center: `<16.482515714849743`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

Decision: event-path adaptation is validated. The next experiment should specifically recover P10 while preserving the new XR-50 center/P5 gains.

# 2026-06-17 XR-51 Validation

## Execution

- Script: `scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh`
- Plan: `docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md`
- Type: no-train checkpoint interpolation/soup diagnostic for P10 recovery after XR-50 center/P5 promotion
- Lane A: XR-50B best-P10 to XR-39 P10 interpolation, alpha `0.05/0.10/0.15/0.20/0.25/0.35`, GPU0
- Lane B: XR-50B best-P10, XR-50B best-P5, and XR-39 P10 tri-soup, weights `80/10/10`, `75/10/15`, `70/10/20`, `70/15/15`, `65/15/20`, `60/20/20`, GPU1
- Static validation:
  - `bash -n scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md`
- Runtime validation:
  - Both lanes completed with exit `0`.
  - All 12 full-test eval summaries were generated.
  - Post-run process check found no XR-51 eval runner still active.
  - Post-run GPU check showed GPU0/GPU1 idle at `15 MiB`, `0%`.

## Results

Active gates before XR-51:

- Center: `<16.482515714849743`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

| candidate | center | P10 | P5 |
|---|---:|---:|---:|
| XR-51B `t60c20p20` | 16.478984827655 | 34.561650446483 | 11.716837085996 |
| XR-51B `t65c15p20` | 16.479341397967 | 34.561650446483 | 11.665816674914 |
| XR-51B `t70c15p15` | 16.479410881656 | 34.621174260548 | 11.738095596858 |
| XR-51B `t70c10p20` | 16.479780200550 | 34.621174260548 | 11.612670407976 |
| XR-51B `t75c10p15` | 16.479912798745 | 34.621174260548 | 11.729592193876 |
| XR-51B `t80c10p10` | 16.480233781678 | 34.576531403405 | 11.780612598147 |
| XR-51A `a0p25` | 16.480831880229 | 34.710459961210 | 11.517007139751 |
| XR-51A `a0p20` | 16.480896479743 | 34.650936160769 | 11.517007139751 |
| XR-51A `a0p15` | 16.481148343427 | 34.517007589340 | 11.633928925650 |
| XR-51A `a0p35` | 16.481268215179 | 34.504252474649 | 11.346939107350 |
| XR-51A `a0p10` | 16.481587026800 | 34.472364732197 | 11.780612598147 |
| XR-51A `a0p05` | 16.482212608201 | 34.442602831977 | 11.978316681726 |

## Judgment

XR-51 promoted center only.

- Center promoted to XR-51B `t60c20p20`: `16.478984827655`.
- P10 remains XR-39 `c25p45f30`: `35.02295998845781`.
- P5 remains XR-50B best-P10 / XR-51A `a0p05` tie: `11.978316681725639`.

New active gates:

- Center: `<16.478984827655`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

Decision: close pure no-train P10 recovery for now. XR-52 should be trainable and should combine XR-39 P10 teacher/reference with explicit XR-50/XR-51 P5 preservation.

# 2026-06-17 XR-52 Validation

## Execution

- Script: `scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh`
- Plan: `docs/resources/xr52_trainable_p10_recovery_from_xr51_plan_2026_06_17.md`
- Type: trainable event-adapter P10 recovery from XR-51 anchors
- Trainable scope:
  - `track_center_heatmap_head.*`
  - `event_adapter.*`
  - `patch_frontend.event_embed.proj.*`
- Lane A: XR-51B `t60c20p20` init, XR-39 P10 teacher, LR `5e-7`, center L2 `0.0060`, P10 boundary `0.008`, state distill `0.00035`, GPU0
- Lane B: XR-51A `a0p05` init, XR-39 P10 teacher, LR `3e-7`, center L2 `0.0070`, P10 boundary `0.005`, state distill `0.00050`, GPU1
- Static validation:
  - `chmod +x scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh`
  - `bash -n scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh b cuda:1`
  - required checkpoint checks for XR-51 `t60c20p20`, XR-51 `a0p05`, and XR-39 `c25p45f30`
  - `git diff --check -- scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh docs/resources/xr52_trainable_p10_recovery_from_xr51_plan_2026_06_17.md docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md docs/track/PROGRESS.md docs/Validation.md docs/track/log.md docs/track/TODO.md`
- Runtime validation:
  - Both lanes passed raw event-count startup.
  - Both lanes loaded the intended init checkpoints.
  - Both lanes reported `trainable_tensors=14`, `trainable_params=1504320/4638746`.
  - Both lanes early-stopped at epoch `7/8`.
  - Both train/eval exits were `0`.
  - `best_metric_track_center_px` was missing and skipped by design.
  - Post-run process check found no XR-52 train/eval processes.
  - Post-run GPU check showed GPU0/GPU1 idle at `15 MiB`, `0%`.

## Results

Active gates before XR-52:

- Center: `<16.478984827655`
- P10: `>35.02295998845781`
- P5: `>11.978316681725639`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-52A best-P10 | 16.483039610726 | 34.517007589340 | 11.625425529480 |
| XR-52A best-P5 | 16.470460832119 | 34.667942953110 | 11.840136425836 |
| XR-52B best-P10 | 16.472449232851 | 34.456633458819 | 12.017432342257 |
| XR-52B best-P5 | 16.479750164918 | 34.427721875054 | 11.929422126498 |

## Judgment

XR-52 promoted center and P5, but did not recover P10.

- Center promoted to XR-52A best-P5: `16.470460832119`.
- P5 promoted to XR-52B best-P10: `12.017432342257`.
- P10 remains XR-39 `c25p45f30`: `35.02295998845781`.

New active gates:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Decision: event-adapter continuation is still productive for center/P5, but low-LR XR-39-teacher P10 recovery is not enough. Next branch should isolate P10 recovery through checkpoint-space recombination with the new XR-52 leaders, or use stronger P10 teacher/model retraining.

# 2026-06-17 XR-53 Preparation Validation

## Execution Readiness

- Script: `scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh`
- Plan: `docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md`
- Type: no-train checkpoint interpolation/soup diagnostic for P10 recovery after XR-52 center/P5 promotion
- Lane A: XR-52A best-P5 to XR-39 P10 interpolation, alpha `0.03/0.06/0.10/0.15/0.20/0.30`, intended GPU0
- Lane B: XR-52A best-P5, XR-52B best-P10, and XR-39 P10 tri-soup, weights `50/35/15`, `45/35/20`, `40/35/25`, `35/35/30`, `30/40/30`, `25/35/40`, intended GPU1
- Static validation:
  - `chmod +x scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh`
  - `bash -n scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh`
  - required checkpoint checks for XR-52A best-P5, XR-52B best-P10, and XR-39 P10
  - `DRY_RUN=1 bash scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh docs/resources/xr52_trainable_p10_recovery_from_xr51_plan_2026_06_17.md docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md docs/track/PROGRESS.md docs/Validation.md docs/track/log.md docs/track/TODO.md`

## Status

XR-53 is prepared but not executed.

Active gates for XR-53:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

# 2026-06-17 XR-53 Validation

## Execution

- Script: `scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh`
- Plan: `docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md`
- Type: no-train checkpoint interpolation/soup diagnostic
- Lane A: XR-52A best-P5 to XR-39 P10 interpolation, alpha `0.03/0.06/0.10/0.15/0.20/0.30`, GPU0
- Lane B: XR-52A best-P5, XR-52B best-P10, and XR-39 P10 tri-soup, weights `50/35/15`, `45/35/20`, `40/35/25`, `35/35/30`, `30/40/30`, `25/35/40`, GPU1
- Runtime validation:
  - Both lanes completed with exit `0`.
  - All 12 full-test eval summaries were generated.
  - Post-run process check found no XR-53 eval processes.
  - Post-run GPU check showed GPU0/GPU1 idle at `15 MiB`, `0%`.

## Results

Active gates before XR-53:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

| candidate | center | P10 | P5 |
|---|---:|---:|---:|
| XR-53A `a0p03` | 16.470736992359 | 34.623300095967 | 11.840136425836 |
| XR-53A `a0p06` | 16.471054373469 | 34.512755877631 | 11.789116014753 |
| XR-53A `a0p10` | 16.471541745322 | 34.512755877631 | 11.738095603670 |
| XR-53A `a0p15` | 16.472256399904 | 34.623300109591 | 11.525510549545 |
| XR-53A `a0p20` | 16.473095047474 | 34.578657252448 | 11.370323467255 |
| XR-53A `a0p30` | 16.475181637491 | 34.616922562463 | 11.310799653190 |
| XR-53B `c50f35p15` | 16.472414144448 | 34.512755877631 | 11.627551378523 |
| XR-53B `c45f35p20` | 16.473200055531 | 34.512755877631 | 11.355442510332 |
| XR-53B `c40f35p25` | 16.474105545453 | 34.608419152669 | 11.310799653190 |
| XR-53B `c35f35p30` | 16.475128199373 | 34.497874947957 | 11.310799653190 |
| XR-53B `c30f40p30` | 16.475147432940 | 34.497874947957 | 11.310799653190 |
| XR-53B `c25f35p40` | 16.477522144999 | 34.497874947957 | 11.266156796047 |

## Judgment

XR-53 did not promote any active gate.

- Best center: XR-53A `a0p03`, `16.470736992359`, missing active center by about `0.000276`.
- Best P10: XR-53A `a0p15`, `34.623300109591`, far below active P10 `35.02295998845781`.
- Best P5: XR-53A `a0p03`, `11.840136425836`, far below active P5 `12.017432342257`.

Active gates remain:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Decision: skip XR-54 near-tie continuation because XR-53 produced no P10 near-tie. Next branch should be XR-55 XR39-anchored expanded-scope P10 model branch.

# 2026-06-17 XR-55 Validation

## Execution

- Script: `scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh`
- Plan: `docs/resources/xr55_xr39_p10_anchor_expanded_scope_plan_2026_06_17.md`
- Type: trainable XR39-anchored expanded-scope P10 branch
- Trainable scope:
  - `track_center_heatmap_head.*`
  - `event_adapter.*`
  - `patch_frontend.event_embed.proj.*`
  - `backbone.attn_stages.5.*`
  - `backbone.mlp_stages.5.*`
  - `backbone.norm.*`
- Lane A: XR-39 P10 init/self-teacher, LR `2e-7`, center L2 `0.0025`, P10 boundary `0.012`, state distill `0.00025`, GPU0
- Lane B: XR-39 P10 init, XR-52B P5 teacher, LR `3e-7`, center L2 `0.0030`, P10 boundary `0.010`, state distill `0.00035`, GPU1
- Static validation:
  - `chmod +x scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh`
  - `bash -n scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh b cuda:1`
  - `git diff --check -- scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh docs/resources/xr55_xr39_p10_anchor_expanded_scope_plan_2026_06_17.md docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md docs/track/PROGRESS.md docs/Validation.md docs/track/log.md docs/track/TODO.md`
- Runtime validation:
  - Both lanes passed raw event-count startup.
  - Both lanes loaded the intended init checkpoints.
  - Both lanes reported `trainable_tensors=32`, `trainable_params=1949568/4638746`.
  - Both lanes completed epoch `8/8`.
  - Both train/eval exits were `0`.
  - `best_metric_track_center_px` was missing and skipped by design.
  - Post-run process check found no XR-55 train/eval processes.
  - Post-run GPU check showed GPU0/GPU1 idle at `15 MiB`, `0%`.

## Results

Active gates before XR-55:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

| lane/checkpoint | center | P10 | P5 |
|---|---:|---:|---:|
| XR-55A best-P10 | 16.501659829276 | 34.399235514232 | 11.324830266408 |
| XR-55A best-P5 | 16.497514723028 | 34.793368141992 | 11.382228217806 |
| XR-55B best-P10 | 16.497369331973 | 34.567177718026 | 11.202381290708 |
| XR-55B best-P5 | 16.500960135460 | 34.399235521044 | 11.312925515856 |

## Judgment

XR-55 did not promote any active gate.

- Best XR-55 P10 was XR-55A best-P5: `34.793368141992`, below active P10 `35.02295998845781`.
- Center/P5 regressed relative to XR-52 leaders.

Active gates remain:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

Decision: close XR39-anchored expanded last-block tuning as no-promotion. Next branch should change supervision directly: explicit P10 teacher retraining or a dedicated P10 calibration/classification head.

# 2026-06-17 XR-56 Validation

## Scope

XR-56 replaces another low-LR/scope expansion P10 recovery with direct metric-aligned soft-threshold supervision.

## Static Validation

- `chmod +x scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
- `bash -n scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
- `python3 -m py_compile src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py`
- `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`
  - Result: `23 passed in 0.64s`
- `DRY_RUN=1 bash scripts/external/run_xr56_soft_threshold_p10_supervision.sh a cuda:0`
- `DRY_RUN=1 bash scripts/external/run_xr56_soft_threshold_p10_supervision.sh b cuda:1`
- `git diff --check -- src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py tests/test_track_center_l2_loss.py scripts/external/run_xr56_soft_threshold_p10_supervision.sh`

## Runtime Evidence

- XR-56A ran on GPU0.
  - Init/teacher: XR-39 P10 `c25p45f30`
  - LR: `7.5e-7`
  - P10 soft threshold: weight `0.006`, margin `10.0`, temperature `1.5`
  - P5 guard: weight `0.001`, margin `5.0`, temperature `1.0`
  - Completed epoch `10/10`; train/eval exits were `0`.
- XR-56B ran on GPU1.
  - Init: XR-52B best-P10/P5 seed
  - Teacher: XR-39 P10 `c25p45f30`
  - LR: `5e-7`
  - P10 soft threshold: weight `0.008`, margin `10.0`, temperature `1.25`
  - P5 guard: weight `0.002`, margin `5.0`, temperature `0.9`
  - Early-stopped at epoch `7/10`; train/eval exits were `0`.

## Full-Test Eval Results

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % |
|---|---|---:|---:|---:|---:|
| A | `best_track_p10` | 8 | 16.49389898266111 | 34.32270488057818 | 11.699830286843437 |
| A | `best_track_p5` | 3 | 16.4924229485648 | 34.62330012321472 | 11.304422106061663 |
| B | `best_track_p10` | 1 | 16.477364584377835 | 34.37670146397182 | 12.044218049730574 |
| B | `best_track_p5` | 5 | 16.4701875601496 | 34.29294293948582 | 12.133503770828247 |

`best_metric_track_center_px.pt` was not produced in either lane, so that eval target was skipped.

## Gate Decision

Previous gates:

- Center: `<16.470460832119`
- P10: `>35.02295998845781`
- P5: `>12.017432342257`

XR-56B `best_track_p5` promoted center and P5. P10 was not recovered.

Current gates:

- Center: `<16.4701875601496`
- P10: `>35.02295998845781`
- P5: `>12.133503770828247`

# 2026-06-17 XR-57 Validation

## Scope

XR-57 adds a bounded residual center-refinement head and routes it into final `track/state` for direct metric visibility.

## Static Validation

- `chmod +x scripts/external/run_xr57_p10_center_refine_calibration.sh`
- `bash -n scripts/external/run_xr57_p10_center_refine_calibration.sh`
- `python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/tracker/track_branch.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py src/hbtxr/loss/stage2.py`
- `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py`
  - Result: `27 passed in 0.62s`
- `PYTHONPATH=src .venv/bin/python -c "... build_model(...)"`.
  - Result: `TrackCenterRefineHead True 4.0`
- `DRY_RUN=1 bash scripts/external/run_xr57_p10_center_refine_calibration.sh a cuda:0`
- `DRY_RUN=1 bash scripts/external/run_xr57_p10_center_refine_calibration.sh b cuda:1`

## Train/Eval Evidence

Full-test eval logs:

- Lane A log: `runs/_logs/xr57_p10_center_refine_calibration_a_gpu0_20260617_075353.log`
- Lane B log: `runs/_logs/xr57_p10_center_refine_calibration_b_gpu1_20260617_075354.log`

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 3 | 16.69362453562873 | 34.38307912690299 | 10.973214619500297 | no promotion |
| A | `best_track_p5` | 7 | 16.715463175092424 | 34.849490649359566 | 10.925595603670393 | no promotion |
| B | `best_track_p10` | 12 | 16.504537062985555 | 34.137755966186525 | 11.207483332497732 | no promotion |
| B | `best_track_p5` | 5 | 16.4997801729611 | 34.190476996558054 | 11.476615987505232 | no promotion |

`best_metric_track_center_px.pt` was not produced in either lane, so that eval target was skipped.

Gate decision against center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`: XR-57 does not promote center, P10, or P5.

## Next Validation Target

XR-57 shows that a small residual correction does not recover the strict P10 gate from the current XR-56/XR-39 anchors. The next validation target should be XR-58 P10 teacher refresh: first produce a deliberately P10-specialized teacher that beats XR-39 test P10, then use it as a student distillation/reference checkpoint.

# 2026-06-17 XR-58 Validation

## Scope

XR-58 trains P10-specialized teacher candidates from the XR-39 P10 checkpoint using expanded trainable scope. Center/P5 preservation is secondary for this branch.

## Static Validation

- `bash -n scripts/external/run_xr58_p10_teacher_refresh.sh`
- `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh a cuda:0`
- `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh b cuda:1`

## Train/Eval Evidence

Full-test eval logs:

- Lane A log: `runs/_logs/xr58_p10_teacher_refresh_a_gpu0_20260617_082242.log`
- Lane B log: `runs/_logs/xr58_p10_teacher_refresh_b_gpu1_20260617_082242.log`

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 4 | 16.503614359242576 | 34.90306201662336 | 11.51275544847761 | no promotion |
| A | `best_track_p5` | 3 | 16.501813726765768 | 34.68920147078378 | 11.259779255730765 | no promotion |
| B | `best_track_p10` | 4 | 16.506438190596445 | 34.84226275852748 | 11.501275873184204 | no promotion |
| B | `best_track_p5` | 3 | 16.49833288192749 | 34.540391949244906 | 11.062075165339879 | no promotion |

`best_metric_track_center_px.pt` was not produced in either lane, so that eval target was skipped.

Gate decision against center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`: XR-58 does not promote center, P10, or P5. Best XR-58 P10 is `34.90306201662336`, about `0.1199` below the P10 gate.

# 2026-06-17 XR-59 Validation

## Scope

XR-59 continues from XR-58A best-P10 with XR-39 as anchored P10 teacher. It tests a narrow LR/P10-soft bracket to close the remaining P10 gap.

## Static Validation

- `bash -n scripts/external/run_xr59_xr58a_teacher_bracket.sh`
- `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh a cuda:0`
- `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh b cuda:1`
- `bash -n scripts/external/eval_xr59_completed_checkpoints.sh`
- `git diff --check -- scripts/external/eval_xr59_completed_checkpoints.sh`

## Train/Eval Evidence

Train logs:

- Lane A log: `runs/_logs/xr59_xr58a_teacher_bracket_a_gpu0_20260617_084640.log`
- Lane B log: `runs/_logs/xr59_xr58a_teacher_bracket_b_gpu1_20260617_084641.log`

Both lanes early-stopped at epoch `7/10`. Post-train eval was run with `HBTXR_DISABLE_CUDNN=1` through `scripts/external/eval_xr59_completed_checkpoints.sh`.

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 1 | 16.520220368249074 | 34.300170864377705 | 11.376701021194458 | no promotion |
| A | `best_track_p5` | 1 | 16.520220368249074 | 34.300170864377705 | 11.376701021194458 | no promotion |
| B | `best_track_p10` | 1 | 16.51065547806876 | 34.43409944261823 | 11.287415306908743 | no promotion |
| B | `best_track_p5` | 1 | 16.51065547806876 | 34.43409944261823 | 11.287415306908743 | no promotion |

Gate decision against center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`: XR-59 does not promote center, P10, or P5. Best XR-59 P10 is `34.43409944261823`, worse than XR-58A best-P10 `34.90306201662336`; scalar P10-soft continuation should be closed.

# 2026-06-17 XR-60 Validation Plan

## Scope

XR-60 adds a default-off multi-candidate P10 calibration head. The branch tests whether routing the top candidate into final `track/state` can recover P10 after XR-59 scalar P10-soft continuation failed.

## Static Validation Targets

- `bash -n scripts/external/run_xr60_p10_candidate_head.sh`: passed.
- `python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/tracker/track_branch.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py src/hbtxr/loss/stage2.py`: passed.
- `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py`: `34 passed in 0.60s`.
- Candidate model-build smoke with `model.heads.track_center_candidate=true`: `TrackCenterCandidateHead True 0.5`.
- `DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh a cuda:0`: passed.
- `DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh b cuda:1`: passed.
- `git diff --check` over modified XR-60 files: passed.

## Launch Evidence

- Initial sandbox launch failed with `torch.cuda.is_available=False` and `cuda_device_count=0` while `nvidia-smi` could see both GPUs.
- Relaunched with approved elevated command prefix `bash scripts/external/run_xr60_p10_candidate_head.sh`.
- XR-60A log: `runs/_logs/xr60_p10_candidate_head_a_gpu0_20260617_172814.log`.
- XR-60B log: `runs/_logs/xr60_p10_candidate_head_b_gpu1_20260617_172813.log`.
- Both lanes passed raw event-count contract and entered epoch `2/12`.

## Gate Decision

Use unchanged gates: center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.

## Closeout Evidence

| Lane | Checkpoint | Epoch | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---:|---|
| A | `best_track_p10` | 2 | 16.526124344553267 | 34.50255186898368 | 11.617772477013725 | no promotion |
| A | `best_track_p5` | 3 | 16.5058109828404 | 34.60331717899867 | 11.337160219464984 | no promotion |
| B | `best_track_p10` | 7 | 16.842585216249738 | 33.49277294022696 | 10.70663298198155 | no promotion |
| B | `best_track_p5` | 5 | 16.717501049382346 | 33.87670159339905 | 10.971939107349941 | no promotion |

XR-60 does not promote center, P10, or P5. The next P0 branch should avoid directly routing the high-LR candidate head into final `track/state`; use auxiliary-only candidate supervision or pivot to paper-backed ellipse/pseudo-label diagnostics.

# 2026-06-17 XR Eye-Tracking Analysis Rewrite Validation

## Scope

The requested rewrite updates all detailed codebase and paper analyses under `anlaysis/xr-eye-tracking` to the reference-document level used by the hardware analysis examples.

## Validation Evidence

- `python3 scripts/external/write_xr_eye_tracking_detailed_analysis.py`: regenerated the analysis set.
- `python3 -m py_compile scripts/external/write_xr_eye_tracking_detailed_analysis.py`: passed.
- Section coverage scan: `18` codebase files include `## 5. 핵심 모듈 / 함수 단위 분석`; `21` paper files include `## 3.1 Paper Section Evidence`.
- `find anlaysis/xr-eye-tracking -maxdepth 3 -type f -name analysis.md -print0 | xargs -0 wc -l | sort -n`: total `8813` lines.
- Representative spot checks: `anlaysis/xr-eye-tracking/codebases/FACET/analysis.md` and `anlaysis/xr-eye-tracking/papers/FACET/analysis.md`.
- Sub-agent note: GPT-5.3-Codex-Spark delegation attempt failed due usage limit, so the main agent completed generation and verification locally.

## Sample Spot Checks

- `anlaysis/xr-eye-tracking/codebases/FACET/analysis.md`: includes source roots, artifact mix, metadata/config refs, directory map, core source refs, README summary, entrypoints, Python symbols, import/dependency clues, config option clues, keyword/line evidence, HGTXR conversion design, priority, and risks.
- `anlaysis/xr-eye-tracking/papers/FACET/analysis.md`: includes problems of prior methods, proposed method, algorithm/loss/inference flow, bounded section evidence, primitive decomposition, result/metric evidence, method evidence, hardware/system relevance, experiment evidence, dataset/result interpretation, ablation axes, HGTXR applicability, and risks.

# 2026-06-18 XR-61 / XR-62 Validation

## XR-61 Closeout Evidence

XR-61 reused the XR-60 runner with `track_center_candidate_as_track_state=false`; this closed the routed-candidate failure mode while keeping candidate loss as auxiliary supervision.

| Lane | Checkpoint | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---|
| A | `best_track_p10` | 16.475529539585114 | 34.356718465260094 | 12.127126216888428 | no promotion |
| A | `best_track_p5` | 16.475529539585114 | 34.356718465260094 | 12.127126216888428 | no promotion |
| B | `best_track_p10` | 16.48864848954337 | 34.39073208400181 | 11.31717723437718 | no promotion |
| B | `best_track_p5` | 16.48967229127884 | 34.58843615395682 | 11.559524168287005 | no promotion |

## XR-62 Scope

XR-62 adds FACET-style training-only track-state auxiliary geometry supervision on top of the heatmap-state main path. It disables candidate routing and tests whether center/P5 preservation can coexist with geometry regularization.

## Static Validation

- `chmod +x scripts/external/run_xr62_facet_geometry_aux_refresh.sh`
- `bash -n scripts/external/run_xr62_facet_geometry_aux_refresh.sh`
- `DRY_RUN=1 bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh a cuda:0`
- `DRY_RUN=1 bash scripts/external/run_xr62_facet_geometry_aux_refresh.sh b cuda:1`

## Train/Eval Evidence

Train logs:

- Lane A: `runs/_logs/xr62_facet_geometry_aux_refresh_a_gpu0_20260618_003307.log`
- Lane B: `runs/_logs/xr62_facet_geometry_aux_refresh_b_gpu1_20260618_003317.log`

Both lanes early-stopped at epoch `7/10`; train/eval exits were `0`.

| Lane | Checkpoint | Center px | P10 % | P5 % | Gate decision |
|---|---|---:|---:|---:|---|
| A | `best_track_p10` | 16.468481131962367 | 34.30782389640808 | 12.052721459524973 | center promotion |
| A | `best_track_p5` | 16.468481131962367 | 34.30782389640808 | 12.052721459524973 | center promotion |
| B | `best_track_p10` | 16.495574762140002 | 34.46598720550537 | 11.370323480878557 | no promotion |
| B | `best_track_p5` | 16.495574762140002 | 34.46598720550537 | 11.370323480878557 | no promotion |

Active gates after XR-62: center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.

# 2026-06-18 XR Eye-Tracking Reference-Level Analysis Validation

## Scope

The user requested the previously generated `anlaysis/xr-eye-tracking` codebase and paper analyses to be expanded to the same depth/style as:

- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/FlexLLM/analysis.md`
- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/EfficientViT-FPGA/analysis.md`

## Validation Evidence

- `python3 -m py_compile scripts/external/write_xr_eye_tracking_detailed_analysis.py`: passed.
- `python3 scripts/external/write_xr_eye_tracking_detailed_analysis.py`: regenerated all analysis files.
- Codebase required-section scan with `rg --files-without-match` returned no missing files for:
  - `Core Source Responsibility Table`
  - `데이터 계약 / HGTXR 호환성 매트릭스`
  - `코드 품질 / 재사용성 이슈`
  - `재현 / 검증 Runbook`
- Paper required-section scan with `rg --files-without-match` returned no missing files for:
  - `데이터셋 / 모델 / 벤치마크`
  - `실험 결과`
  - `Reported Result Evidence`
  - `HGTXR 실험 설계로 변환`
  - `HGTXR 실험 옵션 세부 설계`
- Line-count audit: `39` analysis files, `11574` total lines.
- Representative checks:
  - `anlaysis/xr-eye-tracking/codebases/FACET/analysis.md` includes metadata, core source responsibility, loss/evaluation role classification, contract matrix, quality findings, and runbook.
  - `anlaysis/xr-eye-tracking/papers/FACET/analysis.md` and `papers/EV-Eye/analysis.md` include the new experiment option runbook.
- `git diff --check -- scripts/external/write_xr_eye_tracking_detailed_analysis.py anlaysis/xr-eye-tracking`: passed.

## Sub-agent Evidence

- T-401/T-402 Spark explorer attempts failed due GPT-5.3-Codex-Spark usage limit.
- T-401R/T-402R GPT5.5 explorer agents completed read-only schema extraction.
- The main agent integrated those templates into the generator and regenerated/validated the artifacts.

# 2026-06-18 XR-63 P10 Teacher-Target Oracle Diagnostic Validation

## Scope

XR-63 is a no-train diagnostic. It measures whether sample-wise teacher-target selection among existing leaders has enough headroom to justify a P10 teacher-target construction branch.

Inputs:

- XR-62A center leader eval rows.
- XR-39 `c25p45f30` P10 leader eval rows.
- XR-56B best-P5 leader eval rows.
- XR-58A P10-teacher refresh eval rows.
- `data/_internal/manifests/manifest1/test_manifest.jsonl`.
- XR-62A resolved config for target transform and batch grouping.

## Validation Evidence

- `python3 -m py_compile scripts/external/analyze_p10_teacher_target_oracle.py`: passed.
- `.venv/bin/python scripts/external/analyze_p10_teacher_target_oracle.py --limit 64 --output-json /tmp/xr63_smoke_evalstyle.json --output-md /tmp/xr63_smoke_evalstyle.md`: passed.
- `.venv/bin/python scripts/external/analyze_p10_teacher_target_oracle.py`: passed and wrote:
  - `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json`
  - `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md`
- Metric alignment check passed: the script now computes eval-style mean-of-batch metrics, matching existing `eval_summary.json` values for the individual model rows.

## Result

| Row | Center px | P10 % | P5 % |
|---|---:|---:|---:|
| XR-62A center | 16.468481 | 34.307823 | 12.052721 |
| XR-39 P10 | 16.503941 | 35.022959 | 11.460459 |
| XR-56B P5 | 16.470188 | 34.292942 | 12.133503 |
| XR-58A P10 teacher | 16.503614 | 34.903061 | 11.512755 |
| XR-63 oracle upper bound | 16.040232 | 36.485969 | 13.417517 |

Decision: `GO_teacher_target_construction`.

The oracle is not deployable performance. It proves only that a sample-wise teacher/selector target has enough headroom to beat all active gates if a leakage-safe construction can approximate the selection rule.

## Risks

- Current oracle uses test predictions for diagnosis only. Any training branch must rebuild pseudo targets from train/val-only evidence or clearly separate test from model selection.
- P10-miss recovery by any model is only `3.426791%` by baseline-miss weight, so the largest gain is broad per-sample center selection, not only hard P10 miss flipping.
- A trainable selector may fail to reproduce the oracle without additional supervision features.

## Sub-agent Evidence

- T-404 GPT-5.3-Codex-Spark explorer was attempted for read-only path/key verification.
- T-404 failed due GPT-5.3-Codex-Spark usage limit.
- Main-agent fallback completed implementation and validation.

# 2026-06-18 XR-64 Teacher-Target Construction Validation

## Scope

XR-64 implements the trainable path required by XR-63: sample-wise teacher-target overrides for train/val-only construction, additive losses, and test leakage guards.

## Validation Evidence

- `python3 -m py_compile scripts/external/build_xr64_teacher_target_overrides.py scripts/external/analyze_p10_teacher_target_oracle.py src/hbtxr/data/dataset.py src/hbtxr/config/runtime_config.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py`: passed.
- `bash -n scripts/external/run_xr64_teacher_target_construction.sh`: passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_target_override.py tests/test_track_center_l2_loss.py`: `39 passed`.
- `DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0`: passed; missing override file is warning-only in dry-run.
- `DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1`: passed; missing override file is warning-only in dry-run.
- Builder smoke with `--split test --allow-test-diagnostic-output --limit 64`: passed and wrote `/tmp/xr64_override_smoke.json`.

## Safety Checks

- `cur_state` remains unchanged and remains the metric target.
- Target override is additive: `track_target_override_state` plus explicit override loss weights.
- Dataset rejects target override files on test manifests unless `data.allow_test_target_override=true`.
- XR-64 final test eval clears target override config with `data.track_target_override_path=null`.

## Result

The XR-64 implementation is ready for train/val teacher eval generation and train-only override construction. Training has not been launched yet because the required train override files are not generated.

## Sub-agent Evidence

- T-407 GPT5.5 read-only review completed.
- Recommendations integrated: additive override, test split guard, original `cur_state` preservation, loss tests, builder test, runner dry-runs.

# 2026-06-18 XR-64 Resume Artifact Checker Validation

## Scope

`scripts/external/check_xr64_resume_artifacts.py` is a read-only verifier for the paused XR-64 resume path. It does not run training or evaluation. It checks whether the generated train/val eval rows and override JSON files are complete, structurally valid, split-safe, and free of test-split target leakage.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py`: passed.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py`: `9 passed`.
- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated`: passed with `ready=true`, `generated_complete=false`.
- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed with `resume_status=generated_incomplete`.
- `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh`: passed.

## Checked Conditions

- Required train/val/test manifests, config, reference config, and teacher checkpoints exist.
- Train/val manifest `sample_id` sets are disjoint.
- Eval rows must be list-shaped with `sample_id` and finite 6D `track_state`.
- Override files must contain expected `kind`, `split`, `rule`, `baseline`, `sample_count`, `selection_counts`, and `overrides`.
- Override rows must contain finite 6D target state, weight, source, reason, baseline/teacher errors, and all four teacher error entries.
- Test-named or test-split XR-64 target artifacts fail strict validation.
- Human summary output reports `resume_status`, `ready_to_train`, `can_run_lane`, compact generation/override matrices, and leakage risk.

## Result

Current source prerequisites pass. Generated XR-64 artifacts are still missing, so strict mode is expected to fail until train/val eval rows and override JSON files are produced after experiment execution resumes.

# 2026-06-18 Runs Compatibility Symlink Removal Validation

## Scope

The run directory cleanup removed top-level compatibility symlinks for organized experiment runs while preserving real run data under `runs/XR-*` and `runs/NON_XR/*`.

## Validation Evidence

- `.venv/bin/python scripts/external/maintenance/remove_runs_compat_symlinks.py --execute`: removed `846` top-level run compatibility symlinks and generated `docs/resources/runs_compat_symlink_removal_2026_06_18.md` plus `.json`.
- `find runs -maxdepth 1 -mindepth 1 \( -name 'eval*' -o -name 'raw*' \)`: no output.
- `find runs -xtype l`: no output, so no broken symlink remained.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_resume_status=generated_incomplete`.
- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed with `ready=true`, `generated_complete=false`.
- `.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py`: passed raw contract and found organized Stage1/Stage2 roots under `runs/NON_XR/raw`.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_raw_event_count_training_readiness.py tests/test_raw_event_count_training_result.py tests/test_xr64_resume_artifacts.py tests/test_second_goal_status.py`: `13 passed`.

## Remaining Risk

Legacy queue scripts still contain historical `runs/eval*` glob patterns. These were not migrated because experiments remain paused and the scripts are historical/batch-specific. Shared aliases `runs/_logs`, `runs/diagnostics`, and `runs/interpolated_checkpoints` remain for compatibility.

# 2026-06-20 Current Result Synthesis Validation

## Scope

The current result synthesis records the validated second-goal result interpretation without launching train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_current_result_synthesis.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_current_result_synthesis.py --format summary`: passed with `ok=true`, `oracle_promotable=false`, `direct_submission_comparison_valid=false`, and `xr64_current_status=generated_incomplete`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `current_result_synthesis_exists=true`, `current_result_oracle_promotable=false`, `current_result_direct_submission_comparison_valid=false`, and `current_result_xr64_ready_to_train=false`.
- `.venv/bin/python -m pytest -q tests/test_paper_ref_analysis_coverage.py tests/test_second_goal_experiment_queue.py tests/test_submission_target_gap.py tests/test_metric_protocol_bridge.py tests/test_current_result_synthesis.py tests/test_second_goal_status.py`: `16 passed`.
- `.venv/bin/python scripts/external/check_second_goal_experiment_queue.py --format summary`: passed with P0 queue `XR-64-prep, XR-64A, XR-64B`.
- `.venv/bin/python scripts/external/check_metric_protocol_bridge.py --format summary`: passed with bridge status `blocked_until_metric_frame_and_protocol_match`.

## Result

The current result interpretation is now guarded by a machine-readable artifact and checker. The active goal remains incomplete because experiments are paused, XR-64 generated artifacts are missing, and metric/protocol comparison with the submission target remains blocked.

# 2026-06-20 Second-Goal Objective Trace Validation

## Scope

The objective trace maps the full user-provided second goal to current evidence, missing evidence, blockers, false-completion guardrails, and next allowed actions.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_objective_trace.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary`: passed with `ok=true`, `goal_complete=false`, REQ-1 `complete_as_planning_input`, REQ-2 `planned_not_fully_executed`, REQ-3 `incomplete`, and full ablation matrix count `8`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `objective_trace_exists=true`, `objective_trace_goal_complete=false`, and all requested experiment axes.
- `.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `10 passed` after full-matrix trace and XR-64 build-input readiness updates.

## Sub-agent Evidence

- GPT-5.5 sub-agent `T-TRACE-001` completed a read-only audit.
- The audit recommended explicit fields for `promotion_allowed`, `evidence_artifacts`, `missing_evidence`, `blocked_by`, `next_allowed_action`, `false_completion_guardrails`, and `do_not_claim`.
- Those fields were integrated into `docs/resources/second_goal_objective_trace_2026_06_20.json` and enforced by the checker.

## Result

The trace proves planning progress but also proves the active goal remains incomplete. No train/eval job was launched.

# 2026-06-20 XR-64 Resume Command Manifest Validation

## Scope

The XR-64 resume command manifest encodes the future XR-64-prep command contract without executing training or evaluation.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `ok=true`, `prep_command_count=10`, `eval_command_count=8`, `build_command_count=2`, `expected_eval_rows=8`, `expected_overrides=6`, `eval_inputs_ready=true`, and `build_inputs_ready=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_command_manifest_exists=true`, `xr64_command_manifest_default_allowed_to_run=false`, launch lanes `XR-64A,XR-64B`, `xr64_command_manifest_missing_eval_required_input_count=0`, and `xr64_command_manifest_missing_build_required_input_count=8`.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `7 passed` after eval/build input readiness parity update.
- `.venv/bin/python -m pytest -q tests/test_current_result_synthesis.py tests/test_decide_xr64_postrun_promotion.py tests/test_emit_xr64_resume_commands.py tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_paper_ref_analysis_coverage.py tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_experiment_queue.py tests/test_second_goal_objective_trace.py tests/test_second_goal_status.py tests/test_submission_target_gap.py tests/test_submission_target_source_trace.py tests/test_xr64_postrun_evidence_contract.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `81 passed`.
- `git diff --check`: passed.

## Sub-agent Evidence

- GPT-5.5 sub-agent `T-XR64-MANIFEST-001` completed a read-only audit.
- The audit recommended explicit manifest fields including `manifest_version`, `project_root`, `manifest_root`, `target_dir`, `current_gates`, command-level `env`, `expected_outputs`, `required_inputs`, `depends_on`, and `allowed_to_run`.
- Those fields were integrated into `docs/resources/xr64_resume_command_manifest_2026_06_20.json` and enforced by the checker.

## Result

The manifest is valid as a future execution contract. It is not execution evidence, and no train/eval job was launched.

# 2026-06-20 XR-64 Resume Command Emitter Validation

## Scope

The emitter prints commands from the XR-64 resume command manifest for review. It does not execute experiments.

## Validation Evidence

- `python3 -m py_compile scripts/external/emit_xr64_resume_commands.py`: passed.
- `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section summary --format summary`: passed and reports `execute_supported=false`, `prep_commands=10`, `eval_commands=8`, and `build_commands=2`.
- `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section prep --format commands`: passed and printed train/val-only XR-64-prep commands for review.
- `.venv/bin/python -m pytest -q tests/test_emit_xr64_resume_commands.py`: `3 passed`.

## Result

The emitter is valid as a review/printing tool only. No train/eval job was launched.

# 2026-06-20 Second-Goal Completion Gate Validation

## Scope

The completion gate checks whether the full HGTXR-SW second goal may be marked complete. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `ok=false`, `completion_allowed=false`, and `blocker_count=22`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py`: `4 passed`.

## Sub-agent Evidence

- GPT-5.5 sub-agent `T-COMPLETE-GATE-001` completed a read-only completion audit.
- The audit identified strict completion conditions across XR-64 readiness, objective trace, current result synthesis, metric/protocol bridge, and command-manifest provenance.
- The checker encodes those conditions and preserves current blockers: pause state, missing XR-64 artifacts, no XR-64A/B/C evidence, non-promotable XR-63 oracle, blocked direct submission comparison, and split best-gate ownership.

## Result

The full second goal is not complete. The checker is now the required machine-readable gate before any future completion claim.

# 2026-06-20 XR-64 Prelaunch Packet Validation

## Scope

The XR-64 prelaunch packet bundles the future-only resume path, strict preconditions, command-review entry points, and validation commands. It does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `ok=true`, `allowed_to_execute=false`, `completion_allowed=false`, `completion_blocker_count=22`, `phase_count=5`, and command manifest alignment.
- `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py`: `4 passed`.

## Result

The prelaunch packet is valid as a future-only resume package. It confirms XR-64 still cannot be launched while paused and while generated artifacts are missing.

## 2026-06-21 XR-64 Prelaunch Readiness Parity Validation

The XR-64 prelaunch packet now mirrors command-manifest eval/build input readiness. This keeps the resume packet aligned with the stricter command manifest: eval static inputs are ready, but build inputs are still missing until XR-64-prep generates eval rows. This pass is read-only and does not run train/eval jobs.

Validation commands:

```bash
python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py
git diff --check
```

Expected readiness facts:

- `manifest_eval_inputs_ready=true`
- `manifest_missing_eval_required_input_count=0`
- `manifest_build_inputs_ready=false`
- `manifest_missing_build_required_input_count=8`
- `xr64_prelaunch_eval_inputs_ready=true`
- `xr64_prelaunch_build_inputs_ready=false`

Observed result:

- `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `manifest_eval_inputs_ready=true` and `manifest_build_inputs_ready=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_prelaunch_eval_inputs_ready=true`, `xr64_prelaunch_build_inputs_ready=false`, `xr64_ready_to_train=false`, and `xr64_can_run_lane=false`.
- `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `9 passed`.
- Broader second-goal pytest suite: `84 passed`.
- `git diff --check`: passed.

# 2026-06-21 XR-64 Post-Run Evidence Contract Validation

## Scope

The XR-64 post-run evidence contract defines the required evidence for judging XR-64A/B after future execution. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary`: passed with `ok=true`, `evidence_state=missing_postrun_evidence`, `allowed_to_claim_promotion=false`, `required_lane_count=2`, `required_checkpoint_kind_count=3`, and `required_metric_count=4`.
- `.venv/bin/python -m pytest -q tests/test_xr64_postrun_evidence_contract.py`: `4 passed`.

## Result

The post-run contract is valid as a future result-evidence gate. It confirms XR-64A/B cannot yet be promoted because training runs and test eval summaries are absent. Future eval summaries must include `metric_track_p1_pct` as paper-target bridge evidence.

# 2026-06-21 XR-64 Post-Run Ablation Evidence Contract Validation

## Scope

The XR-64 post-run evidence contract now requires lane-level ablation evidence for the second-goal axes: head, loss, LR, and teacher model training. This pass is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary`: passed with `ablation_axis_count=4` and `ablation_lane_axis_count=2`.
- `.venv/bin/python scripts/external/check_second_goal_ablation_evidence_checklist.py --format summary`: passed with `axis_count=8`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_postrun_ablation_axis_count=4` and `xr64_postrun_ablation_lane_axis_count=2`.
- `.venv/bin/python -m pytest -q tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_status.py`: `15 passed`.
- Broader second-goal pytest suite: `87 passed`.
- `git diff --check`: passed.

## Result

Future XR-64A/B results must now include lane-level `ablation_design`, explicit config key paths for head/loss/LR/teacher controls, LR and optimizer snapshots, teacher provenance, and an ablation attribution report before the result can support second-goal axis claims.

# 2026-06-21 XR-64 Post-Run Promotion Decision Helper Validation

## Scope

The promotion decision helper scores future XR-64A/B test `eval_summary.json` files against the current center/P10/P5 gates. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py`: passed.
- `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed with `decision_status=missing_postrun_evidence`, `candidate_count=0`, `p1_evidence_candidate_count=0`, and `software_promotion_allowed=false`.
- `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py`: `4 passed`.

## Result

The decision helper is ready for future XR-64A/B eval summaries. Current state remains no-evidence/no-promotion. The helper tracks `metric_track_p1_pct` as future paper evidence without using P1 as a current software-promotion gate.

# 2026-06-21 XR-64 Promotion Axis Certification Validation

## Scope

The XR-64 post-run promotion helper now requires ablation proof fields before it can allow software promotion. This aligns the helper with the post-run evidence contract and prevents metric-only promotion decisions from bypassing head/loss/LR/teacher axis evidence. This pass is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed with `axis_certification_required=true`, `axis_claims_allowed=false`, `control_variables_checked=false`, and `rejection_reason=missing_postrun_evidence`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_promotion_axis_certification_required=true`, `xr64_promotion_axis_claims_allowed=false`, and `xr64_promotion_rejection_reason=missing_postrun_evidence`.
- `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_second_goal_status.py`: `11 passed`.
- Broader second-goal pytest suite: `90 passed`.
- `git diff --check`: passed.

## Result

Future candidate `eval_summary.json` files must include strict Boolean `axis_certified=true`, `ablation_axis` covering head/loss/LR/teacher model training, `ablation_changed_keys` as a per-axis key mapping, and `ablation_benchmark_baseline_id=XR-64-prep`. Without those fields, the helper returns invalid candidate evidence and does not allow software promotion.

# 2026-06-21 Metric/Protocol Unblock Contract Validation

## Scope

The metric/protocol unblock contract defines the evidence required before current HGTXR-SW metrics can be compared directly with the `10_submission_initial` paper targets. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_metric_protocol_unblock_contract.py`: passed.
- `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed with `ok=true`, `contract_status=blocked`, `direct_submission_comparison_allowed=false`, `paper_level_completion_allowed=false`, and `required_evidence_count=7`.
- `.venv/bin/python -m pytest -q tests/test_metric_protocol_unblock_contract.py`: `5 passed`.

## Result

The unblock contract is valid as a paper-target comparison guard. Current state remains blocked because coordinate-frame match, sensor-space rows, hybrid scheduler rows, P1 metric coverage, split-protocol provenance, and trained XR-64-or-later evidence are missing.

# 2026-06-21 P1 Metric Coverage Validation

## Scope

The evaluator now exposes P1 hit-rate metrics for search and track predictions. This reduces one metric-coverage blocker but does not unblock direct paper-target comparison because paper-frame/full-test P1 evidence, coordinate-frame match, and hybrid scheduler evaluation are still missing.

## Validation Evidence

- `python3 -m py_compile src/hbtxr/loss/metrics.py scripts/external/check_metric_protocol_bridge.py scripts/external/check_metric_protocol_unblock_contract.py scripts/external/check_current_result_synthesis.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_metric_protocol_bridge.py --format summary`: passed with `current_p1_metric_available=true`, `current_p1_full_test_evidence_available=false`, and `direct_submission_comparison_allowed=false`.
- `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed with `contract_status=blocked`.
- `.venv/bin/python -m pytest -q tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_current_result_synthesis.py tests/test_second_goal_status.py`: `15 passed`.

## Result

`metric_track_p1_pct` and `metric_search_p1_pct` are implemented and covered by focused unit tests. Paper-level comparison remains blocked.

# 2026-06-21 Second-Goal Ablation Evidence Checklist Validation

## Scope

The ablation evidence checklist binds the XR-64-first queue to required controls, evidence, promotion decisions, and rejection rules. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_ablation_evidence_checklist.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_ablation_evidence_checklist.py --format summary`: passed with `ok=true`, `experiment_count=8`, `p0_ids=XR-64-prep,XR-64A,XR-64B`, `axis_count=8`, `software_promotion_allowed=false`, and `paper_level_completion_allowed=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `ablation_evidence_checklist_exists=true`, `ablation_evidence_experiment_count=8`, and `ablation_evidence_missing_eval_rows=8`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_status.py`: `6 passed`.

## Result

The checklist is valid as a future ablation-evidence gate. It confirms that XR-64-prep/A/B remain P0, that P1 is future bridge evidence rather than a software promotion gate, and that current promotion remains blocked while XR-64 generated artifacts are missing.

# 2026-06-21 Second-Goal Completion Readiness Contract Validation

## Scope

The completion readiness contract records the current blocker set, XR-64 readiness state, next resume gate, and no-execute validation rules. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_gate.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, `completion_allowed=false`, `blocker_count=22`, `xr64_ready_to_train=false`, `missing_eval_rows=8`, `missing_overrides=6`, and `leakage_risk=none`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `completion_readiness_contract_exists=true`, `completion_readiness_blocker_count=22`, and `completion_readiness_xr64_ready_to_train=false`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `8 passed`.

## Result

The readiness contract is valid as the current completion-blocker ledger. It confirms that the full second goal cannot be marked complete while experiments are paused, XR-64 generated artifacts are missing, and direct paper-target comparison remains blocked.

# 2026-06-21 Submission Target Source Trace Parser Validation

## Scope

The submission-target source trace checker now validates the paper target values against `10_submission_initial/main.tex` with table-scoped parsing for `tab:algo`. This prevents unrelated `P10`/`P5`/`P1` rows elsewhere in the paper source from being accepted as HBTXR algorithm-table evidence. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_submission_target_source_trace.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_submission_target_source_trace.py --format summary`: passed with `ok=true`, `direct_comparison_valid=false`, `mode_hybrid_p1_pct=99.61`, and `accelerator_range_is_single_target=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports submission source trace values plus `submission_target_source_direct_comparison_valid=false`.
- `.venv/bin/python -m pytest -q tests/test_submission_target_source_trace.py tests/test_submission_target_gap.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_hbtxr_metrics_p1.py`: `32 passed`.
- `.venv/bin/python -m pytest -q tests/test_current_result_synthesis.py tests/test_decide_xr64_postrun_promotion.py tests/test_emit_xr64_resume_commands.py tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_paper_ref_analysis_coverage.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_experiment_queue.py tests/test_second_goal_objective_trace.py tests/test_submission_target_gap.py tests/test_submission_target_source_trace.py tests/test_xr64_postrun_evidence_contract.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `65 passed`.
- `git diff --check`: passed.

## Result

Submission-target source tracing is now robust enough for current completion-gate reporting. Direct comparison to the paper target remains blocked until coordinate-frame, hybrid scheduler, split protocol, P1 full-test evidence, and trained XR-64-or-later evidence are complete.

# 2026-06-21 Paper Target Comparison Evidence Manifest Validation

## Scope

The paper-target comparison evidence manifest converts the abstract metric/protocol unblock contract into concrete file-level evidence requirements. It records which current supporting artifacts exist and which future bridge artifacts must be produced before any direct comparison with the `10_submission_initial` target can be allowed. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary`: passed with `ok=true`, `evidence_state=missing_required_evidence`, `required_future_artifact_count=10`, and `existing_required_future_artifact_count=3`.
- `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed after linking the manifest as a source artifact.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_evidence_state=missing_required_evidence`.
- `.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py`: `7 passed`.
- `.venv/bin/python -m pytest -q tests/test_metric_protocol_unblock_contract.py tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py`: `12 passed`.

## Result

Direct paper-target comparison remains blocked, but the required future evidence is now explicit: 7 evidence classes, 10 future artifacts, 4 current supporting artifacts, and 3 paused-safe doc-prefill future bridge artifacts in the paused state.

# 2026-06-21 Completion Gate Paper-Target Evidence Integration Validation

## Scope

The completion gate, completion-readiness contract, and XR-64 prelaunch packet now enforce the paper-target comparison evidence manifest as a required completion condition. This keeps the concrete file-level bridge evidence from being merely informational in the status reporter. It is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `completion_allowed=false`, `blocker_count=22`, and paper-target blocker IDs present.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, `blocker_count=22`, and paper-target blocker IDs present.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `completion_blocker_count=22`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `completion_readiness_blocker_count=22`, `xr64_prelaunch_completion_blocker_count=22`, and `paper_target_evidence_state=missing_required_evidence`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `18 passed`.

## Result

The paper-target evidence manifest is now a real completion blocker. The second goal cannot be marked complete until the concrete paper-target bridge artifacts exist and direct paper-target comparison is allowed.

# 2026-06-21 Paper-Target Future Artifact Schema Validation

## Scope

The paper-target comparison evidence manifest now defines content schemas for all future bridge artifacts. Future JSON and JSONL files are no longer accepted by existence alone; if present, the checker parses them and verifies required fields. This is read-only and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary`: passed with `ok=true`, `artifact_schema_count=10`, `required_future_artifact_count=10`, and `existing_required_future_artifact_count=3`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_artifact_schema_count=10`.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `completion_allowed=false` and `blocker_count=22`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true` and `paper_target_artifact_schema_count=10`.
- `.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py`: `23 passed`.
- `.venv/bin/python -m pytest -q tests/test_current_result_synthesis.py tests/test_decide_xr64_postrun_promotion.py tests/test_emit_xr64_resume_commands.py tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_paper_ref_analysis_coverage.py tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_experiment_queue.py tests/test_second_goal_objective_trace.py tests/test_second_goal_status.py tests/test_submission_target_gap.py tests/test_submission_target_source_trace.py tests/test_xr64_postrun_evidence_contract.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `78 passed`.
- `git diff --check`: passed.

## Result

The schema layer records `artifact_schema_count=10` and adds regression coverage for malformed future JSONL payloads. Direct paper-target comparison remains blocked because the required future bridge artifacts still do not exist.

# 2026-06-21 XR-64 Ablation Provenance Writer Validation

## Scope

XR-64 future training now has a non-GPU provenance writer that creates the post-run ablation files required by the XR-64 evidence contract and patches test `eval_summary.json` files with the strict axis-certification fields required by the promotion helper. This is infrastructure only; no train/eval jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/write_xr64_ablation_provenance.py scripts/external/decide_xr64_postrun_promotion.py scripts/external/check_xr64_postrun_evidence_contract.py`: passed.
- `bash -n scripts/external/run_xr64_teacher_target_construction.sh`: passed.
- `.venv/bin/python -m pytest -q tests/test_write_xr64_ablation_provenance.py tests/test_decide_xr64_postrun_promotion.py tests/test_xr64_postrun_evidence_contract.py`: `19 passed`.

## Result

Future XR-64A/B runs will emit `train/ablation_lr_manifest.json`, `train/optimizer_config_snapshot.json`, `train/teacher_provenance.json`, and `train/ablation_attribution_report.json`, then patch each test eval summary with `head/loss/lr/teacher_model_training` ablation evidence. Current state still has no XR-64A/B training results.

# 2026-06-21 XR-64 Runner Required Checkpoint Guard Validation

## Scope

XR-64 future training runner now treats the three post-run checkpoint kinds as required by default. Missing `best_track_p10.pt`, `best_track_p5.pt`, `best_metric_track_center_px.pt`, missing test `eval_summary.json`, or missing test `eval_rows.json` will stop the runner instead of silently producing partial post-run evidence. This does not run train/eval jobs.

## Validation Evidence

- `bash -n scripts/external/run_xr64_teacher_target_construction.sh`: passed.
- `.venv/bin/python -m pytest -q tests/test_xr64_runner_contract.py tests/test_write_xr64_ablation_provenance.py tests/test_xr64_postrun_evidence_contract.py tests/test_decide_xr64_postrun_promotion.py tests/test_second_goal_status.py`: `26 passed`.

## Result

Future XR-64A/B post-run evidence is now fail-fast aligned with the post-run evidence contract. Diagnostic partial eval remains available only by explicitly setting `XR64_REQUIRE_ALL_CHECKPOINTS=0`.

# 2026-06-21 XR-64 Post-Run Candidate Collector Validation

## Scope

Added a read-only collector that scans future patched XR-64 test `eval_summary.json` files and prepares `decide_xr64_postrun_promotion.py` candidate inputs. It deduplicates by latest lane/checkpoint by default and does not run train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/collect_xr64_postrun_candidates.py scripts/external/decide_xr64_postrun_promotion.py scripts/external/write_xr64_ablation_provenance.py`: passed.
- `.venv/bin/python -m pytest -q tests/test_collect_xr64_postrun_candidates.py tests/test_decide_xr64_postrun_promotion.py tests/test_write_xr64_ablation_provenance.py`: `16 passed`.
- `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --runs-root /tmp/nonexistent_xr64_runs --format summary`: passed with `candidate_count=0` and `decision_status=missing_postrun_evidence`.

## Result

Future XR-64A/B post-run decision can now be built from discovered certified eval summaries instead of manually assembling `--candidate` paths. Current state still has no XR-64A/B candidates.

# 2026-06-21 XR-64 Candidate Collector Status Integration Validation

## Scope

The second-goal status reporter now surfaces the candidate collector as a tracked post-run artifact and reports the current no-candidate state from the promotion decision placeholder. It does not scan `runs/` or launch train/eval jobs.

## Validation Evidence

- `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/collect_xr64_postrun_candidates.py`: passed.
- `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_collect_xr64_postrun_candidates.py`: `8 passed`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_candidate_collector_exists=true`, `xr64_candidate_collector_current_candidate_count=0`, and `xr64_candidate_collector_current_decision_status=missing_postrun_evidence`.

## Result

Status reporting now shows that post-run candidate collection is ready but no XR-64A/B candidates exist yet.

# 2026-06-21 XR-64 Resume Manifest Post-Run Review Integration

## Scope

The XR-64 future-only resume command manifest now covers the complete review path: prep commands, strict readiness, XR-64A/B launch commands, and post-run candidate/promotion-decision review commands. This remains non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/emit_xr64_resume_commands.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `postrun_command_count=2`.
- `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section postrun --format commands`: passed and printed only `collect_xr64_postrun_candidates.py --format summary` and `--format commands`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_command_manifest_postrun_command_count=2`.
- `.venv/bin/python -m pytest -q tests/test_emit_xr64_resume_commands.py tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `14 passed`.

## Result

Future XR-64 resumption now has a manifest-backed path from preparation through post-run candidate discovery. Current state still has `candidate_count=0`, `missing_eval_rows=8`, and `missing_overrides=6`.

# 2026-06-21 XR-64 Prelaunch Packet Post-Run Phase Integration

## Scope

The XR-64 prelaunch packet now mirrors the resume command manifest through post-run candidate review. The packet has six phases and includes the post-run collector review command. This remains non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `.venv/bin/python -m json.tool docs/resources/xr64_prelaunch_packet_2026_06_20.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `phase_count=6` and `manifest_postrun_command_count=2`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_prelaunch_phase_count=6` and `xr64_prelaunch_postrun_command_count=2`.
- `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py tests/test_xr64_resume_command_manifest.py`: `18 passed`.

## Result

The prelaunch packet now covers status, XR-64-prep eval rows, override generation, strict readiness, XR-64A/B launch, and post-run candidate review. Regression tests reject missing Phase 5 and missing post-run command review. Current state still has no XR-64A/B post-run candidates.

# 2026-06-21 Completion Readiness XR-64 Post-Run Propagation

## Scope

The second-goal completion-readiness contract now verifies that the XR-64 command manifest, prelaunch packet, and post-run candidate state are propagated into readiness status. This remains non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `xr64_command_manifest_postrun_command_count=2`, `xr64_prelaunch_phase_count=6`, `xr64_prelaunch_postrun_command_count=2`, and `xr64_postrun_candidate_count=0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the same readiness fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `30 passed`.
- `git diff --check`: passed.

## Result

Completion readiness now rejects stale prelaunch/post-run state instead of only checking missing eval rows and overrides. Current state remains blocked: `completion_allowed=false`, `missing_eval_rows=8`, `missing_overrides=6`, and `xr64_postrun_candidate_count=0`.

# 2026-06-21 Completion Gate Post-Run Promotion Blocker Propagation

## Scope

The second-goal completion gate, readiness contract, XR-64 prelaunch packet, and status reporter now agree on the stricter post-run promotion blocker state. This remains non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `completion_allowed=false` and `blocker_count=28`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, `blocker_count=28`, `xr64_prelaunch_phase_count=6`, `xr64_prelaunch_postrun_command_count=2`, and `xr64_postrun_candidate_count=0`.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `completion_blocker_count=28`, `phase_count=6`, and `manifest_postrun_command_count=2`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `completion_readiness_blocker_count=28` and `xr64_prelaunch_completion_blocker_count=28`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `30 passed`.
- `git diff --check`: passed.

## Result

Current completion/readiness reporting is synchronized at `28` blockers. The added blocker family is XR-64 post-run promotion evidence: no post-run candidates, no software promotion allowance, no single-model claim allowance, uncertified axis claims, unchecked controls, and no P1 evidence candidate. The active goal remains incomplete while experiments are paused and XR-64 generated artifacts/training evidence are missing.

# 2026-06-21 XR-64 Prep Runner Existing-Artifact Validation

## Scope

The XR-64 prep runner now validates existing eval-row and override JSON files before reusing them through skip paths. This reduces resume risk after partial XR-64-prep attempts. This pass is non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`: passed.
- `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/emit_xr64_resume_commands.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `prep_runner_contract_ok=true` and `prep_runner_contract_check_count=17`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_command_manifest_prep_runner_contract_ok=true` and `xr64_command_manifest_prep_runner_contract_check_count=17`.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_emit_xr64_resume_commands.py tests/test_second_goal_status.py`: `16 passed`.
- `git diff --check`: passed.

## Result

`scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` now validates skipped existing `eval_rows.json` and override JSON files before accepting them. Override generation also validates the four teacher eval-row inputs for the target split before building overrides. The resume command manifest checker statically guards this behavior.

# 2026-06-21 XR-64 Next Prep Command Selector

## Scope

Added a no-execute selector for XR-64-prep that reads the resume command manifest and current generated-artifact state, then reports the next command that is ready after explicit user resume. This does not run train/eval jobs.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/emit_xr64_next_prep_command.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format summary`: passed with `next_id=XR64-EVAL-TRAIN-XR62A`, `ready_after_resume_count=8`, `blocked_count=2`, and `allowed_to_run_now=false`.
- `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands`: passed and printed only the next review command.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the selector fields with `xr64_next_prep_selector_ok=true`.
- `.venv/bin/python -m pytest -q tests/test_emit_xr64_next_prep_command.py tests/test_second_goal_status.py tests/test_emit_xr64_resume_commands.py`: `11 passed`.

## Result

Future XR-64-prep resumption now has a deterministic next-command selector. Current state selects `XR64-EVAL-TRAIN-XR62A`; build commands remain blocked until eval-row artifacts exist. The selector reports `execute_supported=false` and `allowed_to_run_now=false`, preserving the experiment pause directive.

# 2026-06-21 XR-64 Selector Readiness Contract Propagation

## Scope

The second-goal completion-readiness contract now verifies the XR-64 next-prep selector state against the live status reporter. This prevents stale `next_id`, selector counts, or execute-support flags from being treated as current readiness evidence. This remains non-executing infrastructure; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/emit_xr64_next_prep_command.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `xr64_next_prep_selector_next_id=XR64-EVAL-TRAIN-XR62A`, `next_status=ready_after_resume`, and `execute_supported=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the same selector readiness fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_emit_xr64_next_prep_command.py`: `25 passed`.

## Result

Completion readiness now binds the selector fields `ok`, `ready_after_resume_count`, `completed_count`, `blocked_count`, `invalid_count`, `next_id`, `next_status`, `next_allowed_to_run_now`, and `execute_supported`. Current completion remains blocked with `missing_eval_rows=8`, `missing_overrides=6`, and no XR-64 post-run candidates.

# 2026-06-21 Post-XR64 Decision Tree Artifact

## Scope

Added a non-executing post-XR64 decision tree that maps future XR-64A/B result patterns to the smallest next PAPER_REF-backed experiment. This keeps XR-65/XR-66/XR-67/XR-68 deferred until matching evidence exists and preserves the current XR-64-prep gate. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`: passed.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports experiments remain paused with `xr64_ready_to_train=false`, `xr64_missing_eval_rows=8`, and `xr64_missing_overrides=6`.
- `git diff --check`: passed.

## Result

The second-goal plan now has explicit claim levels, lane-to-PAPER_REF evidence trace, and post-XR64 routing. XR-64-prep remains the first executable worker after user resume, and the current next-prep selector remains `XR64-EVAL-TRAIN-XR62A`.

# 2026-06-21 Post-XR64 Decision Tree Checker Integration

## Scope

The post-XR64 decision tree is now validated by a dedicated no-execute checker and surfaced through the second-goal status reporter. This prevents branch routing, claim levels, or XR-64-prep priority from drifting silently. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_post_xr64_decision_tree.py scripts/external/report_second_goal_status.py`: passed.
- `python3 -m json.tool docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `.venv/bin/python scripts/external/check_second_goal_post_xr64_decision_tree.py --format summary`: passed with `ok=true`, `branch_count=7`, `claim_level_count=4`, `minimal_next_worker_after_resume=XR-64-prep`, and `current_next_prep_id=XR64-EVAL-TRAIN-XR62A`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the `post_xr64_decision_tree_*` fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_decision_tree.py tests/test_second_goal_status.py`: `8 passed`.

## Result

Post-XR64 routing now has machine-readable validation. Current status remains paused: XR-64 is not ready to train, `8` eval rows and `6` override files are missing, and direct submission comparison remains blocked.

# 2026-06-21 Post-XR64 Readiness Contract Integration

## Scope

The second-goal completion-readiness contract now binds the post-XR64 decision tree to live status output. This prevents completion-readiness from passing if the decision tree drifts on claim levels, branch count, XR-64-prep priority, next prep id, XR-64 ready state, or direct submission comparison state. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_post_xr64_decision_tree.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with post-XR64 fields including `branch_count=7`, `minimal_next_worker=XR-64-prep`, and `direct_submission_comparison_allowed=false`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_second_goal_post_xr64_decision_tree.py`: `30 passed`.

## Result

Completion readiness now requires the post-XR64 decision tree to remain non-executing and aligned with the current XR-64-first route. Current completion remains blocked with `missing_eval_rows=8`, `missing_overrides=6`, no XR-64 post-run candidates, and blocked paper-target comparison.

# 2026-06-21 Post-XR64 Objective Trace Integration

## Scope

The second-goal objective trace now binds the post-XR64 decision tree directly to REQ-2 experiment-planning evidence. This prevents the objective trace from proving only that an experiment queue exists while omitting the current result-pattern routing plan. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_objective_trace_2026_06_20.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_objective_trace.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary`: passed with `ok=true`, `post_xr64_branch_count=7`, `post_xr64_minimal_next_worker=XR-64-prep`, and `post_xr64_direct_submission_comparison_allowed=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `objective_trace_post_xr64_*` fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py tests/test_second_goal_status.py tests/test_second_goal_post_xr64_decision_tree.py`: `15 passed`.
- `git diff --check`: passed.

## Result

REQ-2 now treats the post-XR64 decision tree as `experiment_planning_only` evidence. The objective trace checker cross-validates the linked tree path, checker/test paths, branch count, claim-level count, XR-64-prep minimal worker, current next prep id, missing XR-64 artifacts, and blocked direct submission comparison. The active second goal remains incomplete.

# 2026-06-21 Paper Target Bridge Artifact Workplan

## Scope

Added a non-executing workplan for the `10` future artifacts required by the paper-target comparison evidence manifest. The workplan separates safe paused-state doc-prefill artifacts from evaluator outputs and trained-candidate evidence that remain blocked until explicit resume. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary`: passed with `ok=true`, `work_package_count=4`, `future_artifact_count=10`, `allowed_while_paused_package_count=2`, and `doc_prefill_artifact_count=3`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_bridge_workplan_*` fields.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_artifact_workplan.py tests/test_second_goal_status.py`: `9 passed`.

## Result

Paper-target bridge work is now ordered into `PTB-0-DOC-PREFILL`, `PTB-1-EVALUATOR-SCHEMA`, `PTB-2-XR64-TRAINED-CANDIDATE`, and `PTB-3-BRIDGE-DECISION`. Only doc prefill and schema planning are allowed while paused. Direct submission comparison remains blocked.

# 2026-06-21 Paper Target Bridge PTB-0 Doc-Prefill Artifacts

## Scope

Created the three paused-safe paper-target bridge source-note artifacts and updated the manifest/workplan/status contracts to reflect their presence. This does not run train/eval/GPU jobs and does not allow direct comparison with `10_submission_initial`.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_second_goal_completion_gate.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary`: passed with `ok=true`, `missing_evidence_count=3`, `partial_evidence_count=4`, and `existing_required_future_artifact_count=3`.
- `.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary`: passed with `ok=true` and `current_existing_future_artifact_count=3`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_existing_required_future_artifact_count=3` and `paper_target_bridge_workplan_current_existing_future_artifact_count=3`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`; completion remains blocked with `blocker_count=28`.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: reports `completion_allowed=false` with `28` blockers.
- `.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_paper_target_bridge_artifact_workplan.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_completion_gate.py`: `45 passed`.
- `git diff --check`: passed.

## Result

`PTB-0-DOC-PREFILL` now has concrete JSON artifacts for split protocol, cur_state target mapping, and accuracy/latency separation. Evidence state remains `missing_required_evidence`; direct paper-target comparison remains blocked until evaluator artifacts and trained XR-64-or-later candidate evidence exist.

# 2026-06-21 Paper Target Bridge PTB-1 Readiness Binding

## Scope

Bound the PTB-1 evaluator schema contract into the second-goal completion-readiness/status path. This makes completion-readiness fail if PTB-1 schema counts or paused-state output-payload absence drift. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_paper_target_bridge_evaluator_schema_contract.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, `paper_target_bridge_evaluator_schema_output_count=5`, and `paper_target_bridge_evaluator_schema_current_existing_output_count=0`.
- `.venv/bin/python scripts/external/check_paper_target_bridge_evaluator_schema_contract.py --format summary`: passed with `ok=true`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports both standalone `paper_target_bridge_evaluator_schema_*` and mirrored `completion_readiness_paper_target_bridge_evaluator_schema_*` fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_paper_target_bridge_evaluator_schema_contract.py`: `35 passed`.

## Result

Completion readiness now requires the PTB-1 schema checker/test in its validation contract and mirrors PTB-1 paused-state counts in status output. Completion remains blocked with `completion_allowed=false`, `blocker_count=28`, `missing_eval_rows=8`, `missing_overrides=6`, and no execution-dependent PTB-1 payload files.

# 2026-06-21 Paper Target Bridge PTB-1 Payload Validator

## Scope

Added a no-create/no-execute PTB-1 payload validator. Default mode verifies that official PTB-1 output paths remain absent while experiments are paused. Future mode, only with explicit `--allow-present`, validates the 5 payload files against the PTB-1 schema contract and checks core type/range/count consistency.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m json.tool docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_paper_target_bridge_payloads.py scripts/external/check_paper_target_bridge_evaluator_schema_contract.py scripts/external/check_second_goal_completion_readiness_contract.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary`: passed with `payload_present_count=0`, `payload_missing_count=5`, `direct_submission_comparison_allowed=false`, and `paper_level_completion_allowed=false`.
- `.venv/bin/python scripts/external/check_paper_target_bridge_evaluator_schema_contract.py --format summary`: passed with `ok=true`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_payloads.py tests/test_paper_target_bridge_evaluator_schema_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `49 passed`.

## Result

PTB-1 now has a schema contract and a payload validation harness. The harness does not generate or bless paper-target evidence while paused; it rejects official payload presence by default and rejects direct-comparison/paper-level-completion flag drift. Direct paper-target comparison remains blocked.

# 2026-06-21 Paper Target Bridge PTB-2 Trained-Candidate Contract Binding

## Scope

Registered the PTB-2 trained-candidate evidence contract as a first-class second-goal artifact and bound it into status/readiness validation. This is a no-execute schema and leakage-guard update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/check_paper_target_bridge_trained_candidate_contract.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_trained_candidate_contract.py --format summary`: passed with `ok=true`, `trained_candidate_output_count=2`, and `current_existing_output_count=0`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true` and PTB-2 payload allowance `false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_bridge_trained_candidate_*` plus mirrored `completion_readiness_paper_target_bridge_trained_candidate_*` fields.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_trained_candidate_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_paper_target_bridge_artifact_workplan.py`: `52 passed`.

## Result

PTB-2 now has a registered contract, status reporter inventory, completion-readiness mirror, workplan source binding, and tests for future payload-pair validation. Official PTB-2 payload files remain absent while experiments are paused. Direct paper-target comparison and paper-level completion remain blocked.

# 2026-06-21 Paper Target Bridge PTB-3 Readiness/Status Regression Binding

## Scope

Hardened the paused-state PTB-3 bridge-decision integration tests so status and completion-readiness outputs cannot silently drop or drift from the PTB-3 decision contract. This is a no-execute validation update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_paper_target_bridge_decision_contract.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_decision_contract.py --format summary`: passed with `ok=true`, `decision_output_count=1`, and `current_existing_output_count=0`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true` and PTB-3 payload allowance `false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports standalone `paper_target_bridge_decision_*` plus mirrored `completion_readiness_paper_target_bridge_decision_*` fields.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_decision_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `49 passed`.

## Result

PTB-3 bridge-decision schema remains allowed while paused, but `bridge_decision.json` payload creation remains blocked. Completion readiness now has regression coverage for PTB-3 execute support, schema-allowed state, output count, paused payload absence, required checker command, required test command, and status summary emission. Direct paper-target comparison remains blocked.

# 2026-06-21 Post-XR64 Follow-Up Experiment Contract Validation

## Scope

Added a no-execute launch/promotion contract for XR-64C, XR-65, XR-66, XR-67, and XR-68. The contract converts the PAPER_REF-backed post-XR64 decision tree into row-specific prerequisites, blocked launch gates, required evidence, forbidden evidence, control variables, dependencies, and rejection rules. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_post_xr64_followup_contract.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_post_xr64_followup_contract.py --format summary`: passed with `ok=true`, `experiment_count=5`, `blocked_launch_count=5`, `p1_count=3`, and `p2_count=2`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true` and follow-up launch allowed `false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports standalone `post_xr64_followup_contract_*` plus mirrored `completion_readiness_post_xr64_followup_contract_*` fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_followup_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `55 passed`.

## Result

All post-XR64 follow-up branches are explicitly blocked while experiments are paused and while XR-64A/B post-run evidence is absent. XR-64C remains diagnostic unless full-test gates improve, XR-65 requires same-scene temporal evidence and bounded center drift, XR-66 requires a no-train confidence/local diagnostic first, XR-67 requires dense or continuous trajectories, and XR-68 requires a stronger stable full-width teacher plus student/hardware evidence.

# 2026-06-21 XR-64 Resume Command Review Safety

## Scope

Hardened the no-execute XR-64 command emitters so `--format commands` prints review-only commented command lines while experiments remain paused. This reduces accidental copy/paste execution risk without changing the manifest, readiness gates, or future resume sequence. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/emit_xr64_next_prep_command.py scripts/external/emit_xr64_resume_commands.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands`: passed and prints the selected XR-64-prep command as `# command: ...`.
- `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section prep --format commands`: passed and prints all prep commands as `# command: ...`.
- `.venv/bin/python -m pytest -q tests/test_emit_xr64_next_prep_command.py tests/test_emit_xr64_resume_commands.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `20 passed`.

## Result

XR-64 command review output remains useful for future resume planning, but no longer emits directly executable shell lines in the default review-oriented command format. The active next command is still `XR64-EVAL-TRAIN-XR62A`, and execution remains blocked until explicit user resume.

# 2026-06-21 XR-64 Post-Run Candidate Matrix Guard

## Scope

Hardened future XR-64 post-run promotion decisions so partial candidate evidence cannot promote a checkpoint. The decision helper now requires the full XR-64A/XR-64B by checkpoint-kind matrix before software promotion, single-model SOTA, axis claims, or control-variable claims can be allowed. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/collect_xr64_postrun_candidates.py scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed with `decision_status=missing_postrun_evidence`, `required_candidate_matrix_complete=false`, and `missing_required_candidate_count=6`.
- `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary`: passed with `candidate_count=0`, `required_candidate_matrix_complete=false`, and `missing_required_candidate_count=6`.
- `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_collect_xr64_postrun_candidates.py tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_status.py`: `25 passed`.

## Result

Future XR-64 promotion review now requires all six required eval summaries: XR-64A and XR-64B each with `best_metric_track_center_px`, `best_track_p10`, and `best_track_p5`. Partial A/B evidence can still be collected for debugging, but it yields `invalid_candidate_evidence` and cannot promote.

# 2026-06-21 XR-64 Override-Cleared Type Guard

## Scope

Hardened future XR-64 post-run promotion decisions so final test eval summaries must prove target overrides were cleared with exact flat eval-summary fields and strict JSON types. This is a no-execute evidence-gate update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/collect_xr64_postrun_candidates.py scripts/external/write_xr64_ablation_provenance.py scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed with `decision_status=missing_postrun_evidence`, `override_cleared_required=true`, `override_cleared_candidate_count=0`, and `missing_required_candidate_count=6`.
- `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary`: passed with `raw_eval_summary_count=526`, `candidate_count=0`, `required_candidate_matrix_complete=false`, and `missing_required_candidate_count=6`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `execution_state=paused_by_user_directive`, `active_goal_complete=false`, `xr64_ready_to_train=false`, `xr64_missing_eval_rows=8`, and `xr64_missing_overrides=6`.
- `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_write_xr64_ablation_provenance.py tests/test_collect_xr64_postrun_candidates.py tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_status.py`: `33 passed`.

## Result

Future XR-64 post-run promotion rejects dotted-only override keys, numeric `0` for `data_allow_test_target_override`, string `"false"`, boolean `false` for numeric loss weights, and string `"0.0"` for numeric loss weights. A clean winning candidate is insufficient if any required XR-64A/B checkpoint-kind candidate has dirty override evidence. Completion remains blocked because XR-64 eval rows/overrides and post-run candidates are still absent.

# 2026-06-21 XR-64 Resume Full-Coverage Gate

## Scope

Hardened the XR-64 strict resume checker so generated train/val teacher eval rows and override JSON files must cover every sample ID in their split manifest. This prevents subset/smoke artifacts from satisfying `ready_to_train`. This is a no-execute readiness-gate update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `ok=true`, `expected_eval_rows=8`, `expected_overrides=6`, `build_inputs_ready=false`, and `errors=0`.
- `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed with `resume_status=generated_incomplete`, `missing_eval_rows=8`, `missing_overrides=6`, and `leakage_risk=none`.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_xr64_resume_command_manifest.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `27 passed`.

## Result

`check_xr64_resume_artifacts.py` now rejects partial train/val teacher eval-row files and partial override JSONs even when the files exist and have valid row shape. `xr64_resume_command_manifest_2026_06_20.*` records this full-coverage contract. Current state remains generated-incomplete because all 8 eval-row files and 6 override JSON files are still absent while experiments are paused.

# 2026-06-21 XR-64 Full-Coverage Readiness Mirror

## Scope

Mirrored the XR-64 strict generated artifact contract into the second-goal completion-readiness contract and status reporter. The mirrored fields cover full eval-row coverage, full override coverage, duplicate sample ID rejection, opposite-split sample ID rejection, subset-artifact rejection, and the strict checker path. This is a no-execute readiness/status update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/check_xr64_resume_artifacts.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, duplicate/opposite split policies `false`, subset-artifact readiness `false`, and strict checker `scripts/external/check_xr64_resume_artifacts.py`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the same strict fields under both `completion_readiness_xr64_command_manifest_*` and `xr64_command_manifest_*`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_resume_command_manifest.py tests/test_xr64_resume_artifacts.py`: `68 passed`.
- `git diff --check -- scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.

## Result

The completion-readiness layer now rejects drift if the strict generated-artifact policy changes or is omitted from the current state mirror. GPT5.5 read-only evaluator feedback identified missing explicit drift tests for full eval and full override coverage; those tests were added before final validation. Current XR-64 state remains `generated_incomplete`: `8` train/val teacher eval rows and `6` train/val override JSON files are still missing while experiments are paused.

# 2026-06-21 XR-64 Expected Generated Count Contract

## Scope

Added explicit train/val cardinality expectations to the XR-64 resume command manifest and checker. This pins future XR-64-prep output counts to the current `manifest1` train/val split sizes: train `5929`, val `844`, `4` teachers, `3` override rules, `27092` total teacher eval-row records, and `20319` total override records. This is a no-execute contract update; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_resume_artifacts.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with expected split rows `5929/844`, teacher count `4`, override rule count `3`, total eval rows `27092`, and total override rows `20319`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the same `xr64_command_manifest_expected_*` count fields.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_resume_artifacts.py`: `71 passed`.

## Result

The XR-64 command manifest now fails if its expected generated counts drift away from the current train/val split manifests. This reduces the risk that future generated files with the correct names but wrong cardinality are mistaken for valid XR-64-prep evidence. Current execution remains paused, and generated artifacts are still absent.

# 2026-06-21 Second-Goal Blocker Artifact Map

## Scope

Added a no-execute blocker-to-artifact map for the active second goal. The map enumerates the current XR-64 generated-artifact blockers, paper-target bridge payload blockers, blocked execution units, ablation-axis implications, and drift-prone status fields. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_blocker_artifact_map_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_blocker_artifact_map.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_blocker_artifact_map.py --format summary`: passed with `blocker_count=28`, `xr64_missing_eval_rows=8`, `xr64_missing_overrides=6`, PTB missing payload counts `5/2/1`, and `ablation_axis_count=6`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `second_goal_blocker_map_*` fields.
- `.venv/bin/python -m pytest -q tests/test_second_goal_blocker_artifact_map.py tests/test_second_goal_status.py`: `10 passed`.
- `git diff --check -- docs/resources/second_goal_blocker_artifact_map_2026_06_21.json docs/resources/second_goal_blocker_artifact_map_2026_06_21.md scripts/external/check_second_goal_blocker_artifact_map.py scripts/external/report_second_goal_status.py tests/test_second_goal_blocker_artifact_map.py tests/test_second_goal_status.py docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.

## Result

The second-goal status chain now has a single machine-checkable map from current blockers to exact missing artifacts. It keeps `execute_supported=false`, `allowed_to_run_now=false`, and direct paper-target comparison blocked while experiments remain paused. Completion remains blocked because the map is not execution evidence: XR-64 still lacks `8` eval-row files and `6` override JSON files, and no XR-64A/B post-run candidate exists.

# 2026-06-21 XR-64 Prep Progress Ledger

## Scope

Added a no-execute XR-64-prep progress ledger that mirrors the live next-prep selector. The ledger tracks all `10` prep units, the current `8` ready-after-resume eval units, the `2` blocked build units, expected row-count evidence, strict readiness requirements, and pause guardrails. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_prep_progress_ledger_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_prep_progress_ledger.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_prep_progress_ledger.py --format summary`: passed with `prep_units=10`, `ready_after_resume_count=8`, `blocked_count=2`, `missing_eval_rows=8`, and `missing_overrides=6`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_prep_ledger_*` fields.
- `.venv/bin/python -m pytest -q tests/test_xr64_prep_progress_ledger.py tests/test_second_goal_status.py`: passed.

## Result

The status chain now has a machine-checkable progress ledger between the command selector and strict XR-64 readiness. It prevents `ready_after_resume` from being interpreted as executable while experiments are paused and keeps XR-64A/B launch blocked until strict generated artifacts exist.

# 2026-06-21 Second-Goal Resume Readiness Matrix

## Scope

Added a no-execute readiness matrix that binds PAPER_REF experiment axes, XR-64 resume gates, post-run promotion blockers, and paper-target bridge blockers into one drift-checked artifact. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_resume_readiness_matrix.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_resume_readiness_matrix.py --format summary`: passed with `axis_count=6`, `resume_gate_count=5`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, missing eval rows `8`, missing overrides `6`, and direct comparison/promotion flags `false`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_resume_readiness_matrix.py`: `6 passed`.

## Result

The second-goal status chain now has an operator/evaluator matrix that prevents the plan from drifting into false completion: PAPER_REF coverage is planning-only, XR-64A/B remain blocked until strict readiness, post-run promotion remains blocked until candidate evidence exists, and direct `10_submission_initial` comparison remains blocked until paper-target bridge evidence is complete.

# 2026-06-21 Resume Readiness Matrix Status Integration

## Scope

Integrated the resume readiness matrix into `scripts/external/report_second_goal_status.py` and `tests/test_second_goal_status.py`. This is a no-execute status integration; no train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_resume_readiness_matrix.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_resume_readiness_matrix.py --format summary`: passed with P0 IDs `XR-64-prep,XR-64A,XR-64B`, missing eval rows `8`, missing overrides `6`, and all promotion/direct-comparison flags `false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and now reports `second_goal_resume_matrix_*` fields; authority links checked `144`, missing `0`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_resume_readiness_matrix.py`: `9 passed`.

## Result

One status command now exposes the resume matrix alongside the existing XR-64 ledger, blocker map, objective trace, and paper-target bridge preflight. Completion remains blocked because this integration is status evidence only; XR-64 still lacks `8` eval-row files, `6` override JSON files, and post-run trained candidate evidence.

# 2026-06-21 XR-64 Prelaunch Latest-Source Cross-Check

## Scope

Extended the existing XR-64 prelaunch packet to cross-check the latest no-execute readiness sources instead of adding a redundant pre-resume gate. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_prelaunch_packet_2026_06_20.json`: passed.
- `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `resume_matrix_axis_count=6`, `resume_matrix_resume_gate_count=5`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, prep ledger counts `8/2`, blocker count `28`, and paper-target preflight counts `7/10`.
- `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py`: `10 passed`.

## Result

The prelaunch packet now validates against the resume readiness matrix, XR-64 prep progress ledger, second-goal blocker map, and paper-target metric bridge preflight. Completion remains blocked because these are readiness/status artifacts only; XR-64 generated eval rows, override JSONs, and XR-64A/B post-run evidence are still missing.

# 2026-06-21 Paper Target Bridge PTB-1 Resume Runbook

## Scope

Added and validated a no-execute PTB-1 resume runbook for future paper-target bridge evaluator payload generation. The runbook fixes the post-resume order for coordinate transform audit, sensor-space eval rows, paper-frame full-test P1, hybrid eval rows, and hybrid metric report while keeping all execution-dependent payload files absent during `paused_by_user_directive`.

## Validation Evidence

- `python3 -m json.tool docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py --format summary`: passed with `sequence_count=5`, `ptb1_output_count=5`, existing payload count `0`, missing payload count `5`, and `future_payload_validation_requires_allow_present=true`.
- `.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary`: passed in paused default mode with payload present count `0` and missing count `5`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_bridge_ptb1_resume_runbook_*` fields; authority links checked `149`, missing `0`.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_ptb1_resume_runbook.py tests/test_second_goal_status.py`: `8 passed`.

## Result

The status chain now has a machine-checkable bridge between the PTB-1 schema contract and future evaluator payload generation. This is not execution evidence: direct paper-target comparison, paper-level completion, and XR-64 train/eval remain blocked until explicit resume produces valid PTB-1 payloads and XR-64-or-later trained candidate evidence.

# 2026-06-21 Paper Target Bridge Resume Dependency Matrix

## Scope

Added a no-execute dependency matrix that binds PTB-0 doc-prefill artifacts, PTB-1 schema/runbook, future PTB-1 payloads, PTB-2 XR-64-or-later trained-candidate evidence, PTB-3 bridge decision, and the strict second-goal completion gate. The matrix keeps direct paper-target comparison and paper-level completion blocked while experiments remain paused.

## Validation Evidence

- `python3 -m json.tool docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_paper_target_bridge_resume_dependency_matrix.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_paper_target_bridge_resume_dependency_matrix.py --format summary`: passed with `dependency_gate_count=6`, `complete_now_gate_count=2`, `blocked_gate_count=4`, execution-dependent payload missing count `8`, `xr64_ready_to_train=false`, and `completion_allowed=false`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `paper_target_bridge_resume_dependency_matrix_*` fields; authority links checked `154`, missing `0`.
- `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_resume_dependency_matrix.py tests/test_second_goal_status.py`: `8 passed`.

## Result

The paper-target bridge now has a single drift-checked dependency gate from paused-safe source notes through PTB-1/2/3 evidence and final completion. This is still readiness infrastructure only: PTB-1 payloads, PTB-2 trained-candidate payloads, PTB-3 bridge decision, XR-64 generated artifacts, and XR-64A/B post-run evidence remain absent.

# 2026-06-21 PAPER_REF Current Ablation Plan Checker

## Scope

Added a no-execute checker and status integration for `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md`. The checker validates prompt-pack metadata, PAPER_REF signal coverage, eight queued experiment rows, P0 IDs, requested axes, lane trace, validation evidence markers, and paused-state no-completion/no-direct-comparison flags.

## Validation Evidence

- `python3 -m py_compile scripts/external/check_second_goal_paper_ref_current_ablation_plan.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_paper_ref_current_ablation_plan.py --format summary`: passed with experiment count `8`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, paper signal count `18`, mapping rows `10`, lane trace count `8`, axis count `8`, and errors `0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and now reports `paper_ref_current_ablation_plan_*` fields; authority links checked `168`, missing `0`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_paper_ref_current_ablation_plan.py tests/test_second_goal_status.py`: `6 passed`.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.

## Result

The current PAPER_REF ablation plan is now drift-checked by automation instead of being only a Markdown plan. This remains planning infrastructure only: no train/eval/GPU jobs were launched, XR-64 generated artifacts remain missing, and second-goal completion remains blocked.

# 2026-06-21 XR-64 Experiment Design Contract

## Scope

Added a non-executing XR-64 design contract that makes XR-64-prep/A/B/C lane parameters, GPU assignment, launch gates, promotion/rejection rules, and post-run evidence requirements independently machine-checkable. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_experiment_design_contract_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_experiment_design_contract.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_experiment_design_contract.py --format summary`: passed with lane count `4`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, axis count `6`, missing eval rows `8`, missing overrides `6`, and errors `0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and now reports `xr64_experiment_design_contract_*` fields with authority links missing `0`.
- `.venv/bin/python -m pytest -q tests/test_xr64_experiment_design_contract.py tests/test_second_goal_status.py`: passed.

## Result

XR-64 P0/P1 lane design is now a checked contract instead of being only embedded inside the broader experiment queue. Completion remains blocked: XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads are still absent.

# 2026-06-21 XR-64 Resume Execution DAG

## Scope

Added a non-executing XR-64 resume execution DAG that binds precheck, prep eval rows, prep override builds, strict readiness, XR-64A/B launch, and post-run review into a machine-checkable dependency order. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_resume_execution_dag_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_resume_execution_dag.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_resume_execution_dag.py --format summary`: passed with node count `17`, phase count `6`, prep nodes `10`, eval nodes `8`, build nodes `2`, launch nodes `2`, postrun nodes `2`, manifest/design contract alignment `true`, and errors `0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and now reports `xr64_resume_execution_dag_*` fields with authority links missing `0`.
- `.venv/bin/python -m pytest -q tests/test_xr64_resume_execution_dag.py tests/test_second_goal_status.py`: passed.

## Result

The future XR-64 resume order is now checked as a DAG instead of being inferred from separated command/design artifacts. Completion remains blocked: DAG readiness is not execution evidence, and XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads are still absent.

# 2026-06-21 Second Goal Accuracy Lift Decision Ladder

## Scope

Added a no-execute decision ladder that binds current metric-specific best results, XR-63 oracle headroom, XR-64 transfer, post-XR64 follow-up branches, and paper-target bridge blockers into one machine-checkable accuracy-lift artifact. No train/eval/GPU jobs were launched.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_accuracy_lift_decision_ladder.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_accuracy_lift_decision_ladder.py --format summary`: passed with stage count `5`, experiment count `8`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, follow-up IDs `XR-64C,XR-65,XR-66,XR-67,XR-68`, missing eval rows `8`, missing overrides `6`, and errors `0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `accuracy_lift_decision_ladder_*` fields with authority links missing `0`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_accuracy_lift_decision_ladder.py tests/test_second_goal_status.py`: `8 passed`.
- `git diff --check -- ...`: passed for touched files.

## Result

The current accuracy strategy now has an explicit ladder: current software gates -> XR-63 oracle signal -> XR-64-prep/A/B transfer -> blocked post-XR64 follow-ups -> paper-target bridge. Completion remains blocked until XR-64 generated artifacts, XR-64A/B post-run candidates, and paper-target bridge payloads exist.

# 2026-06-21 XR-64 Post-Run Tradeoff Scorecard

## Scope

Strengthened the future-only XR-64 post-run promotion helper so candidate selection requires bounded tradeoff. A candidate that improves one gate but severely regresses center/P10/P5 is now classified as diagnostic and cannot become the software-promotion winner.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed with candidate count `0`, bounded tradeoff candidates `0`, unbounded tradeoff candidates `0`, and errors `0`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_promotion_bounded_tradeoff_candidate_count=0`, `xr64_promotion_unbounded_tradeoff_candidate_count=0`, and authority links missing `0`.
- `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_second_goal_status.py`: `19 passed`.
- `git diff --check -- ...`: passed for touched files.

## Result

The future promotion helper now emits candidate-level `promotion_score`, `tradeoff_classification`, `bounded_tradeoff`, `severe_regressions`, and `tradeoff_floors`. Completion remains blocked: no XR-64A/B post-run candidates exist while experiments are paused.

## 2026-06-21 XR-64 Post-Run Promotion Decision Artifact Checker

## Scope

Added a dedicated artifact checker for `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`. The checker validates that the current no-evidence placeholder stays aligned with the read-only promotion helper, current gates, required XR-64A/B candidate matrix, override-clear schema, and bounded-tradeoff floors.

## Validation Evidence

- `python3 -m json.tool docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`: passed.
- `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- `python3 -m py_compile scripts/external/check_xr64_postrun_promotion_decision.py scripts/external/report_second_goal_status.py`: passed.
- `.venv/bin/python scripts/external/check_xr64_postrun_promotion_decision.py --format summary`: passed with `ok=true`, `candidate_count=0`, and `missing_required_candidate_count=6`.
- `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_promotion_decision_ok=true`, `xr64_promotion_decision_error_count=0`, and authority links missing `0`.
- `.venv/bin/python -m pytest -q tests/test_xr64_postrun_promotion_decision.py tests/test_second_goal_status.py`: `8 passed`.

## Result

The promotion decision artifact is now machine-checked. Completion remains blocked because XR-64A/B training runs, six post-run test eval summaries, and paper-target bridge payloads are still absent while experiments remain paused.

## 2026-06-21 Completion Gate Promotion Decision Checker Guard

## Scope

Connected the XR-64 post-run promotion decision artifact checker to the second-goal completion gate. A malformed or stale promotion-decision artifact now blocks completion through `xr64_postrun_promotion_decision_invalid`; a valid current artifact is recorded as guard `xr64_postrun_promotion_decision_valid`.

## Validation Evidence

- `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py`: passed.
- `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `blocker_count=28` and guard `xr64_postrun_promotion_decision_valid`.
- `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`, `blocker_count=28`, and guard `xr64_postrun_promotion_decision_valid`.
- `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py`: `8 passed`.

## Result

Completion remains blocked by missing XR-64 generated artifacts and post-run evidence, but the completion gate now also protects against stale or invalid promotion-decision artifacts.

## 2026-06-21 Stage1 Frame-Search Mid-Run And Warm-Start Runner

## Scope

Recorded the active Stage1 frame-based Search baseline evidence after the minimum 50-epoch requirement was exceeded, and added a follow-up runner for warm-start refinement from the current best lane-B checkpoint. The follow-up runner was validated only with static and dry-run checks; no additional GPU jobs were launched.

## Validation Evidence

- `ps -p 1332213,1332214 -o pid,ppid,sid,stat,etime,cmd`: both Stage1 jobs active with PPID `1` and status `Ssl`.
- `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader`: GPU0 and GPU1 both active, each with `720 MiB` memory used at the check.
- `.venv/bin/python scripts/external/summarize_training_history.py ...`: latest continuation check summarized lane A and lane B through epoch `198`; subsequent reporter checks observed lane A through epoch `229` and lane B through epoch `228`.
- `bash -n scripts/external/run_stage1_frame_search_baseline_matrix.sh`: passed.
- `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh`: passed.
- `DRY_RUN=1 bash scripts/external/run_stage1_frame_search_warmstart_refine.sh`: passed command expansion for S1C warm-start no-distill and S1D warm-start weak self-distill lanes.
- `python3 -m py_compile scripts/external/report_stage1_frame_search_baseline.py`: passed.
- `.venv/bin/python scripts/external/report_stage1_frame_search_baseline.py --format summary`: passed, ranking lane B first with `promote_checkpoint=True`; after the reporter semantics fix, incomplete A/B runs keep `wait_for_300_epoch_completion=True` and `launch_warmstart_refine_when_gpu_free=False`.
- `bash -n scripts/external/run_stage1_frame_search_refine_when_ready.sh`: passed.
- `SKIP_WAIT=1 SKIP_GPU_CHECK=1 ALLOW_INCOMPLETE_BASELINE=1 DRY_RUN=1 DETACH=0 RUN_TAG=watcher_dryrun_20260621_v2 bash scripts/external/run_stage1_frame_search_refine_when_ready.sh`: passed, including reporter readiness and warm-start runner command expansion.
- Real watcher launch check: PID `2087936`, PPID `1`, SID `2087936`, status `Ss`, log `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_v2.log`.
- Stall diagnosis after detached jobs stopped before 300 epochs:
  - `ps -p 1332213,1332214,2087936 -o pid,ppid,sid,stat,etime,cmd`: no matching live processes.
  - `rg -n "Traceback|Error|Exception|Killed|CUDA out|out of memory|RuntimeError|SIG|terminated|KeyboardInterrupt" ...`: no matching error text in the A/B/watcher logs.
  - Reporter still showed `complete_300=False`, so downstream S1C/S1D launch was correctly blocked by the watcher policy.
- Resume validation:
  - Added `LANE_A_RESUME` and `LANE_B_RESUME` support to `scripts/external/run_stage1_frame_search_baseline_matrix.sh`.
  - `bash -n scripts/external/run_stage1_frame_search_baseline_matrix.sh`: passed.
  - `DRY_RUN=1 RUN_TAG=20260621_2258_resume LANE_A_NAME=stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500 LANE_B_NAME=stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 LANE_A_RESUME=runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500 LANE_B_RESUME=runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 bash scripts/external/run_stage1_frame_search_baseline_matrix.sh`: passed.
  - Relaunch outside sandbox: lane A PID `2167860`, lane B PID `2167861`.
  - Resume log evidence: lane A `[resume] ... start_epoch=239`; lane B `[resume] ... start_epoch=238`.
  - Relaunched watcher outside sandbox: PID `2169848`, waiting on `2167860 2167861`.
- Supervisor validation:
  - Added duplicate-launch lock to `scripts/external/run_stage1_frame_search_warmstart_refine.sh`.
  - Added `scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`.
  - `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`: passed.
  - Supervisor launched outside sandbox as PID `2197421`, log `runs/NON_XR/shared/_logs/stage1_frame_search_supervisor_resume2258.log`, waiting on `2167860 2167861`.
  - Supervisor reporter snapshot: lane A epoch `246`, lane B epoch `245`, both `complete_300=False`; lane B remains best with Search P10 `28.22439415050003`.
  - Reporter semantics fix: `launch_warmstart_refine_when_gpu_free` now requires both a promotable checkpoint and no incomplete 300-epoch candidates.

## Result

Lane B is the current Stage1 baseline candidate with best Search P10 `28.2244`, best Search P5 `9.8338`, and best Search center `17.2888 px`, improving over the old Stage1 reference on all three metrics. The active jobs should continue to 300 epochs unless intentionally stopped; follow-up S1C/S1D refinement should wait until a GPU is free.

## 2026-06-21 Stage1 Frame-Search A/B Completion And Follow-Up Launch

## Scope

Recorded the completed 300-epoch A/B Stage1 frame-search baseline result, verified that lane B remains the current promotion checkpoint, and documented the active S1C/S1D follow-up runs.

## Validation Evidence

- `.venv/bin/python scripts/external/report_stage1_frame_search_baseline.py --format summary`: passed.
- Reporter completion evidence:
  - Lane A: `epochs=300`, `complete_300=True`, best Search P10 `25.324573696784253`.
  - Lane B: `epochs=300`, `complete_300=True`, best Search P10 `28.22439415050003`.
  - Promotion checkpoint: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
  - `wait_for_300_epoch_completion=False`.
  - `launch_warmstart_refine_when_gpu_free=True`.
- Host process check:
  - S1C runner PID `2402085` is intentionally stopped to prevent delayed duplicate S1D launch from the inherited-foreground runner.
  - S1C train PID `2402483` is alive on GPU0.
  - S1D train PID `2420379` is alive on GPU1.
- Log progress check:
  - S1C log reached epoch `12/120`; best Search P10 so far `27.9403`.
  - S1D log reached epoch `7/120`; best Search P10 so far `26.9351`.
- GPU check:
  - `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader` showed both RTX 5080 GPUs allocated by the follow-up jobs at the check.
- Launch-path hardening:
  - Patched `scripts/external/run_stage1_frame_search_refine_when_ready.sh` to call the warm-start runner with `DETACH=1`.
  - Patched `scripts/external/run_stage1_frame_search_supervisor_until_complete.sh` to call the warm-start runner with `DETACH=1`.
  - `bash -n scripts/external/run_stage1_frame_search_refine_when_ready.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`: passed.

## Result

The completed A/B baseline is usable as the current Stage1 frame-search reference, with lane B ahead of the old Stage1 reference by `+4.87533706089236 pp` Search P10 and `+1.0306673184880673 px` center improvement. Final promotion is still pending S1C/S1D comparison unless the user explicitly chooses to freeze lane B now.

## 2026-06-21 Stage1 Frame-Search Follow-Up Reporter

## Scope

Added a follow-up reporter for S1C/S1D so baseline replacement decisions are reproducible and gated by the requested minimum 50-epoch evidence threshold.

## Validation Evidence

- `python3 -m py_compile scripts/external/report_stage1_frame_search_followup.py`: passed.
- `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format summary`: passed.
- `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format json`: passed.
- Current summary output:
  - Baseline lane B: P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - S1C: `epochs=23`, `min_ready=False`, best P10 `27.94025216012631`, P5 `10.01010806605501`, center `17.340092240639454`, `promotable_after_min_epoch=False`.
  - S1D: `epochs=18`, `min_ready=False`, best P10 `27.629156058689333`, P5 `9.750674121784714`, center `17.34278841738431`, `promotable_after_min_epoch=False`.
  - Recommendation: `status=wait_min_epochs`, `promote_checkpoint=None`, `continue_training=True`.

## Result

S1C/S1D are running normally but have not yet reached the 50-epoch decision gate and have not beaten lane B on primary P10. The correct current action is to continue training and keep lane B as the active Stage1 frame-search baseline candidate.

## 2026-06-21 Stage1 Follow-Up Min-50 Watcher

## Scope

Added and launched a detached watcher that keeps polling the Stage1 follow-up reporter until S1C/S1D reach the minimum 50-epoch decision gate.

## Validation Evidence

- `bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`: passed.
- `DETACH=0 POLL_SEC=1 MAX_WAIT_SEC=1 RUN_TAG=stage1_followup_min50_dryrun_20260621 bash scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`: generated report JSON/summary and exited with expected timeout code `20` because the 50-epoch gate was not reached.
- `chmod +x scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh scripts/external/report_stage1_frame_search_followup.py`: applied.
- Real detached launch:
  - PID `2485273`, PPID `1`, SID `2485273`, status `Ss`.
  - Log `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755.log`.
  - Report summary `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.txt`.
  - Report JSON `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.json`.
- First poll report:
  - S1C `epochs=23`, `min_ready=False`, P10 delta `-0.284141990373719`.
  - S1D `epochs=18`, `min_ready=False`, P10 delta `-0.5952380918106961`.
  - Status `wait_min_epochs`.

## Result

The min-50 decision gate is now monitored automatically. No promotion is available yet; lane B remains the active baseline candidate while S1C/S1D continue.

# 2026-06-22 Stage1 frame-search continuation validation

## Scope

Validate that the Stage1 frame-search improvement work is still running and that no baseline promotion happens before the 50-epoch gate.

## Evidence

- Host process check:
  - S1C train PID `2402483` alive on GPU0.
  - S1D train PID `2420379` alive on GPU1.
  - Min-50 watcher PID `2485273` alive.
  - Post-followup queue PID `2508026` alive.
- GPU check:
  - GPU0 memory `720 MiB`, active compute process PID `2402483`.
  - GPU1 memory `732 MiB`, active compute process PID `2420379`.
- Reporter check:
  - Baseline lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - S1C completed epochs `67/120`, P10 delta `-0.284141990373719`, P5 delta `+0.21563341032783967`, center improvement delta `+0.05543105107433277`.
  - S1D completed epochs `61/120`, P10 delta `-0.5952380918106961`, P5 delta `+0.09321651818617305`, center improvement delta `-0.05393914906483843`.
  - Status `keep_baseline_unless_later_improves`, `promote_checkpoint=None`, `continue_training=True`.

## Added queue validation

- `scripts/external/run_stage1_frame_search_post_followup_queue.sh` was added to start lower-LR fallback lanes only after S1C/S1D finish or become non-promotable after the min-50 gate.
- It does not stop or modify the active S1C/S1D jobs.
- It requires follow-up completion and GPU free-memory checks before starting S1E/S1F.
- `scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh` now accepts `BASELINE_RUN` and `FOLLOWUP_RUNS`, validated with explicit S1C/S1D run paths.
- The post-followup queue now starts a fallback watcher after S1E/S1F launch, so lower-LR fallback lanes will be judged by the same min-50 promotion gate.
- Original queue PID `2508026` was terminated and replaced by updated queue PID `2638408`; S1C/S1D train PIDs were not stopped.

## Result

The experiment queue is progressing. S1C/S1D have crossed `min_epochs=50`, but no new baseline can be promoted yet because neither beats lane B on primary Search P10.

# 2026-06-22 Stage1 capacity-distill branch validation

## Scope

Validate the Stage1 frame-search accuracy-lift continuation without interrupting active S1C/S1D training. This pass adds the next capacity/distillation branch and confirms it is executable later.

## Evidence

- Follow-up reporter:
  - Baseline lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - S1C epoch `89/120`, best P10 `27.94025216012631`, best P5 `10.049416236157688`, center `17.233418217245138`, not promotable.
  - S1D epoch `82/120`, best P10 `27.715633986131202`, best P5 `10.086478287318968`, center `17.34278841738431`, not promotable.
  - Recommendation `keep_baseline_unless_later_improves`, `promote_checkpoint=None`, `continue_training=True`.
- Host/GPU checks:
  - S1C train PID `2402483` alive on GPU0.
  - S1D train PID `2420379` alive on GPU1.
  - Post-followup queue PID `2638408` alive.
  - S1G large no-distill PID `2723776` alive after launch.
  - S1H large baseline-teacher distill PID `2723777` alive after launch.
  - S1G/S1H min-50 watcher PID `2730067` alive.
  - Capacity-after-followups queue PID `2708973` was stopped after S1G/S1H started to avoid duplicate capacity launch.
- Added artifacts:
  - `scripts/external/run_stage1_frame_search_capacity_distill.sh`.
  - `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
  - `tests/test_model_role_cfg.py`.
  - `src/hbtxr/models/pruning.py` support for `model.teacher.*` role-specific overrides.

## Validation Commands

```bash
bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh
bash -n scripts/external/run_stage1_frame_search_post_followup_queue.sh
bash -n scripts/external/run_stage1_frame_search_capacity_distill.sh
bash -n scripts/external/run_stage1_frame_search_capacity_after_followups.sh
DETACH=0 DRY_RUN=1 POLL_SEC=1 MAX_WAIT_SEC=1 RUN_TAG=stage1_capacity_after_followups_dryrun_20260622 bash scripts/external/run_stage1_frame_search_capacity_after_followups.sh
DETACH=0 POLL_SEC=1 MAX_WAIT_SEC=1 RUN_TAG=stage1_followup_missing_history_dryrun_20260622 FOLLOWUP_RUNS='runs/NON_XR/raw/does_not_exist_a runs/NON_XR/raw/does_not_exist_b' bash scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh
python3 -m py_compile src/hbtxr/models/pruning.py scripts/external/report_stage1_frame_search_followup.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_model_role_cfg.py
PYTHONPATH=src .venv/bin/python -c "from hbtxr.training.model_factory import build_model, create_teacher_model; cfg={'model':{'embed_dim':256,'depth':8,'num_heads':4,'heads':{'active':'search'},'teacher':{'embed_dim':192,'depth':6,'num_heads':3,'head_hidden_dim':192}},'pruning':{'enabled':False},'distillation':{'enabled':True,'teacher_student':True},'data':{'input_size':[256,256]}}; s=build_model(cfg,role='student'); t=create_teacher_model(s,cfg,'cpu'); print('student',s.embed_dim,len(s.backbone.attn_stages)); print('teacher',t.embed_dim,len(t.backbone.attn_stages))"
DRY_RUN=1 DETACH=0 RUN_TAG=stage1_capacity_distill_dryrun_20260622 bash scripts/external/run_stage1_frame_search_capacity_distill.sh
git diff --check -- docs/track/PROGRESS.md docs/track/log.md docs/Validation.md docs/resources/stage1_frame_search_midrun_results_2026_06_21.md src/hbtxr/models/pruning.py tests/test_model_role_cfg.py scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh scripts/external/run_stage1_frame_search_post_followup_queue.sh scripts/external/run_stage1_frame_search_capacity_distill.sh
```

## Result

All focused validation passed. S1G/S1H were launched as a parallel capacity tier after S1C/S1D stayed below lane B beyond the 50-epoch gate. Early epoch-1 metrics are not promotion evidence; the capacity min-50 watcher is the active gate for future comparison.

# 2026-06-22 Stage1 fallback watcher correction validation

## Scope

Validate the corrected active Stage1 frame-search path after discovering that the S1G/S1H capacity branch stopped before the 50-epoch gate and that the initial S1E/S1F watcher used suffix-free paths.

## Evidence

- S1G/S1H logs:
  - S1G stopped during epoch `7/200`.
  - S1H stopped during epoch `6/200`.
  - Failure text: `DataLoader worker ... killed by signal: Killed`.
  - Conclusion: S1G/S1H did not reach the min-50 gate and cannot be compared as promotion candidates.
- Process cleanup:
  - Obsolete S1E/S1F watcher PID `2752001` was terminated.
  - Obsolete S1G/S1H watcher PID `2730067` was terminated.
- Corrected active watcher:
  - PID `2763859`.
  - Log `runs/NON_XR/shared/_logs/stage1_frame_search_fallback_min50_stage1_refine_after_min50_low_lr_20260622_0057_corrected_20260622_0059.log`.
  - Watches actual timestamped run directories:
    - `runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
    - `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
- Current S1E/S1F reporter output:
  - S1E epoch `7/120`, best P10 `27.67969512939453`.
  - S1F epoch `6/120`, best P10 `27.932390464926666`.
  - Baseline Lane B P10 `28.22439415050003`.
  - Status `wait_min_epochs`.
- Capacity retry safety:
  - `scripts/external/run_stage1_frame_search_capacity_distill.sh` now defaults to `NUM_WORKERS=2` instead of `8`.

## Validation Commands

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --min-epochs 50 \
  --target-epochs 120 \
  --followup runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --followup runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --format summary
ps -p 2751991,2751992,2763859 -o pid,ppid,sid,stat,etime,cmd
tail -n 80 runs/NON_XR/shared/_logs/stage1_frame_search_fallback_min50_stage1_refine_after_min50_low_lr_20260622_0057_corrected_20260622_0059.log
```

## Result

The active Stage1 improvement path is now S1E/S1F. Lane B remains the baseline until a later checkpoint passes the same min-50 promotion gate.

# 2026-06-22 Stage1 S1E/S1F live validation

## Evidence

- Reporter command returned `status=wait_min_epochs`.
- S1E completed epoch `9/120`, best P10 `27.67969512939453`.
- S1F completed epoch `9/120`, best P10 `27.932390464926666`.
- Lane B baseline P10 remains `28.22439415050003`.
- Host process check confirmed PIDs `2751991`, `2751992`, and watcher `2763859` alive.
- GPU check confirmed RTX 5080 devices visible and usable.

## Result

S1E/S1F are progressing but have not reached the required `50` epoch decision gate. Current evidence does not support promotion over Lane B.

# 2026-06-22 Stage1 capacity-after queue validation

## Scope

Validate that the next capacity branch is staged behind S1E/S1F completion and uses current timestamped run directories instead of stale paths.

## Evidence

- `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` shell syntax passed.
- Dry-run command with omitted `FOLLOWUP_RUNS` resolved current S1E/S1F paths:
  - `runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
  - `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
- Dry-run returned timeout code `20`, expected because the fallback runs are still below the 50-epoch gate.
- Corrected capacity-after queue PID `2800043` is alive.
- Queue first poll status:
  - S1E epoch `14/120`, best P10 `27.83692773782982`.
  - S1F epoch `13/120`, best P10 `27.932390464926666`.
  - Status `wait_min_epochs`.

## Result

Capacity retry automation is staged safely. It will not launch while S1E/S1F are below the gate, and it will skip capacity retry if an S1E/S1F checkpoint becomes promotable.

# 2026-06-22 Stage1 center-polish runner validation

## Scope

Validate the center-preserve follow-up branch staged after S1E/S1F, without interrupting the active S1E/S1F jobs.

## Evidence

- `scripts/external/run_stage1_frame_search_warmstart_refine.sh` accepts `BEST_METRIC_NAME`.
- `scripts/external/run_stage1_frame_search_center_preserve_polish.sh` was added and made executable.
- Shell syntax passed for:
  - `scripts/external/run_stage1_frame_search_warmstart_refine.sh`.
  - `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
  - `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
- Dry-run command:

```bash
DRY_RUN=1 DETACH=0 RUN_TAG=stage1_centerpolish_dryrun_20260622 \
  bash scripts/external/run_stage1_frame_search_center_preserve_polish.sh
```

- Dry-run produced two expected train commands:
  - no-distill lane with `training.best_metric_name=metric_search_center_px`, LR `1e-5`, `loss.search_xy_weight=2.0`.
  - weak self-distill lane with `training.best_metric_name=metric_search_center_px`, LR `7.5e-6`, EMA `0.999`.
- Queue replacement:
  - Stopped capacity queue PID `2800043`.
  - Launched center-polish queue PID `2819081`.
  - The new queue is alive and waits because S1E/S1F are still below the 50-epoch gate.

## Result

Center-preserve polish is staged as the next branch after S1E/S1F non-promotion. No active training process was stopped.

# 2026-06-22 Stage1 S1E/S1F pre-gate validation

## Evidence

- Reporter command:

```bash
.venv/bin/python scripts/external/report_stage1_frame_search_followup.py \
  --min-epochs 50 \
  --target-epochs 120 \
  --followup runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --followup runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951 \
  --format summary
```

- Reporter result:
  - S1E epoch `23/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `22/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, center `17.28884926831947`.
  - Status `wait_min_epochs`, `promote_checkpoint=None`.
- Host process check confirmed S1E/S1F and both watcher/queue processes alive.

## Result

No baseline promotion. S1E/S1F must reach epoch `50` before any promotion claim is valid.

# 2026-06-22 Stage1 second pre-gate validation

## Evidence

- Reporter result:
  - S1E epoch `26/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `25/120`, best P10 `27.932390464926666`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, center `17.28884926831947`.
  - Status `wait_min_epochs`, `promote_checkpoint=None`.
- Host process check confirmed S1E/S1F and watcher/queue PIDs alive.

## Result

No baseline promotion. Current best fallback remains below Lane B on the primary P10 metric and has not reached the 50-epoch gate.

# 2026-06-22 Stage1 third pre-gate validation

## Evidence

- Reporter result:
  - S1E epoch `31/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `30/120`, best P10 `28.078392352697986`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, center `17.28884926831947`.
  - Status `wait_min_epochs`, `promote_checkpoint=None`.
- Host process check confirmed S1E/S1F and watcher/queue PIDs alive.
- Center-polish queue remains in `wait_min_epochs`.

## Result

No baseline promotion. S1E/S1F are progressing but still below the explicit 50-epoch evaluation gate.

# 2026-06-22 Stage1 fourth pre-gate validation

## Evidence

- Reporter result:
  - S1E epoch `36/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `35/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `wait_min_epochs`, `promote_checkpoint=None`.
- Host process check confirmed S1E/S1F and watcher/queue PIDs alive.
- GPU check showed both GPUs had more than `15 GiB` free memory.

## Result

No baseline promotion. Current candidates are running normally but remain below the 50-epoch gate.

# 2026-06-22 Stage1 S1E/S1F min-50 validation

## Evidence

- Reporter result after both lanes crossed `min_epochs=50`:
  - S1E epoch `55/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `53/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Process check confirmed S1E/S1F and center-polish queue PIDs alive after the min-50 gate check.
- The corrected min-50 watcher had completed by the final process check.

## Result

No baseline promotion at the min-50 gate. Lane B remains the active baseline, while S1E/S1F continue toward 120 epochs.

# 2026-06-22 Stage1 S1E/S1F post-gate validation

## Evidence

- Reporter result:
  - S1E epoch `67/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `64/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Process check confirmed S1E/S1F and center-polish queue PIDs alive.

## Result

No promotion after the post-gate refresh. The current active baseline remains Lane B.

# 2026-06-22 Stage1 post-centerpolish queue validation

## Evidence

- Added `scripts/external/run_stage1_frame_search_after_centerpolish_capacity_queue.sh`.
- Dry-run wait behavior was checked with:

```bash
MAX_WAIT_SEC=0 POLL_SEC=1 DRY_RUN=1 DETACH=0 timeout 3 \
  bash scripts/external/run_stage1_frame_search_after_centerpolish_capacity_queue.sh
```

- The dry-run timed out with status `124` after repeatedly reporting missing center-polish lanes; this is expected before center-polish launches.
- Host launch succeeded:
  - PID `3000547`.
  - Log `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_20260622_0141.log`.
- Process check confirmed the new queue and current S1E/S1F queues are alive.

## Result

Post-centerpolish capacity retry is staged but not running. No additional training job was launched during active S1E/S1F training.

# 2026-06-22 Stage1 continuation plateau validation

## Evidence

- Reporter result:
  - S1E epoch `77/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `74/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct history check found no late best update:
  - S1E latest epoch `77` P10 `26.15116848135894`, below its best P10 `28.218778646217203`.
  - S1F latest epoch `74` P10 `26.965409296863484`, below its best P10 `28.15363937953733`.
- Process check confirmed the active training and queued follow-up processes are alive.

## Result

No baseline promotion. Lane B remains active baseline; S1E/S1F continue toward target completion while queued follow-up branches remain staged.

# 2026-06-22 Stage1 continuation refresh validation

## Evidence

- Reporter result:
  - S1E epoch `80/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `77/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct history rows show no late best update:
  - S1E epoch `80` P10 `26.160153370983195`.
  - S1F epoch `77` P10 `26.10624487894886`.
- Process check confirmed S1E/S1F and both follow-up queues alive.
- Run-directory check confirmed center-polish has not launched yet.

## Result

No baseline promotion. Current training and staged queues are healthy.

# 2026-06-22 Stage1 later continuation validation

## Evidence

- Reporter result:
  - S1E epoch `83/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `79/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct history rows remain below best:
  - S1E epoch `83` P10 `26.089398564032788`.
  - S1F epoch `80` P10 `26.918239575512004`.
- Process check confirmed S1E/S1F and both follow-up queues alive.

## Result

No baseline promotion. Lane B remains active baseline.

# 2026-06-22 Stage1 extended continuation validation

## Evidence

- Reporter result:
  - S1E epoch `93/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `90/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct history rows remain below best:
  - S1E epoch `93` P10 `26.446541462304456`.
  - S1F epoch `90` P10 `26.60938953903486`.
- Process check confirmed S1E/S1F and both follow-up queues alive.

## Result

No baseline promotion. Current branch remains below Lane B while training continues.

# 2026-06-22 Stage1 near-completion validation

## Evidence

- Reporter result:
  - S1E epoch `96/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `92/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct history rows remain below best:
  - S1E epoch `96` P10 `26.581312377497834`.
  - S1F epoch `92` P10 `26.19047675942475`.
- Process check confirmed S1E/S1F and both follow-up queues alive.

## Result

No baseline promotion. Lane B remains active baseline and the current runs continue.

# 2026-06-22 Stage1 queue safety validation

## Evidence

- Shell syntax passed for `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
- Dry-run command:

```bash
DRY_RUN=1 DETACH=0 RUN_TAG=queue_allcomplete_logic_dryrun_20260622 \
  CAPACITY_RUNNER=scripts/external/run_stage1_frame_search_center_preserve_polish.sh \
  CAPACITY_RUN_TAG=stage1_centerpolish_after_s1ef_20260622_0128 \
  MAX_WAIT_SEC=1 POLL_SEC=1 \
  bash scripts/external/run_stage1_frame_search_capacity_after_followups.sh
```

- Dry-run result:
  - Current state remained `min_ready_no_promotion`.
  - The queue did not launch center-polish while S1E/S1F were incomplete.
- Runtime process check after requeue:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.
  - New all-complete center-polish queue PID `3079899` alive.

## Result

Queue safety fix is active. Center-polish will wait for all watched follow-ups to complete before launching.

# 2026-06-22 Stage1 active-run refresh validation

## Evidence

- Reporter result:
  - S1E epoch `107/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `102/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Baseline comparison:
  - Lane B best P10 `28.22439415050003`; S1E remains lower by `0.005615504282825867 pp`.
  - Lane B best center `17.28884926831947`; S1E center is worse by `0.14898580425190744 px`.
  - S1F remains lower than Lane B on P10, P5, and center.
- Queue validation:
  - No center-polish or downstream capacity directories exist, matching the all-complete queue rule.
  - Current follow-ups are incomplete, so next branch must not launch yet.
- Runtime validation:
  - Host `pgrep` sees active S1E/S1F train PIDs and both queue PIDs.
  - Direct `nvidia-smi --query-gpu` succeeded earlier in the turn and showed GPU1 activity; full `nvidia-smi` later failed inside the sandbox with driver-communication error, so GPU table evidence is treated as non-canonical for this refresh.

## Result

No promotion. Current Stage1 work is progressing toward the configured 120-epoch gate; the next experiment is intentionally blocked until both S1E/S1F complete or a promotable checkpoint appears.

# 2026-06-22 Stage1 GPT-5.5 audit validation

## Evidence

- Real sub-agent execution:
  - GPT-5.5 sub-agent `Curie the 2nd` completed read-only audit.
  - Sub-agent reported no file edits and no train/eval launch.
- Reporter result after audit:
  - S1E epoch `113/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `108/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, center `17.307134470849668`.
  - `promote_checkpoint=None`.
  - `continue_training=True`.
- Baseline comparison:
  - Lane B P10 remains higher at `28.22439415050003`.
  - S1E remains below by `0.005615504282825867 pp` and has worse center by `0.14898580425190744 px`.
  - S1F remains below by `0.07075477096269722 pp` and has worse center by `0.01828520253019761 px`.
- Queue validation:
  - No center-polish/capacity run directories exist yet.
  - This matches the queue rule requiring both S1E/S1F to complete target epochs before launching center-polish.

## Result

No baseline promotion. Current behavior is valid: S1E/S1F continue, and queued follow-up experiments wait for the all-complete gate.

# 2026-06-22 Stage1 center-polish launch validation

## Evidence

- S1E/S1F final reporter check:
  - Both follow-ups reached target epoch `120`.
  - Both remained non-promotable against Lane B.
  - `promote_checkpoint=None`.
- Queue behavior:
  - Center-polish launched only after both follow-ups completed.
  - Queue log shows `complete_no_promotion`, successful GPU free check, and launch of `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
- Runtime evidence:
  - Lane I PID `3209126` on GPU0.
  - Lane J PID `3209127` on GPU1.
  - Downstream watcher PID `3222876`.
  - `nvidia-smi --query-gpu` showed both GPUs with active memory use after launch.
- Early metric evidence:
  - Lane I epoch `3/80`: P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J epoch `3/80`: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.

## Result

Launch path is valid and aligned with the Stage1 baseline objective. No promotion is allowed yet because the active center-polish lanes have not reached the 50-epoch gate.

# 2026-06-22 Stage1 center-polish active refresh validation

## Evidence

- Real sub-agent execution:
  - GPT-5.5 sub-agent `Erdos the 2nd` completed read-only audit.
  - Sub-agent reported no file edits and no train/eval launch.
- Reporter result:
  - Lane I epoch `18/80`, min-ready `false`, complete `false`.
  - Lane I best P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J epoch `18/80`, min-ready `false`, complete `false`.
  - Lane J best P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Status `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
- Runtime validation:
  - `pgrep -af` sees Lane I PID `3209126`, Lane J PID `3209127`, downstream watcher PID `3222876`, and data-loader children.
  - `nvidia-smi --query-gpu` shows both GPUs allocated: GPU0 `720 MiB`, GPU1 `732 MiB`; GPU1 showed active utilization in this snapshot.

## Result

Training is active. Lane J is the current leading baseline candidate, but no promotion is valid until the 50-epoch gate is reached.

# 2026-06-22 Stage1 baseline manifest writer validation

## Evidence

- Added `scripts/external/write_stage1_frame_search_baseline_manifest.py`.
- Added `tests/test_stage1_frame_search_baseline_manifest.py`.
- `python3 -m py_compile scripts/external/write_stage1_frame_search_baseline_manifest.py scripts/external/report_stage1_frame_search_followup.py`: passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_stage1_frame_search_baseline_manifest.py`: `3 passed`.
- `.venv/bin/python scripts/external/write_stage1_frame_search_baseline_manifest.py --format summary`: wrote `docs/resources/current_stage1_frame_search_baseline_manifest.json`.
- `python3 -m json.tool docs/resources/current_stage1_frame_search_baseline_manifest.json`: passed.

## Result

The current Stage1 frame-search baseline state is now captured in a durable manifest. It records Lane B as active baseline while promotion is blocked by the 50-epoch gate, and records Lane J as the leading baseline candidate because it improves P10, P5, and center.

# 2026-06-22 Stage1 manifest-hook watcher validation

## Evidence

- Updated `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`:
  - Adds `BASELINE_MANIFEST_WRITER`.
  - Adds `BASELINE_MANIFEST_OUTPUT`.
  - Calls the manifest writer after every reporter poll.
- Static validation:
  - `bash -n scripts/external/run_stage1_frame_search_capacity_after_followups.sh scripts/external/write_stage1_frame_search_baseline_manifest.py`: passed.
  - `python3 -m py_compile scripts/external/write_stage1_frame_search_baseline_manifest.py`: passed.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_stage1_frame_search_baseline_manifest.py`: `3 passed`.
- Dry-run validation:
  - `DRY_RUN=1 DETACH=0 ... MAX_WAIT_SEC=1 ... bash scripts/external/run_stage1_frame_search_capacity_after_followups.sh` printed the manifest hook output.
  - Exit `20` was expected from the forced one-second timeout; the hook executed before timeout.
- Runtime validation:
  - Stopped old watcher PID `3222876`.
  - Launched host watcher PID `3300765`.
  - Log `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_manifest_hook_20260622_0236b.log`.
  - First poll observed Lane I `24/80`, Lane J `23/80`, status `wait_min_epochs`, then wrote `docs/resources/current_stage1_frame_search_baseline_manifest.json`.
  - Later direct reporter refresh observed Lane I `29/80`, Lane J `28/80`, status `wait_min_epochs`.
- Sub-agent validation:
  - GPT-5.5 sub-agent `Pauli the 2nd` completed read-only audit.
  - It confirmed Lane J is the leading center-safe candidate and no promotion is valid before epoch `50`.
  - It confirmed active watcher PID `3300765`.
  - It reported no file edits and no train/eval launch.
- A later GPT-5.5 sub-agent `Pasteur the 2nd` was spawned for a fresh audit but timed out and was closed before completion; no output from it was used for validation.

## Result

The active watcher now preserves baseline-selection state in the manifest. Promotion is still blocked until the 50-epoch gate is reached.

# 2026-06-22 Stage1 50-Epoch Promotion Validation

## Evidence

- Reporter command:
  - `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --baseline runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500 --min-epochs 50 --target-epochs 80 --followup ...centerpolish_nodistill... --followup ...centerpolish_selfdistill... --format summary`
- Reporter result:
  - Lane I no-distill reached at least `53/80`, `min_ready=True`, but `promotable_after_min_epoch=False` because center regressed.
  - Lane J weak self-distill reached at least `51/80`, `min_ready=True`, and `promotable_after_min_epoch=True`.
  - Reporter status `promote_followup`.
  - Promotion checkpoint `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Checkpoint validation:
  - `test -f runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`: passed.
- Manifest validation:
  - `.venv/bin/python scripts/external/write_stage1_frame_search_baseline_manifest.py ... --format summary`: wrote `state=promoted_followup`.
  - `python3 -m json.tool docs/resources/current_stage1_frame_search_baseline_manifest.json`: passed.
- GPU/NVML validation:
  - `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader`: passed after the gate, showing both RTX 5080 GPUs responsive.

## Result

Lane J weak self-distill center-polish is now the validated Stage1 frame-based Search baseline candidate after the minimum 50-epoch gate. Lane I remains non-promotable because its center metric is worse than Lane B.

# 2026-06-22 Stage1 final baseline and XR-64 prep refresh

## Evidence

- Final Stage1 reporter/manifest refresh:
  - Lane I no-distill completed `80/80`, best Search P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J weak self-distill completed `80/80`, best Search P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Lane J remains the promoted baseline because it improves Lane B on P10, P5, and center; Lane I remains non-promotable because center regresses.
- Runtime state:
  - Direct PID checks found no active Stage1 or XR-64 train/eval process.
  - `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader`: passed with both RTX 5080 GPUs idle at `18 MiB` used and `0%` utilization.
- XR-64 prep artifact generation:
  - Ran `XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr62a XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`.
  - Output `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json` exists with `5929` rows and keys `event_state`, `sample_id`, `search_state`, `track_state`.
  - Strict checker now reports `generated_incomplete`, `ready_to_train=false`, `can_run_lane=false`, `missing_eval_rows=7`, `missing_overrides=6`, and `leakage_risk=none`.
- Runner hardening:
  - Updated `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` log naming to include action, split, teacher, PID, and timestamp, preventing same-second parallel log collisions.
  - `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`: passed.
- Regression checks:
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_stage1_frame_search_baseline_manifest.py tests/test_emit_xr64_next_prep_command.py tests/test_xr64_resume_artifacts.py`: `12 passed`.
  - `git diff --check`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format summary`: next id is `XR64-EVAL-TRAIN-XR39`, but `execute_supported=false` and `execution_state=paused_by_user_directive`.

## Result

Stage1 frame-based Search baseline work is complete enough for downstream use. XR-64 is partially resumed but not trainable yet: only `train/xr62a` eval rows are complete, and the remaining seven eval-row files plus six override JSON files must be generated before XR-64A/B training.

# 2026-06-22 XR-64 prep completion and launch blocker

## Evidence

- Generated all XR-64 prep eval rows:
  - Train: `xr62a`, `xr39`, `xr56b`, `xr58a`, each with `5929` rows.
  - Val: `xr62a`, `xr39`, `xr56b`, `xr58a`, each with `844` rows.
- Built all XR-64 override files:
  - Train: `xr64a_conservative_train_overrides.json`, `xr64b_threshold_train_overrides.json`, `xr64c_minerror_train_overrides.json`, each with `5929` rows.
  - Val: `xr64a_conservative_val_overrides.json`, `xr64b_threshold_val_overrides.json`, `xr64c_minerror_val_overrides.json`, each with `844` rows.
- Strict readiness:
  - `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary`: passed with `resume_status=ready_to_train`, `ready_to_train=true`, `can_run_lane=true`, `missing_eval_rows=0`, `missing_overrides=0`, and `leakage_risk=none`.
- XR-64A/B launch attempt:
  - `bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0`: exited `1`.
  - `bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1`: exited `1`.
  - Both logs show `RuntimeError: CUDA device request could not be satisfied` with `torch.cuda.is_available=False` and `cuda_device_count=0`.
- CUDA/NVML instability evidence:
  - Repeated probes alternated between `torch.cuda.is_available=True, count=2` and `torch.cuda.is_available=False, count=0`.
  - `nvidia-smi` alternated between normal RTX 5080 output and return code `9` with driver communication failure.
  - A final five-probe loop produced five consecutive failures: `nvidia-smi` could not communicate with the driver, and torch reported `torch_cuda False count 0` on every probe.

## Result

XR-64 prep is complete and strict-ready. XR-64A/B training is not complete because the launch environment is currently unstable at the CUDA/NVML layer. The next executable action is to stabilize CUDA/NVML, then relaunch `run_xr64_teacher_target_construction.sh a cuda:0` and `b cuda:1`.

# 2026-06-22 XR-64A/B post-run evidence completion

## Evidence

- Host GPU relaunch:
  - Sandbox-local GPU probes failed because `/dev/nvidia*` was not visible inside the sandbox.
  - Host-level probes showed `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`, `nvidia-smi`, and torch CUDA were available.
  - Re-ran `bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0` and `bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1` with host GPU access.
- Training runs:
  - XR-64A run root: `runs/xr64_xr62a_conservative_teacher_target_c255000_lr3e_7_to0_0006_m160k_M384k_r4000kus_p0p5_20260622_050430`.
  - XR-64B run root: `runs/xr64_xr56b_threshold_teacher_target_c255000_lr5e_7_to0_0008_m160k_M384k_r4000kus_p0p5_20260622_050433`.
  - Both lanes reached `XR64_TRAIN_EXIT:0`.
  - Both lanes produced `best_track_p10.pt`, `best_track_p5.pt`, provenance artifacts, and test eval summaries.
- Center checkpoint completion:
  - The runner expected `best_metric_track_center_px.pt`, but trainer only saves P10/P5 plus `best.pt` because `best_metric_name=metric_track_p10_pct`.
  - History showed epoch `7` was the best center epoch for both lanes, and `last.pt` was epoch `7`, so `last.pt` was preserved as `best_metric_track_center_px.pt` for both lanes.
  - Re-ran center checkpoint test evals with `HBTXR_DISABLE_CUDNN=1` after a cuDNN sublibrary mismatch occurred when the variable was omitted.
  - Patched center eval summaries with `write_xr64_ablation_provenance.py --patch-eval-summary`.
- Candidate matrix:
  - `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary`: `ok=true`, `candidate_count=6`, `required_candidate_matrix_complete=true`, `software_promotion_allowed=true`.
  - Corrected promotion decision command: `decision_status=promoted`, `winning_lane=XR-64B`, `winning_checkpoint_kind=best_track_p5`, `center_promoted=true`, `p10_promoted=false`, `p5_promoted=false`, `axis_certified_candidate_count=6`, `override_cleared_candidate_count=6`.
- Test metrics:
  - XR-64A `best_metric_track_center_px`: center `16.46816692352295`, P10 `34.123725230353216`, P5 `12.031462955474854`, P1 `0.9251700946262904`.
  - XR-64A `best_track_p10`: center `16.464686357975005`, P10 `34.3554429258619`, P5 `12.044218056542533`, P1 `1.0153061594281878`.
  - XR-64A `best_track_p5`: center `16.466928049496243`, P10 `34.29379326275417`, P5 `11.986820098331997`, P1 `0.9251700946262904`.
  - XR-64B `best_metric_track_center_px`: center `16.469009229115077`, P10 `34.284439529691426`, P5 `11.909013972963606`, P1 `0.9761905057089669`.
  - XR-64B `best_track_p10`: center `16.474387879031045`, P10 `34.18324905123029`, P5 `12.037840502602714`, P1 `0.9047619342803955`.
  - XR-64B `best_track_p5`: center `16.46796860694885`, P10 `34.50850416592189`, P5 `11.986820098331997`, P1 `0.9251700946262904`.

## Result

XR-64A/B execution and six-candidate post-run evidence are complete. XR-64 improves the center gate to `16.464686357975005` with XR-64A `best_track_p10`, while the promotion helper selects XR-64B `best_track_p5` as the bounded-tradeoff winner. P10 and P5 gates are not promoted, so the next experiment should target P10/P5 recovery without losing the new center gain.

# 2026-06-22 XR-64C min-error diagnostic

## Evidence

- Fixed future checkpoint evidence generation:
  - Updated `src/hbtxr/training/trainer.py` so Stage2 default checkpoint specs include `metric_track_center_px -> best_metric_track_center_px.pt`.
  - Added `tests/test_trainer_checkpoint_specs.py`.
  - Focused checkpoint/postrun tests passed: `26 passed`.
- Ran XR-64C:
  - Command: `bash scripts/external/run_xr64_teacher_target_construction.sh c cuda:0`.
  - Run root: `runs/xr64_xr62a_minerror_teacher_target_c255000_lr3e_7_to0_0010_m160k_M384k_r4000kus_p0p5_20260622_053056`.
  - Training reached `XR64_TRAIN_EXIT:0`.
  - The run produced `best_metric_track_center_px.pt`, `best_track_p10.pt`, and `best_track_p5.pt` without manual checkpoint copying.
- XR-64C test metrics:
  - `best_metric_track_center_px`: center `16.468127271107264`, P10 `34.16836808749608`, P5 `12.031462955474854`, P1 `0.9251700946262904`.
  - `best_track_p10`: center `16.464661524977004`, P10 `34.3554429258619`, P5 `12.044218056542533`, P1 `1.0153061594281878`.
  - `best_track_p5`: center `16.466900491714476`, P10 `34.29379326275417`, P5 `11.986820098331997`, P1 `0.9251700946262904`.

## Result

XR-64C min-error diagnostic slightly improves the observed center best to `16.464661524977004`, but it still does not promote P10 or P5. The next useful experiment is no longer another min-error target replay; it should explicitly recover P10/P5 while preserving the new center level.

# 2026-06-22 XR-67 full-manifest low-similarity weighted recovery

## Evidence

- Validated runner syntax and dry-run before launch:
  - `bash -n scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh b cuda:1`
- Ran XR-67A/B on host GPU access:
  - XR-67A command: `bash scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh a cuda:0`
  - XR-67B command: `bash scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh b cuda:1`
- Training completed:
  - XR-67A reached `XR67_TRAIN_EXIT:0` and `XR67_DONE eval_count=6`.
  - XR-67B reached `XR67_TRAIN_EXIT:0` and `XR67_DONE eval_count=6`.
- Logs:
  - `runs/_logs/xr67_full_lowsim_weighted_a_gpu0_20260622_072844.log`
  - `runs/_logs/xr67_full_lowsim_weighted_b_gpu1_20260622_072853.log`
- Result report:
  - `docs/resources/xr67_fullmanifest_lowsim_weighted_recovery_results_2026_06_22.md`

## Result

XR-67 is executable and completed all required train/eval stages. It does not promote the overall baseline:

- Best full-test center: `16.48420093229839`, worse than center gate `16.464661524977004`.
- Best full-test P10: `34.645834132603234`, below P10 gate `35.02295998845781`.
- Best full-test P5: `12.191752079554966`, above P5 gate `12.133503770828247`.
- The P5-improving checkpoint has center `16.545599697317396`, so it is a P5-recovery candidate only, not a baseline promotion.

Post-documentation validation:

- `bash -n scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh`: passed.
- `python3 -m py_compile scripts/external/compare_xr66_targeted_buckets.py`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_loss_sample_weighting.py tests/test_xr65_runner_contract.py tests/test_trainer_checkpoint_specs.py`: `7 passed`.

Note: pytest emitted a cache warning because the parent-root `.pytest_cache` path is read-only in this sandbox. The tests themselves passed.

# 2026-06-22 XR-68 fixed-LR low-similarity weighted recovery

## Evidence

- Validated runner before corrected launch:
  - `bash -n scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh`: passed.
  - `DRY_RUN=1 bash scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh a cuda:0`: passed.
  - `DRY_RUN=1 bash scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh b cuda:1`: passed.
- Confirmed scheduler override in runner:
  - `training.scheduler.type="$SCHEDULER_TYPE"`
  - `training.scheduler.warmup_epochs=0`
- GPU launch and completion:
  - XR-68A log: `runs/_logs/xr68_full_lowsim_weighted_a_gpu0_20260622_090550.log`
  - XR-68B log: `runs/_logs/xr68_full_lowsim_weighted_b_gpu1_20260622_090552.log`
  - Both lanes reached `XR68_TRAIN_EXIT:0`.
  - Both lanes reached `XR68_DONE eval_count=6`.
- Result report:
  - `docs/resources/xr68_fixedlr_lowsim_weighted_recovery_results_2026_06_22.md`

## Result

XR-68 is executable and completed all required train/eval stages. It does not promote the active baseline:

- Best full-test center: `16.500933163506645`, worse than center gate `16.464661524977004`.
- Best full-test P10: `34.645834132603234`, below P10 gate `35.02295998845781`.
- Best full-test P5: `12.11224525996617`, below P5 gate `12.133503770828247`.

Post-documentation validation:

- `bash -n scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh`: passed.
- `git diff --check`: passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_loss_sample_weighting.py tests/test_xr65_runner_contract.py tests/test_trainer_checkpoint_specs.py`: `7 passed`.

Note: pytest emitted the same cache warning because the parent-root `.pytest_cache` path is read-only in this sandbox. The tests themselves passed.
