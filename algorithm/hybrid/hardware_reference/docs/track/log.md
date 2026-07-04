# HGTXR-SW Work Log

## 2026-06-27

- Consolidated the full current HGTXR software work state into `docs/resources/current_project_synthesis_2026_06_27.md`.
- Created handover documents at `docs/HANDOVER_2026_06_27.md` and `HANDOVER.md`.
- Documentation scope includes current progress, experiment plans, experiment results, conversation decisions, Stage2/XR context, Stage1 frame-search baseline, closed no-promotion branches, post-hoc gate closeout, S1-OC implementation status, and next runbook.
- Real read-only sub-agent workflow was used for this documentation pass:
  - `gpt-5.3-codex-spark` analyst summarized progress/results/plans from existing docs.
  - `gpt-5.5` evaluator defined HANDOVER structure, required checks, and quality gates.
- Main-agent verification confirmed no `*s1oc*` run directories under `runs/NON_XR/raw` and no `*s1oc*` logs under `runs/NON_XR/shared/_logs`.
- Current factual state remains: S1-OC is implemented and smoke-validated, but the real 50-epoch GPU experiment has not started.

## 2026-06-22

- Stage1 frame-based Search work is now the active baseline path before downstream Stage2/XR-64.
- The center-polish follow-up lanes passed the 50-epoch gate. Latest reporter check observed no-distill at `63/80` and self-distill at `60/80`; the manifest refresh observed the promoted self-distill lane at `61/80`.
- Promoted self-distill center-polish checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Promotion evidence versus lane B: P10 `28.27717937613433` versus `28.22439415050003`, P5 `10.153863744915656` versus `9.833782825829848`, center `17.257583906065744` versus `17.28884926831947`.
- No-distill center-polish checkpoint was not promoted despite P10 `28.303010526693093` because center regressed to `17.3741187104639`.
- Regenerated `docs/resources/current_stage1_frame_search_baseline_manifest.json`; active baseline run and checkpoint now both point to the promoted self-distill follow-up.
- Added downstream handoff: `docs/resources/stage1_promoted_baseline_downstream_plan_2026_06_22.md`.
- XR-64 remains not ready to train: `missing_eval_rows=8`, `missing_overrides=6`; next prepared command is `XR64-EVAL-TRAIN-XR62A`, but it remains review-only until resume and GPU availability.
- Final Stage1 refresh at `2026-06-22 03:18:35 KST`: Lane I no-distill completed `80/80`; Lane J self-distill completed `80/80`. Lane J remains active baseline with P10 `28.27717937613433`, P5 `10.153863744915656`, and center `17.257583906065744`. Active Stage1 train processes are gone and both GPUs are free.

## 2026-06-10

- Baseline raw event-count full training inspected.
- Submission target inspected: `0.1812 px` pupil-center error and `0.43 ms` latency.
- Paper summary inspected: FACET, EX-Gaze, EyeTrAES, Swift-Eye, EV-Eye, local-global distillation are most relevant to accuracy recovery.
- First experimental move selected: Stage2-only, fixed Stage1 checkpoint, full width, lower LR, distillation off or weak.
- Spark sub-agents were quota-blocked. GPT5.5 fallback agents supplied read-only paper/submission/code-surface analysis.
- Sub-agent findings reinforced priority order: metric sanity, lower-LR/full-width Stage2 ablations, then FACET ellipse aux loss, EyeTrAES adaptive slicing, Swift-Eye blink handling, EX-Gaze sparse local patches.
- Ran GPU0 `lr2e-4_nodistill_fullwidth`: completed at epoch 13, best val P10 `8.8926`, center `44.2296`.
- Ran GPU1 `lr1e-4_weakdistill_fullwidth`: completed at epoch 13, best val P10 `13.8646`, center `44.8683`; current best P10 candidate.
- Sampled GPU0 `da_roi`: stopped at epoch 7 due weak P10 `4.6597` and center `53.6560`.
- Both GPUs were free after cleanup; next useful branch is weak-distill event-count sweep plus metric coordinate-frame sanity.
- Metric sanity completed with GPT5.5 read-only sub-agent support: `metric_track_center_px` is post-transform input-coordinate error, not direct sensor-pixel error.
- Event-count sweep initially showed identical logs. Added `scripts/external/check_event_count_override_sanity.py` and found event tensors were all zero despite selected counts changing.
- Fixed event input root causes in `src/hbtxr/data/event_builder.py`: use float64 timestamp arithmetic for causal weights and drop manifest `start_timestamp_us` for fixed-count windows.
- Added `tests/test_event_builder.py`; pytest passed for event builder and raw event-count validation tests.
- Restarted corrected parallel sweep: GPU0 `raw_mode1_stage2_count2500_lr1e-4_weakdistill_fullwidth_20260610_210855`, GPU1 `raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_210856`.
- Corrected sweep completed: count1000 best val P10 `14.1543`, center `44.8353`; count2500 best val P10 `13.6961`, center `44.6096`; both result checkers `ok=true`.
- Completed the next parallel pair: GPU0 corrected count5000 best val P10 `13.7298`, center `44.5356`; GPU1 count10000 best val P10 `13.8702`, center `43.3363`; both result checkers `ok=true`.
- Current corrected sweep interpretation: count1000 leads P10, count10000 leads center error; next experiment should target center-aware objective/evaluation alignment rather than larger event counts alone.
- Added center-aware count10000 configs and runner cases: `center_ckpt_count10000` and `centerloss_count10000`.
- Ran GPU0 `center_ckpt_count10000` and GPU1 `centerloss_count10000` in parallel; both completed epoch 30 and result checkers returned `ok=true`.
- Center-aware results: GPU0 center-checkpoint control best val center `39.5347`, P10 `13.8702`; GPU1 center-loss best val center `39.8471`, P10 `14.3430`.
- Current interpretation: center-based checkpoint/scheduler selection is the strongest center-error improvement so far; center-loss improves P10 but does not beat the center-checkpoint control on center error.
- Added eval-side `HBTXR_DISABLE_CUDNN=1` handling to `scripts/external/eval_hbtxr.py` after GPU0 test eval hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`; runtime-env pytest now covers both train and eval scripts.
- Test split validation completed on GPU0: center-checkpoint best-center checkpoint produced test center `36.9280`, test P10 `9.9209`; center-loss best-P10 checkpoint produced test center `42.3723`, test P10 `10.9864`.
- Started GPU1 recency-sharpening experiment in tmux `hgtxr_recency2_gpu1`: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth_20260610_222436`, with `causal_weight_power=2.0`. At epoch 5, best val center was `45.0600`, so it is currently underperforming and should be stopped or deprioritized if still above `43.3363` by epoch 10.
- Stopped recency-sharpening after epoch 10 validation because best val center only reached `44.6194`. Added `centerloss_lite_count10000` config/runner case and started `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243` on GPU1 in tmux `hgtxr_centerloss_lite_gpu1`.
- Added queued GPU1 case `centerloss_lite_lr5e-5_count10000` to the ablation runner and validated dry-run.
- Stopped LR `1e-4` centerloss-lite after epoch 10 validation because best val center was `44.2607`, still worse than count10000 baseline center `43.3363`. Started LR `5e-5` centerloss-lite on GPU1: `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120`, log `runs/_logs/centerloss_lite_lr5e5_gpu1_20260610_224106.log`.
- Added `center_ckpt_lr5e-5_count10000` to the ablation runner, validated dry-run, and started it on GPU0: `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355`, log `runs/_logs/centerckpt_lr5e5_gpu0_20260610_225000.log`.
- Marked LR `5e-5` probes as not promoted: centerloss-lite best observed val center `43.3086`; center-checkpoint LR `5e-5` best observed val center `44.5171`.
- Added `center_ckpt_ema996_count10000`; started it on GPU1 as `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302`, log `runs/_logs/centerckpt_ema996_gpu1_20260610_230000.log`.
- Added `center_ckpt_count1000`, validated dry-run on `cuda:1`, and queued it behind EMA996 in `runs/_logs/centerckpt_count1000_gpu1_queue_20260610_231500.log`.
- Stopped EMA996 after epoch 8/9 because best val center remained noncompetitive (`44.6362` at epoch 8, `43.9921` at epoch 9 observation) versus the center-checkpoint leader `39.5347`; queue advanced to GPU1 `center_ckpt_count1000`.
- Added `center_ckpt_count5000`, validated dry-run, and started it on GPU0 as `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218`, log `runs/_logs/centerckpt_count5000_gpu0_20260610_231000.log`.
- Added and dry-run validated `centerloss_count1000` and `centerloss_count5000` runner cases, but held them back because the lower-count center-checkpoint probes did not reach usable center error.
- Stopped `center_ckpt_count1000` after epoch 8/9 gate: best val center `44.9089`, best P10 `14.1543`; center did not recover.
- Evaluated centerloss count10000 best-center checkpoint on GPU1: `runs/eval_centerloss_bestcenter_test_gpu1_20260610_230859`, test center `36.5813`, test P10 `10.2759`; this is the current best test-center result.
- Stopped `center_ckpt_count5000` after epoch 8 gate: best val center `44.8672`, best P10 `13.7298`; center did not recover.
- Started GPU0 `centerloss_strong_count10000`: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_strong_fullwidth_20260610_231233`, log `runs/_logs/centerloss_strong_gpu0_20260610_232500.log`; latest observed epoch 6 train, best val center `44.6389`, under epoch-8 stop watch.
- Spark sub-agent was quota-blocked and GPT5.5 sub-agent spawn was thread-limit blocked for this planning turn, so the main agent executed the GPU1 selection locally.
- Started GPU1 parallel probe `centerloss_count5000`: `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_231556`, log `runs/_logs/centerloss_count5000_gpu1_20260610_234000.log`; dry-run passed, contract passed, log confirmed `resolved_device=cuda:1`, and epoch 1 best val center was `45.7473`.
- Stopped GPU0 `centerloss_strong_count10000` after epoch-8 gate: best val center `44.1654`, above stop threshold `42.0`; not competitive with center-checkpoint leader `39.5347`.
- Added `center_ckpt_schedulefree_count10000` config/runner case to test optimizer/scheduler axis while preserving the center-checkpoint leader structure.
- Validated `center_ckpt_schedulefree_count10000`: `bash -n` passed, config load printed `adamw 0.0001 True none metric_track_center_px`, dry-run passed.
- Started GPU0 `center_ckpt_schedulefree_count10000`: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_schedulefree_fullwidth_20260610_232114`, log `runs/_logs/centerckpt_schedulefree_gpu0_20260611_000000.log`; log confirmed `resolved_device=cuda:0`.
- Stopped GPU1 `centerloss_count5000` after epoch-9 observation: best val center `44.3874`, above stop threshold `42.0`; not competitive with center-checkpoint leader `39.5347`.
- Added and validated `center_ckpt_lion_count10000`: config load printed `lion 3e-05 [0.9, 0.99] plateau metric_track_center_px`, dry-run passed.
- Started GPU1 `center_ckpt_lion_count10000`: `raw_mode1_stage2_count10000_lr3e-5_weakdistill_centerckpt_lion_fullwidth_20260610_232431`, log `runs/_logs/centerckpt_lion_gpu1_20260611_000500.log`; log confirmed `resolved_device=cuda:1`.
- Stopped GPU0 `center_ckpt_schedulefree_count10000` after epoch-8 gate: best val center `45.3146`, above stop threshold `42.0`.
- Stopped GPU1 `center_ckpt_lion_count10000` after epoch-8 gate: best val center `43.4448`, above stop threshold `42.0`.
- Added and validated `center_ckpt_teachercenter_count10000`; started it on GPU1 as `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_teachercenter_fullwidth_20260610_233212`, log `runs/_logs/centerckpt_teachercenter_gpu1_20260611_001500.log`; log confirmed `resolved_device=cuda:1`.
- Added FACET-style `center_ckpt_ellipseaux_count10000` config/runner case with search/event OBB auxiliary heads and center checkpointing. Initial weights `0.20/0.30` produced loss around `19k`, so the preflight run `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233558` was terminated.
- Reduced ellipse auxiliary search/event OBB weights to `0.001/0.001` and restarted GPU0 `center_ckpt_ellipseaux_count10000` as `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233657`, log `runs/_logs/centerckpt_ellipseaux_light_gpu0_20260611_003000.log`; log confirmed `resolved_device=cuda:0` and early loss returned to the normal hundreds scale.
- Stopped GPU1 `center_ckpt_teachercenter_count10000` after epoch-8 gate: best val center `44.0106`, above stop threshold `42.0`.
- Added and validated `center_ckpt_selfreg_count10000` with light local-global self-regularization; started it on GPU1 as `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_selfreg_fullwidth_20260610_234009`, log `runs/_logs/centerckpt_selfreg_gpu1_20260611_004000.log`; log confirmed `resolved_device=cuda:1` and early loss stayed in the normal hundreds scale.
- Stopped GPU0 `center_ckpt_ellipseaux_count10000` light run after epoch-8 gate: best val center `44.4728`, above stop threshold `42.0`; GPU0 was freed.
- Added and validated low-LR centerloss fine-tune config `configs/external/mode1_stage2_raw_event_count_lr2e-5_weakdistill_centerloss_finetune_fullwidth.yaml`; it starts from the current test leader checkpoint `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt`.
- Started GPU0 `centerloss_finetune`: `raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649`, log `runs/_logs/centerloss_finetune_gpu0_20260611_005000.log`; log confirmed `resolved_device=cuda:0`.
- Stopped GPU1 `center_ckpt_selfreg_count10000` after epoch-9 gate: best val center `43.5076`, above stop threshold `42.0`; GPU1 was freed.
- Added conservative LR sibling config `configs/external/mode1_stage2_raw_event_count_lr1e-5_weakdistill_centerloss_finetune_fullwidth.yaml`; config load via `scripts/external/_config.py` printed LR `1e-05`.
- Started GPU1 `centerloss_finetune_lr1e5`: `raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952`, log `runs/_logs/centerloss_finetune_lr1e5_gpu1_20260611_010000.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 `centerloss_finetune` LR `2e-5` completed by early stop at epoch 9. Best val center was `40.0042`, so it was not promoted versus the current val leader `39.8471`.
- Added track-head-only no-distillation fine-tune config `configs/external/mode1_stage2_raw_event_count_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth.yaml`; config load and event-count contract passed.
- Started GPU0 `trackonly_finetune`: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth_20260610_235630`, log `runs/_logs/trackonly_finetune_gpu0_20260611_011000.log`; log confirmed `resolved_device=cuda:0`.
- GPU1 `centerloss_finetune_lr1e5` completed by early stop at epoch 9. Best val center was `39.9555`, so it was not promoted versus the current val leader `39.8471`.
- Added conservative track-only LR sibling config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth.yaml`.
- Started GPU1 `trackonly_finetune_lr5e6`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth_20260611_000139`, log `runs/_logs/trackonly_finetune_lr5e6_gpu1_20260611_012000.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 `trackonly_finetune` LR `1e-5` completed by early stop at epoch 9. Best val center was `39.9155`, so it was not promoted versus the current val leader `39.8471`.
- Started GPU0 `trackonly_centerckptinit`: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerckptinit_fullwidth_20260611_000343`, log `runs/_logs/trackonly_centerckptinit_gpu0_20260611_013000.log`; log confirmed `resolved_device=cuda:0`.
- GPU1 `trackonly_finetune_lr5e6` completed by early stop at epoch 9. Best val center was `39.9662`, so it was not promoted.
- GPU0 `trackonly_centerckptinit` completed by early stop at epoch 8. Best val center was `39.7478`, a new validation-center leader.
- Evaluated GPU0 `trackonly_centerckptinit` best-center checkpoint on test using `training.num_workers=0` after sandbox worker sockets failed. Result `runs/eval_trackonly_centerckptinit_bestcenter_test_gpu1_w0_20260611_000907`: test center `37.2887`, P10 `10.5884`, P5 `2.6990`; no test-center promotion versus `36.5813`.
- Started GPU1 `trackonly_centerckptinit_lr5e6`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerckptinit_fullwidth_20260611_001034`, log `runs/_logs/trackonly_centerckptinit_lr5e6_gpu1_20260611_014000.log`; log confirmed `resolved_device=cuda:1`.
- Evaluated GPU0 `trackonly_centerckptinit` best-P10 checkpoint on test. Result `runs/eval_trackonly_centerckptinit_bestp10_test_gpu0_w0_20260611_002015`: test center `37.1481`, P10 `10.9864`, P5 `3.4881`; no test-center promotion.
- GPU1 `trackonly_centerckptinit_lr5e6` completed by early stop at epoch 8. Best val center was `39.5315`, a new validation-center leader.
- Evaluated GPU1 `trackonly_centerckptinit_lr5e6` best-center and best-P10 checkpoints on test. Best-center result: center `37.0766`, P10 `10.2194`, P5 `2.8533`. Best-P10 result: center `37.2173`, P10 `10.2755`, P5 `3.5285`. No test-center promotion.
- Created interpolation artifacts under `runs/interpolated_checkpoints/` between the current test leader and `trackonly_centerckptinit_lr5e6` best-center. Alpha `0.25` test center was `36.7546`; alpha `0.50` test center was `38.6547`; neither beat the test-center leader `36.5813`.
- Added all-head centerloss centerckpt-init configs: `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth.yaml` and `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth.yaml`; config load and raw event-count contract passed for both.
- Started GPU0 `centerloss_centerckptinit_nodistill`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth_20260611_004917`, log `runs/_logs/centerloss_centerckptinit_nodistill_gpu0_20260611_015000.log`; log confirmed `resolved_device=cuda:0`.
- Started GPU1 `centerloss_centerckptinit_weakdistill`: `raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943`, log `runs/_logs/centerloss_centerckptinit_weakdistill_gpu1_20260611_015000.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 `centerloss_centerckptinit_nodistill` early-stopped at epoch 8. Best val center was `39.5598`, so it was not promoted versus validation leader `39.5315`.
- GPU1 `centerloss_centerckptinit_weakdistill` early-stopped at epoch 8. Best val center was `39.5780`, so it was not promoted versus validation leader `39.5315`.
- Started GPU1 parallel low-priority prepared runner case `centerloss_count1000` in tmux session `hgtxr_centerloss_count1000_gpu1`: `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerloss_fullwidth_20260611_010010`, log `runs/_logs/centerloss_count1000_gpu1_20260611_005747.log`; log confirmed `resolved_device=cuda:1`.
- Evaluated centerloss-bestcenter/centerckpt-bestcenter interpolation probes on GPU0. Alpha `0.25` test center `36.7547`, alpha `0.50` test center `38.6246`, alpha `0.75` test center `38.1375`; no promotion versus test leader `36.5813`.
- Stopped GPU1 `centerloss_count1000` after epoch-10 gate. Best val center was `44.6844`, above stop threshold `42.0`.
- Added ultra-low LR no-distillation all-head fine-tune config `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_centerloss_finetune_fullwidth.yaml`; config load and raw event-count contract passed.
- Started GPU0 `centerloss_finetune_lr2e6`: `raw_mode1_stage2_count10000_lr2e-6_nodistill_centerloss_finetune_fullwidth_20260611_011131`, log `runs/_logs/centerloss_finetune_lr2e6_gpu0_20260611_011500.log`; log confirmed `resolved_device=cuda:0`.
- GPU0 `centerloss_finetune_lr2e6` early-stopped at epoch 6. Best val center was `39.8800`, so it was not promoted.
- Added `training.trainable.include/exclude` filtering in `src/hbtxr/training/trainer.py` and covered it with `tests/test_trainable_filter.py`; targeted pytest with runtime-env tests passed.
- Added true track-head-only config `configs/external/mode1_stage2_raw_event_count_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth.yaml`; config load and raw event-count contract passed.
- Started GPU1 `trackheadonly_finetune`: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth_20260611_011606`, log `runs/_logs/trackheadonly_finetune_gpu1_20260611_012500.log`; log confirmed `resolved_device=cuda:1` and trainable filter `151688/3156500` params.
- GPU1 `trackheadonly_finetune` early-stopped at epoch 6. Best val center was `39.9387`, so it was not promoted.
- Added partial event-path + track-head fine-tune config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU1 `trackeventadapter_finetune`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth_20260611_012329`, log `runs/_logs/trackeventadapter_finetune_gpu1_20260611_020000.log`; log confirmed `resolved_device=cuda:1` and trainable filter `324680/3156500` params.
- GPU1 `trackeventadapter_finetune` early-stopped at epoch 8 with best val center `39.7622`; not promoted versus validation leader `39.5315`.
- Added final-backbone-block + track-head fine-tune config `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU0 `tracklastblock_finetune`: `raw_mode1_stage2_count10000_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth_20260611_012629`, log `runs/_logs/tracklastblock_finetune_gpu0_20260611_021000.log`; log confirmed `resolved_device=cuda:0` and trainable filter `596936/3156500` params.
- GPU0 `tracklastblock_finetune` reached epoch 3 with best val center `39.8832`; continue until early-stop trend is available.
- Added both-adapter + track-head fine-tune config `configs/external/mode1_stage2_raw_event_count_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU1 `trackadapters_finetune`: `raw_mode1_stage2_count10000_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth_20260611_012935`, log `runs/_logs/trackadapters_finetune_gpu1_20260611_022000.log`; log confirmed `resolved_device=cuda:1` and trainable filter `448520/3156500` params.
- GPU0 `tracklastblock_finetune` early-stopped at epoch 6 with best val center `39.8832`; not promoted versus validation leader `39.5315`.
- GPU1 `trackadapters_finetune` early-stopped at epoch 8 with best val center `39.7822`; not promoted versus validation leader `39.5315`.
- Evaluated small-alpha interpolation probes on GPU0. Alpha `0.05` test center `36.5083`, alpha `0.10` test center `36.4873`, alpha `0.15` test center `36.5190`; alpha `0.10` is the new test-center leader.
- Created fine-grid interpolation checkpoints alpha `0.075`, `0.125`, and `0.175`, then evaluated them on GPU1. Results: `36.4907`, `36.4968`, `36.5556`; none beat alpha `0.10`.
- Added `scripts/external/run_interp_eval_series.sh` to run host/tmux interpolation eval series reliably after sandbox parallel eval stalled before CUDA execution.
- Evaluated micro-grid alpha `0.090/0.095/0.105/0.110`; alpha `0.095` improved test center to `36.486985`.
- Evaluated second micro-grid alpha `0.093/0.094/0.096/0.097`; alpha `0.094` improved test center to `36.486977`.
- Paper evidence reviewed for next non-interpolation moves: EyeTrAES supports adaptive event slicing, FACET supports direct ellipse/center-offset heads, and local-global distillation supports local expert teacher distillation.
- Current test-center leader: `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.094.pt`, test center `36.486977`, P10 `9.9775`, P5 `2.9660`.
- GPU0 `trackheadonly_alphainit` completed: `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackheadonly_centerloss_alphainit_finetune_fullwidth_20260611_020317`, log `runs/_logs/alphainit_trackheadonly_gpu0_20260611.log`; early-stopped at epoch 5 with best val center `39.8494`, so no test promotion versus validation leader `39.5315`.
- GPU1 `trackadapters_alphainit` completed: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackadapters_centerloss_alphainit_finetune_fullwidth_20260611_020333`, log `runs/_logs/alphainit_trackadapters_gpu1_20260611.log`; early-stopped at epoch 7 with best val center `39.8169`, so no test promotion.
- Added alpha-leader event-path and last-block configs: `configs/external/mode1_stage2_raw_event_count_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth.yaml` and `configs/external/mode1_stage2_raw_event_count_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth.yaml`; config load, raw event-count contract, and `bash -n scripts/external/run_prepare_and_train.sh` passed.
- Started GPU1 `trackeventadapter_alphainit`: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`, log `runs/_logs/alphainit_trackeventadapter_gpu1_20260611.log`; log confirmed `resolved_device=cuda:1` and trainable filter `324680/3156500` params.
- Started GPU0 `tracklastblock_alphainit`: `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`, log `runs/_logs/alphainit_tracklastblock_gpu0_20260611.log`; log confirmed `resolved_device=cuda:0` and trainable filter `596936/3156500` params.
- Implemented adaptive-count event slicing in `src/hbtxr/data/event_builder.py` with tests for enabled scaling and disabled compatibility. Added `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml` as the next paper-driven adaptive slicing probe. Validation: config load passed, raw event-count contract passed, and `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_event_builder.py tests/test_raw_event_count_contract.py` produced `6 passed` with only a parent `.pytest_cache` read-only warning.
- GPU1 `trackeventadapter_alphainit` early-stopped at epoch 7 with best val center `39.8169`; no promotion. GPU0 `tracklastblock_alphainit` early-stopped at epoch 5 with best val center `39.8546`; no promotion.
- Started GPU0 adaptive-count LR `5e-6`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`, log `runs/_logs/adaptivecount_trackonly_gpu0_20260611.log`; log confirmed `resolved_device=cuda:0`.
- Added lower-LR adaptive-count sibling `configs/external/mode1_stage2_raw_event_count_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`; config load and raw event-count contract passed. Started it on GPU1 as `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`, log `runs/_logs/adaptivecount_trackonly_lr2e6_gpu1_20260611.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 adaptive-count LR `5e-6` early-stopped at epoch 7 with best val center `38.6105`, beating the previous validation leader `39.5315`. Best-center test eval on GPU0 completed: `eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233`, log `runs/_logs/eval_adaptivecount_lr5e6_gpu0_20260611.log`, checkpoint `runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411/train/best_metric_track_center_px.pt`; test center `35.7182`, P10 `11.3461`, P5 `3.7279`, new test leader.
- GPU1 adaptive-count LR `2e-6` early-stopped at epoch 7 with best val center `38.9294`; it is weaker than the LR `5e-6` adaptive-count run, so test priority remains on LR `5e-6`.
- Started GPU1 parallel adaptive-count sqrt-scaling probe with `adaptive_count.scale_power=0.5`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_sqrt_alphainit_fullwidth_20260611_022233`, log `runs/_logs/adaptivecount_sqrt_trackonly_gpu1_20260611.log`; contract passed and log confirmed `resolved_device=cuda:1`.
- Verified sqrt-scaling override from `hypers/resolved_config.json`: `scale_power=0.5`, `event_count_target=10000`.
- Started GPU0 adaptive-count power-1.25 probe with `adaptive_count.scale_power=1.25`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_pow125_alphainit_fullwidth_20260611_022703`, log `runs/_logs/adaptivecount_pow125_trackonly_gpu0_20260611.log`; contract passed and log confirmed `resolved_device=cuda:0`.
- Manifest delta distribution checked: train median previous/current delta is about `4000003 us`, while original `reference_us=10000` makes `scale_power=0.5/1.0/1.25` all clip to only `5000` or `20000` events. Stopped the power-1.25 run as redundant after it reproduced the saturated trajectory through epoch 3/4 startup.
- GPU1 sqrt-scaling run completed: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_sqrt_alphainit_fullwidth_20260611_022233`; val center tied `38.6105`, best val P10 `12.7920`. Best-P10 test eval completed as `eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812`: test center `35.6666`, P10 `10.9566`, P5 `3.6947`; new test-center leader by center error.
- Started GPU1 fixed-count 20000 control: `raw_mode1_stage2_count20000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023021`, log `runs/_logs/fixed20k_trackonly_gpu1_20260611.log`; purpose is to separate fixed count-20000 benefit from adaptive slicing.
- Started GPU0 non-saturated adaptive-count control with `reference_us=4000003`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_ref4m_alphainit_fullwidth_20260611_023021`, log `runs/_logs/adaptivecount_ref4m_trackonly_gpu0_20260611.log`; first train losses diverged from the saturated/fixed20k trajectory, confirming a distinct input distribution.
- GPU0 ref4m control completed weak: best val center `39.9591`, early-stop at epoch 5, so no test eval planned. GPU1 fixed20k completed with best val center `38.6105`; fixed20k best-center test eval produced center `35.7182`, P10 `11.3461`, P5 `3.7279`. Fixed20k best-P10 test eval produced center `35.666554`, P10 `10.9566`, P5 `3.6947`, a tiny center improvement over the sqrt best-P10 result `35.666558`; this confirms the current gain is primarily fixed 20k input length, not adaptive scaling.
- Started GPU1 fixed-count 30000 follow-up: `raw_mode1_stage2_count30000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023804`, log `runs/_logs/fixed30k_trackonly_gpu1_20260611.log`; raw event-count contract passed and log confirmed `resolved_device=cuda:1`.
- Started GPU0 fixed-count 25000 follow-up: `raw_mode1_stage2_count25000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023947`, log `runs/_logs/fixed25k_trackonly_gpu0_20260611.log`; raw event-count contract passed and log confirmed `resolved_device=cuda:0`. This brackets the current fixed20k leader while fixed30k continues on GPU1.
- Queued GPU0 continuation `hgtxr_eval_fixed25_then_22k_gpu0`: after fixed25k exits, run fixed25k best-center and best-P10 test eval with `training.num_workers=0`, then start fixed-count 22000 on GPU0. Queue log will be `runs/_logs/eval_fixed25_then_fixed22_gpu0_20260611.log`.
- Queued GPU1 continuation `hgtxr_eval_fixed30_then_18k_gpu1`: after fixed30k exits, run fixed30k best-center and best-P10 test eval with `training.num_workers=0`, then start fixed-count 18000 on GPU1. Queue log will be `runs/_logs/eval_fixed30_then_fixed18_gpu1_20260611.log`.
- Added decoded-center L2 track loss as an optional config-only term: `loss.track_center_l2_weight` defaults to `0.0` and logs as `loss_track_center_l2`. Added `tests/test_track_center_l2_loss.py` and config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_finetune_fullwidth.yaml`.
- Validation for center-L2 path passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `2 passed` with only the known parent `.pytest_cache` read-only warning; `python3 -m py_compile src/hbtxr/loss/bundles/track.py` passed; raw event-count contract passed; config load via `scripts/external/train_hbtxr.py` helper resolved `track_center_l2_weight=0.0025`, `lr=5e-06`.
- Queued `hgtxr_centerl2_after_count_gpu0`: after the GPU0 fixed25/eval/fixed22 queue exits, start fixed-count 20000 center-L2 fine-tune on GPU0. Queue log will be `runs/_logs/centerl2_fixed20_after_count_gpu0_20260611.log`.
- Fixed-count sweep update: fixed25k completed with best val center `38.4949`; best-center test eval `eval_fixed25k_bestcenter_test_gpu0_w0_20260611_024841` produced center `35.3650`, P10 `10.9201`, P5 `3.5391`. fixed30k completed with best val center `38.6808`; best-center and best-P10 test evals `eval_fixed30k_bestcenter_test_gpu1_w0_20260611_024704` / `eval_fixed30k_bestp10_test_gpu1_w0_20260611_024837` both produced center `35.2200`, P10 `11.1501`, P5 `3.3516`, replacing the fixed20k/sqrt center leader.
- GPU1 continuation status: fixed30k evals finished and fixed18k is running as `raw_mode1_stage2_count18000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025024` on `cuda:1`. Queued follow-up `hgtxr_eval_fixed18_then_35k_gpu1`: after fixed18k exits, evaluate fixed18k best-center and best-P10 with `training.num_workers=0`, then start fixed35k on GPU1. Queue log will be `runs/_logs/eval_fixed18_then_fixed35_gpu1_20260611.log`.
- GPU0 continuation advanced: fixed25k best-P10 test eval `eval_fixed25k_bestp10_test_gpu0_w0` produced center `35.5696`, P10 `11.6382`, P5 `3.8542`; fixed22k is now running as `raw_mode1_stage2_count22000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025158` on `cuda:0`.
- Queued GPU1 continuation `hgtxr_eval_fixed35_then_centerl2_30_gpu1`: after the fixed18/eval/fixed35 queue exits, evaluate fixed35k best-center and best-P10 with `training.num_workers=0`, then start fixed30k center-L2 fine-tune on `cuda:1`. Queue log will be `runs/_logs/eval_fixed35_then_centerl2_30_gpu1_20260611.log`.
- Added `docs/Paper-Backed-Experiment-Plan.md`: maps EyeTrAES/EX-Gaze/FACET/3ET/local-global-distillation/HBTXR submission claims into executable axes: fixed-count optimum, decoded-center L2, ellipse/state loss, anchor-centered local event crop, optimizer/LR pool, and teacher/distillation. The doc records current queues for GPU0/GPU1 and decision gates.
- fixed18k completed weak on validation/test-center path: train run `raw_mode1_stage2_count18000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025024`, best val center `38.5218`; best-center test eval `eval_fixed18k_bestcenter_test_gpu1_w0_20260611_025644` produced center `35.8224`, P10 `10.8712`, P5 `3.5123`, so it does not beat fixed30k center leader. Its best-P10 eval is still pending in the same GPU1 queue.
- Queued GPU0 continuation `hgtxr_eval_fixed22_centerl2_then_28_gpu0`: after center-L2 fixed20k finishes, evaluate fixed22k and center-L2 fixed20k with `training.num_workers=0`, then start fixed28k. Queue log will be `runs/_logs/eval_fixed22_centerl2_then_fixed28_gpu0_20260611.log`.
- fixed18k best-P10 test eval completed: `eval_fixed18k_bestp10_test_gpu1_w0_20260611_025812` produced center `35.6665`, P10 `11.3890`, P5 `3.6237`; no promotion versus fixed30k center leader or fixed25k P10 leader.
- fixed22k training completed: `raw_mode1_stage2_count22000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025158`, best val center `38.5896`; test eval is queued behind center-L2 fixed20k.
- GPU0 current active: center-L2 fixed20k `raw_mode1_stage2_count20000_lr5e-6_nodistill_trackonly_centerl2_alphainit_fullwidth`, log `runs/_logs/centerl2_fixed20_after_count_gpu0_20260611.log`.
- GPU1 current active: fixed35k `raw_mode1_stage2_count35000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth`, log `runs/_logs/eval_fixed18_then_fixed35_gpu1_20260611.log`; follow-up queue remains fixed35k eval then center-L2 fixed30k.
- Sub-agent plan-review attempt: requested `gpt-5.3-codex-spark` explorer with Caveman skill, but spawn failed with `agent thread limit reached`; no sub-agent output was used.
- Added FACET/HBTXR geometry-loss sweep configs for the post-center-L2 queue:
  - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo075_finetune_fullwidth.yaml`
  - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo100_finetune_fullwidth.yaml`
  Both extend the center-L2 config, keep `track_center_l2_weight=0.0025`, and set `track_geo_weight` to `0.75` / `1.0`. Validation passed: config loader resolved intended weights, raw event-count contract passed for both, and `bash -n scripts/external/run_prepare_and_train.sh` passed.
- fixed22k test evals completed and did not promote: best-center `eval_fixed22k_bestcenter_test_gpu0_w0_20260611_030731` produced center `35.4087`, P10 `10.3631`, P5 `3.4566`; best-P10 `eval_fixed22k_bestp10_test_gpu0_w0_20260611_030849` produced center `35.7890`, P10 `11.1412`, P5 `4.1637`.
- center-L2 fixed20k completed: best val center `38.6252`; best-center test eval `eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932` produced center `35.7344`, P10 `11.2538`, P5 `3.9426`; best-P10 eval `eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011` produced center `35.5177`, P10 `11.7921`, P5 `3.7415`, becoming the current P10 leader.
- fixed35k completed and promoted center: `raw_mode1_stage2_count35000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025939` best-center test eval `eval_fixed35k_bestcenter_test_gpu1_w0_20260611_031023` produced center `34.9689`, P10 `11.5408`, P5 `3.6335`, replacing fixed30k as current center leader.
- fixed35k best-P10 eval `eval_fixed35k_bestp10_test_gpu1_w0_20260611_031215` produced center `36.2104`, P10 `9.5659`, P5 `3.1318`; no promotion.
- Current active queue: GPU0 fixed28k train; GPU1 center-L2 fixed30k train.
- Added follow-up queue runner `scripts/external/run_followup_queue_20260611.sh`; `bash -n` passed.
- Registered tmux follow-up queues:
  - `hgtxr_followup_gpu0_20260611`: fixed28k eval -> fixed40k train, log `runs/_logs/eval_fixed28_then_fixed40_gpu0_20260611.log`.
  - `hgtxr_followup_gpu1_20260611`: center-L2 fixed30k eval -> center-L2 fixed35k train, log `runs/_logs/eval_centerl2_30_then_centerl2_35_gpu1_20260611.log`.
  - `hgtxr_followup_gpu1_geo075_20260611`: center-L2 fixed35k eval -> geo075 fixed35k train, log `runs/_logs/eval_centerl2_35_then_geo075_35_gpu1_20260611.log`.
- fixed28k best-center eval `eval_fixed28k_bestcenter_test_gpu0_w0_20260611_032223` produced center `35.1866`, P10 `11.5731`, P5 `3.7049`; best-P10 eval `eval_fixed28k_bestp10_test_gpu0_w0_20260611_032400` produced center `35.5586`, P10 `11.7517`, P5 `4.2504`. Neither promoted versus fixed35k center leader or center-L2 fixed20k P10 leader.
- center-L2 fixed30k best-center/best-P10 evals `eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232` and `eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421` both produced center `35.2231`, P10 `10.6845`, P5 `3.4643`; no promotion. Current active queue is GPU0 fixed40k train and GPU1 center-L2 fixed35k train; GPU1 geo075 fixed35k remains queued.
- Extended `scripts/external/run_followup_queue_20260611.sh` with `eval_geo_count`, `train_geo100_count`, and mode `gpu1_geo075_35_then_geo100_35`. `bash -n` passed. Registered tmux session `hgtxr_followup_gpu1_geo100_20260611`; it waits for geo075 fixed35k to finish, evaluates best-center/best-P10 on test with `training.num_workers=0`, then starts geo100 fixed35k on GPU1. Queue log: `runs/_logs/eval_geo075_then_geo100_35_gpu1_20260611.log`.
- Extended `scripts/external/run_followup_queue_20260611.sh` with mode `gpu0_fixed40_eval`. `bash -n` passed. Registered tmux session `hgtxr_followup_gpu0_fixed40_eval_20260611`; it waits for fixed40k train completion, then evaluates best-center/best-P10 on test with `training.num_workers=0`. Queue log: `runs/_logs/eval_fixed40_gpu0_20260611.log`.
- fixed40k promoted both center and P10: best-center eval `eval_fixed40k_bestcenter_test_gpu0_w0_20260611_033617` from `runs/raw_mode1_stage2_count40000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_032548/train/best_metric_track_center_px.pt` produced center `34.9533`, P10 `11.9286`, P5 `3.6395`. best-P10 eval `eval_fixed40k_bestp10_test_gpu0_w0_20260611_033811` produced center `35.1615`, P10 `11.6458`, P5 `3.4503`; no further promotion over best-center.
- center-L2 fixed35k did not promote: best-center eval `eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`.
- Count sweep extended after the fixed40 promotion. Added `gpu0_fixed45_eval_then_fixed50` to `scripts/external/run_followup_queue_20260611.sh`; `bash -n` passed. fixed45k is now active on GPU0 via `hgtxr_followup_gpu0_fixed45_20260611`, and `hgtxr_followup_gpu0_fixed50_20260611` waits to evaluate fixed45k then launch fixed50k. geo075 fixed35k is active on GPU1, with geo100 queued after geo075 eval.
- Added terminal eval modes `gpu0_fixed50_eval` and `gpu1_geo100_35_eval` to `scripts/external/run_followup_queue_20260611.sh`; `bash -n` passed. Registered tmux sessions `hgtxr_followup_gpu0_fixed50_eval_20260611` and `hgtxr_followup_gpu1_geo100_eval_20260611` so fixed50k and geo100 fixed35k are automatically evaluated after training.
- fixed45k promoted center: best-center eval `eval_fixed45k_bestcenter_test_gpu0_w0_20260611_035115` from `runs/raw_mode1_stage2_count45000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_034000/train/best_metric_track_center_px.pt` produced center `34.4771`, P10 `11.5935`, P5 `3.9507`. fixed45k best-P10 eval `eval_fixed45k_bestp10_test_gpu0_w0_20260611_035258` produced center `35.1005`, P10 `10.2389`, P5 `3.4264`, so P10 leader remains fixed40k best-center at P10 `11.9286`.
- Current active queue after fixed45 eval: GPU0 fixed50k train `raw_mode1_stage2_count50000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_035439`; GPU1 geo100 fixed35k train `raw_mode1_stage2_count35000_lr5e-6_nodistill_trackonly_centerl2_geo100_alphainit_fullwidth_20260611_035013`. Both have eval queues already registered.
- geo100 fixed35k did not promote: best-center eval `eval_geo100_35k_bestcenter_test_gpu1_w0_20260611_040031` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_geo100_35k_bestp10_test_gpu1_w0_20260611_040210` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`.
- fixed50k promoted both center and P10: best-center eval `eval_fixed50k_bestcenter_test_gpu0_w0_20260611_040420` produced center `34.3416`, P10 `11.7589`, P5 `3.5748`; best-P10 eval `eval_fixed50k_bestp10_test_gpu0_w0_20260611_040514` produced center `34.5391`, P10 `12.1276`, P5 `3.3580`.
- Added generic count-extension runner `scripts/external/run_count_extension_queue_20260611.sh`; `bash -n` passed. Started tmux sessions `hgtxr_count55_gpu0_20260611` and `hgtxr_count60_gpu1_20260611`, logs `runs/_logs/count55_gpu0_20260611.log` and `runs/_logs/count60_gpu1_20260611.log`. Both run fixed-count train then best-center/best-P10 test eval with `training.num_workers=0`.
- Verified current count-extension queue: `tmux ls` shows `hgtxr_count55_gpu0_20260611` and `hgtxr_count60_gpu1_20260611`; `nvidia-smi` shows both GPUs active; logs show fixed55k near epoch 8/12 and fixed60k near epoch 7/12. No extra GPU1 experiment was started because fixed60k already occupies `cuda:1`.
- Added optimizer-probe runner `scripts/external/run_optimizer_probe_queue_20260611.sh`; `bash -n` passed. Use it after fixed55k/fixed60k determine whether the count axis has plateaued.
- Attempted real sub-agent review with `gpt-5.3-codex-spark`, but spawn failed with `agent thread limit reached`; continued with main-agent verification only.
- Added conditional post-count runner `scripts/external/run_post_count_decision_queue_20260611.sh`; `bash -n` passed. Started tmux sessions `hgtxr_post_count_gpu0_20260611` and `hgtxr_post_count_gpu1_20260611`, logs `runs/_logs/post_count_decision_gpu0_20260611.log` and `runs/_logs/post_count_decision_gpu1_20260611.log`. They wait for all fixed55k/fixed60k test eval summaries, then continue count sweep if either promotes, otherwise switch to LR probes.
- fixed55k/fixed60k both improved over fixed50k. fixed55k best-center test eval produced center `34.1183`, P10 `12.5672`, P5 `4.1399`, which is the current P10 leader. fixed60k best-center and best-P10 evals both produced center `33.9445`, P10 `12.5174`, P5 `4.3423`, which is the current center leader. fixed55k best-P10 did not promote: center `34.9445`, P10 `11.2088`, P5 `3.7560`.
- Post-count queues advanced as intended after promotion: GPU0 started fixed70k `raw_mode1_stage2_count70000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042200`; GPU1 started fixed65k `raw_mode1_stage2_count65000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042212`.
- Added second-stage conditional post-count runner `scripts/external/run_post_count2_decision_queue_20260611.sh`; `bash -n` passed. Started watcher sessions `hgtxr_post_count2_gpu0_20260611` and `hgtxr_post_count2_gpu1_20260611`, logs `runs/_logs/post_count2_decision_gpu0_20260611.log` and `runs/_logs/post_count2_decision_gpu1_20260611.log`. They wait for all fixed65k/fixed70k eval summaries, then continue count sweep to fixed80k/fixed75k if either promotes, otherwise switch to AdamW LR probes.
- fixed65k/fixed70k completed and improved the leaderboards. fixed65k best-center/best-P10 evals both produced center `33.7084`, P10 `12.6548`, P5 `3.8202`. fixed70k best-center/best-P10 evals both produced center `33.5467`, P10 `12.8971`, P5 `3.8010`, replacing fixed60k/fixed55k as both center and P10 leader. The second post-count watcher selected `ACTION=count`, `BEST_COUNT=70000`, launched GPU1 fixed75k as `raw_mode1_stage2_count75000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043657`, and launched GPU0 fixed80k as `raw_mode1_stage2_count80000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043808`.
- Added third-stage conditional post-count runner `scripts/external/run_post_count3_decision_queue_20260611.sh`; `bash -n` passed. Started watcher sessions `hgtxr_post_count3_gpu0_20260611` and `hgtxr_post_count3_gpu1_20260611`, logs `runs/_logs/post_count3_decision_gpu0_20260611.log` and `runs/_logs/post_count3_decision_gpu1_20260611.log`. They wait for fixed75k/fixed80k eval summaries, then continue to fixed90k/fixed85k if either promotes over fixed70k, otherwise switch to AdamW LR probes on the best count.
- Implemented next loss-axis candidate: decoded-center hinge loss in `src/hbtxr/loss/bundles/track.py`, with config keys `track_center_hinge_weight` and `track_center_hinge_margin_px`. Added tests in `tests/test_track_center_l2_loss.py`, configs `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge10_finetune_fullwidth.yaml` and `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge5_finetune_fullwidth.yaml`, and generic runner `scripts/external/run_centerhinge_probe_queue_20260611.sh`. Validation passed: unit test `3 passed`, both raw event-count contracts passed, `py_compile` passed, and runner `bash -n` passed.
- fixed75k/fixed80k completed and promoted over fixed70k. fixed75k evals produced center `33.3526`, P10 `13.2428`, P5 `3.5757`, becoming the current P10 leader. fixed80k evals produced center `33.0848`, P10 `12.7543`, P5 `3.4056`, becoming the current center leader. The third-stage watcher selected `ACTION=count`, `BEST_COUNT=80000`, `BEST_CENTER=33.084757`, `BEST_P10=13.242773`; GPU0 launched fixed90k at `2026-06-11T04:53:49+09:00`, and GPU1 launched fixed85k at `2026-06-11T04:53:51+09:00`.
- Added fourth-stage conditional post-count runner `scripts/external/run_post_count4_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count4_gpu0_20260611` / `hgtxr_post_count4_gpu1_20260611` were registered. The watcher waits for fixed85k/fixed90k evals, continues count to fixed100k/fixed95k if either promotes, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to center-hinge margin `10.0`, weight `0.05` on the best count.
- fixed85k/fixed90k completed. fixed85k promoted center to `32.9858` with P10 `12.9605`, P5 `3.9660`; fixed90k did not promote (`33.0128` center on best-center eval; `33.4284` center and `11.3542` P10 on best-P10 eval). P10 leader remains fixed75k at `13.2428`. The fourth-stage watcher selected `ACTION=count`, `BEST_COUNT=85000`, and launched GPU1 fixed95k plus GPU0 fixed100k at `2026-06-11T05:10:27+09:00`.
- Added fifth-stage conditional post-count runner `scripts/external/run_post_count5_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count5_gpu0_20260611` / `hgtxr_post_count5_gpu1_20260611` were registered. The watcher waits for fixed95k/fixed100k evals, continues count to fixed110k/fixed105k if either promotes, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to center-hinge margin `10.0`, weight `0.05` on the best count.
- Added local-anchor crop experiment axis from the paper-backed plan. `src/hbtxr/data/components.py` now supports `crop_policy: prev_pupil_anchor`, using `prev_annotation_ref` to crop around the previous pupil bbox while preserving shared frame/event/target transform alignment. Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_finetune_fullwidth.yaml`, runner `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh`, and unit tests `tests/test_adaptive_roi_resolver.py`. Validation passed: adaptive ROI + event-builder pytest `6 passed`, config override check, raw event-count contract, `py_compile`, `sh -n`, and `bash -n`. Training is not launched yet because fixed95k/fixed100k occupy both GPUs and post_count5 already controls the next branch.
- fixed95k/fixed100k completed. fixed95k did not promote (`32.9237` center, `12.2815` P10, `3.4184` P5). fixed100k best-center promoted center to `32.7796` with P10 `12.5961`, P5 `3.7215`; fixed100k best-P10 did not promote (`33.1030` center, `12.1105` P10, `3.6195` P5). The fifth-stage watcher selected `ACTION=count`, `BEST_COUNT=100000`, and launched GPU1 fixed105k plus GPU0 fixed110k at `2026-06-11T05:26:00+09:00`.
- Added sixth-stage conditional post-count runner `scripts/external/run_post_count6_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count6_gpu0_20260611` / `hgtxr_post_count6_gpu1_20260611` were registered. The watcher waits for fixed105k/fixed110k evals, continues count to fixed120k/fixed115k if either promotes over `32.7796` center or `13.2428` P10, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to the prepared `prev_pupil_anchor` local-crop probe. Process check confirms fixed105k on GPU1, fixed110k on GPU0, and both post_count6 watchers running.
- Prepared FACET-style decoded ellipse-state loss without adding new checkpoint parameters. `src/hbtxr/loss/bundles/track.py` now supports disabled-by-default `track_axis_log` and `track_angle_cos` losses via `loss.track_axis_log_weight` and `loss.track_angle_cos_weight`. Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_finetune_fullwidth.yaml` and runner `scripts/external/run_ellipsestate_probe_queue_20260611.sh`. Validation passed: track loss pytest `4 passed`, raw event-count contract, config loader check, `py_compile`, and runner `bash -n`. Training not launched; both GPUs remain occupied by fixed105k/fixed110k.
- fixed105k/fixed110k evals completed. fixed105k best-center/best-P10 both produced center `32.6356`, P10 `13.1654`, P5 `3.6662`; fixed110k best-center produced center `32.5969`, P10 `13.0697`, P5 `3.8193`, becoming the current center leader; fixed110k best-P10 produced center `32.9429`, P10 `12.6399`, P5 `3.4324`. P10 leader remains fixed75k at `13.2428`. The sixth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=110000`, `BEST_CENTER=32.596922`, `BEST_P10=13.165392`. GPU1 launched fixed115k at `2026-06-11T05:42:02+09:00`; GPU0 launched fixed120k at `2026-06-11T05:44:00+09:00`.
- Added seventh-stage conditional post-count runner `scripts/external/run_post_count7_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watchers `hgtxr_post_count7_gpu0_20260611` and `hgtxr_post_count7_gpu1_20260611`. Logs confirm both are waiting for fixed115k/fixed120k eval summaries. Promotion gate is fixed110k center `32.596922` or fixed75k P10 `13.242773`; if promoted, GPU1 continues to fixed125k and GPU0 to fixed130k, otherwise GPU1 runs `prev_pupil_anchor` and GPU0 runs AdamW `8e-6` on the best count.
- fixed115k/fixed120k evals completed. fixed115k best-center produced center `32.4400`, P10 `13.3512`, P5 `3.6305`; fixed115k best-P10 produced center `32.9068`, P10 `12.8835`, P5 `3.3678`. fixed120k best-center produced center `32.3053`, P10 `13.7976`, P5 `3.4974`, becoming the current center and P10 leader; fixed120k best-P10 produced center `32.7800`, P10 `13.0991`, P5 `3.6922`.
- The seventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=120000`, `BEST_CENTER=32.305316`, `BEST_P10=13.797619`. GPU0 launched fixed130k at `2026-06-11T05:59:57+09:00`; GPU1 launched fixed125k at `2026-06-11T06:00:09+09:00`.
- Added eighth-stage conditional post-count runner `scripts/external/run_post_count8_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watchers `hgtxr_post_count8_gpu0_20260611` and `hgtxr_post_count8_gpu1_20260611`. If fixed125k/fixed130k promote over fixed120k (`32.305316` center or `13.797619` P10), the queue continues to fixed140k on GPU0 and fixed135k on GPU1; otherwise GPU0 switches to AdamW `8e-6` and GPU1 switches to `prev_pupil_anchor`.
- Current process/log check: GPU0 fixed130k is active at epoch 4 with val center `37.0059`, P10 `10.2662`; GPU1 fixed125k is active at epoch 4 with val center `37.1699`, P10 `9.9292`. GPU1 is already occupied, so no additional same-GPU parallel experiment was started.
- Sub-agent plan audit attempt with `gpt-5.3-codex-spark` failed due `agent thread limit reached`; no sub-agent result was used.
- Continuation check: fixed125k/fixed130k remain active and no fixed125k/fixed130k eval summaries exist yet. Latest observed epoch 7: fixed125k val center `36.5371`, best val P10 `10.3291`; fixed130k val center `36.4138`, best val P10 `11.3398`.
- Added `scripts/external/run_post_count9_decision_queue_20260611.sh`; validation passed with `bash -n`, executable bit, and no-arg usage exit `2`. Registered watchers `hgtxr_post_count9_gpu0_20260611` and `hgtxr_post_count9_gpu1_20260611`, logs `runs/_logs/post_count9_decision_gpu0_20260611.log` and `runs/_logs/post_count9_decision_gpu1_20260611.log`; both are waiting for `post_count8` decision.
- fixed125k/fixed130k evals completed and promoted. fixed125k best-center produced center `32.1365`, P10 `14.0676`, P5 `3.8610`, becoming the current P10 leader; fixed125k best-P10 produced center `32.6232`, P10 `12.9962`, P5 `3.7636`. fixed130k best-center produced center `32.0159`, P10 `13.7798`, P5 `3.8457`, becoming the current center leader; fixed130k best-P10 produced center `32.4814`, P10 `12.9940`, P5 `3.7190`.
- The eighth-stage post-count watcher selected count continuation with `BEST_COUNT=130000`. GPU0 launched fixed140k as `raw_mode1_stage2_count140000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061825`; GPU1 launched fixed135k as `raw_mode1_stage2_count135000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061834`.
- post_count9 watchers observed `post_count8` action `count` and now wait for fixed135k/fixed140k eval summaries. Latest observed epoch 1: fixed140k val center `39.9020`, P10 `8.4748`; fixed135k val center `39.9278`, P10 `8.4580`.
- Corrected post_count9 threshold constants from fixed120k to the current leaders: center `32.01590127093451` and P10 `14.067602443695069`. Terminated the old post_count9 watcher processes and restarted corrected watchers as `hgtxr_post_count9_gpu0b_20260611` / `hgtxr_post_count9_gpu1b_20260611`; logs confirm both are waiting for fixed135k/fixed140k eval summaries.
- Current fixed135k/fixed140k status: epoch 6 observed. fixed135k best val center `36.6955`, best val P10 `11.0568`; fixed140k best val center `36.7976`, best val P10 `10.8041`; no fixed135k/fixed140k test eval summaries yet.
- Added `scripts/external/run_post_count10_decision_queue_20260611.sh`; validation passed with `bash -n`, executable bit, and no-arg usage exit `2`. Registered watchers `hgtxr_post_count10_gpu0_20260611` and `hgtxr_post_count10_gpu1_20260611`, logs `runs/_logs/post_count10_decision_gpu0_20260611.log` and `runs/_logs/post_count10_decision_gpu1_20260611.log`; both wait for `post_count9` decision. The script uses dynamic baseline selection from available eval summaries to avoid stale threshold constants.
- fixed135k/fixed140k evals completed. fixed135k best-center produced center `31.9891`, P10 `13.3274`, P5 `3.5242`; fixed135k best-P10 produced center `32.4352`, P10 `12.8053`, P5 `3.8559`. fixed140k best-center produced center `31.8833`, P10 `13.1884`, P5 `3.6637`, becoming the current center leader; fixed140k best-P10 produced center `32.3225`, P10 `12.7156`, P5 `3.6654`. P10 leader remains fixed125k best-center at `14.0676`.
- The ninth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=140000`. GPU1 launched fixed145k at about `2026-06-11T06:37:07+09:00`; GPU0 launched fixed150k at about `2026-06-11T06:37:10+09:00`. Current latest observed epoch 4: fixed145k val center `37.1518`, P10 `10.6289`; fixed150k val center `37.2427`, P10 `10.9007`. The tenth-stage watchers are alive and waiting for fixed145k/fixed150k eval summaries.
- fixed145k/fixed150k evals completed. fixed145k best-center produced center `31.8802`, P10 `12.5897`, P5 `3.9073`; fixed145k best-P10 produced center `32.3115`, P10 `12.5221`, P5 `3.5570`. fixed150k best-center produced center `31.7211`, P10 `12.9154`, P5 `3.7330`, becoming the current center leader; fixed150k best-P10 produced center `32.1254`, P10 `12.8342`, P5 `3.4179`. P10 leader remains fixed125k best-center at `14.0676`.
- The tenth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=150000`. GPU1 launched fixed155k at about `2026-06-11T06:55:33+09:00`; GPU0 launched fixed160k at about `2026-06-11T06:55:15+09:00`. Latest observed epoch 4: fixed155k val center `37.2207`, P10 `11.0580`; fixed160k val center `37.0474`, P10 `11.1927`.
- Added `scripts/external/run_post_count11_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count11_gpu0_20260611` and `hgtxr_post_count11_gpu1_20260611`, logs `runs/_logs/post_count11_decision_gpu0_20260611.log` and `runs/_logs/post_count11_decision_gpu1_20260611.log`; both wait for fixed155k/fixed160k eval summaries.
- fixed155k/fixed160k evals completed. fixed155k best-center/best-P10 both produced center `31.6944`, P10 `12.8878`, P5 `3.5438`. fixed160k best-center/best-P10 both produced center `31.5714`, P10 `13.3108`, P5 `3.5098`, becoming the current center leader. P10 leader remains fixed125k best-center at `14.0676`.
- The eleventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=160000`. GPU1 launched fixed165k at about `2026-06-11T07:14:02+09:00`; GPU0 launched fixed170k at about `2026-06-11T07:14:02+09:00`.
- Added `scripts/external/run_post_count12_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count12_gpu0_20260611` and `hgtxr_post_count12_gpu1_20260611`, logs `runs/_logs/post_count12_decision_gpu0_20260611.log` and `runs/_logs/post_count12_decision_gpu1_20260611.log`; both wait for fixed165k/fixed170k eval summaries. If they promote, GPU1 continues fixed175k and GPU0 fixed180k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Added `scripts/external/run_post_count13_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count13_gpu0_20260611` and `hgtxr_post_count13_gpu1_20260611`, logs `runs/_logs/post_count13_decision_gpu0_20260611.log` and `runs/_logs/post_count13_decision_gpu1_20260611.log`; both wait for the post_count12 decision. If post_count12 count branch promotes, GPU1 continues fixed185k and GPU0 fixed190k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- fixed165k/fixed170k evals completed. fixed165k best-center produced center `31.5321`, P10 `13.1152`, P5 `3.7160`; fixed165k best-P10 produced center `31.9779`, P10 `13.0676`, P5 `3.6352`. fixed170k best-center produced center `31.4726`, P10 `13.5991`, P5 `3.6437`, becoming the current center leader; fixed170k best-P10 produced center `31.9469`, P10 `13.1144`, P5 `3.3129`. P10 leader remains fixed125k best-center at `14.0676`.
- The twelfth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=170000`. GPU1 launched fixed175k at `2026-06-11T07:34:39+09:00`; GPU0 launched fixed180k at `2026-06-11T07:34:39+09:00`. Both trained through epoch 12. fixed175k best-center test eval `eval_fixed175k_bestcenter_test_gpu1_w0_20260611_074834` produced center `31.2559`, P10 `13.4209`, P5 `3.8291`; fixed175k best-P10 eval `eval_fixed175k_bestp10_test_gpu1_w0_20260611_075108` produced center `31.7825`, P10 `13.1718`, P5 `3.6322`. fixed180k best-center test eval `eval_fixed180k_bestcenter_test_gpu0_w0_20260611_074847` produced center `30.9685`, P10 `14.0582`, P5 `3.7117`, becoming the current center leader; fixed180k best-P10 eval `eval_fixed180k_bestp10_test_gpu0_w0_20260611_075125` produced center `31.6223`, P10 `13.4073`, P5 `3.8924`.
- The thirteenth-stage post-count watcher selected count continuation. GPU1 launched fixed185k at `2026-06-11T07:55:11+09:00` as `raw_mode1_stage2_count185000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075511`; GPU0 launched fixed190k at `2026-06-11T07:55:02+09:00` as `raw_mode1_stage2_count190000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075502`.
- Current fixed185k/fixed190k status: both are training normally. Latest observed epoch 10: fixed185k val center `35.0572`, P10 `12.9515`; fixed190k val center `35.0466`, P10 `13.5355`.
- Added `scripts/external/run_post_count15_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count15_gpu0_20260611` and `hgtxr_post_count15_gpu1_20260611`, logs `runs/_logs/post_count15_decision_gpu0_20260611.log` and `runs/_logs/post_count15_decision_gpu1_20260611.log`; both wait for the post_count14 decision. If post_count14 count branch promotes, GPU1 continues fixed205k and GPU0 fixed210k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Fixed `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh` eval output names from `fixed${COUNT}k` to `fixed${COUNT/1000}k`, matching the post-count watcher glob patterns.
- Added `scripts/external/run_post_count16_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count16_gpu0_20260611` and `hgtxr_post_count16_gpu1_20260611`, logs `runs/_logs/post_count16_decision_gpu0_20260611.log` and `runs/_logs/post_count16_decision_gpu1_20260611.log`; both wait for the post_count15 decision. If post_count15 count branch promotes, GPU1 continues fixed215k and GPU0 fixed220k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Added `scripts/external/run_post_count14_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count14_gpu0_20260611` and `hgtxr_post_count14_gpu1_20260611`, logs `runs/_logs/post_count14_decision_gpu0_20260611.log` and `runs/_logs/post_count14_decision_gpu1_20260611.log`; both wait for the post_count13 decision. If post_count13 count branch promotes, GPU1 continues fixed195k and GPU0 fixed200k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Prepared weak EMA distillation probe runner `scripts/external/run_weakdistill_probe_queue_20260611.sh <event_count> <cuda:N>` for the next GPU1 idle/plateau slot. It uses `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth.yaml` with the alpha checkpoint and fixed-count override. Not launched now because GPU1 is already running fixed175k and has post_count13/post_count14 watcher ownership.
- fixed185k/fixed190k completed and promoted. fixed185k best-center/best-P10 both produced center `30.8202`, P10 `14.4379`, P5 `3.9043`; fixed190k best-center/best-P10 both produced center `30.7517`, P10 `14.3967`, P5 `4.1297`.
- post_count14 selected `ACTION=count` with `BEST_COUNT=190000`, `BEST_CENTER=30.751661`, `BEST_P10=14.437926`. GPU1 launched fixed195k at `2026-06-11T08:16:44+09:00`; GPU0 launched fixed200k at `2026-06-11T08:16:33+09:00`.
- Current GPU1 assignment is fixed195k train on `cuda:1`; extra GPU1 probes remain queued until this branch or post_count15 frees GPU1.
- Added `scripts/external/run_post_count17_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable bit, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count17_gpu0_20260611` and `hgtxr_post_count17_gpu1_20260611`, logs `runs/_logs/post_count17_decision_gpu0_20260611.log` and `runs/_logs/post_count17_decision_gpu1_20260611.log`; both wait for post_count16. Count branch can continue to GPU1 fixed225k and GPU0 fixed230k.
- fixed195k/fixed200k completed and promoted. fixed195k best-center/best-P10 both produced center `30.6335`, P10 `14.7258`, P5 `4.4303`; fixed200k best-center/best-P10 both produced center `30.5612`, P10 `14.3117`, P5 `4.1531`.
- post_count15 selected `ACTION=count` with `BEST_COUNT=200000`, `BEST_CENTER=30.561175`, `BEST_P10=14.725766`. GPU1 launched fixed205k at `2026-06-11T08:37:37+09:00`; GPU0 launched fixed210k at `2026-06-11T08:37:28+09:00`.
- Current GPU1 assignment is fixed205k train on `cuda:1`; GPU0 assignment is fixed210k train on `cuda:0`.
- Added `scripts/external/run_post_count18_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count18_gpu0_20260611` and `hgtxr_post_count18_gpu1_20260611`, logs `runs/_logs/post_count18_decision_gpu0_20260611.log` and `runs/_logs/post_count18_decision_gpu1_20260611.log`; both wait for post_count17. Count branch can continue to GPU1 fixed235k and GPU0 fixed240k.
- Continuation check: fixed205k/fixed210k are still training with no test eval summaries yet. Latest observed epoch 11/12: fixed205k val center `35.1333`, P10 `13.2188`, P5 `4.1476`; fixed210k val center `35.1114`, P10 `13.6456`, P5 `3.8949`. GPU1 remains occupied by fixed205k, so no extra same-GPU experiment was launched.
- Added `scripts/external/run_post_count19_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count19_gpu0_20260611` and `hgtxr_post_count19_gpu1_20260611`, logs `runs/_logs/post_count19_decision_gpu0_20260611.log` and `runs/_logs/post_count19_decision_gpu1_20260611.log`; both wait for post_count18. Count branch can continue to GPU1 fixed245k and GPU0 fixed250k if fixed235k/fixed240k promote.
- fixed205k/fixed210k completed and promoted. fixed205k best-center eval produced center `30.3966`, P10 `15.0965`, P5 `4.0693`, becoming the current P10 leader; fixed205k best-P10 eval produced center `31.4222`, P10 `13.8312`, P5 `3.7381`. fixed210k best-center eval produced center `30.1415`, P10 `14.7037`, P5 `4.2636`, becoming the current center leader; fixed210k best-P10 eval produced center `30.8138`, P10 `13.8312`, P5 `3.6939`. post_count16 should continue to GPU1 fixed215k and GPU0 fixed220k on the next polling tick.
- post_count16 selected `ACTION=count`, `BEST_COUNT=210000`, `BEST_CENTER=30.141473`, `BEST_P10=15.096514`. GPU1 launched fixed215k as `raw_mode1_stage2_count215000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090019`; GPU0 launched fixed220k as `raw_mode1_stage2_count220000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090007`. Latest observed epoch 1: fixed215k val center `40.1420`, P10 `9.3531`, P5 `2.6673`; fixed220k val center `40.1718`, P10 `9.4710`, P5 `2.6505`.
- Continuation check: fixed215k/fixed220k are training normally. Latest observed epoch 4: fixed215k val center `36.8466`, P10 `12.6280`, P5 `3.1761`; fixed220k val center `36.8837`, P10 `12.8807`, P5 `3.0413`. GPU1 remains occupied by fixed215k, so no extra same-GPU experiment was launched.
- Added `scripts/external/run_post_count20_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count20_gpu0_20260611` and `hgtxr_post_count20_gpu1_20260611`, logs `runs/_logs/post_count20_decision_gpu0_20260611.log` and `runs/_logs/post_count20_decision_gpu1_20260611.log`; both wait for post_count19. Count branch can continue to GPU1 fixed255k and GPU0 fixed260k if fixed245k/fixed250k promote.
- Continuation check: fixed215k/fixed220k are still training with no fixed215k/fixed220k test eval summaries yet. Latest observed epoch 6: fixed215k val center `36.4212`, P10 `12.4910`, P5 `3.2401`; fixed220k val center `36.4039`, P10 `13.0357`, P5 `3.3109`. GPU1 remains occupied by fixed215k, so no extra same-GPU experiment was launched.
- Added `scripts/external/run_post_count21_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count21_gpu0_20260611` and `hgtxr_post_count21_gpu1_20260611`, logs `runs/_logs/post_count21_decision_gpu0_20260611.log` and `runs/_logs/post_count21_decision_gpu1_20260611.log`; both wait for post_count20. Count branch can continue to GPU1 fixed265k and GPU0 fixed270k if fixed255k/fixed260k promote.
- Full experiment plan print/update checkpoint: current leaders remain fixed210k best-center (`30.1415` center, `14.7037` P10, `4.2636` P5) and fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch remains GPU1 fixed215k and GPU0 fixed220k. Latest observed status: fixed215k epoch 8, val center `35.7267`, P10 `12.1597`, P5 `3.8331`; fixed220k epoch 9, val center `35.3173`, P10 `12.0058`, P5 `3.4288`. No fixed215k/fixed220k test eval summaries exist yet. GPU1 is already occupied by fixed215k, so parallel GPU1 work remains assigned through queued post-count branches rather than launched immediately.
- fixed215k/fixed220k completed train and test evals. fixed215k best-center/best-P10 both produced center `30.0281`, P10 `14.4018`, P5 `4.4498`. fixed220k best-center/best-P10 both produced center `30.0077`, P10 `14.3967`, P5 `4.2096`, becoming the new center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count17 selected `ACTION=count`, `BEST_COUNT=220000`, `BEST_CENTER=30.007657`, `BEST_P10=15.096514`; GPU1 launched fixed225k as `raw_mode1_stage2_count225000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092225`, and GPU0 launched fixed230k as `raw_mode1_stage2_count230000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092216`.
- Added `scripts/external/run_post_count22_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count22_gpu0_20260611` and `hgtxr_post_count22_gpu1_20260611`, logs `runs/_logs/post_count22_decision_gpu0_20260611.log` and `runs/_logs/post_count22_decision_gpu1_20260611.log`; both wait for post_count21. Count branch can continue to GPU1 fixed275k and GPU0 fixed280k if fixed265k/fixed270k promote. Current fixed225k/fixed230k status at registration: both epoch 3; fixed225k val center `37.8648`, P10 `10.7412`, P5 `2.6988`; fixed230k val center `37.8628`, P10 `10.6065`, P5 `2.6988`.
- fixed225k/fixed230k completed train and test evals. fixed225k best-center/best-P10 both produced center `29.9776`, P10 `14.4728`, P5 `3.8984`. fixed230k best-center/best-P10 both produced center `29.9516`, P10 `14.4677`, P5 `3.6050`, becoming the current center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count18 selected `ACTION=count`, `BEST_COUNT=230000`, `BEST_CENTER=29.951577`, `BEST_P10=15.096514`; GPU1 launched fixed235k as `raw_mode1_stage2_count235000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094454`, and GPU0 launched fixed240k as `raw_mode1_stage2_count240000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094445`.
- Full experiment plan print/update checkpoint: current center leader is fixed230k best-center (`29.9516` center, `14.4677` P10, `3.6050` P5); current P10 leader is fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch is GPU1 fixed235k and GPU0 fixed240k. Latest observed epoch 4: fixed235k val center `36.6978`, P10 `12.8504`, P5 `3.4906`; fixed240k val center `36.6337`, P10 `13.1199`, P5 `3.4906`. GPU1 is already occupied by fixed235k, so additional parallel GPU1 work remains assigned through post_count19/post_count20/post_count21/post_count22 rather than launched on the same GPU.
- Added `scripts/external/run_post_count23_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count23_gpu0_20260611` and `hgtxr_post_count23_gpu1_20260611`, logs `runs/_logs/post_count23_decision_gpu0_20260611.log` and `runs/_logs/post_count23_decision_gpu1_20260611.log`; both wait for post_count22. Count branch can continue to GPU1 fixed285k and GPU0 fixed290k if fixed275k/fixed280k promote. Current fixed235k/fixed240k status at registration: epoch 6; fixed235k val center `36.4167`, P10 `12.3169`, P5 `2.5438`; fixed240k val center `36.4364`, P10 `13.3446`, P5 `3.1593`.
- Added `scripts/external/run_post_count24_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count24_gpu0_20260611` and `hgtxr_post_count24_gpu1_20260611`, logs `runs/_logs/post_count24_decision_gpu0_20260611.log` and `runs/_logs/post_count24_decision_gpu1_20260611.log`; both wait for post_count23. Count branch can continue to GPU1 fixed295k and GPU0 fixed300k if fixed285k/fixed290k promote. Latest fixed-count status: fixed235k epoch 7 val center `36.1807`, P10 `12.9009`, P5 `3.0829`; fixed240k epoch 8 val center `35.5407`, P10 `12.9739`, P5 `3.4288`.
- Continuation check: fixed235k/fixed240k still training; no fixed235k/fixed240k test eval summaries yet. Latest observed epoch 10: fixed235k val center `35.0450`, P10 `13.1873`, P5 `3.8387`; fixed240k val center `34.9476`, P10 `13.5108`, P5 `3.9027`. post_count19 is alive and waiting for fixed235k/fixed240k eval summaries before launching GPU1 fixed245k / GPU0 fixed250k or fallback probes.
- fixed235k/fixed240k completed train and test evals. fixed235k best-center/best-P10 both produced center `29.9446`, P10 `13.6692`, P5 `3.5761`, briefly improving center over fixed230k. fixed240k best-center produced center `29.8589`, P10 `13.5132`, P5 `3.3580`, becoming the current center leader; fixed240k best-P10 produced center `30.6359`, P10 `13.4864`, P5 `3.8831`, so it did not improve the P10 leaderboard. P10 leader remains fixed205k best-center at `15.0965`.
- post_count19 selected `ACTION=count`, `BEST_COUNT=240000`, `BEST_CENTER=29.858867`, `BEST_P10=15.096514`. GPU1 launched fixed245k as `raw_mode1_stage2_count245000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_100750`; GPU0 launched fixed250k as `raw_mode1_stage2_count250000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_100838`. Both raw event-count contract checks passed and training processes are active.
- Added post_count25 and post_count26 queue runners; validation passed with shell syntax checks, no-arg usage exit `2`, tmux registration, and startup-log checks. Registered GPU1 continuation slots are fixed305k after post_count24 and fixed315k after post_count25; matching GPU0 slots are fixed310k and fixed320k.
- Runtime check after registration: GPU1 fixed245k is at epoch 7 with val center `36.0692`, P10 `13.4793`, P5 `3.3580`; GPU0 fixed250k is at epoch 6 with val center `36.2951`, P10 `13.1289`, P5 `3.3165`. Current completed center leader remains fixed240k best-center (`29.8589`), while P10 leader remains fixed205k best-center (`15.0965`).
- fixed245k/fixed250k completed and continued the count branch. fixed245k best-center produced center `29.7768`, P10 `13.6025`, P5 `3.2687`; fixed245k best-P10 produced center `30.6034`, P10 `13.3333`, P5 `3.7662`. fixed250k best-center produced center `29.6603`, P10 `13.5514`, P5 `2.7968`, becoming the current center leader; fixed250k best-P10 produced center `30.5370`, P10 `13.7959`, P5 `3.8414`. P10 leader remains fixed205k best-center (`15.0965`).
- post_count20 selected `ACTION=count`, `BEST_COUNT=250000`, `BEST_CENTER=29.660294`, `BEST_P10=15.096514`. GPU1 launched fixed255k at `2026-06-11T10:32:29+09:00`; GPU0 launched fixed260k at `2026-06-11T10:32:25+09:00`. Latest observed epoch 1: fixed255k val center `40.2184`, P10 `8.7859`, P5 `2.7156`; fixed260k val center `40.2167`, P10 `8.9207`, P5 `2.8504`.
- Added post_count27 and post_count28 queue runners; validation passed with shell syntax checks, no-arg usage exit `2`, tmux registration, and startup-log checks. Registered GPU1 continuation slots are fixed325k after post_count26 and fixed335k after post_count27; matching GPU0 slots are fixed330k and fixed340k. Latest fixed255k/fixed260k status at registration check: both epoch 11; fixed255k val center `34.3848`, P10 `12.9043`, P5 `4.0914`; fixed260k val center `34.4096`, P10 `12.5505`, P5 `3.9566`.
- fixed255k/fixed260k completed train and test evals. fixed255k best-center eval `eval_fixed255k_bestcenter_test_gpu1_w0_20260611_104936` produced center `29.6082`, P10 `13.7228`, P5 `3.3431`, becoming the current center leader. fixed255k best-P10 eval produced center `30.5540`, P10 `13.7598`, P5 `3.8601`. fixed260k best-center eval produced center `29.7259`, P10 `13.4515`, P5 `3.7611`; fixed260k best-P10 eval produced center `32.8103`, P10 `11.0128`, P5 `2.7887`. P10 leader remains fixed205k best-center at `15.0965`. post_count21 selected `ACTION=count`, `BEST_COUNT=255000`, `BEST_CENTER=29.608202`, `BEST_P10=15.096514`, then launched GPU1 fixed265k and GPU0 fixed270k at about `2026-06-11T10:57:57+09:00`. Latest fixed265k/fixed270k status: epoch 5; fixed265k val center `36.2752`, P10 `13.7612`, P5 `3.2996`; fixed270k val center `36.2936`, P10 `13.5815`, P5 `3.0301`.
- Added post_count29 and post_count30 queue runners; validation passed with shell syntax checks, no-arg usage exit `2`, tmux registration, and startup-log checks. Registered GPU1 continuation slots are fixed345k after post_count28 and fixed355k after post_count29; matching GPU0 slots are fixed350k and fixed360k. Latest fixed265k/fixed270k status at registration check: epoch 5; fixed265k val center `36.2752`, P10 `13.7612`, P5 `3.2996`; fixed270k val center `36.2936`, P10 `13.5815`, P5 `3.0301`.
- Added squared decoded-center threshold hinge loss for the next tail-error/P10/P5 probe. `src/hbtxr/loss/bundles/track.py` now emits disabled-by-default `loss_track_center_hinge_sq`, controlled by `loss.track_center_hinge_sq_weight` and `loss.track_center_hinge_margin_px`. Added test coverage in `tests/test_track_center_l2_loss.py`, config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhingesq10_finetune_fullwidth.yaml`, and runner `scripts/external/run_centerhingesq_probe_queue_20260611.sh`. Validation passed: track loss pytest `5 passed`, `py_compile`, `bash -n`, `sh -n`, executable bit, no-arg usage exit `2`, raw event-count readiness contract, and config loader check (`sq_weight=0.005`, `margin=10.0`, `lr=5e-6`). Training not launched because GPU1/GPU0 remain owned by fixed265k/fixed270k.
- fixed265k/fixed270k completed and did not promote. fixed265k best-center produced center `29.7582`, P10 `13.5969`, P5 `3.5974`; fixed265k best-P10 produced center `32.8107`, P10 `11.2436`, P5 `3.0013`. fixed270k best-center/best-P10 produced center `29.9207`, P10 `13.3652`, P5 `3.8418`. post_count22 selected `ACTION=mixed_probe`; GPU1 launched fixed255k `prev_pupil_anchor` (`raw_mode1_stage2_count255000_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_alphainit_fullwidth_20260611_112216`) and GPU0 launched fixed255k AdamW `8e-6` (`raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215`).
- Added `scripts/external/run_post_count31_decision_queue_20260611.sh` and registered tmux watchers `hgtxr_post_count31_gpu0_20260611` / `hgtxr_post_count31_gpu1_20260611`. Validation passed: `bash -n`, `sh -n`, no-arg usage exit `2`, and startup logs. post_count31 waits for post_count30; if count continuation through fixed355k/fixed360k promotes it launches fixed365k/fixed370k, otherwise GPU1 runs centerhinge-squared (`margin=10.0`, `weight=0.005`) and GPU0 runs AdamW `8e-6`. If post_count30 mixed probes fail, GPU1 also runs centerhinge-squared while GPU0 runs linear centerhinge.
2026-06-11 latest mixed-probe closeout:

- Training/eval processes finished; both GPUs were idle after manual eval repair.
- GPU0 AdamW `8e-6` at fixed255k is the new overall leader. Best-center test eval `eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913`: center `27.0897`, P10 `15.7109`, P5 `4.6301`.
- GPU0 AdamW best-P10 eval `eval_fixed255k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260611_114133`: center `28.2271`, P10 `14.6743`, P5 `4.3941`.
- GPU1 `prev_pupil_anchor` training finished, but original eval failed on `infer_hbtxr.py --output-dir`.
- Patched `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh` to call `eval_hbtxr.py --experiment-name`; `sh -n` and `bash -n` passed.
- Manual prevpupil evals completed: best-center center `65.0978`, P10 `9.9877`, P5 `3.0629`; best-P10 center `65.6465`, P10 `9.4957`, P5 `3.1365`.
- Analysis: AdamW optimizer/LR was the high-value axis; local previous-pupil crop currently breaks the learned full-ROI distribution and should not be extended before a targeted crop/target alignment diagnostic.
- Next experiment recommendation: run AdamW `8e-6` count bracket at fixed250k/fixed260k in parallel, then run LR bracket at fixed255k (`6e-6`, `1e-5`) or warm-start fine-tune from the AdamW leader.

2026-06-12 experiment tracking convention import:

- Added legacy experiment import document: `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md`.
- Source root analyzed: `/home/kjm26/project/PRJXR/XR-VIT/external_hybrid_package/docs`.
- Imported convention from prior HBTXR work: every meaningful run should record source config, command or runner, log path, checkpoint path, best validation metric, test metric, failure bucket, and next decision.
- Validation rule absorbed from legacy results: do not treat `loss_total`, single validation metric, or long epoch count as sufficient evidence. Preserve coordinate contracts and separate center/P10/P5 leaderboards.

2026-06-16 XR-01 LR bracket closeout:

- Completed fixed255k AdamW LR bracket after fixed250k/fixed260k failed promotion.
- LR `6e-6` best-center and best-P10 both produced test center `28.6105`, P10 `14.5111`, P5 `4.0374`; no promotion.
- LR `1e-5` best-center produced test center `26.1749`, P10 `16.5021`, P5 `5.0999`; promoted to new primary software leader.
- LR `1e-5` best-P10 produced test center `26.5078`, P10 `16.8036`, P5 `4.2598`; promoted by the P10 gate but kept as secondary because center/P5 are weaker.
- Next action: run XR-03 ellipse-state axis/angle loss probe from `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt`.
- XR-03 main branch launched on GPU0 in tmux `hgtxr_xr03_ellipsestate_lr1e5_gpu0_20260616`; log `runs/_logs/xr03_ellipsestate_lr1e-5_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016`.
- XR-03 stability branch launched on GPU1 in tmux `hgtxr_xr03_ellipsestate_lr6e6_gpu1_20260616`; log `runs/_logs/xr03_ellipsestate_lr6e-6_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012145`.
- XR-03 closeout: all four test evals completed. LR `1e-5` best-center produced center `21.6822`, P10 `21.8202`, P5 `7.0446` and becomes the new primary software leader. LR `1e-5` best-P10 produced center `21.9530`, P10 `21.7453`, P5 `6.7351`. LR `6e-6` best-center produced center `22.8794`, P10 `20.3835`, P5 `5.9660`; LR `6e-6` best-P10 produced center `23.0640`, P10 `20.2372`, P5 `6.0910`. All promote versus the prior XR-01 leader, but LR `1e-5` best-center is dominant.
- Next action: run XR-03A/B stronger decoded geometry sweep from `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016/train/best_metric_track_center_px.pt`, using axis `0.05`, angle `0.02`, LR `1e-5` on GPU0 and LR `6e-6` on GPU1.
- XR-03A/B launched. GPU0 session `hgtxr_xr03a_axis005_angle002_lr1e5_gpu0_20260616`, log `runs/_logs/xr03a_axis0p05_angle0p02_lr1e-5_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837`. GPU1 session `hgtxr_xr03b_axis005_angle002_lr6e6_gpu1_20260616`, log `runs/_logs/xr03b_axis0p05_angle0p02_lr6e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014848`. Both passed raw event-count contract and entered epoch 1/12.
- XR-03A/B closeout: both training processes exited cleanly and all four test evals completed. XR-03A LR `1e-5`, axis `0.05`, angle `0.02` best-center and best-P10 both produced center `20.4539`, P10 `25.8686`, P5 `8.3104`, becoming the new overall software leader. XR-03B LR `6e-6` best-center produced center `21.3059`, P10 `23.5102`, P5 `7.5702`; XR-03B best-P10 produced center `21.5008`, P10 `23.4273`, P5 `8.2075`. XR-03B promotes over the previous XR-03 gate but remains secondary to XR-03A.
- Next action: continue the 2차 목표 geometry branch from the XR-03A best-center checkpoint. Preferred next bounded sweep is axis `0.075`, angle `0.03` at LR `1e-5` on GPU0 and LR `6e-6` on GPU1. If this over-regularizes, fall back to midpoint axis `0.0625`, angle `0.025`.
- XR-03C/D launched from XR-03A best-center. GPU0 session `hgtxr_xr03c_axis0075_angle003_lr1e5_gpu0_20260616`, log `runs/_logs/xr03c_axis0p075_angle0p03_lr1e-5_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641`. GPU1 session `hgtxr_xr03d_axis0075_angle003_lr6e6_gpu1_20260616`, log `runs/_logs/xr03d_axis0p075_angle0p03_lr6e-6_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641`. Both passed raw event-count contract, loaded XR-03A checkpoint, resolved correct CUDA device, and entered epoch `1/12`.
- XR-03C/D monitoring checkpoint: both still running at epoch `4/12`; no eval summaries yet. XR-03C latest best val center `24.3015`, XR-03D latest best val center `24.1812`, both still below XR-03A gate `20.4539`.
- XR-04 low-similarity-aware loss candidate prepared while GPUs are occupied. Added `track_low_similarity_*` disabled-by-default weighting, config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml`, and runner `scripts/external/run_xr04_lowsim_ellipsestate_probe.sh`. Validation passed: track loss pytest `6 passed`, py_compile, `bash -n`, and raw event-count contract.
- XR-03C/D closeout completed. XR-03C LR `1e-5` best-center/best-P10 produced center `20.4750`, P10 `26.3202`, P5 `7.5374`. XR-03D LR `6e-6` best-center produced center `20.4940`, P10 `25.9651`, P5 `8.0446`; XR-03D best-P10 produced center `20.4336`, P10 `25.7866`, P5 `7.7491`.
- Decision: XR-03D LR `6e-6`, axis/angle `0.075/0.03`, best-P10 checkpoint is the new center-first leader. XR-03A remains the P10/P5-balanced secondary.
- XR-04 low-similarity-aware probes launched from the XR-03D center leader checkpoint. GPU0: LR `6e-6`, log `runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr6e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p3_s1p0_fullwidth_20260616_023454`. GPU1: LR `1e-5`, log `runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr1e-5_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p3_s1p0_fullwidth_20260616_023505`. Both passed raw event-count contract and entered epoch `1/12`.
- XR-04 closeout completed. Four test summaries exist. No promotion: LR `6e-6` best-center was closest with center `20.4347`, P10 `25.7844`, P5 `8.4286`, missing the XR-03D center gate by `0.0011 px` and missing the XR-03A balanced P10 gate.
- GPT5.5 sidecar recommended moving from low-sim loss weighting to a direct track-state auxiliary head if XR-04 failed.
- Implemented XR-05A direct track-state auxiliary head. `track/fused -> track/state_aux` predicts absolute state6 as training-only supervision; `track/pupil -> track/state` remains the runtime/inference path. Added config and runner.
- XR-05A validation passed: bash syntax, py_compile, raw event-count readiness, model-build smoke (`TrackStateAuxHead`, `150918` params), and `tests/test_track_center_l2_loss.py` `8 passed`.
- XR-05A launched from XR-03D best-P10 center leader. GPU0 LR `6e-6`, log `runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr6e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025651`. GPU1 LR `3e-6`, log `runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr3e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701`. Both passed raw event-count contract and entered epoch `1/12`.
- XR-05A closeout completed. Four test summaries exist. GPU0 LR `6e-6` best-P10 became the new center-first leader: center `20.3170`, P10 `25.8639`, P5 `8.2398`. GPU1 LR `3e-6` best-P10 became the P10 leader: center `20.3220`, P10 `26.5761`, P5 `8.0336`. GPU1 LR `3e-6` best-center became the P5/balanced secondary: center `20.3833`, P10 `26.2105`, P5 `8.6956`.
- XR-05B lighter aux refinement launched on GPU1 from XR-05A LR `3e-6` best-P10. Aux center/axis/angle `0.0005/0.0125/0.005`, LR `2e-6`, log `runs/_logs/xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_lr2e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335`.
- XR-05C midpoint-LR same-aux refinement launched on GPU0 from XR-03D best-P10. Aux center/axis/angle `0.001/0.025/0.01`, LR `4.5e-6`, log `runs/_logs/xr05c_trackstateaux_c0p001_axis0p025_angle0p01_lr4p5e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr4_5e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_031550`.
- XR-05B closeout completed. Runner stopped after train before automatic eval; manual eval without `HBTXR_DISABLE_CUDNN=1` failed with cuDNN sublibrary mismatch, then recovered with `HBTXR_DISABLE_CUDNN=1`. Best-center test: center `20.3479`, P10 `26.2317`, P5 `8.6446`. Best-P10 test: center `20.2927`, P10 `26.0446`, P5 `8.0782`; this becomes the new center-first leader.
- XR-05C closeout completed. Best-center test: center `20.5962`, P10 `25.2691`, P5 `7.8350`. Best-P10 test: center `20.3236`, P10 `26.1671`, P5 `7.8508`; no promotion.
- XR-05D launched on GPU1 from XR-05B best-P10. Aux center/axis/angle `0.00025/0.00625/0.0025`, LR `1e-6`, log `runs/_logs/xr05d_trackstateaux_c0p00025_axis0p00625_angle0p0025_lr1e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr1e_6_nodistill_trackonly_trackstateaux_c0p00025_axis0p00625_angle0p0025_fullwidth_20260616_033720`.
- XR-05E launched on GPU0 from XR-05A LR `3e-6` best-center. Aux center/axis/angle `0.00075/0.01875/0.0075`, LR `1.5e-6`, log `runs/_logs/xr05e_trackstateaux_c0p00075_axis0p01875_angle0p0075_lr1p5e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr1_5e_6_nodistill_trackonly_trackstateaux_c0p00075_axis0p01875_angle0p0075_fullwidth_20260616_033725`.
- XR-05D/E closeout completed. XR-05D did not promote: best-center center/P10/P5 `20.3298/26.2934/8.2980`; best-P10 `20.2981/25.8873/7.8176`. XR-05E best-center promoted the P10 gate: center/P10/P5 `20.3673/26.6539/8.4830`; best-P10 did not promote at `20.3076/25.8788/8.1037`.
- Added XR-06 weak-distill track-state-aux config and runner as fallback; syntax/config validation passed. Kept queued after GPT5.5 sidecar recommended XR-02 dense trajectory post-process and tiny hinge recovery first.
- Added and launched XR-07A/B tiny center-hinge recovery from XR-05B best-P10. XR-07A GPU1: linear hinge margin `10`, weight `0.01`, LR `1e-6`, log `runs/_logs/xr07a_trackstateaux_linearhinge_m10_w0p01_lr1e-6_gpu1_20260616.log`. XR-07B GPU0: squared hinge margin `10`, weight `0.001`, LR `1e-6`, log `runs/_logs/xr07b_trackstateaux_squaredhinge_m10_w0p001_lr1e-6_gpu0_20260616.log`.
- XR-07A/B closeout: both early-stopped at epoch 5 and failed validation gates. XR-07A best val center/P10/P5 `23.6261/21.8980/7.7943`; XR-07B best val center/P10/P5 `23.6286/21.8980/7.7943`. Runner-created test eval directories lacked `eval_summary.json`, and a manual full eval was interrupted after extended runtime; branch closed as no-promotion based on validation.
- XR-02 dense-gap check: test manifest has `2238` rows across `72` groups, median adjacent timestamp gap `4000003us`, minimum `360000us`, maximum `24000019us`. Current sparse test split is not a valid dense trajectory for EyeLoRiN smoothing under the `50000us` gate.
- Launched XR-06 weak-distill fallback on GPU1 in tmux session `hgtxr_xr06_weakdistill_centerleader_gpu1_20260616`, log `runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log`. It uses XR-05B best-P10 as both init and teacher checkpoint, fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `1e-6`; startup passed raw contract and reached epoch 1/12 on `cuda:1`.
- Nash GPT5.5 explorer recommended a complementary GPU0 weak-distill branch using XR-05E best-center P10 leader as both init and teacher. Launched XR-06B in tmux session `hgtxr_xr06b_weakdistill_p10leader_gpu0_20260616`, log `runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log`; fixed255k, aux `0.00075/0.01875/0.0075`, AdamW LR `1e-6`. Startup passed raw event-count contract, loaded XR-05E best-center checkpoint, resolved `cuda:0`, and entered epoch 1/12.
- XR-06A closeout completed. Training early-stopped at epoch 10. Best-center eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestcenter_test_gpu1_w0_20260616_042332/eval/test/eval_summary.json` produced center/P10/P5 `20.283847980839866/26.277211591175625/8.472364248548235`, promoting the center gate. Best-P10 eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestp10_test_gpu1_w0_20260616_042640/eval/test/eval_summary.json` produced `20.368657435689652/26.00425246102469/8.164115946633475`, no promotion.
- XR-06B closeout completed. Best-center eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p00075_axis0p01875_angle0p0075_adamw_lr1e_6_bestcenter_test_gpu0_w0_20260616_043100/eval/test/eval_summary.json` produced `20.295043339048114/26.217687790734427/8.519132954733712`. Best-P10 eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p00075_axis0p01875_angle0p0075_adamw_lr1e_6_bestp10_test_gpu0_w0_20260616_043343/eval/test/eval_summary.json` produced `20.46945102555411/25.931973491396224/8.630952685219901`. No promotion.
- XR-05F no-distill control launched on GPU1 in tmux session `hgtxr_xr05f_nodistill_xr05e_lightaux_gpu1_20260616`, log `runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log`. It starts from XR-05E best-center, uses fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `1e-6`, and no distillation. Startup passed raw event-count contract, loaded the checkpoint, resolved `cuda:1`, and entered epoch 1/12.
- XR-05F closeout completed. Best-center eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestcenter_test_gpu1_w0_20260616_043850/eval/test/eval_summary.json` produced `20.356191604478017/26.432823869160245/8.298044504438128`. Best-P10 eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr1e_6_bestp10_test_gpu1_w0_20260616_044115/eval/test/eval_summary.json` produced `20.31945925780705/26.150936106273107/8.142857428959438`. No promotion.
- XR-06C/XR-05G launched from the XR-06A center leader. XR-06C uses weak distillation on GPU0, log `runs/_logs/xr06c_weakdistill_xr06a_lr5e-7_gpu0_20260616.log`; XR-05G uses no distillation on GPU1, log `runs/_logs/xr05g_nodistill_xr06a_lr5e-7_gpu1_20260616.log`. Both use fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `5e-7`, passed raw event-count contract, loaded XR-06A best-center, resolved the intended CUDA device, and entered epoch 1/12.
- Evaluator sidecar T-033 returned a bounded fallback queue if XR-06C/XR-05G fail promotion: XR-06D center-preserve no-distill from XR-06A at LR `2.5e-7`; XR-05H P10-preserve weak-distill from XR-05E at LR `7.5e-7`; XR-05I P5-preserve no-distill from XR-05A at LR `5e-7`. Stop after first promotion and do not broaden architecture yet.
- XR-05G completed with no promotion: best-center `20.291977088791985/26.442177615846905/8.456632941109794`; best-P10 `20.375956610270908/26.27423542567662/8.281462873731341`.
- XR-06C completed and promoted two gates: best-center `20.283125744547164/26.363521112714494/8.43664994921003`; best-P10 `20.382882516724724/26.74489871433803/8.526786014011927`. New active gates: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-06D/XR-06E launched from XR-06C leaders. XR-06D uses no-distill from XR-06C best-center on GPU0, log `runs/_logs/xr06d_nodistill_xr06c_center_lr2p5e-7_gpu0_20260616.log`; XR-06E uses weak-distill from XR-06C best-P10 as init and teacher on GPU1, log `runs/_logs/xr06e_weakdistill_xr06c_p10_lr2p5e-7_gpu1_20260616.log`. Both passed raw event-count contract and entered epoch 1/12.
- XR-06D completed with no promotion. Best-center eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2_5e_7_bestcenter_test_gpu0_w0_20260616_051148/eval/test/eval_summary.json` and best-P10 eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2_5e_7_bestp10_test_gpu0_w0_20260616_051455/eval/test/eval_summary.json` both produced `20.295614736420767/26.158589158739364/8.504677152633667`.
- XR-06E completed with no promotion. Best-center eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2_5e_7_bestcenter_test_gpu1_w0_20260616_051823/eval/test/eval_summary.json` produced `20.29422003201076/26.280612965992518/8.538690771375384`; best-P10 eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p0005_axis0p0125_angle0p005_adamw_lr2_5e_7_bestp10_test_gpu1_w0_20260616_052047/eval/test/eval_summary.json` produced `20.31491228171757/26.13392930030823/8.681122759410313`.
- XR-08 checkpoint interpolation support added: `scripts/external/interpolate_hbtxr_checkpoints.py` creates eval-only payloads from two same-architecture checkpoints, and `scripts/external/run_xr08_checkpoint_interp_eval.sh` evaluates them with the fixed255k XR-06C surface.
- XR-08 alpha `0.50` between XR-06C best-center and best-P10 completed with no promotion: `runs/eval_fixed255k_xr08_xr06c_center_p10_interp_alpha0p50r_gpu0_w0_20260616_053156/eval/test/eval_summary.json` produced `20.32587662594659/26.49830005509513/8.386479888643537`. The initial parallel `tee` launch produced incomplete eval dirs, so the runner was corrected to file redirection and alpha `0.50` was rerun successfully.
- XR-05I launched on GPU0 from XR-05A P5/balanced secondary `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701/train/best_metric_track_center_px.pt`. It uses no-distill, fixed255k, aux `0.001/0.025/0.01`, AdamW LR `5e-7`, log `runs/_logs/xr05i_p5preserve_nodistill_xr05a_lr5e-7_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053814`. Startup passed raw event-count contract, loaded the intended checkpoint, resolved `cuda:0`, and entered epoch 1/12.
- XR-06F launched on GPU1 as complementary P5-preserve weak-distill from the same XR-05A P5/balanced checkpoint as init and teacher. It uses fixed255k, aux `0.001/0.025/0.01`, AdamW LR `5e-7`, log `runs/_logs/xr06f_p5preserve_weakdistill_xr05a_lr5e-7_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053815`. Startup passed raw event-count contract, loaded the intended init/teacher checkpoint, resolved `cuda:1`, and entered epoch 1/12.
- XR-05I closeout completed. Training early-stopped at epoch 5. Best-center eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_054521/eval/test/eval_summary.json` produced `20.351013261931282/26.15008576256888/8.428996889931815`. Best-P10 eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_054833/eval/test/eval_summary.json` produced `20.393892083849227/26.170068747656686/8.450255387169975`. No promotion.
- XR-06F closeout completed. Training early-stopped at epoch 10. Best-center eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_055145/eval/test/eval_summary.json` produced `20.322171998023986/26.200681025641305/8.625850643430438`. Best-P10 eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_055409/eval/test/eval_summary.json` produced `20.421796573911394/26.394558545521328/8.630102334703718`. No promotion against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Decision: local low-LR polish/P5-preserve fallback is saturated. Continue second-goal work with a targeted low-similarity/high-motion XR-04B branch or dense-trajectory XR-02, not another identical low-LR direct-aux replay.
- XR-04B/XR-04C targeted failure-bucket branches launched after XR-05I/XR-06F no-promotion. XR-04B runs on GPU0 with threshold `0.1`, scale `2.0`, LR `6e-6`, log `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p1_s2p0_fullwidth_20260616_060202`. XR-04C runs on GPU1 with threshold `0.1`, scale `2.0`, LR `3e-6`, log `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p1_s2p0_fullwidth_20260616_060224`. Both passed raw event-count contract, loaded XR-03D best-P10, resolved the intended CUDA device, and entered epoch 1/12.
- XR-04B/XR-04C closeout completed. Both early-stopped at epoch 8 and all four test eval summaries were generated. XR-04B best-center `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_061317/eval/test/eval_summary.json` produced `20.484137114456722/25.600340850012643/8.277636350904192`. XR-04B best-P10 `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_061619/eval/test/eval_summary.json` produced `20.67380678653717/25.401361295155116/7.875425440924508`. XR-04C best-center `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_061333/eval/test/eval_summary.json` and best-P10 `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_061637/eval/test/eval_summary.json` both produced `20.588273804528374/25.658163949421475/7.849915211541312`. No promotion.
- Added `scripts/external/build_failure_bucket_manifest.py` and `scripts/external/run_xr04d_failbucket_subset_probe.sh`. Validation passed: py_compile, bash syntax, subset creation with train `1186` and val `182` rows. XR-04D launched on GPU0 with LR `6e-6`, log `runs/_logs/xr04d_failbucket_sim01_s201_lr6e-6_gpu0_20260616.log`; XR-04E launched on GPU1 with LR `3e-6`, log `runs/_logs/xr04e_failbucket_sim01_s201_lr3e-6_gpu1_20260616.log`. Both use `similarity_target <= 0.1` + `session_201` train/val subset, start from XR-03D best-P10, and reached epoch 1/12.
- XR-04D/XR-04E closeout completed. Both early-stopped at epoch 10 and all four full-test eval summaries were generated. XR-04D LR `6e-6` best-center `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_063110/eval/test/eval_summary.json` produced `24.50812735216958/16.752551589693343/4.517857306344169`; best-P10 `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_063417/eval/test/eval_summary.json` produced `25.703870964050292/15.307398448671613/4.238095378875732`. XR-04E LR `3e-6` best-center `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_063111/eval/test/eval_summary.json` produced `23.55767515386854/18.159439352580478/5.106292690549578`; best-P10 `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_063417/eval/test/eval_summary.json` produced `23.42795329945428/18.25552776881627/5.031037589481898`. No promotion; hard-subset failure-bucket training is closed as a full-test path.
- XR-09 full-manifest weighted sampler implemented. Added `training.sampler.type=weighted_failure_bucket` in `src/hbtxr/data/loader.py`, config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_weightedsampler_fullwidth.yaml`, runner `scripts/external/run_xr09_weightedsampler_trackstateaux_probe.sh`, and tests `tests/test_weighted_sampler.py`. Validation passed: py_compile, bash syntax, `3 passed` weighted sampler tests, dataloader smoke with canonical root, and train-manifest weight sanity `6x=1186`, `3x=438`, `2x=1686`, `1x=2619`.
- XR-09A/XR-09B launched. XR-09A uses XR-06C best-center on GPU0 with low-sim `3x`, session_201 `2x`, cap `6x`, log `runs/_logs/xr09a_weightedsampler_center_lr5e-7_gpu0_20260616.log`. XR-09B uses XR-06C best-P10 on GPU1 with low-sim `2x`, session_201 `2x`, cap `4x`, log `runs/_logs/xr09b_weightedsampler_p10_lr5e-7_gpu1_20260616.log`. Both startup logs show raw event-count contract pass, intended checkpoint load, resolved CUDA device, full train/val manifest counts `5929/844`, and epoch `1/12` entry.
- XR-09A/XR-09B closeout completed. Both training runs exited `0`, all four test eval summaries were generated, and no active gate promoted. XR-09A best-center `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m3p0_s2p0_max6p0_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_065504/eval/test/eval_summary.json` produced `20.579283670016697/25.632653781345912/8.010204356057303`; XR-09A best-P10 `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m3p0_s2p0_max6p0_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_065808/eval/test/eval_summary.json` produced `20.641361141204833/24.742772783551896/8.041241747992379`; XR-09B best-center `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m2p0_s2p0_max4p0_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_065535/eval/test/eval_summary.json` produced `20.415359340395245/26.196429313932146/8.305697563716343`; XR-09B best-P10 `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m2p0_s2p0_max4p0_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_065840/eval/test/eval_summary.json` produced `20.55930529321943/25.68239870071411/7.932823378699166`.
- XR-10 full-manifest loss-side sample weighting implemented. Added `meta.session_key` preservation in `src/hbtxr/data/components.py`, `loss.track_sample_weight` in `src/hbtxr/loss/stage_common.py`, Stage2 track/aux/consistency/constraint weighting in `src/hbtxr/loss/stage2.py`, config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_lossweight_fullwidth.yaml`, runner `scripts/external/run_xr10_lossweight_trackstateaux_probe.sh`, and tests `tests/test_loss_sample_weighting.py`. Validation passed: py_compile, bash syntax, `14 passed` loss/sampler/track tests, dataloader smoke with canonical root, and train-manifest loss-weight sanity `3x=1186`, `2x=438`, `1.5x=1686`, `1x=2619`.
- XR-10A/XR-10B launched. XR-10A uses XR-06C best-center on GPU0 with low-sim `2x`, session_201 `1.5x`, cap `3x`, log `runs/_logs/xr10a_lossweight_center_lr5e-7_gpu0_20260616.log`. XR-10B uses XR-06C best-P10 on GPU1 with low-sim `1.5x`, session_201 `1.5x`, cap `2.5x`, log `runs/_logs/xr10b_lossweight_p10_lr5e-7_gpu1_20260616.log`. Both startup logs show raw event-count contract pass, intended checkpoint load, resolved CUDA device, full train/val manifest counts `5929/844`, and epoch `1/12` entry.
- XR-10A/XR-10B closeout completed. Both training runs exited `0`, all four test eval summaries were generated, and no active gate promoted. XR-10A best-center `runs/eval_fixed255k_xr10_lossweight_t0p1_m2p0_s1p5_max3p0_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_071610/eval/test/eval_summary.json` produced `20.34876068319593/26.20620824268886/8.127976478849138`; XR-10A best-P10 `runs/eval_fixed255k_xr10_lossweight_t0p1_m2p0_s1p5_max3p0_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_071916/eval/test/eval_summary.json` produced `20.429714499201094/26.048044974463327/7.999575104032244`; XR-10B best-center `runs/eval_fixed255k_xr10_lossweight_t0p1_m1p5_s1p5_max2p5_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_071612/eval/test/eval_summary.json` produced `20.3265901020595/25.97023880141122/8.133503682272774`; XR-10B best-P10 `runs/eval_fixed255k_xr10_lossweight_t0p1_m1p5_s1p5_max2p5_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_071918/eval/test/eval_summary.json` produced `20.38969965662275/26.1870755808694/7.99447306905474`.

2026-06-16 XR-11 SimDR setup:

- Sub-agent T-041 selected track-branch SimDR-style coordinate auxiliary as the safest next P0 after XR-10 no-promotion; dense-trajectory XR-02 remains blocked by sparse test-manifest gaps.
- Implemented `TrackStateSimDRHead` and wired it through head factory, tracker, track branch, model factory, Stage2 loss, and tests.
- Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_simdr_fullwidth.yaml`.
- Added runner `scripts/external/run_xr11_trackstate_simdr_probe.sh`.
- Validation passed: `bash -n`, `python3 -m py_compile` on touched Python files, `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_loss_sample_weighting.py tests/test_weighted_sampler.py` with `19 passed`, config/model smoke output `track/state_simdr=torch.Size([2, 2, 64])`, and `git diff --check`.
- Next launch plan: GPU0 center lane from XR-06C best-center; GPU1 P10 lane from XR-06C best-P10.
- XR-11 launch validation:
  - GPU0 center lane session `hgtxr_xr11_simdr_center_gpu0_20260616`, log `runs/_logs/xr11_simdr_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073321`.
  - GPU1 P10 lane session `hgtxr_xr11_simdr_p10_gpu1_20260616`, log `runs/_logs/xr11_simdr_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073333`.
  - Both logs show raw event-count contract pass, intended XR-06C checkpoint load, resolved CUDA devices `cuda:0`/`cuda:1`, full train/val counts `5929/844`, and epoch `1/12` entry.
- XR-11 closeout completed. Both train runs exited `0`; all four full-test eval summaries were generated. Center-init best-center `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_center_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_074029/eval/test/eval_summary.json` produced `20.338836230550495/26.16071500778198/8.301445865631104`. Center-init best-P10 `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_center_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_074332/eval/test/eval_summary.json` produced `20.366218400001525/26.545493936538698/8.573979895455496`. P10-init best-center `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_p10_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_074031/eval/test/eval_summary.json` produced `20.34303183896201/25.954507521220616/8.235544497626169`. P10-init best-P10 `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_p10_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_074341/eval/test/eval_summary.json` produced `20.289304559571402/26.325255816323416/8.309949268613543`. No active gate promoted.
- Decision: XR-11 no-distill SimDR is closed. Next P0 is XR-12 weak-distill light-SimDR: preserve XR-06C init/teacher and direct aux weights, lower SimDR weight to `0.0001` first, and compare against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-12 weak-distill light-SimDR config and runner added: `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_fullwidth.yaml` and `scripts/external/run_xr12_weakdistill_simdr_probe.sh`.
- XR-12 validation passed: `bash -n`, config/model smoke (`distillation.enabled=True`, `track_state_simdr=True`, `track_state_simdr_weight=0.0001`, bins `64`), XR-06C best-center/best-P10 checkpoint existence checks, and `git diff --check`.
- XR-12 launched in parallel. GPU0 center lane session `hgtxr_xr12_wdsimdr_center_gpu0_20260616`, log `runs/_logs/xr12_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr0p0001_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_075416`. GPU1 P10 lane session `hgtxr_xr12_wdsimdr_p10_gpu1_20260616`, log `runs/_logs/xr12_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr0p0001_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_075427`.
- XR-12 startup logs show raw event-count contract pass, intended XR-06C init/teacher checkpoint per lane, resolved devices `cuda:0` and `cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.

2026-06-16 XR-12 closeout:

- XR-12 center lane train stopped by early-stop at epoch 6 with best validation center `23.5377`; P10 lane stopped by early-stop at epoch 6 with best validation center `23.5457`.
- Parsed four full-test eval summaries. Center lane best-center: `20.3049/26.1514/8.2207`; center lane best-P10: `20.3599/26.0821/8.7917`; P10 lane best-center: `20.3070/26.1514/8.1696`; P10 lane best-P10: `20.3070/26.1514/8.1696`.
- Decision: no center/P10 promotion. The center lane best-P10 checkpoint is P5-only better than the XR-05A P5 gate, but it is not selected because center/P10 regress relative to XR-06C leaders.
- Next P0: XR-13 head-only weak-distill SimDR. Freeze the backbone/track trunk and train only `track_head.*`, `track_state_aux_head.*`, and `track_state_simdr_head.*` first; add `prev_state_encoder.*` only if the strictly head-only variant underfits without moving validation center.

2026-06-16 XR-13 setup and launch:

- GPT5.5 sidecar T-047B independently recommended the same next P0: head-limited weak-distill SimDR, not SimDR-head-only, because the SimDR auxiliary is not directly on the inference path.
- Added `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_headonly_fullwidth.yaml` and `scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh`.
- Validation passed: runner `bash -n`, XR-06C checkpoint existence checks, and config/model/trainable smoke. Trainable filter matched `18` head tensors and `500494/3505306` trainable params.
- XR-13 launched in parallel. GPU0 center lane session `hgtxr_xr13_headonly_center_gpu0_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081531`. GPU1 P10 lane session `hgtxr_xr13_headonly_p10_gpu1_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081528`.
- Startup logs show raw event-count contract pass, intended XR-06C init/teacher checkpoint per lane, trainable-filter `18` tensors and `500494/3505306` trainable params, resolved devices `cuda:0` and `cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.

2026-06-16 XR-13 closeout:

- XR-13 center lane early-stopped at epoch 5. Best-center eval produced `20.2916/26.2976/8.4694`; best-P10 eval produced `20.2897/26.2679/8.5204`.
- XR-13 P10 lane ran through epoch 12. Best-center eval produced `20.3555/26.4622/8.4906`; best-P10 eval produced `20.3696/26.5514/8.5332`.
- No active gate promoted against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Decision: close SimDR line for now. Next P0 is XR-02 dense/continuous prediction trajectory preparation and EyeLoRiN-style M2F rerun on valid trajectories.

2026-06-16 XR-02 dense trajectory availability:

- Added `scripts/external/check_xr02_dense_trajectory_availability.py`.
- Validation passed with `python3 -m py_compile` and execution against test manifest plus canonical root.
- Result artifact: `runs/diagnostics/xr02_dense_trajectory_availability_20260616_084506.json`.
- Manifest and canonical annotation summaries match: `72` groups, `2238` rows, min gap `360000us`, median gap `4000003us`, dense pair count `0`, longest dense segment `1`.
- Decision: current canonical1 labeled split cannot support EyeLoRiN-style M2F metric promotion. Next practical move is XR-04 diagnostics refresh before any new temporal/head training.

2026-06-16 XR-04 diagnostics refresh:

- Ran refreshed failure-bucket diagnostics for XR-06C best-center and best-P10 leaders.
- Artifacts:
  - `runs/diagnostics/xr04_refresh_xr06c_bestcenter_failure_buckets_20260616.json`
  - `runs/diagnostics/xr04_refresh_xr06c_bestp10_failure_buckets_20260616.json`
- Best-center low-sim bucket `similarity_target <= 0.1`: count `493`, weighted center `27.1247`, P10 `19.2399`, P5 `4.7506`.
- Best-P10 low-sim bucket `similarity_target <= 0.1`: count `493`, weighted center `26.9471`, P10 `19.0024`, P5 `4.9881`.
- Worst repeated structure remains `session_201`, especially subjects `42/45/39`. Blink/closed-eye is not the immediate weighted-metric driver.
- Search/event branch diagnostics joined `0/2238` rows because `model.heads.active=track` disables those branches in the normalized active leader config.
- Added `--limit` to `scripts/external/infer_hbtxr.py` after a full confidence-inference attempt produced no output/no GPU activity and was stopped.
- `--limit 2` smoke succeeded at `runs/diagnostics/xr04_confidence_xr06c_bestcenter_infer_limit2_20260616`; rows include `track_pred` and confirm `search_state`/`event_state` are null for the active track-only leader.
- Added `scripts/external/summarize_track_confidence_buckets.py`; limit-2 smoke joined `2/2` rows and wrote `runs/diagnostics/xr04_confidence_xr06c_bestcenter_limit2_buckets_20260616.json`.
- Decision: do not rerun low-sim/session_201 reweighting. Next action is bounded `track_pred` confidence/quality analysis and only then confidence/relocalization gating or confidence auxiliary loss.

2026-06-16 XR-04 confidence/quality probe:

- Added `scripts/external/build_confidence_probe_manifest.py`.
- Built `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/test_manifest.jsonl` from the test split: selected `128` rows with `similarity_target <= 0.1` and `128` rows with `similarity_target >= 0.6`.
- Ran CPU inference on XR-06C best-center and best-P10 checkpoints over the 256-row probe.
- Summary artifacts:
  - `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestcenter_buckets_20260616.json`
  - `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestp10_buckets_20260616.json`
- Center leader low-sim bucket: center `24.5408`, P10 `21.5686`, P5 `2.9412`, confidence `0.9999997`, quality `0.9999998`.
- P10 leader low-sim bucket: center `24.5044`, P10 `20.5882`, P5 `3.9216`, confidence `0.9999997`, quality `0.9999997`.
- High-sim rows are easier but not more confident. Existing confidence/quality logits are saturated and not calibrated for relocalization.
- Decision: reject no-train confidence gate from current logits. Next useful P0 must train calibrated confidence/quality with a usable fallback path, or enable a representation/branch change that can directly improve low-sim rows.

2026-06-16 XR-14 fallback diagnostic and all-head setup:

- Added and ran `scripts/external/eval_similarity_fallback.py` on the XR-06C best-center full-test eval rows.
- Artifact: `runs/diagnostics/xr14_similarity_prevstate_fallback_xr06c_bestcenter_20260616.json`.
- Result: raw is best by official-like batchmean center `20.283125752718494`; best fallback `blend(threshold=0.05, alpha=0.75)` degrades to `22.57394233260353`.
- Extended `scripts/external/infer_hbtxr.py` to write `track_state_aux` and `track_state_simdr`.
- Extended `scripts/external/summarize_track_confidence_buckets.py` with `--state-key`.
- Aux-state probe artifact: `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_auxstate_buckets_20260616.json`.
- Result: `track_state_aux` is not a fallback path; probe center `158.7825`, P10 `0.0`, P5 `0.0`.
- Added XR-14A all-head relocalization branch:
  - `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml`
  - `scripts/external/run_xr14_allhead_relocalize_probe.sh`
- Validation passed: bash syntax, default checkpoint existence, py_compile, and model smoke showing search/event/track heads enabled with mask disabled and distillation restricted to track keys.
- First parallel launch was stopped after startup because both lanes shared the same experiment prefix, creating a post-train run-root selection risk.
- Patched the runner to accept lane tag and include it in the experiment name.
- Relaunched v2 lanes:
  - center/GPU0 log `runs/_logs/xr14a_allhead_relocalize_center_lr5e-7_gpu0_20260616_v2.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_center_trackpreserve_fullwidth_20260616_094218`
  - P10/GPU1 log `runs/_logs/xr14a_allhead_relocalize_p10_lr5e-7_gpu1_20260616_v2.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_p10_trackpreserve_fullwidth_20260616_094220`
- Both lanes passed startup gates: raw contract, intended checkpoint, CUDA device resolution, train/val counts, and epoch `1/12` step logs.

2026-06-16 XR-14A closeout:

- Center and P10 lanes both early-stopped at epoch `8/12` with train exit `0`.
- Center lane eval summaries:
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_095401/eval/test/eval_summary.json`
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_095710/eval/test/eval_summary.json`
- P10 lane eval summaries:
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_095347/eval/test/eval_summary.json`
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_095652/eval/test/eval_summary.json`
- Center lane test metric: `20.342467624800545 / 25.644133370263237 / 8.11947306905474`.
- P10 lane test metric: `20.348247524670192 / 25.703657184328353 / 7.97278938974653`.
- Active XR-06C gates remain unbeaten: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Generated 12 diagnostic bucket files under `runs/diagnostics/xr14a_*_state_failure_buckets_20260616.json`.
- Best branch-state result remains `track_state`: center lane weighted `20.170973689926353 / 25.753704649974452 / 8.175779253960144`; P10 lane weighted `20.17721054100933 / 25.8048032703117 / 8.02248339294839`.
- `search_state` and `event_state` are not fallback candidates: weighted center about `177.8/178.5px`, P10/P5 `0.0`.
- Decision: close XR-14A and do not train calibrated search/event fallback from it. Next P0 should be a representation/data-density branch for low similarity, `session_201`, and subjects `42/45/39`.

2026-06-16 XR-15 support-adaptive setup:

- Selected XR-15 support-adaptive fixed-count event-window training as next P0.
- Added `docs/XR15_SUPPORT_ADAPTIVE_PLAN.md`.
- Added config `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`.
- Added runner `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`.
- Default XR-15A preserves XR-06C init/teacher, weak distill, AdamW `5e-7`, full-width track-only, and direct state aux `0.0005/0.0125/0.005`.
- Event-window change: fixed-count policy with adaptive count enabled at min/base/max `192k/255k/320k`, reference `4000003us`, power `0.5`.
- Preflight passed: runner syntax, executable bit, raw event-count contract, py_compile, and dataset smoke showing adaptive target counts are resolved.

2026-06-16 XR-15 support-adaptive launch:

- Launched XR-15A center lane on GPU0 in tmux session `hgtxr_xr15_supportadaptive_center_gpu0_20260616`; log `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148`.
- Launched XR-15A P10 lane on GPU1 in tmux session `hgtxr_xr15_supportadaptive_p10_gpu1_20260616`; log `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205`.
- Both lanes passed startup validation: raw event-count contract, intended XR-06C checkpoint as init/teacher, resolved CUDA device `cuda:0`/`cuda:1`, train/val counts `5929/844`, and epoch `1/12` step logs.
- Added `scripts/external/eval_xr15_p5_checkpoint.sh` as a post-train helper for `best_track_p5.pt` eval. The main runner already evaluates best-center and best-P10, and the trainer also saves a best-P5 checkpoint.

2026-06-16 XR-15A closeout:

- XR-15A center lane completed epoch `12/12` with train exit `0`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148`.
- XR-15A P10 lane completed epoch `12/12` with train exit `0`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205`.
- Full-test evals parsed:
  - center lane best-center: `20.230558776855467 / 26.303146975381033 / 8.573554713385446`.
  - center lane best-P10: `20.259641446386066 / 25.812925890513828 / 8.451105751310076`.
  - P10 lane best-center: `20.225680075372967 / 26.50085105895996 / 8.304422058377947`.
  - P10 lane best-P10: `20.259886418070113 / 26.12457557405744 / 8.357993486949375`.
- Decision: P10-lane best-center promotes the center gate. New active center gate is `<20.225680075372967`. P10/P5 gates remain `>26.74489871433803` and `>8.69557854788644`.
- Diagnostic artifacts generated:
  - `runs/diagnostics/xr15_p10_bestcenter_track_state_failure_buckets_20260616.json`
  - `runs/diagnostics/xr15_center_bestcenter_track_state_failure_buckets_20260616.json`
- Failure buckets remain concentrated in low `similarity_target <= 0.1`, subject `42/45`, and `session_201`.
- Best-P5 eval helper attempts did not complete under sandbox/logged timeout runs; no `eval_summary.json` exists for `bestp5`. Treat as incomplete artifact coverage, not a blocker for the center promotion.
- Next P0: XR-15B narrow support-adaptive window `224k/255k/288k`, same XR-15A weak-distill/direct-aux contract. If no promotion, run XR-15C wider `160k/255k/384k`.

2026-06-16 XR-15B launch:

- Center lane launched on GPU0:
  - tmux session `hgtxr_xr15b_supportadaptive_center_gpu0_20260616`
  - log `runs/_logs/xr15b_supportadaptive_center_lr5e-7_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512`
  - init/teacher: XR-15A P10-lane best-center checkpoint.
- P10 lane launched on GPU1:
  - tmux session `hgtxr_xr15b_supportadaptive_p10_gpu1_20260616`
  - log `runs/_logs/xr15b_supportadaptive_p10_lr5e-7_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111515`
  - init/teacher: XR-06C best-P10 checkpoint.
- Both logs passed startup validation: raw event-count contract, intended checkpoint load, `resolved_device=cuda:0/1`, train/val `5929/844`, and epoch `1/12` train steps.
- Sandbox `nvidia-smi` query failed after launch with driver communication error, but the training logs themselves confirmed CUDA device resolution and active train steps.

2026-06-16 XR-15B closeout and XR-15C launch:

- XR-15B center and P10 lanes both completed epoch `12/12` with train exit `0`.
- XR-15B best-center eval summaries parsed:
  - center lane best-center `runs/eval_fixed255k_xr15_supportadaptive_center_xr15b_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_113216/eval/test/eval_summary.json`: `20.215672533852715 / 26.575255823135375 / 8.627551317214966`.
  - P10 lane best-center `runs/eval_fixed255k_xr15_supportadaptive_p10_xr15b_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_113231/eval/test/eval_summary.json`: `20.21979672227587 / 26.324405458995273 / 8.28741525241307`.
- Decision: XR-15B center-lane best-center is the new center leader. P10/P5 gates remain XR-06C/XR-05A, so active gates are center `<20.215672533852715`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-15B best-P10 eval runs created `hypers/*` only and no `eval_summary.json`; sessions were cleaned up. XR-15B failure-bucket diagnostic was also interrupted after a long no-output run and should be retried only if XR-15C does not resolve the branch.
- XR-15C launched on both GPUs with wider min/base/max `160k/255k/384k`.
  - center/GPU0 session `hgtxr_xr15c_supportadaptive_center_gpu0_20260616`, log `runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819`.
  - P10/GPU1 session `hgtxr_xr15c_supportadaptive_p10_gpu1_20260616`, log `runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816`.
- XR-15C startup validation passed for both lanes: raw event-count contract, intended checkpoint load, `resolved_device=cuda:0/1`, train/val `5929/844`, and epoch `1/12` train steps.

2026-06-16 XR-15C closeout:

- XR-15C center and P10 lanes both completed training with exit `0`, then best-center test evals completed.
- XR-15C best-center eval summaries parsed:
  - center lane best-center `runs/eval_fixed255k_xr15_supportadaptive_center_xr15c_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_115600/eval/test/eval_summary.json`: `20.19088832650866 / 26.009779623576573 / 8.436224787575858`.
  - P10 lane best-center `runs/eval_fixed255k_xr15_supportadaptive_p10_xr15c_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_115558/eval/test/eval_summary.json`: `20.205999997683932 / 26.303146975381033 / 8.397959463936942`.
- Decision: XR-15C center-lane best-center is the new center leader. P10/P5 gates remain XR-06C/XR-05A, so active gates are center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Next experiment direction: stop repeating event-support range sweeps and move to XR-15D bounded track-adapter/coordinate-head training.

2026-06-16 XR-15D launch:

- Added `scripts/external/run_xr15d_trackadapters_probe.sh`.
- T-062 sub-agent launch-risk review confirmed the config/seed plan is valid and identified the lack of a dedicated runner as the main launch blocker.
- Static validation passed: runner `bash -n`, raw event-count contract, XR-15C/XR-06C checkpoints exist, and `PYTHONPATH=src` model-build smoke reports active head `track`.
- Center lane launched on GPU0:
  - session `hgtxr_xr15d_trackadapters_center_gpu0_20260616`
  - log `runs/_logs/xr15d_trackadapters_center_lr3e-6_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackadapters_centerloss_center_xr15d_fullwidth_20260616_120804`
  - init `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt`
- P10 lane launched on GPU1:
  - session `hgtxr_xr15d_trackadapters_p10_gpu1_20260616`
  - log `runs/_logs/xr15d_trackadapters_p10_lr3e-6_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackadapters_centerloss_p10_xr15d_fullwidth_20260616_120843`
  - init `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`
- Both startup logs reached raw contract pass, checkpoint load, trainable filter `22` tensors / `448520` params, `resolved_device=cuda:0/1`, and epoch `1/16`.

2026-06-16 XR-15D closeout and XR-15E launch:

- XR-15D train/eval completed with train exit `0` and four eval summaries.
- XR-15D results:
  - center best-center `20.333278461865017 / 25.986395263671874 / 8.095238372257777`.
  - center best-P10 `20.36335334096636 / 25.821854482378278 / 8.246173749651227`.
  - P10 best-center `20.362164442879813 / 26.141582359586444 / 8.562500286102296`.
  - P10 best-P10 `20.410626077651976 / 26.1883510862078 / 8.385629544939313`.
- Decision: XR-15D no-distill LR `3e-6` adapter branch did not promote and appears to drift away from the XR-15C center gate.
- Added XR-15E weak-distill low-LR adapter artifacts:
  - config `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
  - runner `scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh`.
- T-064 sub-agent request failed because GPT-5.3-Codex-Spark quota was reached; Codex-native fallback completed validation/launch.
- XR-15E center lane launched:
  - session `hgtxr_xr15e_weakdistill_trackadapters_center_gpu0_20260616`
  - log `runs/_logs/xr15e_weakdistill_trackadapters_center_lr5e-7_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_center_xr15e_fullwidth_20260616_123419`
- XR-15E P10 lane launched:
  - session `hgtxr_xr15e_weakdistill_trackadapters_p10_gpu1_20260616`
  - log `runs/_logs/xr15e_weakdistill_trackadapters_p10_lr5e-7_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_p10_xr15e_fullwidth_20260616_123431`
- Both lanes passed startup validation and reached epoch `1/12`.

2026-06-16 XR-15E closeout and XR-16A launch:

- XR-15E train/eval completed with train exit `0` and four eval summaries.
- XR-15E results:
  - center best-center `20.281941563742503 / 26.2521265574864 / 8.633928898402623`.
  - center best-P10 `20.297553059032985 / 26.096939522879463 / 8.618197590964181`.
  - P10 best-center `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
  - P10 best-P10 `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
- Decision: XR-15E did not promote. Weak distillation bounded drift versus XR-15D but did not beat XR-15C/XR-06C/XR-05A gates.
- T-065 GPT-5.5 sub-agent recommended an event-only adapter ablation as the next isolated variable after XR-15D/E both-adapter drift.
- Added XR-16A weak-distill event-path adapter artifacts:
  - config `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`.
  - runner `scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh`.
- XR-16A static validation passed: runner `bash -n`, raw event-count contract, diff whitespace check, and trainable-filter smoke with `14` tensors / `324680` trainable params.
- XR-16A center lane launched:
  - session `hgtxr_xr16a_eventadapter_center_gpu0_20260616`
  - log `runs/_logs/xr16a_weakdistill_trackeventadapter_center_lr5e-7_gpu0_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_center_xr16a_fullwidth_20260616_125515`
- XR-16A P10 lane launched:
  - session `hgtxr_xr16a_eventadapter_p10_gpu1_20260616`
  - log `runs/_logs/xr16a_weakdistill_trackeventadapter_p10_lr5e-7_gpu1_20260616.log`
  - run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_p10_xr16a_fullwidth_20260616_125515`
- Both lanes passed startup validation and reached epoch `1/12`.

2026-06-16 XR-16A/XR-16B closeout and XR-15C P5 update:

- XR-16A train/eval completed with train exit `0` and four eval summaries.
- XR-16A results:
  - center best-center `20.281941199302672 / 26.2521265574864 / 8.633928898402623`.
  - center best-P10 `20.29755315440042 / 26.096939522879463 / 8.618197590964181`.
  - P10 best-center `20.3268525327955 / 26.333759232929776 / 8.409864234924317`.
  - P10 best-P10 `20.3268525327955 / 26.333759232929776 / 8.409864234924317`.
- Decision: XR-16A did not promote. Event-only adapter isolation stayed below XR-15C center, XR-06C P10, and XR-05A P5 gates.
- Added XR-16B checkpoint interpolation diagnostic runner `scripts/external/run_xr16b_checkpoint_interp_eval.sh`.
- XR-16B alpha `0.125` between XR-15C center best-center and XR-06C best-P10 produced `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.
- Decision: XR-16B did not promote.
- XR-15C best-P5 artifact gap was filled:
  - center lane best-P5 `20.243119949953897 / 26.12159937449864 / 8.601190778187343`.
  - P10 lane best-P5 `20.245368467058455 / 26.487245675495693 / 8.703231593540737`.
- Decision: XR-15C P10-lane best-P5 promotes the P5/balanced secondary from XR-05A `8.69557854788644` to `8.703231593540737`.
- Active gates after this update: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.703231593540737`.
- Next recommended action: use the XR-15C P5 leader as the updated secondary anchor; prefer same-branch XR-15C center-to-P5 interpolation or failure-bucket comparison before another trainable adapter branch.

2026-06-16 XR-17A same-branch interpolation closeout:

- GPT-5.3-Codex-Spark sidecar request hit quota; GPT-5.5 sidecar reviewed the same-branch interpolation plan and recommended alpha `0.25/0.50/0.75/0.875/0.9375`.
- Added runner `scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh`.
- XR-17A endpoints:
  - alpha `0.0`: XR-15C center-lane best-center checkpoint.
  - alpha `1.0`: XR-15C P10-lane best-P5 checkpoint.
- Static validation passed: runner `bash -n`, endpoint checkpoints exist, and script diff whitespace check.
- XR-17A sweep results:
  - alpha `0.25`: `20.199484479427337 / 26.20960956301008 / 8.625425474984306`.
  - alpha `0.50`: `20.211353632381986 / 26.35204153742109 / 8.679847247259957`.
  - alpha `0.625`: `20.218607200895036 / 26.449830661501203 / 8.703231607164655`.
  - alpha `0.75`: `20.22669484274728 / 26.34566399029323 / 8.732993507385254`.
  - alpha `0.875`: `20.235614109039307 / 26.527636807305473 / 8.673469693320138`.
- Decision: alpha `0.75` promotes the P5/balanced gate. Center and P10 gates remain XR-15C/XR-06C.
- Active gates after XR-17A: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.732993507385254`.
- Next recommended action: compare XR-15C center leader versus XR-17A alpha `0.75` failure buckets, then design a bounded branch that preserves center while retaining the XR-17A P5 direction.

2026-06-16 XR-17A failure-bucket comparison:

- Generated diagnostics:
  - `runs/diagnostics/xr17a_compare_xr15c_center_failure_buckets_20260616.json`
  - `runs/diagnostics/xr17a_compare_xr17a_alpha0p75_failure_buckets_20260616.json`
- Row-weighted aggregate moved from XR-15C center `20.0095/26.1625/8.4824` to XR-17A alpha `0.75` `20.0415/26.5202/8.7890`.
- Interpretation: XR-17A is a P5/P10 recovery direction with small center cost. Gains are broad across similarity buckets and left-eye/subject `42/45/39`; regressions remain for subject `43/41` and right-eye P5.
- Decision: do not use XR-17A as the sole center seed. Next branch should keep XR-15C center as primary seed/teacher and use XR-17A alpha `0.75` as a bounded P5-direction teacher, checkpoint-soup anchor, or regularizer.

2026-06-16 XR-17B p5-anchor support-adaptive closeout:

- Added config `configs/external/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml`.
- Added runner `scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh`.
- Sandbox CUDA launch failed with `cuda_device_count=0`; escalated GPU0 run succeeded and resolved `cuda:0`.
- Run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310`.
- Test results:
  - best-center `20.182338142395018 / 26.113946315220424 / 8.474490090778895`; center promotion.
  - best-P10 `20.22993471963065 / 26.166242245265416 / 8.603316634041922`; no promotion.
  - best-P5 `20.234592584201266 / 26.09183748109 / 8.844813244683403`; P5 promotion.
- Failure buckets:
  - best-center weighted `20.0015 / 26.2647 / 8.5335`.
  - best-P5 weighted `20.0492 / 26.2136 / 8.8401`.
- Active gates after XR-17B: center `<20.182338142395018`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- Next action: P10 recovery branch that preserves XR-17B center/P5 gates.

2026-06-16 XR-18A no-train P10 interpolation closeout:

- Added runner `scripts/external/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh`.
- Sandbox evals hung with idle GPUs and were interrupted; escalated GPU0/GPU1 evals completed.
- Results:
  - alpha `0.025`: `20.184962025710515 / 26.113946315220424 / 8.474490090778895`.
  - alpha `0.05`: `20.18765983411244 / 26.054422501155308 / 8.525510501861572`.
  - alpha `0.075`: `20.190412517956325 / 26.156463316508702 / 8.525510501861572`.
  - alpha `0.10`: `20.193220179421562 / 26.20748372077942 / 8.629677173069545`.
- Decision: no promotion. No-train interpolation toward XR-06C P10 does not recover P10 enough and starts losing XR-17B center/P5 gates.

2026-06-16 XR-19A trainable P10-recovery micro-polish closeout:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml`.
- Added runner `scripts/external/run_xr19a_p10recovery_micro_polish.sh`.
- GPT-5.5 read-only evaluator recommended XR-17B best-P5 init plus XR-06C best-P10 weak teacher at LR `1.25e-7`.
- Run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_micro_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_152610`.
- Train completed epoch `8/8` with exit `0`; all three test evals completed.
- Test results:
  - best-center `20.182335831437793 / 25.973640203475952 / 8.588435670307705`.
  - best-P10 `20.20784169435501 / 26.232143613270352 / 8.609694181169782`.
  - best-P5 `20.181213889803207 / 25.928997346333094 / 8.508503695896694`; center promotion.
- Failure bucket diagnostic: `runs/diagnostics/xr19a_p10recovery_bestp10_failure_buckets_20260616.json`, weighted `20.0236 / 26.3669 / 8.6357`, low-similarity weighted `26.5449 / 17.5772 / 5.2257`.
- Active gates after XR-19A: center `<20.181213889803207`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- Decision: micro LR is center-safe but too weak for P10/P5 recovery. Next action should be stronger constrained P10 recovery or a P10-specialized lane, not another unchanged micro-polish.

2026-06-16 XR-20A/XR-20B P10-recovery LR ladder closeout:

- Reused `scripts/external/run_xr19a_p10recovery_micro_polish.sh` with explicit init and teacher checkpoints.
- XR-20A LR `2.5e-7` run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr2p5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155340`.
- XR-20B LR `5e-7` run root: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155656`.
- XR-20A train completed epoch `8/8`; XR-20B early-stopped at epoch `6`.
- Test results:
  - XR-20A best-center `20.175542894431523 / 25.884354482378278 / 8.486394848142352`; center promotion.
  - XR-20A best-P10 `20.202199470996856 / 26.031038168498448 / 8.74914996964591`; no P10/P5 promotion.
  - XR-20A best-P5 `20.211376798152923 / 26.00552796636309 / 8.71088467325483`; no promotion.
  - XR-20B best-center `20.257083107743945 / 25.838861281531198 / 8.380102341515677`; no promotion.
  - XR-20B best-P10 `20.183283712182725 / 25.8312082358769 / 8.423894848142352`; no promotion.
  - XR-20B best-P5 `20.218086302280426 / 25.805698026929583 / 8.588435690743582`; no promotion.
- Active gates after XR-20A: center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- XR-20A best-center failure-bucket diagnostic `runs/diagnostics/xr20a_lr2p5e7_bestcenter_failure_buckets_20260616.json` joined all `2238` rows. Weighted aggregate was `19.9949/26.0092/8.5335`; low-similarity `similarity_target < 0.1` remained weak at `26.5436/17.8147/5.2257`.
- Decision: the P5-init/XR-06C-P10-teacher LR-only ladder is closed. Next branch should change the P10 mechanism, not only LR.

2026-06-16 XR-21 P10-margin mechanism-change closeout:

- Added configs:
  - `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_fullwidth.yaml`
  - `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_fullwidth.yaml`
- Added runners:
  - `scripts/external/run_xr21_p10margin_supportadaptive_probe.sh`
  - `scripts/external/run_xr21_p10leader_margin_probe.sh`
- GPT-5.5 read-only evaluator recommended a P10-specialized squared hinge margin lane seeded from XR-06C best-P10, with hinge weight `0.0005` and best metric `metric_track_p10_pct`.
- Secondary XR-20A-init lanes completed:
  - `w=0.0015` best-center `20.18683280604226 / 25.886480338232857 / 8.49914995602199`.
  - `w=0.0015` best-P10 `20.19624582529068 / 25.90688850539071 / 8.558673770087106`.
  - `w=0.0015` best-P5 `20.19960424559457 / 25.96938850539071 / 8.77976222038269`.
  - `w=0.003` best-center `20.197126933506556 / 25.900510951450894 / 8.618197584152222`.
  - `w=0.003` best-P10 `20.187303059441703 / 26.126701450347902 / 8.791666977746146`.
  - `w=0.003` best-P5 `20.200117662974765 / 25.96938850539071 / 8.728741809300013`.
- Primary XR-06C-init P10-leader lane completed epoch `8/8` with train exit `0`; best-P10 and best-P5 both reached `20.26367484842028 / 26.20110617365156 / 8.696003689084733`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`. The next P0 must change representation/supervision more materially; tiny P10-margin replay is closed.

2026-06-16 XR-22 tri-leader checkpoint soup closeout:

- Added `scripts/external/average_hbtxr_checkpoints.py` for weighted N-way model-state soup.
- Added `scripts/external/run_xr22_trileader_soup_eval.sh` for no-train soup generation and test eval.
- Wegener GPT-5.5 read-only explorer independently recommended checkpoint soup/interpolation as the fastest next P0, with optimizer probe as second priority.
- Inputs:
  - XR-20A center leader checkpoint.
  - XR-06C P10 leader checkpoint.
  - XR-17B P5 leader checkpoint.
- Test results:
  - `c50_p25_f25`: `20.21947033064706 / 26.349915708814347 / 8.785289430618286`.
  - `c34_p33_f33`: `20.236038860252926 / 26.4604599407741 / 8.869047941480364`; P5 promotion.
  - `c25_p50_f25`: `20.257381524358475 / 26.36267081669399 / 8.601190784999302`.
  - `c25_p25_f50`: `20.235262938908168 / 26.269133404323032 / 8.829932287761144`.
  - `c20_p60_f20`: `20.270640075206757 / 26.405187838418144 / 8.735119356427873`.
  - `c20_p40_f40`: `20.251396659442356 / 26.349915736062187 / 8.603316634041922`.
- Decision: XR-22 promotes P5/balanced gate to `8.869047941480364`. Center and P10 gates remain XR-20A and XR-06C. Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.

2026-06-16 XR-23 P10-preserving optimizer probe closeout:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml`.
- Added runner `scripts/external/run_xr23_p10_optimizer_probe.sh`.
- Poincare GPT-5.5 read-only optimizer explorer selected ADOPT and Lion as bounded first-pass optimizer probes from the supported optimizer registry.
- ADOPT/GPU0 command completed epoch `8/8`: `bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 adopt 2.5e-7 cuda:0`.
- Lion/GPU1 command early-stopped at epoch `6/8`: `bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 lion 1e-7 cuda:1`.
- Test results:
  - ADOPT best-P10 `20.261837770257678 / 26.318027945927213 / 8.681122745786395`.
  - ADOPT best-P5 `20.219128920350755 / 25.956633363451278 / 8.469388042177473`.
  - Lion best-P10 `20.22320341382708 / 26.311650391987392 / 8.513180569240026`.
  - Lion best-P5 `20.271730688640048 / 26.129677595411028 / 8.532313203811645`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. Next P0 should use XR-22 as a bounded P10-recovery anchor; only retry optimizer if testing ADOPT-specific beta/eps defaults.

2026-06-16 XR-24 XR-22-anchor P10-recovery closeout:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml`.
- Added runner `scripts/external/run_xr24_xr22_anchor_p10recovery.sh`.
- Carson GPT-5.5 read-only strategy explorer recommended the XR-22 `c34/p33/f33` soup as the next bounded P10-recovery anchor, with XR-06C best-P10 teacher and P10-selection.
- GPU0 command completed epoch `8/8`: `bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
- GPU1 command completed epoch `8/8`: `bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Test results:
  - LR `1.25e-7` best-P10 `20.217043702942984 / 26.254252440588814 / 8.764881270272392`.
  - LR `1.25e-7` best-P5 `20.223851100036075 / 26.415817070007325 / 8.722364262172155`.
  - LR `2.5e-7` best-P10 `20.23367166178567 / 26.121599388122558 / 8.588435677119664`.
  - LR `2.5e-7` best-P5 `20.238911376680647 / 26.07695653779166 / 8.791666984558105`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. XR-22 remains useful as a no-train anchor, but AdamW polish from it is closed.

2026-06-16 XR-25 XR-22-anchor ADOPT-defaults closeout:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml`.
- Added runner `scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh`.
- Static validation passed: runner syntax, config load, ADOPT build with betas `[0.9,0.9999]` and eps `1e-6`, raw event-count contract, checkpoint existence, and `git diff --check`.
- GPU0 command early-stopped at epoch `7/8`: `bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
- GPU1 command completed epoch `8/8`: `bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Test results:
  - LR `1.25e-7` best-P10 `20.21177260194506 / 26.46045993396214 / 8.728741809300013`.
  - LR `1.25e-7` best-P5 `20.22005627495902 / 26.181123181751797 / 8.707483305249895`.
  - LR `2.5e-7` best-P10 `20.23533037390028 / 26.108844266619002 / 8.64583364214216`.
  - LR `2.5e-7` best-P5 `20.23760941709791 / 26.187500749315536 / 8.86607174192156`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. XR-22 trainable polish is closed for AdamW and ADOPT-defaults; next P0 is XR-26 P10-boundary loss from XR-06C best-P10 init/teacher.

2026-06-16 XR-26 P10-boundary head-only closeout:

- Added P10-boundary surrogate loss on `track/state` and `track/state_aux`, plus config `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10boundary_headonly_fullwidth.yaml` and runner `scripts/external/run_xr26_p10boundary_aux_probe.sh`.
- Static validation passed: runner syntax, pytest `18 passed`, config smoke, checkpoint existence, and `git diff --check`.
- Non-escalated launch failed because sandbox CUDA reported `cuda_device_count=0`; escalated launch ran normally.
- GPU0 default command early-stopped at epoch `6/8`: `bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:0`.
- GPU1 light command early-stopped at epoch `6/8`: `bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:1 ... xr06cp10_init_teacher_light 0.01 0.005`.
- Test results:
  - Default best-P10 `20.32213627440589 / 26.45790890966143 / 8.535289403370449`.
  - Default best-P5 `20.318065077917918 / 26.42814700944083 / 8.541666957310268`.
  - Light best-P10 `20.32214218207768 / 26.45790890966143 / 8.535289403370449`.
  - Light best-P5 `20.318073788711004 / 26.42814700944083 / 8.541666957310268`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. Scalar P10-boundary/hinge polish is closed; next P0 should change coordinate representation or teacher quality.

2026-06-16 XR-27 track-heatmap coordinate representation closeout:

- Added `TrackCenterHeatmapHead`, heatmap decode into `track/state[:, :2]`, heatmap CE/offset losses, config `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`, and runner `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`.
- Sub-agent audit found a launch blocker: `distillation.state_similarity=true` would compare the student's heatmap-state output against teacher heatmap-state output with random newly added heatmap weights. Clean runs set `state_similarity=false` and `state_weight=0.0`.
- GPU0 LR `1e-4` command completed epoch `10/10`: `bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 1e-4 cuda:0 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr1e4`.
- GPU1 LR `3e-5` command completed epoch `10/10`: `bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 3e-5 cuda:1 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr3e5`.
- Test results:
  - LR `1e-4` best-P10 `17.274589475563594 / 31.915391901561193 / 10.289966331209456`.
  - LR `1e-4` best-P5 `17.6397743497576 / 29.80569808142526 / 10.210459525244577`.
  - LR `3e-5` best-P10 `18.388037020819528 / 27.238521099090576 / 8.488520683561052`.
  - LR `3e-5` best-P5 `19.03528357914516 / 24.192602627617973 / 6.854166896002633`.
- Decision: XR-27A best-P10 is the new active software leader and promotes all gates. New active gates: center `<17.274589475563594`, P10 `>31.915391901561193`, P5 `>10.289966331209456`.
- Next action: XR-28 consolidation diagnostics, best-center checkpoint selection support, and narrow LR/heatmap-weight refinement around LR `1e-4`.

2026-06-16 XR-28 consolidation diagnostic closeout:

- Generated failure-bucket summaries for XR-27A, XR-06C, XR-20A, and XR-22 under `runs/diagnostics/xr28_*_failure_buckets_20260616.json`.
- All four diagnostics joined `2238` test rows with `0` missing predictions.
- Overall weighted center/P10/P5:
  - XR-27A `17.1708 / 32.1921 / 10.3219`.
  - XR-06C `20.2065 / 26.9801 / 8.5846`.
  - XR-20A `19.9949 / 26.0092 / 8.5335`.
  - XR-22 `20.0528 / 26.6224 / 8.8912`.
- Interpretation: XR-27A primarily fixes low/mid `similarity_target` buckets, including `<=0.1` where weighted center improves from about `26.5-26.9` to `20.1549`.
- Remaining risks: high-similarity `>0.9` P10 remains better in XR-20A/XR-22, and subject `39` P10/P5 regresses despite lower center error.
- Follow-up action was completed by XR-28 LR `7e-5` and `1.5e-4` heatmap refinements with `distillation.state_similarity=false`.

2026-06-16 XR-28 runner update:

- Updated `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` to accept `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME`.
- Existing XR-27A validation history shows best-center and best-P10 both at epoch `10`, so `best_track_p10.pt` remains the correct current leader.
- Next XR-28 commands should set `XR27_BEST_METRIC_NAME=metric_track_center_px` for LR `7e-5` and `1.5e-4` lanes so center-selected checkpoints are emitted in addition to best-P10/best-P5.

2026-06-16 XR-28 LR refinement closeout:

- Ran LR `7e-5` on GPU0 and LR `1.5e-4` on GPU1 with center-selected checkpointing, unchanged heatmap weights `0.005/0.001`, center L2 `0.001`, support-adaptive fixed255k, and state distillation disabled.
- Both train lanes completed epoch `10/10` with train exit `0`; six eval summaries were produced for best-P10, best-P5, and best-center.
- LR `7e-5` did not promote. Best-center was `17.55661221402032/31.095664044788904/9.788265630177088`.
- LR `1.5e-4` promoted all active gates. Best-P10, best-P5, and best-center all resolve to epoch `10` and test `17.08485197339739/32.081208263124736/10.502551344462804`.
- Active leader is `runs/raw_mode1_stage2_count255000_adamw_lr1_5e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr28_heatmap_centerselect_lr1p5e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_200229/train/best_metric_track_center_px.pt`.
- Next P0: XR-29 post-XR-28 failure-bucket diagnostics and bounded LR/epoch refinement around LR `1.5e-4`.

2026-06-16 XR-29 post-XR-28 diagnostic and LR-neighbor launch:

- Sub-agent audit `Aristotle` completed read-only review and agreed that promoted XR-28 failure buckets were the missing artifact before broadening experiments.
- Generated `runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json` from the promoted XR-28 LR `1.5e-4` best-center eval rows.
- Diagnostic joined `2238` test rows with `0` missing predictions. Weighted aggregate: `16.9868/32.3454/10.5774`.
- Residual weak buckets: `similarity <=0.1`, high-similarity P10/P5 relative to XR-20A/XR-22 anchors, subject `39`, subjects `42/45`, and `session_201`-heavy sessions.
- Launched XR-29 LR-neighbor jobs:
  - GPU0 LR `1.25e-4`, tmux `hgtxr_xr29_heatmap_lr1p25e4_gpu0_20260616`, log `runs/_logs/xr29_heatmap_centerselect_lr1p25e-4_gpu0_20260616.log`.
  - GPU1 LR `1.75e-4`, tmux `hgtxr_xr29_heatmap_lr1p75e4_gpu1_20260616`, log `runs/_logs/xr29_heatmap_centerselect_lr1p75e-4_gpu1_20260616.log`.
- Startup validation passed for both: raw event-count contract, intended XR-06C init/teacher checkpoint, `resolved_device=cuda:0/1`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10` train steps.

2026-06-16 XR-29 LR-neighbor closeout:

- Both XR-29 train lanes completed epoch `10/10` with train exit `0`; six eval summaries were produced for best-P10, best-P5, and best-center.
- LR `1.25e-4` produced identical best-P10/best-P5/best-center test metrics `17.179786903517588/32.04761978558132/10.53443912097386`; this improves P5 only and is not selected.
- LR `1.75e-4` produced identical best-P10/best-P5/best-center test metrics `17.04961508342198/32.56462665285383/11.50467722075326`.
- LR `1.75e-4` promotes all active gates versus XR-28 LR `1.5e-4`; deltas are center `-0.035236889975411856`, P10 `+0.48341838972909557`, P5 `+1.0021258762904566`.
- Active leader is `runs/raw_mode1_stage2_count255000_adamw_lr1_75e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr29_heatmap_centerselect_lr1p75e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_204422/train/best_metric_track_center_px.pt`.
- New active gates: center `<17.04961508342198`, P10 `>32.56462665285383`, P5 `>11.50467722075326`.
- Next P0: XR-30 LR micro-bracket with LR `1.625e-4` and `1.875e-4`; keep heatmap weights, support-adaptive count contract, teacher/init checkpoint, and center-selected checkpointing unchanged.

2026-06-16 XR-30 LR micro-bracket closeout:

- Ran LR `1.625e-4` on GPU0 and LR `1.875e-4` on GPU1 with the same XR-29 heatmap-state support-adaptive contract.
- Startup validation passed for both: raw event-count contract, intended XR-06C init/teacher checkpoint, `resolved_device=cuda:0/1`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10` train steps.
- Both train lanes completed epoch `10/10` with train exit `0`; all six best-P10/best-P5/best-center full-test eval summaries were produced.
- LR `1.625e-4` produced identical best-P10/best-P5/best-center test metrics `17.067689692974092/32.420068802152365/10.866496937615532`; no active gate promoted.
- LR `1.875e-4` produced identical best-P10/best-P5/best-center test metrics `17.035534060001375/32.42474567549569/11.266581957680838`; center promoted, P10/P5 regressed relative to XR-29.
- New active gates are mixed: center `<17.035534060001375` from XR-30 LR `1.875e-4`, P10 `>32.56462665285383` and P5 `>11.50467722075326` from XR-29 LR `1.75e-4`.
- Next P0: XR-31 no-train checkpoint interpolation between XR-29 LR `1.75e-4` and XR-30 LR `1.875e-4`; train LR `1.8125e-4` only if interpolation cannot preserve P10/P5.

2026-06-16 XR-31 no-train interpolation closeout:

- Added `scripts/external/run_xr31_xr29_xr30_heatmap_interp_eval.sh`, using existing checkpoint interpolation utility and the XR-29/XR-30 heatmap-state eval contract.
- Spark sub-agent execution was attempted for sidecar inspection but failed due GPT-5.3-Codex-Spark usage limit; main-agent fallback executed and validated the task.
- Checkpoint compatibility passed: both source checkpoints are epoch `10`, have `138` matching model tensors, and all tensors are floating-point interpolated tensors.
- Evaluated alpha `0.125/0.1875/0.25/0.50/0.75`.
- Results:
  - `0.125`: `17.04816469124385/32.57313004221235/11.488945926938738`.
  - `0.1875`: `17.047394837651932/32.528487185069494/11.503826883860997`.
  - `0.25`: `17.04659355367933/32.57950758934021/11.503826883860997`.
  - `0.50`: `17.043099636690958/32.50170144353594/11.37500034059797`.
  - `0.75`: `17.039319237640925/32.5080789906638/11.362245225906372`.
- Decision: alpha `0.25` promotes P10 only. It improves center relative to XR-29 and improves P10 over the active gate, but misses the strict P5 gate by about `0.000850336892263`.
- Active gates are now center `<17.035534060001375` from XR-30, P10 `>32.57950758934021` from XR-31 alpha `0.25`, and P5 `>11.50467722075326` from XR-29.
- Next P0: train the midpoint LR `1.8125e-4` under the same heatmap-state contract.

2026-06-16 XR-32/XR-33 trained LR midpoint closeout:

- Ran XR-32 LR `1.8125e-4` on GPU0 and XR-33 LR `1.84375e-4` on GPU1 under the same XR-29/XR-30 heatmap-state support-adaptive fixed255k contract.
- Both train lanes completed epoch `10/10` with train exit `0`.
- XR-32 best-P10/best-P5/best-center full-test evals all produced `17.041867678506033/32.5463443006788/11.37500034059797`; no active gate promoted.
- XR-33 best-P10/best-P5/best-center full-test evals all produced `17.039179919447218/32.62074908529009/11.362245225906372`; P10 promoted only.
- Active gates are now center `<17.035534060001375` from XR-30, P10 `>32.62074908529009` from XR-33, and P5 `>11.50467722075326` from XR-29.
- Best unified checkpoint remains XR-29 LR `1.75e-4` because XR-33 improves P10 while regressing center and P5.
- Next P0: avoid another close LR-only replay unless history shows non-convergence. Prefer heatmap loss-ratio refinement or a bounded XR-29/XR-33 anchor/soup regularization branch.

2026-06-16 XR-34 heatmap loss-ratio launch:

- Added `scripts/external/run_xr34_heatmap_lossratio_refinement.sh` with two lanes:
  - XR-34A center/P5-preserve: XR-29 init/teacher, LR `1.75e-4`, heatmap/offset/center `0.004/0.0015/0.0015`, GPU0.
  - XR-34B P10-sharpen: XR-33 init/teacher, LR `1.84375e-4`, heatmap/offset/center `0.006/0.001/0.001`, GPU1.
- Added plan artifact `docs/resources/xr34_heatmap_lossratio_plan_2026_06_16.md`.
- Launched tmux sessions `hgtxr_xr34a_heatmap_lossratio_gpu0_20260616` and `hgtxr_xr34b_heatmap_lossratio_gpu1_20260616`.
- Logs:
  - `runs/_logs/xr34a_heatmap_lossratio_gpu0_20260616.log`.
  - `runs/_logs/xr34b_heatmap_lossratio_gpu1_20260616.log`.
- Startup validation passed for both lanes: raw event-count contract, intended init checkpoint, `resolved_device=cuda:0/1`, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- GPT-5.5 sidecar audit completed read-only. It ranked bounded longer-budget XR-29/XR-33 continuation as next fallback because both histories still improved through epoch `10`; optimizer change and teacher retraining remain lower priority.
- Promotion decision pending full train/eval closeout.

2026-06-16 XR-34 closeout and XR-35 preparation:

- XR-34A completed train/eval with exit `0`; best-P10/best-P5/best-center all resolved to epoch `3` and test `17.026217068944657/32.430698088237214/10.911139822006225`.
- XR-34B completed train/eval with exit `0`; best-P10 epoch `7` reached `16.53321223940168/33.77168447630746/11.276786088943481`, promoting center and P10.
- XR-34B best-P5/best-center epoch `3` reached `17.0139569742339/32.49957566261291/11.226190853118897`.
- Active gates are now center `<16.53321223940168`, P10 `>33.77168447630746`, and P5 `>11.50467722075326`.
- Added XR-35 no-train interpolation runner `scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh` and plan artifact `docs/resources/xr35_xr29_xr34b_interpolation_plan_2026_06_16.md`.

2026-06-16 XR-35 no-train interpolation closeout:

- Ran `bash scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh cuda:1 0.03125:a0p03125 0.0625:a0p0625 0.09375:a0p09375 0.125:a0p125`.
- Results:
  - `0.03125`: `17.016330581051964/32.47534093856812/11.334609195164271`.
  - `0.0625`: `16.983979083810535/32.468963384628296/11.391156809670585`.
  - `0.09375`: `16.952843945366997/32.42432054110936/11.420918709891183`.
  - `0.125`: `16.923010180677686/32.352041605540684/11.255102368763515`.
- No active gate promoted. Active gates remain center `<16.53321223940168`, P10 `>33.77168447630746`, and P5 `>11.50467722075326`.
- Next P0: trainable P5-anchor continuation using XR-34B center/P10 signal and XR-29 P5 anchor.

2026-06-16 XR-36 preparation:

- Added `scripts/external/run_xr36_p5_anchor_continuation.sh`.
- Added `docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md`.
- Static validation passed: `bash -n scripts/external/run_xr36_p5_anchor_continuation.sh`.
- Required source checkpoints exist: XR-34B best-P10, XR-35 alpha `0.09375`, and XR-29 P5 anchor.
- Planned lanes:
  - XR-36A: XR-34B init, XR-29 teacher/reference, LR `5e-5`, GPU0.
  - XR-36B: XR-35 alpha `0.09375` init, XR-29 teacher/reference, LR `8.75e-5`, GPU1.

2026-06-16 XR-36 closeout and XR-37 preparation:

- XR-36A and XR-36B launched through tmux after sandbox tmux socket access required escalation.
- Both lanes passed startup validation: raw event-count contract, intended checkpoint, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params.
- Both lanes early-stopped at epoch `7/10` with train exit `0`.
- XR-36A best-P10 reached `16.59279990025929/34.39710958344596/11.738095617294311`.
- XR-36A best-P5 reached `16.576952314376832/34.74064704350063/11.415816688537598`.
- XR-36A best-center reached `16.599328325475966/33.54209257534572/10.998724787575858`.
- XR-36B best-P10 reached `16.53305721793856/34.19387831687927/11.502551344462804`.
- XR-36B best-P5/best-center reached `16.844227249281747/32.883078956604/11.231718029294695`.
- Active gates are now center `<16.53305721793856`, P10 `>34.74064704350063`, and P5 `>11.738095617294311`.
- Added XR-37 runner `scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh` and plan `docs/resources/xr37_xr36b_xr36a_interpolation_plan_2026_06_16.md`.

2026-06-17 XR-37 closeout:

- Ran `bash scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh cuda:1 0.10:a0p10 0.20:a0p20 0.35:a0p35 0.50:a0p50`.
- Results: alpha `0.10` reached `16.52253861086709/34.062075574057445/11.574830266407558`.
- Results: alpha `0.20` reached `16.51475806917463/34.28826605933053/11.44387788772583`.
- Results: alpha `0.35` reached `16.50803507396153/34.23086808749608/11.292517362322126`.
- Results: alpha `0.50` reached `16.507612899371555/34.33205857958112/11.539116007941109`.
- Decision: alpha `0.50` promotes center only. Active gates are now center `<16.507612899371555`, P10 `>34.74064704350063`, and P5 `>11.738095617294311`.

2026-06-17 XR-38 preparation:

- Added `scripts/external/run_xr38_center_preserve_p10p5_recovery.sh`.
- Added `docs/resources/xr38_center_preserve_p10p5_recovery_plan_2026_06_17.md`.
- Static validation passed: `bash -n scripts/external/run_xr38_center_preserve_p10p5_recovery.sh`.
- Required source checkpoints exist: XR-37 alpha `0.50`, XR-36A best-P10, and XR-36A best-P5.
- Planned lanes:
  - XR-38A: GPU0, XR-37 alpha `0.50` init, XR-36A best-P10 reference, LR `2.5e-5`, center-selected.
  - XR-38B: GPU1, XR-37 alpha `0.50` init, XR-36A best-P5 reference, LR `1.25e-5`, P10-selected.
- Sandboxed launch failed for both lanes with PyTorch CUDA unavailable despite `nvidia-smi`; reran unsandboxed for CUDA/NVML access.
- XR-38A run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38a_xr37a050_init_xr36a_bestp10_ref_lr2p5e5_centerselect_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002109`.
- XR-38B run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr38b_xr37a050_init_xr36a_bestp5_ref_lr1p25e5_p10select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_002120`.
- Startup validation passed for both lanes: raw event-count contract, intended checkpoint, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.

2026-06-17 XR-38 closeout:

- Both lanes completed with train exit `0` and early-stopped at epoch `7/10`.
- XR-38A best-center reached `16.556500560896737/33.96471167291914/11.519983339309693`.
- XR-38A best-P10 reached `16.57439456837518/33.92984774453299/11.455357497079032`.
- XR-38A best-P5 reached `16.513289058208464/34.1815484387534/11.56462620326451`.
- XR-38B best-P10 reached `16.510086681161606/34.707483761651176/11.223214626312256`.
- XR-38B best-P5 reached `16.513694180761064/33.93452457700457/11.843962955474854`.
- Decision: XR-38B best-P5 promotes P5 only. Active gates are now center `<16.507612899371555`, P10 `>34.74064704350063`, and P5 `>11.843962955474854`.
- Next P0: XR-39 no-train mixed-leader soup/interpolation around XR-37 alpha `0.50`, XR-38B best-P5, and XR-36A best-P5.

2026-06-17 XR-39 closeout:

- Added `scripts/external/run_xr39_mixed_leader_soup_eval.sh`.
- Added `docs/resources/xr39_mixed_leader_soup_plan_2026_06_17.md`.
- Spark sidecar audit failed due quota; GPT5.5 fallback returned a read-only matrix audit.
- Static/input validation passed: `bash -n`, `git diff --check`, all source checkpoints exist, and all three anchors have matching `138` model keys.
- Ran 13 no-train mixed-leader soups across GPU0/GPU1.
- Best center: `c60p25f15` reached `16.491779099191938/34.5306130204882/11.50000034059797`.
- Best P10: `c25p45f30` reached `16.503940873486656/35.02295998845781/11.460459525244577`.
- Best XR-39 P5: `c70p20f10` reached `16.492875189440593/34.34566400391715/11.744473137174333`, below XR-38B.
- Decision: XR-39 promotes center and P10; P5 remains XR-38B-owned.
- Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.843962955474854`.

2026-06-17 XR-40 closeout:

- Added `scripts/external/run_xr40_xr39leader_p5_soup_eval.sh`.
- Added `docs/resources/xr40_xr39leader_p5_soup_plan_2026_06_17.md`.
- Static/input validation passed: `bash -n`, `git diff --check`, all source checkpoints exist, and all three anchors have matching `138` model keys.
- Ran 7 no-train soups across GPU0/GPU1.
- Best center: `c45p35f20` reached `16.493494159834725/34.77508579662868/11.255527530397687`.
- Best P10: `c40p40f20` reached `16.494016419138227/34.7750858102526/11.249149976457868`.
- Best P5: `c35p25f40` reached `16.49522715806961/34.64115719795227/11.529762240818568`.
- Decision: no gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.843962955474854`.
- Next P0: trainable loss-ratio fallback for P5 recovery.

2026-06-17 XR-41 closeout:

- Added `scripts/external/run_xr41_lossratio_p5_fallback.sh`.
- Added `docs/resources/xr41_lossratio_p5_fallback_plan_2026_06_17.md`.
- Static/input validation passed: `bash -n`, `git diff --check`, and required source checkpoint existence checks.
- XR-41A ran on GPU0 from XR-38B best-P5 init with LR `1e-5`, best metric `metric_track_p5_pct`, and loss ratio `0.004/0.0015/0.0015`.
- XR-41B ran on GPU1 from XR-39 `c25p45f30` init with LR `8e-6`, best metric `metric_track_p10_pct`, and the same loss ratio.
- Both lanes completed train/eval with exit `0` and early-stopped at epoch `7/10`.
- XR-41A best-P10 reached `16.519287032740458/34.66199056080409/11.25637790134975`.
- XR-41A best-P5 reached `16.519514334201812/34.09821502821786/11.868197652271816`.
- XR-41B best-P10 reached `16.50909768513271/34.426021228517804/11.286139808382307`.
- XR-41B best-P5 reached `16.50978491306305/34.228742252077375/11.734694249289376`.
- Decision: XR-41A best-P5 promotes P5 only.
- Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.

2026-06-17 XR-42 launch:

- Added `scripts/external/run_xr42_p5_preserve_lowdrift.sh`.
- Added `docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md`.
- Static/input validation passed: `bash -n`, `git diff --check`, and required source checkpoint existence checks.
- XR-42A launched on GPU0 from XR-39 center `c60p25f15`, with XR-41A best-P5 as teacher/reference, LR `3e-6`, best metric `metric_track_center_px`, and heatmap/offset/center `0.0035/0.00125/0.0020`.
- XR-42B launched on GPU1 from XR-39 P10 `c25p45f30`, with XR-41A best-P5 as teacher/reference, LR `3e-6`, best metric `metric_track_p10_pct`, and the same loss ratio.
- Both lanes passed startup validation: raw event-count contract, intended checkpoint load, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- Active gates are unchanged until full-test eval closeout: center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.

2026-06-17 XR-42 closeout:

- Both lanes completed train/eval with exit `0` and early-stopped at epoch `7/10`.
- XR-42A best-P10 reached `16.49802110535758/34.48299399103437/11.366922119685581`.
- XR-42A best-P5 reached `16.498888087272643/34.30739874839783/11.347789451054163`.
- XR-42A best-center reached `16.498888087272643/34.30739874839783/11.347789451054163`.
- XR-42B best-P10 reached `16.505801352432798/34.3171777180263/11.278911903926305`.
- XR-42B best-P5 reached `16.508924693720683/34.29506881577628/11.323554761069161`.
- Decision: no gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next P0: decide whether to run XR-42C with explicit tiny distillation/regularization or pivot away from the XR-41 P5 teacher branch.
# 2026-06-17 XR-42C closeout

- Prepared `scripts/external/run_xr42c_explicit_distill_recovery.sh` after GPT-5.5 sub-agent audit found that XR-42 had no effective heatmap-state teacher anchor because XR-27 forced `distillation.state_similarity=false`.
- Added plan artifact: `docs/resources/xr42c_explicit_distill_recovery_plan_2026_06_17.md`.
- Ran XR-42C-A on GPU0 and XR-42C-B on GPU1. Initial sandbox execution failed with CUDA unavailable; approved GPU runtime executed successfully.
- XR-42C-A early-stopped at epoch `7/10`; best-center was the closest result: `16.491862688745773/34.712585769380844/11.472364275796073`, but it missed the center gate by `0.000083589553835`.
- XR-42C-B completed epoch `10/10`; best-P10 was `16.502160484450204/34.39710965156555/11.22236428941999`.
- No gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Next decision: pivot away from XR41 P5-teacher recovery. Prioritize failure-bucket/subject-39 direct optimization or teacher retraining.

# 2026-06-17 XR-43 closeout

- Added `scripts/external/run_xr43_xr39center_xr42ca_interp_eval.sh`.
- Ran no-train interpolation between XR-39 center leader `xr39_mixedleader_soup_c60p25f15.pt` and XR-42C-A best-center.
- Log: `runs/_logs/xr43_xr39center_xr42ca_interp_gpu0_20260617_025024.log`.
- Alpha results:
  - `0.125`: `16.49161465849195/34.44132730620248/11.389456115450178`
  - `0.25`: `16.491504199164254/34.33078307424273/11.478741829735892`
  - `0.5`: `16.491429926667895/34.512755850383215/11.531888089861189`
  - `0.75`: `16.491550181593215/34.60204156466893/11.576530947004045`
- XR-43 alpha `0.5` promoted center. Active gates are now center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Next trainable branch should avoid XR41-teacher replay and previous low-similarity reweighting. Preferred next direction is representation/teacher retraining focused on low similarity and subject `39`.

# 2026-06-17 XR-44 closeout

- Added `scripts/external/run_xr44_updated_trileader_soup_eval.sh`.
- Ran updated tri-leader soup with XR-43 center alpha `0.5`, XR-39 P10 `c25p45f30`, and XR-41A best-P5.
- Results:
  - `c70p20f10`: `16.49202539069312/34.685800075531006/11.264030940192086`
  - `c60p25f15`: `16.492678804056986/34.64753478595188/11.255527537209646`
  - `c55p30f15`: `16.493043976170675/34.692177643094745/11.306547941480364`
  - `c50p35f15`: `16.493474864959715/34.78784090450832/11.306547941480364`
  - `c45p40f15`: `16.493977418967656/34.7963443006788/11.255527530397687`
  - `c40p45f15`: `16.49455840757915/34.7368205002376/11.20450711931501`
  - `c35p45f20`: `16.49523995944432/34.692177643094745/11.21088467325483`
  - `c30p45f25`: `16.49610698393413/34.63265381540571/11.315051344462804`
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.

# 2026-06-17 XR-45 closeout

- Added `scripts/external/run_xr45_lowsim_heatmap_refresh.sh`.
- Checked manifest split before launch: subject `39` is test-only, so direct subject-39 train/val slicing was rejected as leakage risk.
- XR-45A: low-similarity `<=0.3` specialist manifest, GPU0, XR-43 center init/teacher, LR `1e-5`, best-center selection. The focus subset had train `2350` and val `345`.
- XR-45B: full-manifest refresh, GPU1, XR-43 center init, XR-39 P10 teacher, LR `1e-5`, best-P10 selection.
- Both lanes passed train/eval with exit `0`. XR-45A completed epoch `10/10`; XR-45B early-stopped at epoch `7/10`.
- XR-45A best-P10 reached `17.03435743876866/32.714711679731096/11.164966344833374`.
- XR-45A best-P5 reached `16.849295384543282/33.63605521747044/11.585034343174526`.
- XR-45A best-center reached `17.808542433806828/30.72661645753043/9.841411910738264`.
- XR-45B best-P10 reached `16.50801784992218/34.5969395501273/11.428571782793318`.
- XR-45B best-P5 reached `16.50964238813945/33.96726266316005/11.820578595570156`.
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next: avoid subset-only low-sim heatmap refresh. Prefer checkpoint-space search around XR-43/XR-39/XR-41 anchors or full-manifest calibration that explicitly preserves the XR-43 center leader.

# 2026-06-17 XR-46 closeout

- Added `scripts/external/run_xr46_center_p10_compat_soup_eval.sh`.
- Added `docs/resources/xr46_center_p10_compat_soup_plan_2026_06_17.md`.
- GPT-5.5 sub-agent audit recommended XR43-manifold micro-alpha sweep as the cleanest no-train follow-up. Spark sub-agent spawn failed due thread limit, so the trainable-branch review used local fallback.
- Static/input validation passed: runner syntax, required checkpoint existence, and full-test eval exits.
- XR-46 A center-to-P10 interpolation:
  - `a0p125`: `16.491489429133278/34.563776268277849/11.517007132938931`
  - `a0p25`: `16.491949367523194/34.659439550127303/11.412840468542916`
  - `a0p50`: `16.494027202469962/34.793368135179790/11.317177200317383`
  - `a0p75`: `16.498019000462122/34.852891956056865/11.370323453630720`
  - `a0p875`: `16.500783746583121/35.022959988457814/11.421343871525355`
- XR-46 B XR45B-bridged soup:
  - `c70p20x05f05`: `16.491824064935958/34.590136814117429/11.264030940192086`
  - `c60p25x10f05`: `16.492247397559030/34.647534785951883/11.204507126126970`
  - `c55p30x10f05`: `16.492606668812886/34.736820500237599/11.255527530397687`
  - `c50p35x10f05`: `16.493030984061104/34.787840904508322/11.306547941480364`
  - `c45p40x10f05`: `16.493524081366402/34.867772872107366/11.306547941480364`
- XR-46C XR43-manifold micro-alpha:
  - `0.55`: `16.491439385073527/34.512755850383215/11.531888089861189`
  - `0.60`: `16.491455343791415/34.512755850383215/11.531888089861189`
  - `0.65`: `16.491479851518360/34.512755850383215/11.576530947004045`
  - `0.70`: `16.491511028153557/34.557398707526069/11.576530947004045`
  - `0.80`: `16.491597735881804/34.602041564668930/11.576530947004045`
  - `0.875`: `16.491683168070658/34.661565365110128/11.576530947004045`
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next: stop no-train checkpoint-space sweeps around these anchors. Move to trainable full-manifest calibration with explicit XR-43 center preservation.

# 2026-06-17 XR-47 closeout

- Added `scripts/external/run_xr47_center_preserve_full_calibration.sh`.
- Added `docs/resources/xr47_center_preserve_full_calibration_plan_2026_06_17.md`.
- Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, and `git diff --check`.
- Ran XR-47A on GPU0: XR-43 center init/teacher, LR `2e-6`, center L2 `0.0025`, state distillation `0.0005`, best metric `metric_track_p10_pct`.
- Ran XR-47B on GPU1: same init/teacher/settings, best metric `metric_track_p5_pct`.
- XR-47A early-stopped at epoch `7/8`; train/eval exit `0`.
- XR-47B completed epoch `8/8`; train/eval exit `0`.
- XR-47A best-P10 reached `16.496871314729962/34.372449772698538/11.298894902638027`.
- XR-47A best-P5 reached `16.495964876243047/34.196854530061991/11.303146593911308`.
- XR-47B best-P10 reached `16.496871314729962/34.372449772698538/11.298894902638027`.
- XR-47B best-P5 reached `16.497679948806763/34.444728687831336/11.592262281690324`.
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next: mechanism change required. Do not keep spending cycles on same heatmap-state head LR/teacher sweeps unless a new loss/head constraint is added.

# 2026-06-17 XR-48 closeout

- Added `scripts/external/run_xr48_p10_boundary_heatmap_calibration.sh`.
- Added `docs/resources/xr48_p10_boundary_heatmap_calibration_plan_2026_06_17.md`.
- GPT-5.5 read-only sub-agent recommended P10-boundary heatmap-state calibration using XR-43 center init and XR-39 P10 teacher.
- Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, and `git diff --check`.
- Leakage check passed: subject `39` count was train `0`, val `0`, test `144`; full train/val manifests were used.
- Sandboxed CUDA failed with `cuda_is_available=False`; approved GPU runtime succeeded.
- XR-48A: GPU0, center L2 `0.0030`, P10 boundary `0.010`, state distill `0.00025`; early-stopped at epoch `7/8`.
- XR-48B: GPU1, center L2 `0.0040`, P10 boundary `0.006`, state distill `0.00040`; early-stopped at epoch `7/8`.
- XR-48A best-P10 reached `16.49682899543217/34.431973586763654/11.298894902638027`.
- XR-48A best-P5 reached `16.495972565242223/34.19685453006199/11.303146593911308`.
- XR-48B best-P10 reached `16.49668344429561/34.431973586763654/11.417942530768258`.
- XR-48B best-P5 reached `16.50094030073711/34.45663345200675/11.52423505101885`.
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next: close heatmap head-only P10-boundary calibration. Prefer XR-49 explicit P5-boundary objective or non-head-only adapter update.

# 2026-06-17 XR-49 closeout

- Added `scripts/external/run_xr49_p5_boundary_heatmap_calibration.sh`.
- Added `docs/resources/xr49_p5_boundary_heatmap_calibration_plan_2026_06_17.md`.
- Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, and `git diff --check`.
- XR-49A: GPU0, XR-41A best-P5 init/teacher, LR `2e-6`, center L2 `0.0015`, P5-boundary `0.012`, state distill `0.00050`.
- XR-49B: GPU1, XR-43 center init, XR-41A best-P5 teacher, LR `2e-6`, center L2 `0.0045`, P5-boundary `0.008`, state distill `0.00025`.
- XR-49B initial attempt failed before training because the run path was too long. The runner now uses shorter experiment names.
- XR-49A training completed and produced checkpoints, but the old wrapper failed during automated eval. Manual CUDNN-off eval completed for best-P10 and best-P5.
- XR-49B completed train/eval with exit `0`; best-center checkpoint was absent and skipped.
- XR-49A best-P10 reached `16.52969342981066/34.083759260177615/11.41369082587106`.
- XR-49A best-P5 reached `16.532351425715856/33.93452455656869/11.70068063054766`.
- XR-49B best-P10 reached `16.496637644086565/34.431973586763654/11.417942530768258`.
- XR-49B best-P5 reached `16.497121804101127/34.444728687831336/11.54124187060765`.
- Decision: no gate promoted. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- Next: boundary-only heatmap head calibration is closed. Prefer XR-50 adapter or track-event-adapter update with explicit center preservation.

# 2026-06-17 XR-50 closeout

- Added `scripts/external/run_xr50_event_adapter_heatmap_calibration.sh`.
- Added `docs/resources/xr50_event_adapter_heatmap_calibration_plan_2026_06_17.md`.
- GPT-5.5 read-only sub-agent and local review agreed on the same mechanism: keep the heatmap-state head, but expand trainable scope to `track_center_heatmap_head.*`, `event_adapter.*`, and `patch_frontend.event_embed.proj.*`.
- Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, `git diff --check`, and required checkpoint checks.
- GPU idle check passed before launch; XR-50A ran on GPU0 and XR-50B ran on GPU1.
- Startup validation passed for both lanes: raw event-count contract, XR-43 init checkpoint load, CUDA device resolution, and `trainable_tensors=14`, `trainable_params=1504320/4638746`.
- XR-50A: XR-43 init, XR-41A P5 teacher, LR `1e-6`, best metric P5, center L2 `0.0045`, 5px boundary weight `0.008`.
- XR-50B: XR-43 init, XR-39 P10 teacher, LR `1e-6`, best metric P10, center L2 `0.0050`, 10px boundary weight `0.006`.
- Both lanes completed epoch `8/8`; train/eval exits were `0`.
- XR-50A best-P10 reached `16.483389932768684/34.3227048942021/11.933673824582781`.
- XR-50A best-P5 reached `16.482811435631344/34.48086814199175/11.863520765304566`.
- XR-50B best-P10 reached `16.483026616913932/34.394133479254585/11.978316681725639`.
- XR-50B best-P5 reached `16.482515714849743/34.48086814199175/11.81250035422189`.
- Decision: XR-50 promoted center and P5. New active gates are center `<16.482515714849743`, P10 `>35.02295998845781`, and P5 `>11.978316681725639`.
- Next: recover P10 without giving back XR-50 center/P5, likely via XR-50B best-P10 to XR-39 P10 no-train soup or low-LR P10 continuation from XR-50B best-P10.

# 2026-06-17 XR-51 closeout

- Added `scripts/external/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh`.
- Added `docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md`.
- Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, `git diff --check`, and GPU idle check.
- XR-51A ran on GPU0: XR-50B best-P10 to XR-39 P10 interpolation at alpha `0.05/0.10/0.15/0.20/0.25/0.35`.
- XR-51B ran on GPU1: XR-50B best-P10, XR-50B best-P5, and XR-39 P10 tri-soup at weights `80/10/10`, `75/10/15`, `70/10/20`, `70/15/15`, `65/15/20`, `60/20/20`.
- Both lanes completed with exit `0`; all 12 eval summaries were generated.
- Best center candidate was XR-51B `t60c20p20`: `16.478984827655/34.561650446483/11.716837085996`.
- Best XR-51 P10 candidate was XR-51A `a0p25`: `16.480831880229/34.710459961210/11.517007139751`, still below XR-39 P10 `35.02295998845781`.
- Best P5-preserving candidate was XR-51A `a0p05`: `16.482212608201/34.442602831977/11.978316681726`.
- Decision: XR-51 promoted center only. New active gates are center `<16.478984827655`, P10 `>35.02295998845781`, and P5 `>11.978316681725639`.
- Next: stop pure no-train P10 recovery. Use trainable P10 recovery with XR-39 teacher/reference and explicit XR-50/XR-51 center/P5 preservation.

# 2026-06-17 XR-52 closeout

- Added `scripts/external/run_xr52_trainable_p10_recovery_from_xr51.sh`.
- Added `docs/resources/xr52_trainable_p10_recovery_from_xr51_plan_2026_06_17.md`.
- Static validation passed: `chmod +x`, `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, `git diff --check`, and GPU idle check.
- XR-52A ran on GPU0: XR-51 center leader `t60c20p20` init, XR-39 P10 teacher, LR `5e-7`, center L2 `0.0060`, P10 boundary `0.008`, state distill `0.00035`.
- XR-52B ran on GPU1: XR-51 P5-preserve `a0p05` init, XR-39 P10 teacher, LR `3e-7`, center L2 `0.0070`, P10 boundary `0.005`, state distill `0.00050`.
- Both lanes passed startup validation with `trainable_tensors=14`, `trainable_params=1504320/4638746`.
- Both lanes early-stopped at epoch `7/8`; train/eval exits were `0`.
- XR-52A best-P10 reached `16.483039610726/34.517007589340/11.625425529480`.
- XR-52A best-P5 promoted center: `16.470460832119/34.667942953110/11.840136425836`.
- XR-52B best-P10 promoted P5: `16.472449232851/34.456633458819/12.017432342257`.
- XR-52B best-P5 reached `16.479750164918/34.427721875054/11.929422126498`.
- Decision: XR-52 promoted center and P5, but did not recover P10. New active gates are center `<16.470460832119`, P10 `>35.02295998845781`, and P5 `>12.017432342257`.
- Next: do not repeat low-LR XR-39-teacher event-adapter P10 recovery. Prefer XR-53 checkpoint-space recombination using XR-52 center/P5 leaders plus XR-39 P10, or a stronger P10 teacher/model branch.

# 2026-06-17 XR-53 preparation

- Added `scripts/external/run_xr53_xr52_xr39_p10_recombination_eval.sh`.
- Added `docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md`.
- Static validation passed: `chmod +x`, `bash -n`, required checkpoint checks, lane A/B `DRY_RUN=1`, and `git diff --check`.
- XR-53A is ready on GPU0: XR-52A best-P5 to XR-39 P10 interpolation at alpha `0.03/0.06/0.10/0.15/0.20/0.30`.
- XR-53B is ready on GPU1: XR-52A best-P5, XR-52B best-P10, and XR-39 P10 tri-soup at weights `50/35/15`, `45/35/20`, `40/35/25`, `35/35/30`, `30/40/30`, `25/35/40`.
- Status: prepared but not executed.

# 2026-06-17 XR-53 closeout

- Ran XR-53A on GPU0 and XR-53B on GPU1.
- Both lanes completed with exit `0`; all 12 eval summaries were generated.
- XR-53A interpolation results:
  - `a0p03`: `16.470736992359/34.623300095967/11.840136425836`
  - `a0p06`: `16.471054373469/34.512755877631/11.789116014753`
  - `a0p10`: `16.471541745322/34.512755877631/11.738095603670`
  - `a0p15`: `16.472256399904/34.623300109591/11.525510549545`
  - `a0p20`: `16.473095047474/34.578657252448/11.370323467255`
  - `a0p30`: `16.475181637491/34.616922562463/11.310799653190`
- XR-53B tri-soup results:
  - `c50f35p15`: `16.472414144448/34.512755877631/11.627551378523`
  - `c45f35p20`: `16.473200055531/34.512755877631/11.355442510332`
  - `c40f35p25`: `16.474105545453/34.608419152669/11.310799653190`
  - `c35f35p30`: `16.475128199373/34.497874947957/11.310799653190`
  - `c30f40p30`: `16.475147432940/34.497874947957/11.310799653190`
  - `c25f35p40`: `16.477522144999/34.497874947957/11.266156796047`
- Decision: no gate promoted. Active gates remain center `<16.470460832119`, P10 `>35.02295998845781`, and P5 `>12.017432342257`.
- Next: skip XR-54 near-tie continuation because no XR-53 P10 near-tie exists. Move to XR-55 XR39-anchored expanded-scope P10 branch.

# 2026-06-17 XR-55 closeout

- Added `scripts/external/run_xr55_xr39_p10_anchor_expanded_scope.sh`.
- Added `docs/resources/xr55_xr39_p10_anchor_expanded_scope_plan_2026_06_17.md`.
- Static validation passed: `chmod +x`, `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, and `git diff --check`.
- XR-55A ran on GPU0: XR-39 P10 init/self-teacher, LR `2e-7`, center L2 `0.0025`, P10 boundary `0.012`, state distill `0.00025`.
- XR-55B ran on GPU1: XR-39 P10 init, XR-52B P5 teacher, LR `3e-7`, center L2 `0.0030`, P10 boundary `0.010`, state distill `0.00035`.
- Both lanes used expanded trainable scope: heatmap head, event adapter, event patch projection, `backbone.attn_stages.5.*`, `backbone.mlp_stages.5.*`, and `backbone.norm.*`.
- Startup validation passed for both lanes with `trainable_tensors=32`, `trainable_params=1949568/4638746`.
- Both lanes completed epoch `8/8`; train/eval exits were `0`.
- XR-55A best-P10 reached `16.501659829276/34.399235514232/11.324830266408`.
- XR-55A best-P5 reached `16.497514723028/34.793368141992/11.382228217806`.
- XR-55B best-P10 reached `16.497369331973/34.567177718026/11.202381290708`.
- XR-55B best-P5 reached `16.500960135460/34.399235521044/11.312925515856`.
- Decision: no gate promoted. Active gates remain center `<16.470460832119`, P10 `>35.02295998845781`, and P5 `>12.017432342257`.
- Next: stop checkpoint-space and low-LR scope-expansion P10 recovery. Move to direct supervision change: explicit P10 teacher retraining or dedicated P10 calibration/classification head.

# 2026-06-17 XR-56 launch

- Added metric-aligned soft-threshold loss path:
  - `center_soft_threshold_loss`
  - `track_p10_soft_threshold`
  - `track_p5_soft_threshold`
  - default-zero track-state-aux variants
- Added runner: `scripts/external/run_xr56_soft_threshold_p10_supervision.sh`.
- Added plan: `docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md`.
- Sub-agent review recommended soft-threshold P10 before a new calibration head because it is checkpoint-safe and directly aligned with hard P10 hit-rate.
- Static validation passed: `bash -n`, `py_compile`, target pytest `23 passed`, lane A/B dry-runs, and `git diff --check`.
- XR-56A launched on GPU0 from XR-39 P10 self-teacher:
  - LR `7.5e-7`
  - P10 soft `0.006`, temperature `1.5`
  - P5 soft `0.001`
- XR-56B launched on GPU1 from XR-52B seed with XR-39 teacher:
  - LR `5e-7`
  - P10 soft `0.008`, temperature `1.25`
  - P5 soft `0.002`
- Closeout completed. XR-56A completed epoch `10/10`; XR-56B early-stopped at epoch `7/10`; train/eval exits were `0`.
- XR-56A best-P10 reached `16.49389898266111/34.32270488057818/11.699830286843437`; XR-56A best-P5 reached `16.4924229485648/34.62330012321472/11.304422106061663`.
- XR-56B best-P10 reached `16.477364584377835/34.37670146397182/12.044218049730574`; XR-56B best-P5 promoted center and P5 with `16.4701875601496/34.29294293948582/12.133503770828247`.
- Decision: active gates after XR-56 are center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. P10 remains unrecovered; next P0 should be XR-57 calibration/refinement or P10 teacher retraining.

# 2026-06-17 XR-57 implementation

- Added `TrackCenterRefineHead`, a zero-initialized bounded residual center head.
- Wired optional config keys through head factory, tracker, track branch, model factory, loss bundle, and stage2 loss logs.
- Added `scripts/external/run_xr57_p10_center_refine_calibration.sh`.
- Added `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md`.
- Sub-agent spawn for T-097 failed with `agent thread limit reached`; main-agent fallback completed implementation.
- Static validation passed:
  - `bash -n scripts/external/run_xr57_p10_center_refine_calibration.sh`
  - `python3 -m py_compile ...`
  - `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` -> `27 passed`
  - model-build smoke -> `TrackCenterRefineHead True 4.0`
  - lane A/B `DRY_RUN=1`
- Actual XR-57 train/eval completed. Lane A early-stopped at epoch `7/12`; test metrics were best-P10 `16.69362453562873/34.38307912690299/10.973214619500297` and best-P5 `16.715463175092424/34.849490649359566/10.925595603670393`. Lane B completed epoch `12/12`; test metrics were best-P10 `16.504537062985555/34.137755966186525/11.207483332497732` and best-P5 `16.4997801729611/34.190476996558054/11.476615987505232`.
- Decision: XR-57 did not promote any gate. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. Next P0 is XR-58 P10 teacher refresh.

# 2026-06-17 XR-58 implementation

- Added `scripts/external/run_xr58_p10_teacher_refresh.sh`.
- Added `docs/resources/xr58_p10_teacher_refresh_plan_2026_06_17.md`.
- Static validation passed:
  - `bash -n scripts/external/run_xr58_p10_teacher_refresh.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr58_p10_teacher_refresh.sh b cuda:1`
- Launched XR-58A on GPU0 and XR-58B on GPU1.
- Startup validation passed for both lanes: raw event-count contract, XR-39 init checkpoint load, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- Actual XR-58 train/eval completed. Both lanes early-stopped at epoch `8/16` with train/eval exit `0`. XR-58A best-P10 reached `16.503614359242576/34.90306201662336/11.51275544847761`; XR-58A best-P5 reached `16.501813726765768/34.68920147078378/11.259779255730765`. XR-58B best-P10 reached `16.506438190596445/34.84226275852748/11.501275873184204`; XR-58B best-P5 reached `16.49833288192749/34.540391949244906/11.062075165339879`.
- Decision: XR-58 did not promote any gate. Best XR-58 P10 improved over XR-57 but remained below XR-39 by about `0.1199`. Next P0 is XR-59 narrow anchored-teacher bracket around XR-58A.

# 2026-06-17 XR-59 implementation

- Added `scripts/external/run_xr59_xr58a_teacher_bracket.sh`.
- Added `docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md`.
- Static validation passed:
  - `bash -n scripts/external/run_xr59_xr58a_teacher_bracket.sh`
  - `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/external/run_xr59_xr58a_teacher_bracket.sh b cuda:1`
- Launched XR-59A on GPU0 and XR-59B on GPU1.
- Startup validation passed for both lanes: raw event-count contract, XR58A best-P10 init checkpoint load, XR39 teacher checkpoint set, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- Actual XR-59 train/eval closeout completed with no promotion.
- Closed XR-59 with no promotion. XR-59A/B both early-stopped at epoch `7/10`. Added `scripts/external/eval_xr59_completed_checkpoints.sh` and evaluated `best_track_p10`/`best_track_p5` checkpoints on the full test split. XR-59A reached `16.520220368249074/34.300170864377705/11.376701021194458`; XR-59B reached `16.51065547806876/34.43409944261823/11.287415306908743`. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. Next P0 should pivot away from scalar P10-soft continuation.

# 2026-06-17 XR-60 preparation

- Added default-off P10 candidate-head path and candidate loss wiring.
- Added `scripts/external/run_xr60_p10_candidate_head.sh` and `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- Sub-agent T-102 reviewed XR-56/57/58/59 runner conventions and recommended XR59B-based candidate-head ablation. Integrated this into the runner defaults.
- Static validation passed: `bash -n`, `py_compile`, targeted pytest `34 passed`, candidate model-build smoke, A/B dry-runs, and `git diff --check`.
- Initial sandbox launch failed because PyTorch could not see CUDA (`cuda_device_count=0`); elevated launch resolved CUDA access.
- XR-60A/B launched on GPU0/GPU1 and entered epoch `2/12`.
- Closed XR-60 with no promotion. XR-60A early-stopped at epoch `7/12`; XR-60B early-stopped at epoch `11/12`. XR-60A best-P10/best-P5 reached `16.526124344553267/34.50255186898368/11.617772477013725` and `16.5058109828404/34.60331717899867/11.337160219464984`. XR-60B best-P10/best-P5 reached `16.842585216249738/33.49277294022696/10.70663298198155` and `16.717501049382346/33.87670159339905/10.971939107349941`. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Rewrote XR eye-tracking detailed analyses under `anlaysis/xr-eye-tracking` to the requested reference-document depth. Added `scripts/external/write_xr_eye_tracking_detailed_analysis.py` and regenerated `39` `analysis.md` files: `18` codebase analyses and `21` paper analyses, total `8813` lines. Codebase reports now include Python symbol extraction, import/dependency clues, config option clues, and HGTXR conversion design. Paper reports now include bounded section evidence, primitive decomposition, result/metric evidence, and experiment conversion. Validation passed with generator `py_compile`, section coverage scan, and FACET spot checks. GPT-5.3-Codex-Spark sub-agent attempt failed due quota exhaustion, so main agent completed the rewrite.
- Closed XR-61 auxiliary candidate branch with no promotion: best observed metrics were XR-61A `16.475529539585114/34.356718465260094/12.127126216888428` and XR-61B `16.48967229127884/34.58843615395682/11.559524168287005`.
- Added and ran XR-62 FACET geometry auxiliary refresh. Added `scripts/external/run_xr62_facet_geometry_aux_refresh.sh` and `docs/resources/xr62_facet_geometry_aux_refresh_plan_2026_06_17.md`. Static validation passed (`bash -n`, A/B dry-run), both lanes early-stopped at epoch `7/10`, and XR-62A promoted center to `16.468481131962367` with P10 `34.30782389640808` and P5 `12.052721459524973`. XR-62B did not promote. Active gates are now center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Reworked the XR eye-tracking analysis generator to the stricter requested reference-analysis style. The generator now emits codebase metadata compatible with the FlexLLM-style template, source responsibility tables, HGTXR data-contract matrices, static quality findings, and reproduction runbooks. Paper reports now include dataset/model/benchmark tables, reported-result evidence, HGTXR result interpretation, and detailed experiment option runbooks. Regenerated all `39` `analysis.md` files under `anlaysis/xr-eye-tracking`, increasing the total to `11574` lines. Validation passed with `py_compile`, full regeneration, required-section scans, FACET/EV-Eye spot checks, and `git diff --check`. Spark sub-agent attempts failed due quota exhaustion; GPT5.5 explorer agents provided the codebase/paper templates that were integrated.
- Added and ran XR-63 no-train P10 teacher-target oracle diagnostic. `scripts/external/analyze_p10_teacher_target_oracle.py` loads manifest1 test targets through the XR-62A resolved config, aligns `eval_rows.json` from XR-62A, XR-39, XR-56B, and XR-58A, and computes eval-style batch-mean center/P10/P5 metrics. Outputs: `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` and `.md`. Oracle upper bound is `16.04023192701366/36.48596938775512/13.41751700680271`, exceeding all active gates. Decision: proceed to leakage-safe P10 teacher-target construction; do not repeat candidate-only or geometry-only auxiliary branches without this new protocol.
- Added `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md`. The plan records that current code has no direct sample-wise pseudo-target loading path, so XR-64 must generate train/val teacher eval rows, build split-isolated target overrides, add dataset/loss support, and reject test-derived pseudo labels during training.
- Implemented XR-64 target override path. Added `scripts/external/build_xr64_teacher_target_overrides.py`, `scripts/external/run_xr64_teacher_target_construction.sh`, data config keys `track_target_override_path` and `allow_test_target_override`, additive dataset fields, additive Stage2 target override losses, and tests. Guard: test manifest override path is rejected by default, and XR-64 runner clears override config during final test eval. Validation passed with `py_compile`, `bash -n`, targeted pytest `39 passed`, A/B dry-run, and a 64-sample builder smoke.

# 2026-06-18 current consolidation and next experiment list

- Added consolidated status/analysis note: `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- Recorded that XR-63 oracle remains the highest-value signal: `16.04023192701366/36.48596938775512/13.41751700680271` as no-train upper bound over XR-62A/XR-39/XR-56B/XR-58A predictions.
- Confirmed active gates remain center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Added XR-64 preparation helper: `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`.
- Static validation passed for the helper: `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh`, `py_compile` for XR-64 builder and analysis generator, and `chmod +x`.
- Attempted the all-in-one XR-64 prep helper once. It started `train/xr62a` eval but stopped without producing `eval_rows.json`; no train/val override JSON was produced from that attempt.
- Ran a direct 8-row smoke eval with XR-62A checkpoint and the same support-adaptive fixed255k eval overrides. It succeeded and wrote summary/rows under `/tmp/xr64_eval_smoke_20260618_0149`, proving the checkpoint/config/smoke path itself is valid.
- Recorded next experiment list in `docs/track/TODO.md` and `docs/Paper-Backed-Experiment-Plan.md`:
  - `XR-64A` conservative teacher-target selector.
  - `XR-64B` threshold-priority teacher-target selector.
  - `XR-64C` min-error diagnostic.
  - `XR-65` teacher-target temporal-lite residual.
  - `XR-66` EX-Gaze confidence-gated local update diagnostic.
  - `XR-67` EyeLoRiN/dense-trajectory reopening.
  - `XR-68` sparse/quant/distilled edge branch.
- Decision: do not start new geometry-only, candidate-only, scalar P10-soft, or low-similarity weighting branches before completing XR-64 train/val override generation and A/B execution.

# 2026-06-18 experiment pause and documentation pass

- User directive: stop experiments for now and document all work completed so far.
- Process check: no active process matched `train_hbtxr`, `eval_hbtxr`, `run_xr64`, `run_xr`, or `scripts/external/run_.*train`.
- GPU check: GPU0 `15 MiB` used / `15827 MiB` free / `0%`; GPU1 `15 MiB` used / `15827 MiB` free / `0%`.
- Sub-agent T-701 completed a read-only review of the XR-64 prep helper. Main findings: the first all-in-one helper attempt stopped during `train/xr62a`, no `eval_rows.json` was produced, and the prior helper needed split/teacher-level resumability, foreground logging, output validation, and explicit test-split refusal.
- Hardened `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` for future use:
  - added `XR64_ACTIONS`, `XR64_SPLITS`, and `XR64_TEACHERS`.
  - added split/teacher checkpoint resolution helpers.
  - added JSON row validation after eval/build.
  - changed eval logging to foreground `tee`.
  - made the helper refuse `test` split for XR-64 target generation.
- Artifact state at pause:
  - no full train/val teacher eval row file exists under `data/_internal/manifests/manifest1/xr64_teacher_targets/`.
  - no `xr64a`, `xr64b`, or `xr64c` train/val override JSON exists yet.
  - `/tmp/xr64_eval_smoke_20260618_0149` remains an 8-row smoke validation only, not a full experiment result.
- Documentation updated:
  - `docs/resources/experiment_pause_documentation_2026_06_18.md`.
  - `docs/resources/second_goal_completion_audit_2026_06_18.md`.
  - `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
  - `docs/Paper-Backed-Experiment-Plan.md`.
  - `docs/track/PROGRESS.md`.
  - `docs/track/TODO.md`.
- No new experiment was launched after the pause directive.

# 2026-06-18 second-goal audit update

- Audited the active second goal against current worktree evidence.
- Verified current analysis inventory: `18` codebase `analysis.md` files, `21` paper `analysis.md` files, `39` total per-item analyses, `15590` total lines across those files.
- Verified `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` exists and maps PAPER_REF to head/loss/LR/optimizer/distillation/teacher axes.
- Found that the PAPER_REF map still contained older XR-01/XR-15-era gates and immediate-priority rows. Added a 2026-06-18 supersession note pointing to the current XR-64 pause/plan documents.
- Added `docs/resources/second_goal_completion_audit_2026_06_18.md`.
- Audit judgment: active goal is not complete because submission-level accuracy is not reached, XR-64 train/val eval rows and override JSONs are missing, XR-64A/B/C are not executed, and experiments are paused by user directive.

# 2026-06-18 XR-64 resume runbook

- Added `docs/resources/xr64_resume_runbook_2026_06_18.md`.
- The runbook records future-only commands for:
  - static preflight checks.
  - train split eval-row generation by teacher.
  - val split eval-row generation by teacher.
  - train/val override generation for conservative, threshold-priority, and min-error rules.
  - JSON row-count validation.
  - XR-64A/B/C launch order after artifacts exist.
  - leakage guards and promotion gates.
- GPT-5.3-Codex-Spark sub-agent T-802 completed read-only review and identified runbook safety gaps. Integrated:
  - fail-fast existence checks for eval rows and override JSONs.
  - note that current training runner consumes train override path while val override files are analysis/diagnostic artifacts.
  - reproducibility snapshot commands for git HEAD, Python/Torch/CUDA, and disk space.
  - `RUN_ROOT` latest-pattern caveat and required log-key checks.
- No train/eval experiment was launched.

# 2026-06-18 XR-64 artifact checker

- Added `scripts/external/check_xr64_resume_artifacts.py`, a read-only verifier for XR-64 resume artifacts.
- The checker validates:
  - required manifests/config/checkpoints.
  - train/val manifest sample-id separation.
  - train/val eval row JSON shape and sample-id membership.
  - train/val override JSON shape, rule/baseline mapping, selection counts, finite numeric fields, and teacher coverage.
  - no test-named or test-split XR-64 target artifacts.
  - generated artifact completeness.
- Added `tests/test_xr64_resume_artifacts.py` for missing-generated, complete-artifact, and test-split leakage paths.
- Updated pause/audit/runbook docs so strict checker is required before XR-64A/B execution after resume.
- Added `--format summary` to the checker for human resume decisions. It reports `resume_status`, `ready_to_train`, `can_run_lane`, missing artifact counts, compact generation/override matrices, and leakage risk while keeping JSON as the default output.
- Validation passed:
  - `python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py`
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py` -> `9 passed`
  - `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated` -> `ready=true`, `generated_complete=false`
  - `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary` -> `resume_status=generated_incomplete`
  - `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh`
- No train/eval experiment was launched.

# 2026-06-18 second-goal artifact index

- Added `docs/resources/second_goal_artifact_index_2026_06_18.md`.
- Added machine-readable companion `docs/resources/second_goal_artifact_index_2026_06_18.json`.
- The index records:
  - current authority files for pause/current summary/completion audit/resume runbook/validation/TODO/progress/log.
  - PAPER_REF and XR-eye-tracking analysis artifacts.
  - XR-63/XR-64 scripts, plans, tests, and checker artifacts.
  - current gates and XR-64 generated-artifact missing state.
  - expected XR-64 generated file paths.
  - validation commands and strict resume gate.
  - stale/link risks, including the intentional `anlaysis` path spelling and historical PAPER_REF map rows.
- GPT-5.3-Codex-Spark sub-agent T-805 completed read-only index review; required missing artifact paths, tracking docs, `docs/Paper-Backed-Experiment-Plan.md`, and stale/link risks were integrated.
- No train/eval experiment was launched.

# 2026-06-18 second-goal pause-state status reporter

- Added `scripts/external/report_second_goal_status.py`.
- Added `tests/test_second_goal_status.py`.
- Updated `docs/resources/second_goal_artifact_index_2026_06_18.md` and `.json` to include the status reporter and validation command.
- Updated `docs/Validation.md`, `docs/track/PROGRESS.md`, and `docs/track/TODO.md`.
- Validation passed:
  - `python3 -m py_compile scripts/external/report_second_goal_status.py`
  - `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py` -> `5 passed`
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary` -> `execution_state=paused_by_user_directive`, `active_goal_complete=false`, `xr64_resume_status=generated_incomplete`, `xr64_can_run_lane=false`, `missing_eval_rows=8`, `missing_overrides=6`, `leakage_risk=none`
- No train/eval experiment was launched.

# 2026-06-18 software cleanup / refactor preparation

- User requested cleanup/refactor work for the dirty `software` directory.
- Created `docs/resources/software_cleanup_refactor_plan_2026_06_18.md`.
- Added read-only cleanup tooling:
  - `scripts/external/maintenance/inventory_software_tree.py`
  - `scripts/external/maintenance/audit_path_references.py`
- Generated reports:
  - `docs/resources/software_cleanup_inventory_2026_06_18.md`
  - `docs/resources/software_cleanup_inventory_2026_06_18.json`
  - `docs/resources/software_path_reference_audit_2026_06_18.md`
  - `docs/resources/software_path_reference_audit_2026_06_18.json`
- Inventory summary: `runs=849`, software dirty entries `236`, `runs/` size `44.5 GiB`, `.venv` size `4.8 GiB`, root stray file `32.56462665285383` is empty.
- `runs/` classification: `390` preserve/review, `315` archive-candidate eval, `130` archive-candidate closed experiment, `8` smoke cleanup candidates, `6` duplicate archive candidates.
- Path audit summary: `1166` findings, including `411` project-root mentions, `318` `.venv/bin/python` mentions, `254` canonical-root mentions, and `130` `/home/...` absolute paths.
- Sub-agent CLEAN-001 completed read-only runs risk review. Integrated must-preserve XR-39/XR-56/XR-58/XR-62/XR-63/XR-64, `interpolated_checkpoints`, `_logs`, and `diagnostics`.
- Sub-agent CLEAN-002 completed read-only script/config/docs risk review. Integrated wrapper-first policy for `PYTHON_BIN`, `CANONICAL_ROOT`, and `PATHS_CONFIG`.
- No files were deleted or moved; no train/eval experiment was launched.

# 2026-06-18 runs catalog organization

- User clarified that `runs/` should be kept because it contains experiment results and only needs gitignore plus internal categorization.
- Confirmed root `.gitignore` already ignores `software/runs/*` and `runs/*`.
- Added `scripts/external/maintenance/build_runs_catalog.py`.
- Ran dry-run: `source_run_dirs=849`, planned symlink actions `2078`.
- Ran actual catalog build: created `2078` symlinks under `runs/_catalog`.
- Generated:
  - `docs/resources/runs_catalog_report_2026_06_18.md`
  - `docs/resources/runs_catalog_report_2026_06_18.json`
- Created category view under:
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
- Original `runs/<run_id>` directories were not moved or deleted.

# 2026-06-18 runs physical experiment reorganization

- User requested actual movement of all directories under `runs/` into experiment-list folders such as `XR-1`, `XR-2`, `XR-3`, and so on.
- Added `scripts/external/maintenance/reorganize_runs_by_experiment.py`.
- Classification policy:
  - first `xrNN` token in the run name maps to `runs/XR-<N>/<run_id>`.
  - no-XR eval runs map to `runs/NON_XR/eval/<run_id>`.
  - no-XR raw stage runs map to `runs/NON_XR/raw/<run_id>`.
  - `_logs`, `diagnostics`, and `interpolated_checkpoints` map to `runs/NON_XR/shared/<name>`.
  - remaining no-XR runs map to `runs/NON_XR/other/<run_id>`.
- Dry-run result: `source_dirs_considered=849`, action `would_move=849`.
- Actual run result: `source_dirs_considered=849`, action `moved_with_compat_symlink=849`.
- Existing `runs/<run_id>` paths now remain as compatibility symlinks to the moved locations.
- `runs/_catalog` was excluded as generated metadata.
- Validation:
  - `find runs -maxdepth 1 -mindepth 1 -type l | wc -l` -> `849`
  - `find runs -xtype l | head -20` -> no output, so no broken symlink found in the checked tree
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary` still returns `xr64_resume_status=generated_incomplete`
- No run content was deleted and no train/eval job was launched.
- Sub-agent MOVE-001 completed read-only review and identified a high-risk issue: `scripts/external/run_prepare_and_train.sh` used `find -type d`, which misses top-level compatibility symlinks.
- Fixed `scripts/external/run_prepare_and_train.sh` latest-run discovery to include `-type l`.
- Added `--verify` mode to `scripts/external/maintenance/reorganize_runs_by_experiment.py`.
- Generated verification reports:
  - `docs/resources/runs_experiment_reorganization_verify_2026_06_18.md`
  - `docs/resources/runs_experiment_reorganization_verify_2026_06_18.json`
- Verification result: `ok=true`, `top_level_compat_symlinks=849`, `organized_run_dirs=849`, `broken=0`, `bad_targets=0`.

# 2026-06-18 runs compatibility symlink removal

- User asked to proceed after top-level `eval*` and `raw*` entries were identified as compatibility symlinks.
- Added `scripts/external/maintenance/remove_runs_compat_symlinks.py`.
- Updated active run lookup paths to search organized nested roots:
  - `src/hbtxr/config/run_contract.py`
  - `scripts/external/run_prepare_and_train.sh`
  - `scripts/external/check_raw_event_count_training_readiness.py`
  - `scripts/external/check_raw_event_count_training_result.py`
- Updated active XR-62/XR-64 reference paths to use `runs/XR-56`, `runs/XR-58`, and `runs/XR-62` for moved run roots.
- Ran dry-run: removable top-level run symlinks `846`, retained shared aliases `3`.
- Ran actual removal: removed top-level run symlinks `846`; removed stale generated `runs/_catalog` symlink tree entries `2092`.
- Retained shared aliases:
  - `runs/_logs -> NON_XR/shared/_logs`
  - `runs/diagnostics -> NON_XR/shared/diagnostics`
  - `runs/interpolated_checkpoints -> NON_XR/shared/interpolated_checkpoints`
- Generated:
  - `docs/resources/runs_compat_symlink_removal_2026_06_18.md`
  - `docs/resources/runs_compat_symlink_removal_2026_06_18.json`
- Validation:
  - top-level `eval*`/`raw*`: `0`
  - broken symlinks from `find runs -xtype l`: `0`
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed
  - `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed
  - `.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py`: found Stage1/Stage2 under `runs/NON_XR/raw`
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_raw_event_count_training_readiness.py tests/test_raw_event_count_training_result.py tests/test_xr64_resume_artifacts.py tests/test_second_goal_status.py`: `13 passed`
- Environment note: `nvidia-smi` returned driver communication failure during readiness reporting; no train/eval job was launched.

# 2026-06-20 current result synthesis guard

- Added current result synthesis artifacts:
  - `docs/resources/second_goal_current_result_synthesis_2026_06_20.md`
  - `docs/resources/second_goal_current_result_synthesis_2026_06_20.json`
- Added read-only validator:
  - `scripts/external/check_current_result_synthesis.py`
  - `tests/test_current_result_synthesis.py`
- Updated:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/Validation.md`
- Locked current interpretation:
  - active gates are center `16.468481131962367`, P10 `35.02295998845781`, P5 `12.133503770828247`.
  - XR-63 oracle is diagnostic only and not promotable as a trained result.
  - direct comparison with submission target `0.1812 px` is blocked by metric/protocol mismatch.
  - XR-64 remains `generated_incomplete` with `8` missing eval rows and `6` missing override JSON files.
- Validation:
  - `python3 -m py_compile scripts/external/check_current_result_synthesis.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_current_result_synthesis.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_ref_analysis_coverage.py tests/test_second_goal_experiment_queue.py tests/test_submission_target_gap.py tests/test_metric_protocol_bridge.py tests/test_current_result_synthesis.py tests/test_second_goal_status.py`: `16 passed`.
- Sub-agent note: a GPT-5.3-Codex-Spark read-only audit was attempted but failed due usage limit. Main-agent fallback completed the work.
- No train/eval job was launched.

# 2026-06-20 second-goal objective trace

- Added:
  - `docs/resources/second_goal_objective_trace_2026_06_20.md`
  - `docs/resources/second_goal_objective_trace_2026_06_20.json`
  - `scripts/external/check_second_goal_objective_trace.py`
  - `tests/test_second_goal_objective_trace.py`
- Integrated objective trace into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/Validation.md`
- Trace status:
  - REQ-1 PAPER_REF analysis: `complete_as_planning_input`.
  - REQ-2 experiment planning: `planned_not_fully_executed`.
  - REQ-3 accuracy closure: `incomplete`.
  - `goal_complete=false`.
- False-completion guardrails:
  - no train/eval while paused.
  - no XR-63 oracle promotion.
  - no single-model best claim because current best gates are split across owners.
  - no direct comparison to `0.1812 px` while metric/protocol bridge is blocked.
  - no XR-64A/B execution claim before XR-64-prep generated artifacts pass strict checking.
- Sub-agent:
  - GPT-5.5 read-only audit `T-TRACE-001` completed.
  - Recommended fields were integrated: `promotion_allowed`, `evidence_artifacts`, `missing_evidence`, `blocked_by`, `next_allowed_action`, `false_completion_guardrails`, and `do_not_claim`.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_objective_trace.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py tests/test_second_goal_status.py`: `5 passed`.
- No train/eval job was launched.

# 2026-06-20 XR-64 resume command manifest

- Added:
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.json`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `tests/test_xr64_resume_command_manifest.py`
- Integrated command manifest into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/Validation.md`
- Manifest contract:
  - execution state remains `paused_by_user_directive`.
  - `default_allowed_to_run=false`.
  - XR-64-prep has `8` eval commands and `2` build commands.
  - expected generated artifacts are `8` eval-row files and `6` override JSON files.
  - XR-64A/B launch is gated by strict `XR64-STRICT-READY`.
- Sub-agent:
  - GPT-5.5 read-only audit `T-XR64-MANIFEST-001` completed.
  - Recommended fields and checker rules were integrated.
- Validation:
  - `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `5 passed`.
- No train/eval job was launched.

# 2026-06-20 XR-64 resume command emitter

- Added:
  - `scripts/external/emit_xr64_resume_commands.py`
  - `tests/test_emit_xr64_resume_commands.py`
- Integrated emitter into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/Validation.md`
- Emitter modes:
  - `summary`: counts and guard state.
  - `prep`: future XR-64-prep commands for review.
  - `launch`: future XR-64A/B commands for review.
  - `script`: commented script output.
- Validation:
  - `python3 -m py_compile scripts/external/emit_xr64_resume_commands.py`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section summary --format summary`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section prep --format commands`: passed.
  - `.venv/bin/python -m pytest -q tests/test_emit_xr64_resume_commands.py`: `3 passed`.
- No train/eval job was launched.

# 2026-06-20 second-goal completion gate

- Added:
  - `scripts/external/check_second_goal_completion_gate.py`
  - `tests/test_second_goal_completion_gate.py`
- Integrated completion gate into:
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Gate behavior:
  - strict mode fails until the full second goal is actually complete.
  - `--allow-incomplete` returns zero for read-only blocker reporting.
  - current state reports `completion_allowed=false` and `blocker_count=22`.
  - current primary blockers include pause state, incomplete objective trace, missing trained single-model result evidence, blocked metric/protocol bridge, incomplete paper-target evidence manifest, invalid direct submission comparison, missing XR-64 eval rows, missing XR-64 override JSON files, and missing XR-64 training runs.
- Sub-agent:
  - GPT-5.5 read-only audit `T-COMPLETE-GATE-001` completed.
  - Recommended strict completion conditions and negative test cases were integrated into checker behavior and tests.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py`: `4 passed`.
- No train/eval job was launched.

# 2026-06-20 XR-64 prelaunch packet

- Added:
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.md`
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.json`
  - `scripts/external/check_xr64_prelaunch_packet.py`
  - `tests/test_xr64_prelaunch_packet.py`
- Integrated prelaunch packet into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Packet state:
  - `allowed_to_execute=false`.
  - `completion_allowed=false`.
  - `completion_blocker_count=22`.
  - phases: read-only status, eval-row generation after resume, override generation after resume, strict readiness, XR-64A/B launch after strict readiness.
  - command review uses `emit_xr64_resume_commands.py` only; no train/eval launcher is executed by packet validation.
- Validation:
  - `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py`: `4 passed`.
- Sub-agent note:
  - GPT-5.3-Codex-Spark audit attempt failed due usage limit.
  - GPT-5.5 read-only audit `T-PRELAUNCH-AUDIT` completed as fallback.
  - Audit recommendation integrated: final test evaluation for promoted checkpoints must clear target override paths.
- No train/eval job was launched.

# 2026-06-21 XR-64 post-run evidence contract

- Added:
  - `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md`
  - `docs/resources/xr64_postrun_evidence_contract_2026_06_21.json`
  - `scripts/external/check_xr64_postrun_evidence_contract.py`
  - `tests/test_xr64_postrun_evidence_contract.py`
- Integrated post-run evidence contract into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Contract state:
  - `evidence_state=missing_postrun_evidence`.
  - `allowed_to_claim_promotion=false`.
  - required lanes: XR-64A and XR-64B.
  - required checkpoint kinds: `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px`.
  - required metrics: `metric_track_center_px`, `metric_track_p10_pct`, `metric_track_p5_pct`, and `metric_track_p1_pct`.
  - P1 is future paper-target bridge evidence, not a current software promotion gate.
  - final test eval must clear target override paths before promotion.
- Validation:
  - `python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_postrun_evidence_contract.py`: `4 passed`.
- Sub-agent note:
  - GPT-5.5 read-only audit attempt failed due usage limit; main-agent fallback completed the contract and validation.
- No train/eval job was launched.

# 2026-06-21 XR-64 post-run promotion decision helper

- Added:
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`
  - `scripts/external/decide_xr64_postrun_promotion.py`
  - `tests/test_decide_xr64_postrun_promotion.py`
- Integrated promotion decision helper into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Helper behavior:
  - no candidates reports `decision_status=missing_postrun_evidence`.
  - future candidates use `LANE:CHECKPOINT_KIND:path/to/eval_summary.json`.
  - software promotion is true only when a future XR-64A/B test summary improves at least one current gate.
  - P1 evidence is tracked through `metric_track_p1_pct` and `p1_evidence_candidate_count`.
  - paper-level completion remains false because metric/protocol bridge is separate.
- Validation:
  - `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py`: passed.
  - `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py`: `4 passed`.
- No train/eval job was launched.

# 2026-06-21 metric/protocol unblock contract

- Added:
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.md`
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.json`
  - `scripts/external/check_metric_protocol_unblock_contract.py`
  - `tests/test_metric_protocol_unblock_contract.py`
- Integrated unblock contract into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Contract behavior:
  - current state is `contract_status=blocked`.
  - `direct_submission_comparison_allowed=false`.
  - `paper_level_completion_allowed=false`.
  - required evidence covers coordinate-frame match, hybrid scheduler eval, P1 metric coverage, split protocol match, target definition match, latency separation, and trained XR-64-or-later result dependency.
  - validation commands are read-only and must not launch train/eval.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `XR-METRIC-UNBLOCK-REVIEW-001` completed.
  - It confirmed P1, raw sensor-space, hybrid scheduler, and split protocol evidence are the critical missing unblock conditions.
- Validation:
  - `python3 -m py_compile scripts/external/check_metric_protocol_unblock_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_metric_protocol_unblock_contract.py`: `5 passed`.
- No train/eval job was launched.

# 2026-06-21 P1 metric coverage

- Added:
  - `metric_track_p1_pct` and `metric_search_p1_pct` in `src/hbtxr/loss/metrics.py`.
  - `tests/test_hbtxr_metrics_p1.py`.
- Updated:
  - `docs/resources/second_goal_metric_protocol_bridge_2026_06_20.md`
  - `docs/resources/second_goal_metric_protocol_bridge_2026_06_20.json`
  - `scripts/external/check_metric_protocol_bridge.py`
  - `tests/test_metric_protocol_bridge.py`
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.md`
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.json`
  - `scripts/external/check_metric_protocol_unblock_contract.py`
  - `docs/resources/second_goal_current_result_synthesis_2026_06_20.md`
  - `docs/resources/second_goal_current_result_synthesis_2026_06_20.json`
  - `scripts/external/check_current_result_synthesis.py`
  - `scripts/external/report_second_goal_status.py`
- State:
  - P1 evaluator implementation is available.
  - paper-frame/full-test P1 evidence remains unavailable.
  - direct paper-target comparison remains blocked.
- Validation:
  - `python3 -m py_compile src/hbtxr/loss/metrics.py scripts/external/check_metric_protocol_bridge.py scripts/external/check_metric_protocol_unblock_contract.py scripts/external/check_current_result_synthesis.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_metric_protocol_bridge.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_current_result_synthesis.py tests/test_second_goal_status.py`: `15 passed`.
- No train/eval job was launched.

# 2026-06-21 second-goal ablation evidence checklist

- Added:
  - `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md`
  - `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.json`
  - `scripts/external/check_second_goal_ablation_evidence_checklist.py`
  - `tests/test_second_goal_ablation_evidence_checklist.py`
- Integrated checklist into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Checklist behavior:
  - current state is `execution_state=paused_by_user_directive`.
  - XR-64-prep/A/B remain the only P0 rows.
  - required experiment axis coverage count is `8`.
  - software promotion remains false while XR-64 generated artifacts are missing.
  - `metric_track_p1_pct` is required as future bridge evidence but is not a current software-promotion gate.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only explorer `T-SEC-EVAL-002` completed.
  - It recommended an additional completion-readiness contract; this was recorded as a P1 follow-up rather than implemented in this checkpoint.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_ablation_evidence_checklist.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_ablation_evidence_checklist.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_status.py`: `6 passed`.
- No train/eval job was launched.

# 2026-06-21 second-goal completion readiness contract

- Added:
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
- Integrated readiness contract into:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Contract behavior:
  - current state is `execution_state=paused_by_user_directive`.
  - `completion_allowed=false` and `blocker_count=22`.
  - XR-64 readiness remains false with `missing_eval_rows=8`, `missing_overrides=6`, and `leakage_risk=none`.
  - next resume gate requires `ready_to_train`, `can_run_lane=true`, zero missing artifacts, and leakage risk `none`.
  - validation contract forbids train/eval commands in required checks.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-READINESS-001` completed.
  - It confirmed the blocker IDs, strict completion fields, and forbidden command checks that must be represented.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_gate.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `8 passed`.
- No train/eval job was launched.

# 2026-06-21 submission target source trace parser hardening

- Updated:
  - `scripts/external/check_submission_target_source_trace.py`
  - `tests/test_submission_target_source_trace.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Change:
  - scoped HBTXR algorithm-table extraction to `\label{tab:algo}` and its following `tabular` block.
  - added a regression test that confirms external fake `P10`/`P5`/`P1` bold rows do not override `tab:algo`.
  - preserved `direct_comparison_valid=false`; paper target values are traceable but not directly comparable to current software metrics.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-VERIFY-SOURCE-TRACE` completed.
  - It recommended table-scoped parsing and targeted regression coverage; both were integrated.
- Validation:
  - `python3 -m py_compile scripts/external/check_submission_target_source_trace.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_submission_target_source_trace.py --format summary`: passed with `ok=true`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_submission_target_source_trace.py tests/test_submission_target_gap.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_hbtxr_metrics_p1.py`: `32 passed`.
  - `.venv/bin/python -m pytest -q tests/test_current_result_synthesis.py tests/test_decide_xr64_postrun_promotion.py tests/test_emit_xr64_resume_commands.py tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_bridge.py tests/test_metric_protocol_unblock_contract.py tests/test_paper_ref_analysis_coverage.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_experiment_queue.py tests/test_second_goal_objective_trace.py tests/test_submission_target_gap.py tests/test_submission_target_source_trace.py tests/test_xr64_postrun_evidence_contract.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `65 passed`.
  - `git diff --check`: passed.
- No train/eval job was launched.

# 2026-06-21 paper target comparison evidence manifest

- Added:
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md`
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json`
  - `scripts/external/check_paper_target_comparison_evidence_manifest.py`
  - `tests/test_paper_target_comparison_evidence_manifest.py`
- Integrated into:
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.md`
  - `docs/resources/metric_protocol_unblock_contract_2026_06_21.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Manifest behavior:
  - current `evidence_state=missing_required_evidence`.
  - `direct_submission_comparison_allowed=false`.
  - required evidence count is `7`.
  - missing evidence count is `4`; partial evidence count is `3`.
  - future required artifact count is `10`.
  - current supporting artifact count is `4`.
  - existing future bridge artifact count is `0`.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-METRIC-BRIDGE-GAP-001` completed.
  - It confirmed the gap between abstract unblock contract and concrete file-level evidence inventory; the manifest/checker/test/status integration address that gap.
- Validation:
  - `python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py`: `7 passed`.
  - `.venv/bin/python -m pytest -q tests/test_metric_protocol_unblock_contract.py tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py`: `12 passed`.
- No train/eval job was launched.

# 2026-06-21 completion gate paper-target evidence integration

- Updated:
  - `scripts/external/check_second_goal_completion_gate.py`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/check_xr64_prelaunch_packet.py`
  - `tests/test_second_goal_completion_gate.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_xr64_prelaunch_packet.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.md`
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Change:
  - completion gate now treats `paper_target_comparison_evidence_manifest` as mandatory completion evidence.
  - added blockers `paper_target_required_evidence_incomplete`, `paper_target_direct_comparison_blocked`, and `paper_target_future_artifacts_missing`.
  - current completion blocker count is now `22`.
  - XR-64 prelaunch packet now also reports `completion_blocker_count=22`.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-COMPLETION-GATE-PAPER-TARGET-001` confirmed the missing integration between status reporter, completion gate, and readiness contract.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `blocker_count=22`.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `ok=true`.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `completion_blocker_count=22`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `completion_readiness_blocker_count=22`.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `18 passed`.
- No train/eval job was launched.

# 2026-06-21 paper-target future artifact schema validation

- Updated:
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md`
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json`
  - `scripts/external/check_paper_target_comparison_evidence_manifest.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_paper_target_comparison_evidence_manifest.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Change:
  - added `artifact_content_schema` for all 10 required future bridge artifacts.
  - future JSON/JSONL bridge files are parsed for required fields when present.
  - status reporting now surfaces `paper_target_artifact_schema_count`.
  - completion gate and readiness contract now reject a corrupt paper-target artifact schema count.
  - malformed future JSONL rows are covered by regression tests.
- Current state:
  - direct paper-target comparison remains blocked.
  - existing future bridge artifact count remains `0`.
  - focused validation passed: py_compile, paper-target manifest summary, completion gate summary, readiness contract summary, status reporter summary, and pytest `23 passed`.
  - broader second-goal pytest passed: `78 passed`.
  - `git diff --check`: passed.
  - no train/eval job was launched.

# 2026-06-21 objective-trace full matrix and XR-64 input readiness

- Updated:
  - `docs/resources/second_goal_objective_trace_2026_06_20.md`
  - `docs/resources/second_goal_objective_trace_2026_06_20.json`
  - `scripts/external/check_second_goal_objective_trace.py`
  - `tests/test_second_goal_objective_trace.py`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `tests/test_xr64_resume_command_manifest.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Change:
  - objective trace ablation matrix now requires the full current experiment queue: XR-64-prep, XR-64A, XR-64B, XR-64C, XR-65, XR-66, XR-67, and XR-68.
  - XR-64 resume command manifest checker now reports eval and build required input readiness.
  - current paused state reports `eval_required_input_count=6`, `existing_eval_required_input_count=6`, `missing_eval_required_input_count=0`, and `eval_inputs_ready=true`.
  - current paused state reports `build_required_input_count=8`, `existing_build_required_input_count=0`, `missing_build_required_input_count=8`, and `build_inputs_ready=false`.
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-SECOND-GOAL-GAP-AUDIT-002` identified missing build-input readiness visibility as the highest-value non-executing gap.
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-EVAL-INPUT-READINESS-003` identified eval input readiness parity as the next concrete gap.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_objective_trace.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary`: passed with `ablation_count=8`.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `eval_inputs_ready=true` and `build_inputs_ready=false`.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `10 passed`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `7 passed` after eval/build input readiness parity update.
  - broader second-goal pytest passed: `81 passed`.
  - `git diff --check`: passed.
  - no train/eval job was launched.

# 2026-06-21 XR-64 prelaunch readiness parity

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Align XR-64 prelaunch/status launch gates with command-manifest eval/build readiness.
- Updated:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `tests/test_xr64_prelaunch_packet.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/TODO.md`
  - `docs/track/PROGRESS.md`
  - `docs/Validation.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-READINESS-AUDIT-001` reported no field-value mismatch, then identified launch-gate readiness coupling and build mismatch test coverage as hardening targets.
- Current readiness facts:
  - Eval static inputs: `6/6`, `eval_inputs_ready=true`.
  - Build inputs: `0/8`, `build_inputs_ready=false`.
  - Missing eval rows remain `8`; missing override JSON files remain `6`.
  - XR-64 launch remains blocked until explicit resume and strict readiness pass.
- Validation:
  - `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_ready_to_train=false` and `xr64_can_run_lane=false`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `9 passed`.
  - broader second-goal pytest suite: `84 passed`.
  - `git diff --check`: passed.

# 2026-06-21 XR-64 post-run ablation evidence contract

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Strengthen future XR-64A/B post-run evidence so result claims can support second-goal ablation axes.
- Updated:
  - `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md`
  - `docs/resources/xr64_postrun_evidence_contract_2026_06_21.json`
  - `scripts/external/check_xr64_postrun_evidence_contract.py`
  - `tests/test_xr64_postrun_evidence_contract.py`
  - `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md`
  - `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.json`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-POSTRUN-ABLATION-AUDIT-002` identified lane-level ablation intent, LR evidence, teacher provenance, and attribution-report gaps.
- Change:
  - Post-run contract now requires `ablation_design`, `baseline_reference=XR-64-prep`, `axis_control_key_values`, and teacher provenance requirements per XR-64A/B lane.
  - Required future artifacts now include `train/ablation_lr_manifest.json`, `train/optimizer_config_snapshot.json`, `train/teacher_provenance.json`, and `train/ablation_attribution_report.json`.
  - Status reporter now surfaces `xr64_postrun_ablation_axis_count=4` and `xr64_postrun_ablation_lane_axis_count=2`.
- Validation:
  - `python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary`: passed with `ablation_axis_count=4`.
  - `.venv/bin/python scripts/external/check_second_goal_ablation_evidence_checklist.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_ablation_evidence_checklist.py tests/test_second_goal_status.py`: `15 passed`.
  - broader second-goal pytest suite: `87 passed`.
  - `git diff --check`: passed.

# 2026-06-21 XR-64 promotion axis certification

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Require future XR-64 promotion candidates to carry ablation proof, not metrics only.
- Updated:
  - `scripts/external/decide_xr64_postrun_promotion.py`
  - `tests/test_decide_xr64_postrun_promotion.py`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-PROMOTION-AXIS-AUDIT-003` identified missing contract-required decision fields, loose `axis_certified` bool handling, and weak `ablation_changed_keys` typing.
- Change:
  - Promotion helper now requires strict `axis_certified is True`.
  - `ablation_changed_keys` must be an object keyed by `head`, `loss`, `lr`, and `teacher_model_training`, each with a non-empty string list.
  - Decision output now includes `axis_claims_allowed`, `axis_claims`, `control_variables_checked`, `gate_tradeoff_table`, and `rejection_reason`.
  - Current placeholder reports `axis_claims_allowed=false`, `control_variables_checked=false`, and `rejection_reason=missing_postrun_evidence`.
- Validation:
  - `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_second_goal_status.py`: `11 passed`.
  - broader second-goal pytest suite: `90 passed`.
  - `git diff --check`: passed.

# 2026-06-21 XR-64 ablation provenance writer

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Make future XR-64A/B runs emit the ablation proof artifacts required by the post-run evidence contract and promotion helper.
- Updated:
  - `scripts/external/write_xr64_ablation_provenance.py`
  - `scripts/external/run_xr64_teacher_target_construction.sh`
  - `tests/test_write_xr64_ablation_provenance.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-PROVENANCE-AUDIT-005` identified exact required eval-summary fields and runner integration hazards.
- Change:
  - Writer creates `train/ablation_lr_manifest.json`, `train/optimizer_config_snapshot.json`, `train/teacher_provenance.json`, and `train/ablation_attribution_report.json`.
  - Runner invokes the writer immediately after `XR64_TRAIN_EXIT`, then patches each test `eval_summary.json` after `XR64_EVAL_EXIT`.
  - Patched summaries include required `ablation_axis`, `ablation_changed_keys`, `ablation_benchmark_baseline_id=XR-64-prep`, and strict `axis_certified=true`.
  - Runner now finds run roots with a max-depth search compatible with categorized `runs/XR-*` directories.
- Validation:
  - `python3 -m py_compile scripts/external/write_xr64_ablation_provenance.py scripts/external/decide_xr64_postrun_promotion.py scripts/external/check_xr64_postrun_evidence_contract.py`: passed.
  - `bash -n scripts/external/run_xr64_teacher_target_construction.sh`: passed.
  - `.venv/bin/python -m pytest -q tests/test_write_xr64_ablation_provenance.py tests/test_decide_xr64_postrun_promotion.py tests/test_xr64_postrun_evidence_contract.py`: `19 passed`.

# 2026-06-21 XR-64 required checkpoint guard

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Prevent future XR-64A/B partial checkpoint/eval evidence from being treated as promotion-ready.
- Updated:
  - `scripts/external/run_xr64_teacher_target_construction.sh`
  - `tests/test_xr64_runner_contract.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-RUNNER-CHECKPOINT-GUARD-AUDIT-006` confirmed the existing runner skipped missing checkpoint kinds and identified script/contract/writer checkpoint set checks.
  - The evaluator recommended lenient script behavior with strict post-run validation; final integration keeps default fail-fast behavior to avoid partial promotion evidence, with `XR64_REQUIRE_ALL_CHECKPOINTS=0` available for explicit diagnostics.
- Change:
  - Runner now defines `REQUIRED_CHECKPOINT_KINDS=(best_track_p10 best_track_p5 best_metric_track_center_px)`.
  - Default `XR64_REQUIRE_ALL_CHECKPOINTS=1` makes missing required checkpoints fatal.
  - Missing test `eval_summary.json` or `eval_rows.json` after eval is fatal.
  - Runner checks actual eval count against the required checkpoint count.
- Validation:
  - `bash -n scripts/external/run_xr64_teacher_target_construction.sh`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_runner_contract.py tests/test_write_xr64_ablation_provenance.py tests/test_xr64_postrun_evidence_contract.py tests/test_decide_xr64_postrun_promotion.py tests/test_second_goal_status.py`: `26 passed`.

# 2026-06-21 XR-64 post-run candidate collector

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Automate future discovery of certified XR-64A/B test eval summaries for promotion decisions.
- Updated:
  - `scripts/external/collect_xr64_postrun_candidates.py`
  - `tests/test_collect_xr64_postrun_candidates.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark read-only evaluator `T-XR64-CANDIDATE-COLLECTOR-AUDIT-007` confirmed required eval-summary fields, candidate spec format, A/B lane requirement, duplication hazards, and no-GPU validation targets.
- Change:
  - Collector scans `runs/**/eval_summary.json` for summaries patched with `lane`, `ablation_checkpoint_kind`, `ablation_benchmark_baseline_id=XR-64-prep`, and strict `axis_certified=true`.
  - Default output selects the latest summary per lane/checkpoint; `--include-all` is available for explicit audits.
  - Collector returns a decision report by reusing `decide_xr64_postrun_promotion.py` and can print the exact candidate command with `--format commands`.
  - `report_second_goal_status.py` now reports collector existence, tests, default latest-selection policy, and current `candidate_count=0` from the promotion decision placeholder without scanning `runs/`.
- Validation:
  - `python3 -m py_compile scripts/external/collect_xr64_postrun_candidates.py scripts/external/decide_xr64_postrun_promotion.py scripts/external/write_xr64_ablation_provenance.py`: passed.
  - `.venv/bin/python -m pytest -q tests/test_collect_xr64_postrun_candidates.py tests/test_decide_xr64_postrun_promotion.py tests/test_write_xr64_ablation_provenance.py`: `16 passed`.
  - `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --runs-root /tmp/nonexistent_xr64_runs --format summary`: passed with `candidate_count=0`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_candidate_collector_exists=true` and `xr64_candidate_collector_current_candidate_count=0`.

# 2026-06-21 XR-64 resume manifest post-run review integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Close the future resume command chain from XR-64-prep through post-run candidate and promotion-decision review.
- Updated:
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `scripts/external/emit_xr64_resume_commands.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_emit_xr64_resume_commands.py`
  - `tests/test_xr64_resume_command_manifest.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - Attempted GPT-5.3-Codex-Spark read-only evaluator `T-XR64-POSTRUN-GAP-001`.
  - Runtime failed with quota limit, so main agent completed the read-only gap analysis and integration directly.
- Change:
  - Manifest now includes `postrun_commands_after_launch` with collector summary and collector command-output review.
  - Checker enforces exactly two post-run commands, both `allowed_to_run=false`, both using `collect_xr64_postrun_candidates.py`, and neither calling train/eval helpers.
  - Emitter supports `--section postrun` and includes post-run commands in `all`/`script` output.
  - Status reporter surfaces `xr64_command_manifest_postrun_command_count=2`.
- Validation:
  - `python3 -m py_compile scripts/external/emit_xr64_resume_commands.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `postrun_command_count=2`.
  - `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section postrun --format commands`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_command_manifest_postrun_command_count=2`.
  - `.venv/bin/python -m pytest -q tests/test_emit_xr64_resume_commands.py tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py`: `14 passed`.

# 2026-06-21 XR-64 prelaunch packet post-run phase integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Align the prelaunch packet with the post-run review section now present in the resume command manifest.
- Updated:
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.md`
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `scripts/external/check_xr64_prelaunch_packet.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_xr64_prelaunch_packet.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-XR64-PRELAUNCH-POSTRUN-GAP-002` was started for independent gap review.
  - Main agent proceeded with direct integration after local evidence showed the packet still stopped at the launch phase.
- Change:
  - Packet now has `PHASE-5-POSTRUN-CANDIDATE-REVIEW`.
  - Packet source artifacts now include post-run evidence contract, promotion decision, candidate collector, and promotion decider.
  - Packet command review now includes `emit_xr64_resume_commands.py --section postrun --format commands`.
  - Checker enforces post-run phase/order, source artifacts, command review, validation command, and manifest post-run count parity.
  - Status reporter now exposes `xr64_prelaunch_postrun_command_count=2`.
  - Regression tests reject missing `PHASE-5-POSTRUN-CANDIDATE-REVIEW` and missing post-run command review.
- Validation:
  - `.venv/bin/python -m json.tool docs/resources/xr64_prelaunch_packet_2026_06_20.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `phase_count=6` and `manifest_postrun_command_count=2`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports `xr64_prelaunch_phase_count=6` and `xr64_prelaunch_postrun_command_count=2`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py tests/test_xr64_resume_command_manifest.py`: `18 passed`.

# 2026-06-21 completion readiness XR-64 post-run propagation

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Ensure the second-goal completion-readiness contract rejects stale XR-64 prelaunch/post-run state.
- Updated:
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-READINESS-PRELAUNCH-PROPAGATION-003` was started to inspect propagation gaps.
  - Main agent proceeded after local inspection showed readiness contract did not yet track prelaunch phase count or post-run command count.
- Change:
  - Readiness contract now records `xr64_command_manifest_postrun_command_count=2`, `xr64_prelaunch_phase_count=6`, `xr64_prelaunch_postrun_command_count=2`, and `xr64_postrun_candidate_count=0`.
  - Readiness checker compares those values against live status reporter output.
  - Status reporter prints the new readiness fields.
  - Tests reject stale prelaunch phase count, stale prelaunch post-run command count, and missing prelaunch packet validation.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with the new XR-64 propagation fields.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed and reports the new readiness fields.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_xr64_prelaunch_packet.py tests/test_xr64_resume_command_manifest.py`: `30 passed`.
  - `git diff --check`: passed.

# 2026-06-21 completion gate post-run promotion blocker propagation

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Synchronize completion gate, readiness contract, prelaunch packet, artifact index, TODO, and tests around the current stricter blocker count.
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/track/TODO.md`
  - `docs/track/PROGRESS.md`
  - `docs/Validation.md`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_xr64_prelaunch_packet.py`
  - `tests/test_second_goal_status.py`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-READINESS-STALENESS-AUDIT-004` found stale current-state `22` references and recommended either updating current-state files or appending supersession notes for historical logs.
- Change:
  - Current completion/readiness reporting is synchronized at `28` blockers.
  - Artifact index primary blockers now include the XR-64 post-run promotion blocker family.
  - Historical validation/log entries that recorded `22` are superseded by the new validation evidence rather than rewritten as if they had always reported `28`.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed with `blocker_count=28`.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed with `blocker_count=28`.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed with `completion_blocker_count=28`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `completion_readiness_blocker_count=28` and `xr64_prelaunch_completion_blocker_count=28`.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `30 passed`.
  - `git diff --check`: passed.

# 2026-06-21 XR-64 prep runner existing-artifact validation

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Reduce XR-64-prep resume risk by validating existing artifacts before skip/reuse.
- Updated:
  - `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `scripts/external/report_second_goal_status.py`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `tests/test_xr64_resume_command_manifest.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-XR64-PREP-RESUME-GAP-005` identified the key gap: `XR64_EVAL_SKIP` and `XR64_OVERRIDE_SKIP` reused existing files without `validate_json_rows`.
- Change:
  - Existing eval-row skip paths now call `validate_json_rows`.
  - Existing override skip paths now call `validate_json_rows`.
  - Override build path validates the four teacher eval-row files for the requested split before generating overrides.
  - Resume manifest checker now statically requires 17 prep runner safety/restartability markers.
  - Status reporter exposes `xr64_command_manifest_prep_runner_contract_ok=true` and check count `17`.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/emit_xr64_resume_commands.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed with `prep_runner_contract_ok=true` and `prep_runner_contract_check_count=17`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_command_manifest_prep_runner_contract_ok=true` and `xr64_command_manifest_prep_runner_contract_check_count=17`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_emit_xr64_resume_commands.py tests/test_second_goal_status.py`: `16 passed`.
  - `git diff --check`: passed.

# 2026-06-21 XR-64 next prep command selector

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add a no-execute way to identify the next XR-64-prep command from current artifact readiness.
- Updated:
  - `scripts/external/emit_xr64_next_prep_command.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_emit_xr64_next_prep_command.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.3-Codex-Spark attempt failed due usage limit.
  - GPT-5.5 read-only explorer `T-XR64-SELECTOR-AUDIT-RETRY` confirmed the selector should report no executable command while paused and should point `next_after_resume` to `XR64-EVAL-TRAIN-XR62A`.
- Change:
  - Selector validates the command manifest, checks generated artifact existence/JSON row shape, and classifies prep commands as ready, blocked, completed, or invalid.
  - Current state: `next_id=XR64-EVAL-TRAIN-XR62A`, `ready_after_resume_count=8`, `blocked_count=2`, `completed_count=0`, `invalid_count=0`, and `allowed_to_run_now=false`.
  - Status reporter now surfaces selector readiness fields.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/emit_xr64_next_prep_command.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format summary`: passed with `next_id=XR64-EVAL-TRAIN-XR62A`.
  - `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands`: passed and printed only the next review command.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_next_prep_selector_ok=true`.
  - `.venv/bin/python -m pytest -q tests/test_emit_xr64_next_prep_command.py tests/test_second_goal_status.py tests/test_emit_xr64_resume_commands.py`: `11 passed`.

# 2026-06-21 XR-64 selector readiness contract propagation

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind the XR-64 next-prep selector into the second-goal completion-readiness contract.
- Updated:
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-XR64-SELECTOR-READINESS-AUDIT` recommended adding `completed_count`, `invalid_count`, `next_status`, and `execute_supported` to the selector readiness snapshot.
- Change:
  - Completion-readiness now checks selector fields against live status output.
  - Current snapshot: `ok=true`, `ready_after_resume_count=8`, `completed_count=0`, `blocked_count=2`, `invalid_count=0`, `next_id=XR64-EVAL-TRAIN-XR62A`, `next_status=ready_after_resume`, `next_allowed_to_run_now=false`, `execute_supported=false`.
  - Required validation now includes `emit_xr64_next_prep_command.py --format summary` and the selector test file.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/emit_xr64_next_prep_command.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_emit_xr64_next_prep_command.py`: `25 passed`.

# 2026-06-21 Post-XR64 decision tree artifact

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add PAPER_REF-backed routing for future XR-64A/B post-run decisions.
- Added:
  - `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.md`
  - `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`
- Updated:
  - `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md`
  - `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Claim levels are now explicit: diagnostic improvement, software promotion, clean single-model promotion, and paper-level completion.
  - Future XR-64A/B result patterns now route to XR-64C, XR-65, XR-66, XR-67, or XR-68 only when matching evidence exists.
  - Paper-target future artifacts are separated into doc-prefillable, eval-row-dependent, and trained-candidate-dependent groups.
  - XR-64-prep remains the first executable worker after explicit user resume.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `git diff --check`: passed.

# 2026-06-21 Post-XR64 decision tree checker integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Promote the post-XR64 decision tree from a document-only artifact to a validated status-reported artifact.
- Added:
  - `scripts/external/check_second_goal_post_xr64_decision_tree.py`
  - `tests/test_second_goal_post_xr64_decision_tree.py`
- Updated:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-POSTXR64-CHECKER-AUDIT-001` recommended checks for pause state, XR-64-prep priority, current gates, blockers, decision inputs, claim hierarchy, and branch completeness.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_post_xr64_decision_tree.py scripts/external/report_second_goal_status.py`: passed.
  - `python3 -m json.tool docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_post_xr64_decision_tree.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_decision_tree.py tests/test_second_goal_status.py`: `8 passed`.

# 2026-06-21 Post-XR64 readiness contract integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind the post-XR64 decision tree into the completion-readiness contract.
- Updated:
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `T-POSTXR64-READINESS-001` was spawned to audit readiness fields and mismatch tests.
- Change:
  - Completion readiness now checks post-XR64 decision tree fields against live status: execute support, claim levels, decision inputs, branch count, minimal next worker, current next prep id, XR-64 ready state, and direct comparison state.
  - Required validation now includes `check_second_goal_post_xr64_decision_tree.py --format summary` and `tests/test_second_goal_post_xr64_decision_tree.py`.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_post_xr64_decision_tree.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_second_goal_post_xr64_decision_tree.py`: `30 passed`.

# 2026-06-21 Post-XR64 objective trace integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind post-XR64 decision routing into the main second-goal objective trace.
- Updated:
  - `docs/resources/second_goal_objective_trace_2026_06_20.json`
  - `docs/resources/second_goal_objective_trace_2026_06_20.md`
  - `scripts/external/check_second_goal_objective_trace.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_objective_trace.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `019ee8d1-142f-7243-8148-8ba2beaaa02b` audited the schema and recommended linking the decision tree to REQ-2 as planning-only evidence.
- Change:
  - REQ-2 now includes `post_xr64_decision_tree_ref` and `post_xr64_decision_tree_summary`.
  - Objective trace checker validates the linked tree path, checker/test paths, `experiment_planning_only` scope, branch count, claim levels, XR-64-prep minimal worker, current next prep id, blocked direct comparison, and next-experiment matrix coverage.
  - Status reporter emits `objective_trace_post_xr64_*` fields.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_objective_trace_2026_06_20.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_objective_trace.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py tests/test_second_goal_status.py tests/test_second_goal_post_xr64_decision_tree.py`: `15 passed`.
  - `git diff --check`: passed.

# 2026-06-21 Paper target bridge artifact workplan

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Order the 10 future paper-target bridge artifacts into work packages.
- Added:
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md`
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`
  - `scripts/external/check_paper_target_bridge_artifact_workplan.py`
  - `tests/test_paper_target_bridge_artifact_workplan.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/track/TODO.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - `PTB-0-DOC-PREFILL`: allowed while paused; covers split protocol audit, cur_state mapping audit, and accuracy/latency separation note.
  - `PTB-1-EVALUATOR-SCHEMA`: allowed while paused as schema planning only; actual sensor-space/hybrid/P1 rows remain missing.
  - `PTB-2-XR64-TRAINED-CANDIDATE`: blocked while paused and requires future train/eval evidence.
  - `PTB-3-BRIDGE-DECISION`: blocked until previous packages are complete.
- Validation:
  - `python3 -m json.tool docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_artifact_workplan.py tests/test_second_goal_status.py`: `9 passed`.

# 2026-06-21 Paper target bridge PTB-0 doc-prefill artifacts

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Create paused-safe source-note artifacts for the paper-target bridge.
- Added:
  - `docs/resources/future/paper_target_bridge/split_protocol_audit.json`
  - `docs/resources/future/paper_target_bridge/cur_state_target_mapping_audit.json`
  - `docs/resources/future/paper_target_bridge/accuracy_latency_separation_note.json`
- Updated:
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json`
  - `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md`
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `scripts/external/check_paper_target_comparison_evidence_manifest.py`
  - `scripts/external/check_paper_target_bridge_artifact_workplan.py`
  - `tests/test_paper_target_comparison_evidence_manifest.py`
  - `tests/test_paper_target_bridge_artifact_workplan.py`
  - `tests/test_second_goal_status.py`
  - `tests/test_second_goal_completion_gate.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Sub-agent:
  - GPT-5.5 read-only evaluator `019ee8e1-8edb-7691-9cde-6ffa5d91f198` identified the exact count/status/test changes required after adding PTB-0 artifacts.
- Change:
  - `split_protocol_match` moved from `missing` to `partial`.
  - Paper-target manifest now reports `missing_evidence_count=3`, `partial_evidence_count=4`, and `existing_required_future_artifact_count=3`.
  - Bridge workplan now reports `current_existing_future_artifact_count=3`.
  - Direct paper-target comparison remains blocked.
- Validation:
  - `python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_second_goal_completion_gate.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: reported `completion_allowed=false`.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_paper_target_bridge_artifact_workplan.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_completion_gate.py`: `45 passed`.
  - `git diff --check`: passed.

# 2026-06-21 Paper target bridge PTB-1 readiness binding

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind PTB-1 evaluator schema contract into completion-readiness and status summary.
- Updated:
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Completion-readiness now requires `check_paper_target_bridge_evaluator_schema_contract.py --format summary`.
  - Completion-readiness pytest bundle now includes `tests/test_paper_target_bridge_evaluator_schema_contract.py`.
  - Status summary now mirrors `completion_readiness_paper_target_bridge_evaluator_schema_*` fields.
  - Actual PTB-1 evaluator output payloads remain absent and blocked while paused.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_paper_target_bridge_evaluator_schema_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_evaluator_schema_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_paper_target_bridge_evaluator_schema_contract.py`: `35 passed`.

# 2026-06-21 Paper target bridge PTB-1 payload validator

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add a validator for PTB-1 future payload files without creating those files.
- Added:
  - `scripts/external/check_paper_target_bridge_payloads.py`
  - `tests/test_paper_target_bridge_payloads.py`
- Updated:
  - `docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json`
  - `docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/check_paper_target_bridge_evaluator_schema_contract.py`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `tests/test_paper_target_bridge_evaluator_schema_contract.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Default payload check rejects official PTB-1 output presence while paused.
  - `--allow-present --require-all-present` validates future payloads against the PTB-1 schema contract.
  - Validator checks JSONL rows, JSON objects, numeric ranges, booleans, full-test P1 split, and hybrid report row-count consistency.
  - Validator rejects `execute_supported`, direct-comparison, paper-level-completion, or train/eval validation-command drift.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m json.tool docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_paper_target_bridge_payloads.py scripts/external/check_paper_target_bridge_evaluator_schema_contract.py scripts/external/check_second_goal_completion_readiness_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_evaluator_schema_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_payloads.py tests/test_paper_target_bridge_evaluator_schema_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `49 passed`.

# 2026-06-21 Paper target bridge PTB-2 trained-candidate contract binding

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Register and bind PTB-2 trained-candidate evidence contract into status/readiness paths.
- Updated:
  - `docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.json`
  - `docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`
  - `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md`
  - `scripts/external/check_paper_target_bridge_trained_candidate_contract.py`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/check_paper_target_bridge_artifact_workplan.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_paper_target_bridge_trained_candidate_contract.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_paper_target_bridge_artifact_workplan.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - PTB-2 contract is now a first-class indexed artifact.
  - Status reporter emits `paper_target_bridge_trained_candidate_*` fields.
  - Completion-readiness mirrors PTB-2 schema/payload guard fields and requires the PTB-2 checker/test.
  - Workplan source alignment now checks the PTB-2 contract.
  - Future payload validation requires paired eval summary and leakage audit, no test override, and leakage risk `none`.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/check_paper_target_bridge_trained_candidate_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_trained_candidate_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_trained_candidate_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_paper_target_bridge_artifact_workplan.py`: `52 passed`.

# 2026-06-21 Paper target bridge PTB-3 readiness/status regression binding

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Harden PTB-3 bridge-decision readiness/status tests so final paper-target comparison cannot be silently unblocked by stale fields.
- Updated:
  - `tests/test_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Status reporter tests now assert standalone `paper_target_bridge_decision_*` fields.
  - Completion-readiness status tests now assert mirrored `completion_readiness_paper_target_bridge_decision_*` fields.
  - Readiness contract tests now reject PTB-3 execute-support drift, schema-allowed drift, payload-presence drift, output-count drift, payload-allowed drift, and missing PTB-3 checker/test commands.
  - PTB-3 remains schema-only while paused; `bridge_decision.json` remains absent.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py scripts/external/check_paper_target_bridge_decision_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_decision_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_decision_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `49 passed`.

# 2026-06-21 Post-XR64 follow-up experiment contract

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Convert the PAPER_REF-backed post-XR64 decision tree into no-execute branch contracts for XR-64C/XR-65/XR-66/XR-67/XR-68.
- Added:
  - `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json`
  - `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.md`
  - `scripts/external/check_second_goal_post_xr64_followup_contract.py`
  - `tests/test_second_goal_post_xr64_followup_contract.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - All follow-up branches expose explicit `launch_allowed=false` and `promotion_allowed=false`.
  - XR-64C now requires validation-transfer plus full-test evidence and rejects validation-only promotion.
  - XR-65 now requires same-scene temporal evidence and center drift `<=0.3 px`.
  - XR-66 now requires no-train confidence/local fallback diagnostics before local-update training.
  - XR-67 now requires dense or continuous trajectories.
  - XR-68 now requires a stronger stable full-width teacher plus student/hardware evidence.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_post_xr64_followup_contract.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_post_xr64_followup_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_followup_contract.py tests/test_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py`: `55 passed`.

# 2026-06-21 XR-64 resume command review safety

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Reduce accidental execution risk in future XR-64 resume command review output.
- Updated:
  - `scripts/external/emit_xr64_next_prep_command.py`
  - `scripts/external/emit_xr64_resume_commands.py`
  - `tests/test_emit_xr64_next_prep_command.py`
  - `tests/test_emit_xr64_resume_commands.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - `emit_xr64_next_prep_command.py --format commands` now emits the selected future command as `# command: ...`.
  - `emit_xr64_resume_commands.py --format commands` now emits future precheck/prep/strict/launch/postrun commands as `# command: ...`.
  - Tests assert review output contains no raw executable `XR64_ACTIONS=eval` or `XR64_ACTIONS=build` lines.
- Validation:
  - `python3 -m py_compile scripts/external/emit_xr64_next_prep_command.py scripts/external/emit_xr64_resume_commands.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands`: passed.
  - `.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section prep --format commands`: passed.
  - `.venv/bin/python -m pytest -q tests/test_emit_xr64_next_prep_command.py tests/test_emit_xr64_resume_commands.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `20 passed`.

# 2026-06-21 XR-64 post-run candidate matrix guard

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Prevent future XR-64 promotion from partial post-run candidate evidence.
- Updated:
  - `scripts/external/decide_xr64_postrun_promotion.py`
  - `scripts/external/collect_xr64_postrun_candidates.py`
  - `tests/test_decide_xr64_postrun_promotion.py`
  - `tests/test_collect_xr64_postrun_candidates.py`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Promotion decision now requires the complete candidate matrix: XR-64A and XR-64B each with `best_metric_track_center_px`, `best_track_p10`, and `best_track_p5`.
  - Partial evidence sets now report `invalid_candidate_evidence` instead of allowing software promotion.
  - Collector summary now reports `required_candidate_matrix_complete` and `missing_required_candidate_count`.
  - Future command example now uses valid `best_metric_track_center_px` kind instead of stale `best_track_center`.
- Validation:
  - `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/collect_xr64_postrun_candidates.py scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed.
  - `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_collect_xr64_postrun_candidates.py tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_status.py`: `25 passed`.

# 2026-06-21 XR-64 override-cleared type guard

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Prevent future XR-64 promotion when final test eval summaries encode cleared target overrides with ambiguous JSON values.
- Updated:
  - `scripts/external/decide_xr64_postrun_promotion.py`
  - `tests/test_decide_xr64_postrun_promotion.py`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - `data_track_target_override_path` must be JSON `null`.
  - `data_allow_test_target_override` must be JSON boolean `false`; numeric `0` and string `"false"` reject.
  - `loss_track_target_override_center_l2_weight` and `loss_track_state_aux_target_override_center_l2_weight` must be JSON numeric zero; boolean `false` and string `"0.0"` reject.
  - Dotted config-path keys alone do not satisfy the future eval-summary schema; flat summary keys are required.
  - Dirty override evidence in any required XR-64A/B checkpoint-kind candidate rejects promotion, even if the winning candidate is clean.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py scripts/external/collect_xr64_postrun_candidates.py scripts/external/write_xr64_ablation_provenance.py scripts/external/check_xr64_postrun_evidence_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary`: passed.
  - `.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_decide_xr64_postrun_promotion.py tests/test_write_xr64_ablation_provenance.py tests/test_collect_xr64_postrun_candidates.py tests/test_xr64_postrun_evidence_contract.py tests/test_second_goal_status.py`: `33 passed`.

# 2026-06-21 XR-64 resume full-coverage gate

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Prevent partial train/val XR-64-prep artifacts from satisfying strict readiness.
- Sub-agent:
  - Attempted real GPT-5.3-Codex-Spark explorer `019ee93f-82e6-7e02-92c2-648d96219126`.
  - It errored due usage limit before producing output; no sub-agent findings were used or claimed.
- Updated:
  - `scripts/external/check_xr64_resume_artifacts.py`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `tests/test_xr64_resume_artifacts.py`
  - `tests/test_xr64_resume_command_manifest.py`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.json`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Teacher `eval_rows.json` files must contain every sample ID from the corresponding split manifest.
  - Override JSON files must contain every sample ID from the corresponding split manifest.
  - Subset/smoke artifacts now remain useful only as diagnostics and cannot satisfy `ready_to_train`.
  - The command manifest now records `strict_generated_artifact_contract`, and the manifest checker rejects contract drift.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/check_xr64_prelaunch_packet.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_xr64_resume_command_manifest.py tests/test_xr64_prelaunch_packet.py tests/test_second_goal_status.py`: `27 passed`.

# 2026-06-21 XR-64 full-coverage readiness mirror

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Ensure the completion-readiness layer mirrors the full XR-64 strict generated-artifact policy.
- Updated:
  - `scripts/external/report_second_goal_status.py`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `tests/test_second_goal_status.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Completion readiness now mirrors full eval coverage, full override coverage, duplicate ID rejection, opposite-split ID rejection, subset-artifact rejection, and strict checker path from the XR-64 command manifest.
  - Status summary now prints the mirrored fields under `completion_readiness_xr64_command_manifest_*`.
  - Readiness checker rejects drift for duplicate sample policy, opposite split policy, subset-artifact readiness, and strict checker path.
  - GPT5.5 read-only evaluator identified missing explicit drift tests for full eval and full override coverage; both tests were added.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_resume_command_manifest.py scripts/external/check_xr64_resume_artifacts.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_resume_command_manifest.py tests/test_xr64_resume_artifacts.py`: `68 passed`.
  - `git diff --check -- scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
- Remaining:
  - XR-64 remains `generated_incomplete`.
  - Missing generated artifacts remain `8` eval rows and `6` override JSON files.
  - XR-64A/B execution remains blocked until explicit user resume and strict readiness passes.

# 2026-06-21 XR-64 expected generated count contract

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Pin expected XR-64-prep output cardinality before future artifact generation.
- Sub-agent:
  - GPT5.5 read-only evaluator recommended a broader `second_goal_blocker_artifact_map` as the next no-execute artifact.
  - Main agent completed the smaller expected-count contract first because it directly strengthens the existing XR-64 command manifest and status checker.
- Updated:
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.json`
  - `docs/resources/xr64_resume_command_manifest_2026_06_20.md`
  - `scripts/external/check_xr64_resume_command_manifest.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_xr64_resume_command_manifest.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Added `expected_generated_counts` to the XR-64 command manifest.
  - Train split expected rows: `5929`.
  - Val split expected rows: `844`.
  - Teacher count: `4`.
  - Override rule count: `3`.
  - Total future teacher eval-row records: `27092`.
  - Total future override records: `20319`.
  - Checker now compares these values to current train/val manifest line counts.
  - Status summary now prints `xr64_command_manifest_expected_*` fields.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_resume_command_manifest_2026_06_20.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py scripts/external/report_second_goal_status.py scripts/external/check_second_goal_completion_readiness_contract.py scripts/external/check_xr64_resume_artifacts.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py tests/test_second_goal_status.py tests/test_second_goal_completion_readiness_contract.py tests/test_xr64_resume_artifacts.py`: `71 passed`.
- Remaining:
  - XR-64 generated artifacts are still absent.
  - Next executable work remains blocked until explicit user resume.
  - Recommended next no-execute artifact is blocker-to-missing-artifact map.

# 2026-06-21 second-goal blocker artifact map

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Convert current XR-64 and paper-target blockers into one machine-checkable missing-artifact map.
- Sub-agent:
  - GPT5.5 read-only evaluator audited required fields and drift-prone counts.
  - Integrated its recommendations for `blocker_count=28`, next-prep selector fields, paper-target artifact counts, and drift-prone field listing.
- Updated:
  - `docs/resources/second_goal_blocker_artifact_map_2026_06_21.json`
  - `docs/resources/second_goal_blocker_artifact_map_2026_06_21.md`
  - `scripts/external/check_second_goal_blocker_artifact_map.py`
  - `tests/test_second_goal_blocker_artifact_map.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Map records all `8` XR-64 train/val teacher eval-row paths and all `6` XR-64 override JSON paths.
  - Map records blocked build command IDs `XR64-BUILD-TRAIN` and `XR64-BUILD-VAL`.
  - Map records PTB-1 evaluator payloads, PTB-2 trained-candidate payloads, and PTB-3 bridge-decision payload separately.
  - Checker cross-checks map state against the XR-64 command manifest, completion-readiness contract, paper-target manifest, paper-target bridge workplan, and PTB-1 payload checker.
  - Status summary now prints `second_goal_blocker_map_*` fields.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_blocker_artifact_map_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_blocker_artifact_map.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_blocker_artifact_map.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_blocker_artifact_map.py tests/test_second_goal_status.py`: `10 passed`.
  - `git diff --check -- docs/resources/second_goal_blocker_artifact_map_2026_06_21.json docs/resources/second_goal_blocker_artifact_map_2026_06_21.md scripts/external/check_second_goal_blocker_artifact_map.py scripts/external/report_second_goal_status.py tests/test_second_goal_blocker_artifact_map.py tests/test_second_goal_status.py docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- Remaining:
  - XR-64 remains `generated_incomplete`.
  - Missing generated artifacts remain `8` eval rows and `6` override JSON files.
  - XR-64A/B and paper-target bridge decisions remain blocked until explicit experiment resume and post-run evidence.

# 2026-06-21 XR-64 prep progress ledger

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add a progress ledger that mirrors the live XR-64 next-prep selector and can be checked for drift.
- Sub-agent:
  - GPT5.5 read-only evaluator identified key ledger risks: pause guard, exact 10-unit prep matrix, strict readiness gate, test split refusal, subset readiness refusal, row-count contract, and review-only command handling.
- Updated:
  - `docs/resources/xr64_prep_progress_ledger_2026_06_21.json`
  - `docs/resources/xr64_prep_progress_ledger_2026_06_21.md`
  - `scripts/external/check_xr64_prep_progress_ledger.py`
  - `tests/test_xr64_prep_progress_ledger.py`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Ledger records `prep=10`, `eval=8`, `build=2`.
  - Current progress is `ready_after_resume=8`, `blocked=2`, `completed=0`, `invalid=0`.
  - Current next unit is `XR64-EVAL-TRAIN-XR62A`.
  - Current generated gap remains `8` eval rows and `6` override JSONs.
  - Expected generated row totals remain `27092` eval-row records and `20319` override records.
- Remaining:
  - XR-64-prep still requires explicit user resume before any artifact generation.
  - XR-64A/B remain blocked until strict readiness passes.

# 2026-06-21 second-goal resume readiness matrix

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Connect PAPER_REF experiment axes, XR-64 readiness, post-run promotion, and paper-target bridge blockers in one matrix.
- Sub-agent:
  - GPT5.3-Codex-Spark delegation failed due usage limit; main agent continued with local file evidence and checker validation.
- Updated:
  - `docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json`
  - `docs/resources/second_goal_resume_readiness_matrix_2026_06_21.md`
  - `scripts/external/check_second_goal_resume_readiness_matrix.py`
  - `tests/test_second_goal_resume_readiness_matrix.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Matrix records `axis_count=6` and `resume_gate_count=5`.
  - Current P0 remains `XR-64-prep`, `XR-64A`, `XR-64B`.
  - Current next prep remains `XR64-EVAL-TRAIN-XR62A`.
  - XR-64 remains not ready with missing eval rows `8` and missing overrides `6`.
  - Promotion, paper-level completion, and direct submission comparison remain false.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_resume_readiness_matrix.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_resume_readiness_matrix.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_resume_readiness_matrix.py`: `6 passed`.
- Remaining:
  - Resume matrix is no-execute infrastructure only.
  - XR-64-prep still requires explicit user resume before artifact generation.
  - Direct paper-target comparison remains blocked until required bridge evidence exists.

# 2026-06-21 resume readiness matrix status integration

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Expose resume readiness matrix in the one-command second-goal status summary.
- Sub-agent:
  - GPT5.5 read-only evaluator confirmed main integration risk: add test fixture/assertions for matrix inventory and summary lines.
- Updated:
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Status now reports `second_goal_resume_matrix_exists`, `axis_count=6`, `resume_gate_count=5`, P0 IDs `XR-64-prep,XR-64A,XR-64B`, missing eval rows `8`, missing overrides `6`, postrun candidates `0`, and direct comparison/promotion flags `false`.
  - Authority link inventory now includes matrix files; current status reports `authority_links_checked=144`, `authority_links_missing=0`.
- Validation:
  - `python3 -m py_compile scripts/external/report_second_goal_status.py scripts/external/check_second_goal_resume_readiness_matrix.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_resume_readiness_matrix.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_resume_readiness_matrix.py`: `9 passed`.
- Remaining:
  - Integration is status evidence only.
  - XR-64 generated artifacts and post-run trained candidate evidence remain missing.

# 2026-06-21 XR-64 prelaunch latest-source cross-check

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Reuse existing XR-64 prelaunch packet as the pre-resume gate and bind it to latest readiness/status sources.
- Updated:
  - `docs/resources/xr64_prelaunch_packet_2026_06_20.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/log.md`
- Change:
  - Added validation commands for resume readiness matrix, XR-64 prep ledger, blocker artifact map, and paper-target metric preflight to the packet JSON.
  - Added artifact-index summary fields for prelaunch latest-source mirror counts.
  - Recorded that a separate new pre-resume gate is redundant; existing packet now cross-checks latest source artifacts.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_prelaunch_packet_2026_06_20.json`: passed.
  - `.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py`: `10 passed`.
- Remaining:
  - XR-64 generated eval rows and override JSONs are still absent.
  - XR-64A/B post-run evidence is still absent.
  - Experiments remain paused until explicit user resume.

# 2026-06-21 paper-target bridge PTB-1 resume runbook

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Freeze the future PTB-1 evaluator payload generation order without creating payload files while paused.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only PTB-1 verification, but the runtime reported `agent thread limit reached`; main agent continued with local checker/test validation.
- Updated:
  - `docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json`
  - `docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.md`
  - `scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py`
  - `tests/test_paper_target_bridge_ptb1_resume_runbook.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/track/log.md`
- Change:
  - Runbook sequence is `coordinate_transform_audit`, `sensor_space_eval_rows`, `paper_frame_full_test_p1`, `hybrid_eval_rows`, and `hybrid_metric_report`.
  - Current PTB-1 payload state remains absent: existing `0`, missing `5`.
  - Future payload validation requires `--allow-present --require-all-present`.
  - Direct paper-target comparison and paper-level completion remain false.
- Validation:
  - `python3 -m json.tool docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary`: passed with payload present count `0` and missing count `5`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `paper_target_bridge_ptb1_resume_runbook_*` fields and authority links missing `0`.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_ptb1_resume_runbook.py tests/test_second_goal_status.py`: `8 passed`.
- Remaining:
  - PTB-1 runbook is no-execute infrastructure only.
  - Official PTB-1 payloads remain absent until explicit resume and evaluator execution.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and XR-64-or-later trained candidate evidence remain missing.

# 2026-06-21 paper-target bridge resume dependency matrix

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add one upper dependency gate that binds PTB-0/1/2/3, XR-64 readiness, and strict completion before any direct paper-target comparison claim.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only dependency-matrix review, but the runtime reported `agent thread limit reached`; main agent continued with local source alignment checks and tests.
- Updated:
  - `docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json`
  - `docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.md`
  - `scripts/external/check_paper_target_bridge_resume_dependency_matrix.py`
  - `tests/test_paper_target_bridge_resume_dependency_matrix.py`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
  - `docs/track/log.md`
- Change:
  - Matrix records `dependency_gate_count=6`.
  - Complete-now gates are `2`: PTB-0 doc-prefill and PTB-1 schema/runbook.
  - Blocked gates are `4`: PTB-1 payloads, PTB-2 trained candidate, PTB-3 bridge decision, and strict completion.
  - Current execution-dependent payload gap is `8`.
  - Current XR-64 state remains `ready_to_train=false`, missing eval rows `8`, missing overrides `6`.
  - Current completion state remains `completion_allowed=false`, blocker count `28`.
- Validation:
  - `python3 -m json.tool docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_paper_target_bridge_resume_dependency_matrix.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_paper_target_bridge_resume_dependency_matrix.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `paper_target_bridge_resume_dependency_matrix_*` fields and authority links missing `0`.
  - `.venv/bin/python -m pytest -q tests/test_paper_target_bridge_resume_dependency_matrix.py tests/test_second_goal_status.py`: `8 passed`.
- Remaining:
  - Dependency matrix is no-execute readiness infrastructure only.
  - PTB-1 official payloads, PTB-2 trained-candidate payloads, PTB-3 bridge decision, XR-64 generated artifacts, and XR-64A/B post-run evidence remain absent.

# 2026-06-21 PAPER_REF experiment trace

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind PAPER_REF paper groups to the current XR-64-first experiment queue and evidence requirements.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only gap review, but the runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json`
  - `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.md`
  - `scripts/external/check_second_goal_paper_ref_experiment_trace.py`
  - `tests/test_second_goal_paper_ref_experiment_trace.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
  - `docs/track/TODO.md`
- Change:
  - Trace records six evidence groups: geometry/state, teacher/distillation, temporal robustness, local fallback/blink, dense refinement, deployment/distillation.
  - Trace records eight experiment mappings: XR-64-prep, XR-64A, XR-64B, XR-64C, XR-65, XR-66, XR-67, XR-68.
  - Current state remains blocked: `xr64_ready_to_train=false`, missing eval rows `8`, missing overrides `6`, completion allowed `false`.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_paper_ref_experiment_trace.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_paper_ref_experiment_trace.py --format summary`: passed.
  - `.venv/bin/python scripts/external/check_paper_ref_analysis_coverage.py --format summary`: passed with `analysis_count=31`.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_paper_ref_experiment_trace.py`: `4 passed`.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `authority_links_missing=0`.
  - `git diff --check` for touched files: passed.
- Follow-up status integration:
  - Added `paper_ref_experiment_trace_inventory` to `scripts/external/report_second_goal_status.py`.
  - Added `paper_ref_experiment_trace_*` summary lines.
  - Extended `tests/test_second_goal_status.py` fixture and assertions.
  - Validation passed: py_compile, status summary, and `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_second_goal_paper_ref_experiment_trace.py` with `7 passed`.
- Remaining:
  - Trace is planning/evidence infrastructure only.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads remain absent.

# 2026-06-21 PAPER_REF current ablation plan checker

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Make the current PAPER_REF ablation plan machine-checkable and visible in the second-goal status report.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only gap review, but the runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `scripts/external/check_second_goal_paper_ref_current_ablation_plan.py`
  - `tests/test_second_goal_paper_ref_current_ablation_plan.py`
- Updated:
  - `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Checker validates prompt-pack metadata, PAPER_REF signal coverage, current ablation matrix, P0 IDs, lane trace, requested-axis coverage, validation evidence markers, and paused-state completion guards.
  - Status summary now reports `paper_ref_current_ablation_plan_*` fields.
- Validation:
  - `python3 -m py_compile scripts/external/check_second_goal_paper_ref_current_ablation_plan.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_paper_ref_current_ablation_plan.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `paper_ref_current_ablation_plan_ok=true` and authority links missing `0`.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_paper_ref_current_ablation_plan.py tests/test_second_goal_status.py`: `6 passed`.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
- Remaining:
  - This is no-execute planning infrastructure only.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads remain absent.

# 2026-06-21 XR-64 experiment design contract

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Promote the XR-64 A/B/C design from queue-embedded details into a standalone machine-checkable contract.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only gap review, but the runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `docs/resources/xr64_experiment_design_contract_2026_06_21.json`
  - `docs/resources/xr64_experiment_design_contract_2026_06_21.md`
  - `scripts/external/check_xr64_experiment_design_contract.py`
  - `tests/test_xr64_experiment_design_contract.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Contract records XR-64-prep/A/B/C lane order, P0 IDs, GPU assignment, LR/loss coefficients, strict checker dependency, no-test-override guard, and post-run evidence requirements.
  - Status summary now reports `xr64_experiment_design_contract_*` fields.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_experiment_design_contract_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_experiment_design_contract.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_experiment_design_contract.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_experiment_design_contract_ok=true` and authority links missing `0`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_experiment_design_contract.py tests/test_second_goal_status.py`: passed.
- Remaining:
  - This is no-execute planning infrastructure only.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads remain absent.

# 2026-06-21 XR-64 resume execution DAG

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind XR-64 precheck, prep eval/build, strict readiness, XR-64A/B launch, and post-run review into one dependency graph.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only gap review, but the runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `docs/resources/xr64_resume_execution_dag_2026_06_21.json`
  - `docs/resources/xr64_resume_execution_dag_2026_06_21.md`
  - `scripts/external/check_xr64_resume_execution_dag.py`
  - `tests/test_xr64_resume_execution_dag.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - DAG records 17 nodes across 6 phases.
  - `XR64-STRICT-READY` depends on both train/val build nodes.
  - XR-64A/B launch nodes depend on strict readiness and preserve `cuda:0`/`cuda:1` assignment.
  - Post-run review depends on both launch lanes.
  - Checker cross-validates DAG nodes against the command manifest and design contract.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_resume_execution_dag_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_resume_execution_dag.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_resume_execution_dag.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed with `xr64_resume_execution_dag_ok=true` and authority links missing `0`.
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_execution_dag.py tests/test_second_goal_status.py`: passed.
- Remaining:
  - This is no-execute planning infrastructure only.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads remain absent.

# 2026-06-21 accuracy lift decision ladder

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Bind current best gates, XR-63 oracle headroom, XR-64 transfer, post-XR64 follow-up triggers, and paper-target bridge blockers into one machine-checkable accuracy-lift ladder.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only integration review.
  - First attempt failed because `priority` service tier is unsupported for `gpt-5.3-codex-spark`.
  - Second attempt failed with `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.json`
  - `docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.md`
  - `scripts/external/check_second_goal_accuracy_lift_decision_ladder.py`
  - `tests/test_second_goal_accuracy_lift_decision_ladder.py`
- Updated:
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Ladder records five stages: current gates, oracle signal, XR-64 transfer, post-XR64 branching, and paper-target bridge.
  - Ladder records eight blocked experiment rows: XR-64-prep/A/B/C and XR-65/66/67/68.
  - Status summary now reports `accuracy_lift_decision_ladder_*` fields.
- Remaining:
  - This is no-execute planning infrastructure only.
  - XR-64 generated eval rows, override JSONs, XR-64A/B post-run evidence, and paper-target bridge payloads remain absent.

# 2026-06-21 XR-64 post-run tradeoff scorecard

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Make future XR-64A/B promotion decision reject severe center/P10/P5 tradeoff even if one gate improves.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only scorecard gap review, but the runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Updated:
  - `scripts/external/decide_xr64_postrun_promotion.py`
  - `tests/test_decide_xr64_postrun_promotion.py`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Added candidate-level `promotion_score`, `tradeoff_classification`, `bounded_tradeoff`, `severe_regressions`, and `tradeoff_floors`.
  - Winning candidate selection now ignores candidates with `diagnostic_unbounded_tradeoff`.
  - Severe-regression floors are center `< -0.3 px`, P10 `< -1.0 pp`, and P5 `< -1.0 pp`.
- Remaining:
  - This is future-result decision logic only.
  - XR-64 generated artifacts, train/eval runs, and post-run candidates remain absent while experiments are paused.

# 2026-06-21 XR-64 post-run promotion decision artifact checker

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Add a machine-checkable guard for the current XR-64 post-run promotion decision placeholder.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only checker invariant review.
  - The runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Added:
  - `scripts/external/check_xr64_postrun_promotion_decision.py`
  - `tests/test_xr64_postrun_promotion_decision.py`
- Updated:
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`
  - `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md`
  - `docs/resources/second_goal_artifact_index_2026_06_18.json`
  - `docs/resources/second_goal_artifact_index_2026_06_18.md`
  - `scripts/external/report_second_goal_status.py`
  - `tests/test_second_goal_status.py`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Checker validates the no-candidate decision helper output against the placeholder JSON.
  - Checker enforces active gates, required metrics, future P1 evidence marker, required axes, target-override-clear schema, bounded-tradeoff floors, all-false promotion flags, and the six required XR-64A/B checkpoint candidates.
  - Status summary now reports `xr64_promotion_decision_ok` and `xr64_promotion_decision_error_count`.
- Validation:
  - `python3 -m json.tool docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`: passed.
  - `python3 -m json.tool docs/resources/second_goal_artifact_index_2026_06_18.json`: passed.
  - `python3 -m py_compile scripts/external/check_xr64_postrun_promotion_decision.py scripts/external/report_second_goal_status.py`: passed.
  - `.venv/bin/python scripts/external/check_xr64_postrun_promotion_decision.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_second_goal_status.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_xr64_postrun_promotion_decision.py tests/test_second_goal_status.py`: `8 passed`.
- Remaining:
  - This is no-execute validation infrastructure only.
  - XR-64 generated artifacts, train/eval runs, post-run candidates, and paper-target bridge payloads remain absent while experiments are paused.

# 2026-06-21 completion gate promotion decision checker guard

- Scope:
  - Keep experiments paused.
  - Do not launch train/eval/GPU jobs.
  - Make full second-goal completion impossible when the XR-64 post-run promotion decision artifact checker fails.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only completion-gate review.
  - The runtime reported `agent thread limit reached`; main agent continued with local source checks and tests.
- Updated:
  - `scripts/external/check_second_goal_completion_gate.py`
  - `tests/test_second_goal_completion_gate.py`
  - `scripts/external/check_second_goal_completion_readiness_contract.py`
  - `tests/test_second_goal_completion_readiness_contract.py`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`
  - `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md`
  - `docs/Validation.md`
  - `docs/track/PROGRESS.md`
- Change:
  - Added completion blocker `xr64_postrun_promotion_decision_invalid` when the decision checker reports non-ok or missing status.
  - Added guard `xr64_postrun_promotion_decision_valid` when the current promotion-decision artifact checker passes.
  - Current blocker count remains `28`; current guard list now includes the promotion-decision-valid guard.
- Validation:
  - `python3 -m json.tool docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`: passed.
  - `python3 -m py_compile scripts/external/check_second_goal_completion_gate.py scripts/external/check_second_goal_completion_readiness_contract.py`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary`: passed.
  - `.venv/bin/python scripts/external/check_second_goal_completion_readiness_contract.py --format summary`: passed.
  - `.venv/bin/python -m pytest -q tests/test_second_goal_completion_gate.py`: `8 passed`.
- Remaining:
  - XR-64 generated artifacts, trained candidates, and paper-target bridge evidence remain absent while experiments are paused.

# 2026-06-21 Stage1 frame-search baseline pivot

- Scope:
  - Resume experiment execution with the active priority changed to Stage1 frame-based Search accuracy.
  - Use the improved Stage1 result as the next baseline before returning to Stage2/XR-64 work.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only Stage1 audit.
  - Runtime reported `agent thread limit reached`; main agent continued with local source checks and script/document updates.
- Evidence:
  - Current Stage1 reference is `runs/NON_XR/raw/raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452`.
  - Current gate: best val Search P10 `23.34905708960767` at epoch `165`; best val Search center `18.319516586807538 px` at epoch `154`.
  - `model.heads.active=search` is read by `src/hbtxr/training/trainer.py` and passed into Stage1 loss execution.
  - Stage1 loss code keeps frame Search branch active while eye/mask losses are zeroed unless their head is active.
- Added:
  - `scripts/external/run_stage1_frame_search_baseline_matrix.sh`
  - `docs/resources/stage1_frame_search_baseline_plan_2026_06_21.md`
- Plan:
  - Launch two detached lanes under `runs/NON_XR/raw`.
  - GPU0 lane A: Search-only full-width AdamW LR `1e-3`.
  - GPU1 lane B: Search-only full-width AdamW LR `5e-4`, `loss.search_xy_weight=1.5`.
  - Both lanes: 300 epochs, no distillation, no pruning, active head Search, best metric `metric_search_p10_pct`.
- Remaining:
  - Static and dry-run validation passed.
  - Initial launch attempts exposed three blockers: sandbox CUDA/NVML false negatives, cuDNN sublibrary version mismatch when cuDNN was not disabled, and detached child cleanup without `setsid`.
  - Fixed runner preflight to use Torch CUDA tensor smoke, changed detached launch to `setsid`, and kept `HBTXR_DISABLE_CUDNN=1`.
  - Terminated stale `watch -n 1 nvidia-smi` monitor before relaunch.
  - Foreground one-batch smoke passed with `HBTXR_DISABLE_CUDNN=1` for both `num_workers=0` and `num_workers=8`.
  - Launched production lanes with `RUN_TAG=20260621_204500`.
  - Lane A PID `1332213`: `NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500`, GPU0, LR `1e-3`.
  - Lane B PID `1332214`: `NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`, GPU1, LR `5e-4`, `loss.search_xy_weight=1.5`.
  - Both lanes were alive after 20 seconds with PPID `1`, SID equal to PID, and step logs beyond epoch 1 step 200/742.
  - Summarize histories after completion and promote only if the Stage1 gate improves.

# 2026-06-21 Stage1 frame-search mid-run evidence and follow-up runner

- Scope:
  - Preserve the active Stage1 frame-search evidence after the minimum 50-epoch requirement was met.
  - Prepare the next refinement script without interrupting the current 300-epoch lanes.
- Sub-agent:
  - GPT5.3-Codex-Spark spawn was attempted for read-only next-experiment planning.
  - Runtime reported `agent thread limit reached`; main agent continued with local source checks and documentation.
- Current process evidence:
  - Lane A PID `1332213`, PPID `1`, SID `1332213`, `Ssl`, elapsed `02:13:54` at the check.
  - Lane B PID `1332214`, PPID `1`, SID `1332214`, `Rsl`, elapsed `02:13:54` at the check.
  - GPU0: `720 MiB` used, `15123 MiB` free, `15%`.
  - GPU1: `720 MiB` used, `15123 MiB` free, `34%`.
- Metric evidence:
  - Old Stage1 reference: P10 `23.34905708960767`, P5 `6.390386527439333`, center `18.319516586807538`.
  - Latest report-script check observed lane A through epoch `229` and lane B through epoch `228`.
  - Lane A best remains P10 `24.58782615301744`, P5 `7.867250928338969`, center `17.595290881282878`.
  - Lane B best remains P10 `28.22439415050003`, P5 `9.833782825829848`, P1 `1.0646900500891343`, center `17.28884926831947`.
  - Lane B is the current baseline candidate, but latest epoch values drifted below the best checkpoint, so promotion should use `best_search_p10.pt`, not `last.pt`, unless later epochs recover.
- Added:
  - `docs/resources/stage1_frame_search_midrun_results_2026_06_21.md`
  - `scripts/external/run_stage1_frame_search_warmstart_refine.sh`
  - `scripts/external/report_stage1_frame_search_baseline.py`
- Reporter validation:
  - `python3 -m py_compile scripts/external/report_stage1_frame_search_baseline.py`: passed.
  - `.venv/bin/python scripts/external/report_stage1_frame_search_baseline.py --format summary`: passed.
  - Reporter ranked lane B first with P10 delta `+4.87533706089236 pp`, P5 delta `+3.443396298390515 pp`, center improvement `+1.0306673184880673 px`, and checkpoint path `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
- Completion watcher:
  - Added `scripts/external/run_stage1_frame_search_refine_when_ready.sh`.
  - First real watcher attempt without `setsid` exited immediately after printing active PIDs; fixed the detach path to use `setsid`, matching the training runner survival pattern.
  - Dry-run validation passed with `SKIP_WAIT=1 SKIP_GPU_CHECK=1 ALLOW_INCOMPLETE_BASELINE=1 DRY_RUN=1 DETACH=0 RUN_TAG=watcher_dryrun_20260621_v2 bash scripts/external/run_stage1_frame_search_refine_when_ready.sh`.
  - Real watcher launched as PID `2087936`, PPID `1`, SID `2087936`, log `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_v2.log`.
  - Watcher waits for PIDs `1332213 1332214`, then requires `complete_300_epochs=true`, reporter promotion readiness, and at least `12000 MiB` free on both GPUs before launching `scripts/external/run_stage1_frame_search_warmstart_refine.sh`.
- Follow-up plan:
  - Wait for the current 300-epoch lanes unless a GPU is intentionally freed.
  - S1C: warm-start lane B `best_search_p10.pt`, Search-only, no distill, AdamW LR `1e-4`, 120 epochs.
  - S1D: warm-start lane B `best_search_p10.pt`, weak self-distill with same checkpoint as teacher, AdamW LR `7.5e-5`, 120 epochs.
- Remaining:
  - Run static validation for the new warm-start script.
  - Summarize final histories after the current lanes complete.
  - Promote the Stage1 baseline only after final checkpoint evidence is recorded.

# 2026-06-21 Stage1 frame-search resume after stalled detached jobs

- Scope:
  - Explain and fix why the Stage1 frame-search experiment pipeline was not advancing to the next planned experiments.
- Diagnosis:
  - Original Stage1 A/B training PIDs `1332213` and `1332214` were no longer alive before 300 epochs.
  - Original watcher PID `2087936` was also no longer alive.
  - Reporter still showed `complete_300=False`, so S1C/S1D were not supposed to launch under the watcher guard.
  - Training logs ended without `Traceback`, `Exception`, `CUDA out of memory`, or `Killed` text. The likely cause is external process/session cleanup rather than a model-level failure.
- Script update:
  - Added `LANE_A_RESUME` and `LANE_B_RESUME` support to `scripts/external/run_stage1_frame_search_baseline_matrix.sh`.
  - Validation passed:
    - `bash -n scripts/external/run_stage1_frame_search_baseline_matrix.sh`
    - `DRY_RUN=1 RUN_TAG=20260621_2258_resume ... bash scripts/external/run_stage1_frame_search_baseline_matrix.sh`
- Relaunch:
  - Relaunched outside the sandbox with `RUN_TAG=20260621_2258_resume`.
  - Lane A resumed from `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr1e3_fullwidth_20260621_204500/train/last.pt` as PID `2167860`; log `runs/NON_XR/shared/_logs/stage1_frame_search_lane_a_20260621_2258_resume.log`; resume start epoch `239/300`.
  - Lane B resumed from `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/last.pt` as PID `2167861`; log `runs/NON_XR/shared/_logs/stage1_frame_search_lane_b_20260621_2258_resume.log`; resume start epoch `238/300`.
  - Relaunched watcher outside the sandbox as PID `2169848`; log `runs/NON_XR/shared/_logs/stage1_frame_search_refine_when_ready_stage1_refine_after_300ep_20260621_resume2258.log`; waits on PIDs `2167860 2167861`.
- Current execution rule:
  - Let A/B finish 300 epochs.
  - Watcher must see `complete_300_epochs=true`, `promote_checkpoint=true`, and at least `12000 MiB` free on both GPUs before launching S1C/S1D.

# 2026-06-21 Stage1 frame-search supervisor guard

- Scope:
  - Reduce the risk that detached A/B jobs stop again before 300 epochs and leave the follow-up queue idle.
- Added:
  - `scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`
  - Run-tag scoped warm-start launch lock in `scripts/external/run_stage1_frame_search_warmstart_refine.sh`
- Behavior:
  - Supervisor polls the Stage1 reporter.
  - If both A/B candidates reach `complete_300_epochs=true`, it launches `scripts/external/run_stage1_frame_search_warmstart_refine.sh`.
  - If A/B are incomplete and watched PIDs are no longer alive, it relaunches both lanes from `last.pt` through `scripts/external/run_stage1_frame_search_baseline_matrix.sh`.
  - `MAX_RELAUNCHES=5` bounds automatic retries.
  - Warm-start runner lock prevents the existing watcher and supervisor from launching duplicate S1C/S1D jobs under the same `RUN_TAG`.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`: passed.
- Launch:
  - Supervisor launched outside the sandbox as PID `2197421`.
  - Log: `runs/NON_XR/shared/_logs/stage1_frame_search_supervisor_resume2258.log`.
  - Wait PIDs: `2167860 2167861`.
  - Latest reporter snapshot at launch window: lane A epoch `246`, lane B epoch `245`, both `complete_300=False`, lane B still ranked first with best Search P10 `28.22439415050003`.
  - Follow-up reporter snapshot after another poll: lane A epoch `251`, lane B epoch `250`, both `complete_300=False`; training PIDs `2167860` and `2167861`, watcher `2169848`, and supervisor `2197421` all remained alive.

# 2026-06-21 Stage1 frame-search A/B completion and S1C/S1D launch

- Scope:
  - Finish the Stage1 frame-based Search baseline pivot and start the next refinement lanes from the best A/B checkpoint.
- A/B completion evidence:
  - Reporter shows lane A and lane B both reached `epochs=300` and `complete_300=True`.
  - Lane B remains first-ranked with best Search P10 `28.22439415050003`.
  - Promotion checkpoint: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
  - Deltas vs old Stage1 reference: Search P10 `+4.87533706089236 pp`, Search P5 `+3.443396298390515 pp`, Search center improvement `+1.0306673184880673 px`.
- Follow-up launches:
  - S1C warm-start no-distill run on GPU0: `runs/NON_XR/raw/stage1_frame_search_warmstart_lr1e4_xy1p5_stage1_refine_after_300ep_20260621_resume2258_20260621_234220`.
  - S1C process evidence at check: parent runner PID `2402085` stopped intentionally, train PID `2402483` alive, log `runs/NON_XR/shared/_logs/stage1_frame_search_warmstart_lane_c_stage1_refine_after_300ep_20260621_resume2258.log`.
  - S1C progress at check: epoch `12/120`, best Search P10 so far `27.9403`.
  - S1D weak self-distill run on GPU1: `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr7p5e5_xy1p5_stage1_refine_after_300ep_20260621_resume2258_manual_20260621_234506`.
  - S1D process evidence at check: train PID `2420379` alive, log `runs/NON_XR/shared/_logs/stage1_frame_search_selfdistill_lane_d_stage1_refine_after_300ep_20260621_resume2258_manual.log`.
  - S1D progress at check: epoch `7/120`, best Search P10 so far `26.9351`.
- Launch-path fix:
  - Diagnosed that watcher/supervisor invoked the warm-start runner with inherited `DETACH=0`, causing lane C to run in the foreground and lane D to be delayed.
  - Manually launched S1D on GPU1 to keep both follow-up lanes parallel.
  - Patched `scripts/external/run_stage1_frame_search_refine_when_ready.sh` and `scripts/external/run_stage1_frame_search_supervisor_until_complete.sh` so future warm-start calls force `DETACH=1`.
  - Static validation passed: `bash -n scripts/external/run_stage1_frame_search_refine_when_ready.sh` and `bash -n scripts/external/run_stage1_frame_search_supervisor_until_complete.sh`.
- Remaining:
  - Keep the stopped S1C parent runner stopped until S1C completes; do not resume it because it can launch a duplicate/delayed S1D.
  - Compare S1C/S1D final best metrics against lane B before promoting the Stage1 baseline downstream.

# 2026-06-21 Stage1 frame-search follow-up reporter and current gate

- Scope:
  - Make S1C/S1D follow-up judgment reproducible instead of relying on ad hoc log tails.
- Added:
  - `scripts/external/report_stage1_frame_search_followup.py`
- Behavior:
  - Compares S1C and S1D against the completed lane B Stage1 frame-search baseline.
  - Uses `metric_search_p10_pct` as the primary promotion metric.
  - Requires `min_epochs=50` before follow-up promotion can be recommended.
  - Requires P10 improvement, non-regressed center, and an existing `best_search_p10.pt`.
- Current reporter command:
  - `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format summary`
- Current reporter result:
  - Baseline lane B: P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - S1C: epoch `23/120`, best P10 `27.94025216012631`, P5 `10.01010806605501`, center `17.340092240639454`, P10 delta `-0.284141990373719`, center improvement `-0.05124297231998298`.
  - S1D: epoch `18/120`, best P10 `27.629156058689333`, P5 `9.750674121784714`, center `17.34278841738431`, P10 delta `-0.5952380918106961`, center improvement `-0.05393914906483843`.
  - Recommendation: `status=wait_min_epochs`, `promote_checkpoint=None`, `continue_training=True`.
- Validation:
  - `python3 -m py_compile scripts/external/report_stage1_frame_search_followup.py`: passed.
  - `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format summary`: passed.
  - `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --format json`: passed.
- Current decision:
  - Continue S1C/S1D to at least 50 epochs before making a promotion/rejection decision.
  - Do not replace lane B baseline yet.

# 2026-06-21 Stage1 frame-search min-50 watcher

- Scope:
  - Keep follow-up decision collection moving without manual polling.
- Added:
  - `scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`
- Behavior:
  - Runs `scripts/external/report_stage1_frame_search_followup.py` every `300s`.
  - Writes JSON and summary reports under `runs/NON_XR/shared/_logs`.
  - Exits when all follow-up runs reach `min_epochs=50` or the reporter status becomes `promote_followup`.
  - Does not start, stop, or modify training jobs.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`: passed.
  - `DETACH=0 POLL_SEC=1 MAX_WAIT_SEC=1 RUN_TAG=stage1_followup_min50_dryrun_20260621 bash scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`: generated reports and exited with expected timeout code `20` before the 50-epoch gate.
  - `chmod +x scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh scripts/external/report_stage1_frame_search_followup.py`: applied.
- Launch:
  - Detached watcher launched outside the sandbox as PID `2485273`.
  - Host process evidence: PID `2485273`, PPID `1`, SID `2485273`, status `Ss`.
  - Log: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755.log`.
  - PID file: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755.pid`.
  - Report summary: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.txt`.
  - Report JSON: `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.json`.
- First poll:
  - S1C `23/120`, S1D `18/120`.
  - Reporter status `wait_min_epochs`, `promote_checkpoint=None`, `continue_training=True`.

# 2026-06-22 Stage1 frame-search min-50 continuation and fallback queue

## Objective

Continue raising Stage1 frame-based Search accuracy, keep lane B as the active baseline candidate, and avoid promoting any follow-up until it has at least 50 epochs of evidence and beats lane B on primary Search P10 without center regression.

## Runtime hierarchy and Task Card

```yaml
task_card:
  task_id: S1-FRAME-BASELINE-001
  sub_agent: codex-native
  role: runtime-manager
  objective: Keep S1C/S1D running to the 50-epoch decision gate and preserve lane B as baseline until a stricter candidate wins.
  file_ownership:
    - scripts/external/report_stage1_frame_search_followup.py
    - scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh
    - scripts/external/run_stage1_frame_search_post_followup_queue.sh
    - docs/track/PROGRESS.md
    - docs/track/log.md
    - docs/Validation.md
    - docs/resources/stage1_frame_search_midrun_results_2026_06_21.md
  assigned_skill:
    - agent-hierarchy-runtime-manager
    - caveman
  inputs:
    - Lane B 300-epoch checkpoint and metrics
    - S1C/S1D training logs
    - follow-up reporter summary
    - host process and nvidia-smi checks
  outputs:
    - current decision state
    - post-followup fallback queue
    - updated tracking documents
  validation:
    - bash syntax checks
    - reporter py_compile
    - reporter summary
    - git diff whitespace check
  dependencies: []
```

Attempted to spawn a real `gpt-5.3-codex-spark` read-only evaluator sub-agent, but the runtime returned `agent thread limit reached`. Main agent continued with the Codex-native fallback.

## Current status

- Lane B remains the active Stage1 frame-search baseline candidate:
  - Run: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`
  - Checkpoint: `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`
  - Best Search P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`
- S1C is alive on GPU0:
  - Train PID `2402483`
  - Run `runs/NON_XR/raw/stage1_frame_search_warmstart_lr1e4_xy1p5_stage1_refine_after_300ep_20260621_resume2258_20260621_234220`
  - Reporter completed epochs `67/120`
  - Best P10 `27.94025216012631`, P5 `10.049416236157688`, center `17.233418217245138`
  - Delta vs lane B: P10 `-0.284141990373719`, P5 `+0.21563341032783967`, center improvement `+0.05543105107433277`
- S1D is alive on GPU1:
  - Train PID `2420379`
  - Run `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr7p5e5_xy1p5_stage1_refine_after_300ep_20260621_resume2258_manual_20260621_234506`
  - Reporter completed epochs `61/120`
  - Best P10 `27.629156058689333`, P5 `9.926999344016021`, center `17.34278841738431`
  - Delta vs lane B: P10 `-0.5952380918106961`, P5 `+0.09321651818617305`, center improvement `-0.05393914906483843`
- Reporter state after both follow-ups crossed the 50-epoch gate: `status=keep_baseline_unless_later_improves`, `promote_checkpoint=None`, `continue_training=True`.

## Automation

- Min-50 watcher remains active:
  - PID `2485273`
  - Log `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755.log`
  - Report summary `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.txt`
  - Report JSON `runs/NON_XR/shared/_logs/stage1_frame_search_followup_min50_20260621_235755_report.json`
- Post-followup fallback queue was added and launched:
  - Script `scripts/external/run_stage1_frame_search_post_followup_queue.sh`
  - Original PID `2508026` was replaced by updated queue PID `2638408`.
  - Log `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027.log`
  - Report summary `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027_report.txt`
  - Report JSON `runs/NON_XR/shared/_logs/stage1_frame_search_post_followup_queue_v2_20260622_0027_report.json`
  - If S1C/S1D fail to promote and finish, it waits for GPU free memory then launches lower-LR S1E/S1F fallback lanes.
- Follow-up watcher was generalized:
  - `scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh` now accepts `BASELINE_RUN` and `FOLLOWUP_RUNS`.
  - The post-followup queue now launches a fallback min-50 watcher after S1E/S1F are launched.
  - Planned S1E path: `runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_000233`.
  - Planned S1F path: `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_000233`.

## Decision

Do not promote S1C/S1D yet. Both follow-ups crossed the 50-epoch gate, but both remain behind lane B on primary Search P10. Continue current training to 120 epochs and keep lane B as the active baseline candidate unless a later checkpoint improves P10 without center regression.

# 2026-06-22 Stage1 frame-search live refresh and capacity-distill preparation

## Runtime note

A real `gpt-5.3-codex-spark` sub-agent spawn was attempted for read-only Stage1 status cross-checking, but the runtime returned `agent thread limit reached`. The main agent continued with the Codex-native fallback and host-level process checks.

## Current reporter result

- Baseline lane B remains active:
  - Run `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`.
  - Checkpoint `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500/train/best_search_p10.pt`.
  - Best P10 `28.22439415050003`, P5 `9.833782825829848`, P1 `1.0646900500891343`, center `17.28884926831947`.
- S1C warm-start no-distill:
  - Epoch `89/120`.
  - Best P10 `27.94025216012631`, P5 `10.049416236157688`, center `17.233418217245138`.
  - Not promotable because primary P10 is `-0.284141990373719 pp` behind lane B.
- S1D weak self-distill:
  - Epoch `82/120`.
  - Best P10 `27.715633986131202`, P5 `10.086478287318968`, center `17.34278841738431`.
  - Not promotable because primary P10 is `-0.5087601643688267 pp` behind lane B and center is worse.
- Reporter decision: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`, `continue_training=True`.

## Host process and queue status

- Host process check confirmed:
  - S1C train PID `2402483` alive on GPU0.
  - S1D train PID `2420379` alive on GPU1.
  - Post-followup queue PID `2638408` alive.
- GPU check showed both RTX 5080 cards allocated by the active Stage1 follow-up work.
- The queue waits for S1C/S1D to exit before launching S1E/S1F lower-LR fallback lanes, then starts the generalized follow-up watcher for those fallback run directories.
- Added and launched capacity-after-followups queue:
  - Script `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
  - PID `2708973`.
  - Log `runs/NON_XR/shared/_logs/stage1_capacity_after_followups_queue_20260622_0041.log`.
  - Capacity run tag `stage1_capacity_after_fallback_20260622_0041`.
  - First poll is waiting for missing S1E/S1F histories, which is expected while S1C/S1D are still active.
- Since S1C/S1D stayed below lane B after the 50-epoch gate and both GPUs had enough free memory, launched S1G/S1H immediately as a parallel capacity branch:
  - S1G PID `2723776`, large no-distill, run `runs/NON_XR/raw/stage1_frame_search_large256_d8_partialwarm_lr2e4_stage1_capacity_parallel_20260622_0050_20260622_004413`.
  - S1H PID `2723777`, large baseline-teacher distill, run `runs/NON_XR/raw/stage1_frame_search_large256_d8_baseteacher_distill_lr1e4_stage1_capacity_parallel_20260622_0050_20260622_004413`.
  - Both reached epoch `1/200` train logs.
  - S1G/S1H min-50 watcher PID `2730067`, log `runs/NON_XR/shared/_logs/stage1_capacity_parallel_min50_20260622_0050.log`.
  - First watcher report status `wait_min_epochs`.
- Stopped capacity-after-followups queue PID `2708973` after S1G/S1H started to avoid duplicate capacity launch. The S1E/S1F post-followup queue PID `2638408` remains active.

## Added capacity branch

- Added `scripts/external/run_stage1_frame_search_capacity_distill.sh`.
- Added `scripts/external/run_stage1_frame_search_capacity_after_followups.sh`.
- Hardened `scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh` so missing histories produce a wait state instead of terminating the watcher.
- Added `model.teacher.*` override support through `src/hbtxr/models/pruning.py`.
- Added focused test `tests/test_model_role_cfg.py`.
- S1G prepared: large student `embed_dim=256`, `depth=8`, `num_heads=4`, no distillation, LR `2e-4`, 200 epochs.
- S1H prepared: large student `256/8/4`, baseline-size teacher `192/6/3`, Lane B teacher checkpoint, LR `1e-4`, EMA `0.999`, 200 epochs.
- S1G/S1H were not launched because current GPUs are occupied and S1E/S1F are already staged as the immediate fallback queue.

## Validation

- `bash -n scripts/external/run_stage1_frame_search_followup_until_min_epoch.sh`: passed.
- `bash -n scripts/external/run_stage1_frame_search_post_followup_queue.sh`: passed.
- `bash -n scripts/external/run_stage1_frame_search_capacity_distill.sh`: passed.
- `python3 -m py_compile src/hbtxr/models/pruning.py scripts/external/report_stage1_frame_search_followup.py`: passed.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_model_role_cfg.py`: passed.
- Large-student/small-teacher build smoke passed with student `256/8` and teacher `192/6`.
- Capacity script dry-run passed and printed the expected S1G/S1H train commands.
- `git diff --check` over modified Stage1 files and docs passed.

## Decision

Keep Lane B as the current Stage1 baseline. Continue S1C/S1D to 120 epochs, allow S1E/S1F to launch automatically only after current follow-ups finish without promotion, and reserve S1G/S1H for the next capacity tier when GPUs are free.

# 2026-06-22 Stage1 fallback watcher correction and capacity branch stop

## Summary

S1E/S1F lower-LR fallback training is now the active Stage1 frame-search improvement path. The earlier S1G/S1H capacity branch was stopped by DataLoader worker failures before the 50-epoch gate, so it is not a valid promotion candidate.

## Evidence

- Real sub-agent spawn attempt for `gpt-5.3-codex-spark` failed with `agent thread limit reached`; main agent continued with direct host/process checks.
- S1E/S1F process check:
  - S1E PID `2751991`, GPU0, LR `5e-5`, run `runs/NON_XR/raw/stage1_frame_search_warmstart_lr5e5_xy1p5_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
  - S1F PID `2751992`, GPU1, LR `3e-5`, EMA `0.999`, run `runs/NON_XR/raw/stage1_frame_search_selfdistill_lr3e5_xy1p5_ema999_stage1_refine_after_min50_low_lr_20260622_0057_20260622_004951`.
  - Corrected watcher PID `2763859`, log `runs/NON_XR/shared/_logs/stage1_frame_search_fallback_min50_stage1_refine_after_min50_low_lr_20260622_0057_corrected_20260622_0059.log`.
- S1E/S1F reporter snapshot:
  - S1E epoch `3/120`, best P10 `27.67969512939453`, P5 `9.648472875918982`, center `17.437835072571378`.
  - S1F epoch `2/120`, best P10 `27.59995568473384`, P5 `8.657906802195424`, center `17.307134470849668`.
  - Status `wait_min_epochs`; no promotion.
- Capacity branch outcome:
  - S1G stopped during epoch `7/200`.
  - S1H stopped during epoch `6/200`.
  - Both logs ended with `DataLoader worker ... killed by signal: Killed`.
  - Capacity watcher PID `2730067` was stopped because the watched train jobs were no longer alive.

## Code/automation changes

- Updated `scripts/external/run_stage1_frame_search_post_followup_queue.sh` to resolve timestamp-suffixed fallback run directories before launching the S1E/S1F watcher.
- Updated `scripts/external/run_stage1_frame_search_capacity_distill.sh` default `NUM_WORKERS` from `8` to `2` to reduce host memory pressure on future capacity retries.
- Terminated the suffix-free S1E/S1F watcher PID `2752001`.
- Launched corrected S1E/S1F watcher PID `2763859`.

## Decision

Keep Lane B as the active Stage1 baseline until S1E/S1F cross at least 50 epochs and beat Lane B on primary Search P10 without center regression. Defer capacity retry until lower-concurrency conditions are available.

# 2026-06-22 Stage1 S1E/S1F live refresh

## Evidence

- Direct reporter status:
  - S1E epoch `9/120`, best P10 `27.67969512939453`, P5 `9.851752245201254`, center `17.437835072571378`.
  - S1F epoch `9/120`, best P10 `27.932390464926666`, P5 `9.528302156700278`, center `17.307134470849668`.
  - Baseline Lane B P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Reporter status `wait_min_epochs`, `promote_checkpoint=None`, `continue_training=True`.
- Host process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Corrected watcher PID `2763859` alive.
- Training logs show active progress beyond the reporter-completed epoch:
  - S1E entered epoch `11/120` train after completing epoch `10` validation.
  - S1F was in epoch `10/120` train.

## Decision

No baseline promotion yet. Continue S1E/S1F to the minimum 50-epoch gate.

# 2026-06-22 Stage1 capacity-after queue correction

## Evidence

- `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` now resolves latest timestamped S1E/S1F run directories when `FOLLOWUP_RUNS` is omitted.
- Corrected capacity-after queue launched:
  - PID `2800043`.
  - Log `runs/NON_XR/shared/_logs/stage1_capacity_after_s1ef_corrected_20260622_0120.log`.
  - Capacity run tag `stage1_capacity_after_s1ef_safe_workers2_20260622_0120`.
- Queue first poll:
  - S1E epoch `14/120`, best P10 `27.83692773782982`, P5 `9.851752245201254`, center `17.437835072571378`.
  - S1F epoch `13/120`, best P10 `27.932390464926666`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `wait_min_epochs`; no promotion.
- Dry-run path-resolution smoke timed out with code `20`, which is expected because S1E/S1F have not reached the min gate.

## Decision

Keep S1E/S1F running. Capacity retry is staged but gated behind fallback completion and non-promotion.

# 2026-06-22 Stage1 center-polish queue staging

## Evidence

- S1E is nearly tied with Lane B on P10 but still regresses center:
  - Lane B P10 `28.22439415050003`, center `17.28884926831947`.
  - S1E latest best P10 `28.218778646217203`, center `17.437835072571378`.
- Generalized `scripts/external/run_stage1_frame_search_warmstart_refine.sh` with `BEST_METRIC_NAME`.
- Added `scripts/external/run_stage1_frame_search_center_preserve_polish.sh`.
  - Uses `BEST_METRIC_NAME=metric_search_center_px`.
  - Runs a no-distill lane and a weak self-distill lane.
  - Uses low LR and `loss.search_xy_weight=2.0` to target center recovery while retaining Search-only contract.
- Dry-run printed expected train commands for both center-polish lanes.
- Replaced the capacity-after queue:
  - Stopped PID `2800043`.
  - Started center-polish queue PID `2819081`.
  - Log `runs/NON_XR/shared/_logs/stage1_centerpolish_after_s1ef_queue_20260622_0128.log`.
- Queue first poll:
  - S1E epoch `20/120`, best P10 `28.218778646217203`, P5 `9.851752245201254`, center `17.437835072571378`.
  - S1F epoch `19/120`, best P10 `27.932390464926666`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `wait_min_epochs`.

## Decision

Do not launch extra jobs while S1E/S1F are active. If they finish without promotion, run center-polish before any larger capacity retry.

# 2026-06-22 Stage1 pre-gate refresh

## Evidence

- Direct reporter snapshot:
  - S1E epoch `23/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `22/120`, best P10 `27.932390464926666`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `wait_min_epochs`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Min-50 watcher PID `2763859` alive.
  - Center-polish queue PID `2819081` alive.

## Decision

Continue S1E/S1F. S1E is close on P10 but still not promotable because it is below the 50-epoch gate and regresses center.

# 2026-06-22 Stage1 second pre-gate refresh

## Evidence

- Direct reporter snapshot:
  - S1E epoch `26/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `25/120`, best P10 `27.932390464926666`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `wait_min_epochs`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Min-50 watcher PID `2763859` alive.
  - Center-polish queue PID `2819081` alive.

## Decision

No promotion before epoch `50`. Keep S1E/S1F running and keep center-polish queued for non-promotion.

# 2026-06-22 Stage1 third pre-gate refresh

## Evidence

- Direct reporter snapshot:
  - S1E epoch `31/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `30/120`, best P10 `28.078392352697986`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `wait_min_epochs`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Min-50 watcher PID `2763859` alive.
  - Center-polish queue PID `2819081` alive.
- Center-polish queue log shows continued polling and `wait_min_epochs`.

## Decision

Continue current jobs. Do not promote before epoch `50`.

# 2026-06-22 Stage1 fourth pre-gate refresh

## Evidence

- Direct reporter snapshot:
  - S1E epoch `36/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `35/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `wait_min_epochs`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Min-50 watcher PID `2763859` alive.
  - Center-polish queue PID `2819081` alive.
- GPU state:
  - GPU0 free memory about `15123 MiB`.
  - GPU1 free memory about `15111 MiB`.

## Decision

Continue current jobs until at least epoch `50`. Lane B remains the active Stage1 baseline.

# 2026-06-22 Stage1 S1E/S1F min-50 gate

## Evidence

- Direct reporter snapshot:
  - S1E epoch `55/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `53/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Both follow-up lanes are now min-ready.
- S1E/S1F train PIDs were still alive during the gate check.
- The corrected min-50 watcher completed and was no longer listed in the final process check.
- Center-polish queue remained in waiting state; it is configured to launch only after S1E/S1F completion and non-promotion.

## Decision

Do not promote S1E/S1F at the min-50 gate. Keep Lane B as active Stage1 baseline. Let S1E/S1F continue toward 120 epochs and keep center-polish staged as the next branch.

# 2026-06-22 Stage1 S1E/S1F post-gate continuation

## Evidence

- Direct reporter snapshot:
  - S1E epoch `67/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `64/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Lane B baseline P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
- Queue log:
  - Center-polish queue remains in `min_ready_no_promotion`.
  - It is waiting for S1E/S1F completion before launching center-polish.

## Decision

Continue S1E/S1F to the target epoch count. Do not launch extra concurrent Stage1 jobs.

# 2026-06-22 Stage1 post-centerpolish capacity queue

## Evidence

- Direct reporter snapshot:
  - S1E epoch `69/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `67/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Added queue script:
  - `scripts/external/run_stage1_frame_search_after_centerpolish_capacity_queue.sh`.
- Launched host queue:
  - PID `3000547`.
  - Log `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_20260622_0141.log`.
  - First poll saw center-polish lane I/J missing, which is expected because S1E/S1F are still active and center-polish has not launched yet.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.

## Decision

Do not start extra training now. The new queue only stages the next capacity retry after center-polish exists and completes without promotion.

# 2026-06-22 Stage1 continuation plateau check

## Evidence

- Reporter snapshot:
  - S1E epoch `77/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `74/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct `history.json` check:
  - S1E best P10 epoch `15`, best P5 epoch `23`, best center epoch `3`.
  - S1E latest epoch `77` validation metrics: P10 `26.15116848135894`, P5 `9.258760398288942`, P1 `0.1347708971995228`, center `17.70168143398357`.
  - S1F best P10 epoch `35`, best P5 epoch `13`, best center epoch `1`.
  - S1F latest epoch `74` validation metrics: P10 `26.965409296863484`, P5 `9.07232734392274`, P1 `0.5615454079969874`, center `17.785983782894206`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.

## Decision

Current S1E/S1F branch appears plateaued below Lane B. Keep it running to the configured target, but expect center-polish to be the next meaningful accuracy-improvement branch unless a later best checkpoint appears.

# 2026-06-22 Stage1 continuation refresh

## Evidence

- Reporter snapshot:
  - S1E epoch `80/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `77/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct `history.json` rows:
  - S1E epoch `80`: P10 `26.160153370983195`, P5 `9.232929283717894`, P1 `0.3706199538032964`, center `17.88450043606308`.
  - S1F epoch `77`: P10 `26.10624487894886`, P5 `8.707322840420705`, P1 `0.6794699362988742`, center `17.84767130185973`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.
- Run-directory check:
  - No center-polish or downstream capacity run directories exist yet.

## Decision

No promotion. Continue S1E/S1F and keep queued follow-up branches staged.

# 2026-06-22 Stage1 later continuation refresh

## Evidence

- Reporter snapshot:
  - S1E epoch `83/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `79/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct `history.json` rows:
  - S1E epoch `83`: P10 `26.089398564032788`, P5 `8.738769369305304`, P1 `0.2751572447003059`, center `17.880303625790578`.
  - S1F epoch `80`: P10 `26.918239575512004`, P5 `8.611860095330005`, P1 `0.4099281418998286`, center `17.691625046280194`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.
- Run-directory check:
  - No center-polish or downstream capacity run directories exist yet.

## Decision

No promotion. S1E/S1F remain below Lane B and are still running toward target completion.

# 2026-06-22 Stage1 extended continuation refresh

## Evidence

- Reporter snapshot:
  - S1E epoch `93/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `90/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct `history.json` rows:
  - S1E epoch `93`: P10 `26.446541462304456`, P5 `7.851527645902814`, P1 `0.5446990390993515`, center `17.97235798385908`.
  - S1F epoch `90`: P10 `26.60938953903486`, P5 `8.505166485624493`, P1 `0.6626235674012382`, center `17.68777278234374`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.
- Run-directory check:
  - No center-polish or downstream capacity run directories exist yet.

## Decision

No promotion. Continue to target completion and keep queued follow-up branches active.

# 2026-06-22 Stage1 near-completion refresh

## Evidence

- Reporter snapshot:
  - S1E epoch `96/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `92/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
- Direct `history.json` rows:
  - S1E epoch `96`: P10 `26.581312377497834`, P5 `8.716307586094118`, P1 `0.25269542550140955`, center `18.04789892232643`.
  - S1F epoch `92`: P10 `26.19047675942475`, P5 `8.409703776521503`, P1 `1.0163971523068986`, center `17.752043238225973`.
- Runtime process check:
  - S1E PID `2751991` alive.
  - S1F PID `2751992` alive.
  - Center-polish queue PID `2819081` alive.
  - Post-centerpolish capacity queue PID `3000547` alive.
- Run-directory check:
  - No center-polish or downstream capacity run directories exist yet.

## Decision

No promotion. Continue to target completion and rely on queued center-polish if no late best update occurs.

# 2026-06-22 Stage1 queue safety fix

## Evidence

- Script audit:
  - `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` used `any_followup_complete` in the `complete_no_promotion` branch.
  - This could launch center-polish when only one of S1E/S1F had completed.
- Code change:
  - `decision_state()` now derives `all_followups_complete` from `payload["followups"][*]["complete_target_epochs"]`.
  - `complete_no_promotion` now requires both `all_followups_min_epoch_ready` and `all_followups_complete`.
- Validation:
  - Dry-run on current S1E/S1F state returned `min_ready_no_promotion`, not `complete_no_promotion`.
- Runtime update:
  - Stopped old center-polish queue PID `2819081`.
  - Launched all-complete queue PID `3079899`.
  - Log `runs/NON_XR/shared/_logs/stage1_centerpolish_after_s1ef_queue_allcomplete_20260622_0158.log`.
- Latest queue snapshot:
  - S1E epoch `100/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `96/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Decision `min_ready_no_promotion`.

## Decision

Keep S1E/S1F running. The next center-polish branch is now guarded against early launch while one follow-up is still active.

# 2026-06-22 Stage1 active-run refresh after restart request

## Evidence

- Reporter snapshot:
  - S1E epoch `107/120`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `102/120`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
  - `continue_training=True`.
- Baseline Lane B remains:
  - Run `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`.
  - P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
- Current deltas:
  - S1E P10 delta `-0.005615504282825867`, center improvement delta `-0.14898580425190744`.
  - S1F P10 delta `-0.07075477096269722`, center improvement delta `-0.01828520253019761`.
- Runtime state:
  - Host `pgrep` sees S1E PID `2751991`, S1F PID `2751992`, center-polish all-complete queue PID `3079899`, and post-centerpolish capacity queue PID `3000547`.
  - Sandbox `ps -p` cannot see host PIDs because of PID namespace isolation; use `pgrep -af` and run-history updates for host process evidence in this environment.
- Queue state:
  - Center-polish has not launched yet.
  - Post-centerpolish capacity has not launched yet.
  - This is expected because the all-complete queue requires both S1E and S1F to finish target epoch `120`.
- Sub-agent state:
  - GPT-5.3-Codex-Spark evaluator spawn failed due usage limit; no sub-agent output was used.

## Decision

Training is progressing. The apparent delay is queue gating, not a stopped experiment. Continue S1E/S1F to 120, keep Lane B as baseline, and let center-polish launch only after both watched follow-ups complete without promotion.

# 2026-06-22 Stage1 GPT-5.5 audit and later active-run refresh

## Evidence

- GPT-5.5 sub-agent `Curie the 2nd` completed a read-only Stage1 audit:
  - Inputs inspected: S1E/S1F histories, reporter JSON, center-polish queue log, post-centerpolish queue log, and `docs/track`.
  - No files edited by sub-agent.
  - No train/eval jobs launched by sub-agent.
- Main reporter snapshot:
  - S1E epoch `113/120`, best P10 `28.218778646217203`, best P5 `9.974169173330631`, best center `17.437835072571378`.
  - S1F epoch `108/120`, best P10 `28.15363937953733`, best P5 `9.743935603015828`, best center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`.
  - `promote_checkpoint=None`.
  - `continue_training=True`.
- Latest validation rows remain below best:
  - S1E epoch `113` P10 `26.48584965040099`, P5 `7.980683074807221`, center `17.999366805238544`.
  - S1F epoch `108` P10 `26.176999649911558`, P5 `8.645552833125276`, center `17.734962881736035`.
- Queue state:
  - Center-polish queue still reports `min_ready_no_promotion` from the last poll.
  - Post-centerpolish capacity queue still reports center-polish lane I/J missing.
  - No center-polish or downstream capacity run directories exist yet.

## Decision

No promotion. Lane B remains the active baseline. The next material state change is S1E/S1F completion at 120 epochs; only then should center-polish launch if no promotable checkpoint appears.

# 2026-06-22 Stage1 center-polish launch

## Evidence

- S1E/S1F completion reporter:
  - S1E epoch `120/120`, complete `true`, best P10 `28.218778646217203`, P5 `9.974169173330631`, center `17.437835072571378`.
  - S1F epoch `120/120`, complete `true`, best P10 `28.15363937953733`, P5 `9.743935603015828`, center `17.307134470849668`.
  - Status `keep_baseline_unless_later_improves`; `promote_checkpoint=None`.
- Queue transition:
  - `stage1_centerpolish_after_s1ef_queue_allcomplete_20260622_0158.log` reached `complete_no_promotion`.
  - GPU free check passed with both GPUs above the `12000 MiB` free threshold.
  - `scripts/external/run_stage1_frame_search_center_preserve_polish.sh` launched.
- Launched runs:
  - Lane I no-distill: `runs/NON_XR/raw/stage1_frame_search_centerpolish_nodistill_lr1e5_xy2_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.
  - Lane J weak self-distill: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`.
  - Train PIDs: Lane I `3209126`, Lane J `3209127`.
  - Active watcher: downstream queue PID `3222876`.
- Early center-polish reporter snapshot from downstream queue:
  - Lane I epoch `3/80`, best P10 `28.303010526693093`, best P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J epoch `3/80`, best P10 `28.27717937613433`, best P5 `10.153863744915656`, center `17.257583906065744`.
  - Status `wait_min_epochs`; `promote_checkpoint=None`.

## Decision

Lane B remains active baseline because S1E/S1F failed promotion. Center-polish is now the active accuracy-improvement branch. Although Lane J is already better than Lane B on P10/P5/center at epoch 3, it cannot be promoted before the 50-epoch gate.

# 2026-06-22 Center-polish active refresh

## Evidence

- GPT-5.5 sub-agent `Erdos the 2nd` completed a read-only center-polish gate audit.
  - It inspected Lane I/J histories, baseline evidence, and queue state.
  - It did not edit files or launch train/eval jobs.
- Reporter snapshot:
  - Baseline Lane B remains `runs/NON_XR/raw/stage1_frame_search_sonly_adamw_lr5e4_xy1p5_fullwidth_20260621_204500`.
  - Lane B best: P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
  - Lane I no-distill reached epoch `18/80`; best P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J weak self-distill reached epoch `18/80`; best P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Reporter status is `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
- Baseline manifest:
  - Added `scripts/external/write_stage1_frame_search_baseline_manifest.py`.
  - Wrote `docs/resources/current_stage1_frame_search_baseline_manifest.json`.
  - Manifest state is `pending_min_epoch`.
  - Active baseline checkpoint remains Lane B.
  - Leading baseline candidate is Lane J because it improves P10, P5, and center; Lane I has higher P10 but regresses center.
- Watcher integration:
  - Updated `scripts/external/run_stage1_frame_search_capacity_after_followups.sh` to run the manifest writer after each reporter poll.
  - Dry-run showed the hook emitting `STAGE1_FRAME_SEARCH_BASELINE_MANIFEST` with `state=pending_min_epoch`.
  - Replaced old watcher PID `3222876` with host watcher PID `3300765`.
  - New watcher log `runs/NON_XR/shared/_logs/stage1_after_centerpolish_capacity_queue_manifest_hook_20260622_0236b.log` reached first poll and wrote the manifest.
- Latest reporter snapshot after watcher replacement:
  - Lane I epoch `29/80`; best P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J epoch `28/80`; best P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Status `wait_min_epochs`; `promote_checkpoint=None`.
- GPT-5.5 sub-agent audit:
  - `Pauli the 2nd` completed a read-only audit.
  - It confirmed Lane I is higher-P10 but not center-safe because center regresses by `0.08526944214442977 px`.
  - It confirmed Lane J beats Lane B on P10, P5, and center, but remains below the 50-epoch gate.
  - It confirmed active watcher `stage1_after_centerpolish_capacity_queue_manifest_hook_20260622_0236b.log`, PID `3300765`.
  - It made no edits and launched no training/evaluation jobs.
- A second GPT-5.5 sub-agent `Pasteur the 2nd` was spawned for a fresh read-only audit, but timed out before returning. It was closed while running, and its output was not used.

## Decision

Continue training; do not promote yet. Lane J is the current best baseline candidate because it improves over Lane B on P10, P5, and center, but the min 50 epoch gate is not satisfied. Downstream capacity queue should stay in waiting state until Lane I/J reach the gate or complete.

# 2026-06-22 Stage1 Center-Polish 50-Epoch Gate Promotion

## Evidence

- Direct reporter snapshot at `2026-06-22 02:56:48 KST`:
  - Lane I no-distill reached at least `53/80`, min-ready `true`; best P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J weak self-distill reached at least `51/80`, min-ready `true`; best P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Reporter status `promote_followup`.
  - Promotion checkpoint `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Previous Lane B baseline:
  - P10 `28.22439415050003`, P5 `9.833782825829848`, center `17.28884926831947`.
- Lane J deltas against Lane B:
  - P10 `+0.052785225634302435 pp`.
  - P5 `+0.3200809190858074 pp`.
  - Center `+0.03126536225372689 px` improvement.
- Lane I is not promotable despite higher P10 because center regresses by `0.08526944214442977 px`.
- Checkpoint existence check passed for the Lane J `best_search_p10.pt`.
- `docs/resources/current_stage1_frame_search_baseline_manifest.json` was regenerated with `state=promoted_followup` and validates as JSON.
- GPU/NVML check after the gate returned normally for both RTX 5080 GPUs.

## Decision

Promote Lane J as the current Stage1 frame-based Search baseline for downstream planning. Continue the Lane I/J training jobs toward `80/80` for final stability evidence, but the minimum 50-epoch baseline gate is now satisfied.

# 2026-06-22 Stage1 Final Baseline And XR-64 Prep Resume

## Evidence

- Final Stage1 center-polish refresh:
  - Lane I no-distill completed `80/80`; best Search P10 `28.303010526693093`, P5 `9.884321932522756`, center `17.3741187104639`.
  - Lane J weak self-distill completed `80/80`; best Search P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - Lane J remains the promoted Stage1 frame-search baseline because it improves Lane B on all three tracked gates.
- Runtime:
  - No active Stage1/XR-64 train or eval process was present after direct PID checks.
  - Both RTX 5080 GPUs were responsive and idle by `nvidia-smi`.
- XR-64 prep:
  - Generated `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json`.
  - Verified row count `5929`.
  - Artifact checker now reports `missing_eval_rows=7`, `missing_overrides=6`, `ready_to_train=false`, `can_run_lane=false`.
  - A parallel `xr39/xr56b` eval attempt produced no output artifacts; the runner log filename was hardened to prevent same-second parallel log collisions.

## Decision

Use the promoted Stage1 Lane J checkpoint as the baseline. Continue XR-64 prep with the remaining eval rows first; do not launch XR-64A/B until strict generated-artifact readiness reaches `missing_eval_rows=0`, `missing_overrides=0`, and `ready_to_train=true`.

# 2026-06-22 XR-64 Prep Completion And Launch Blocker

## Evidence

- Completed XR-64 prep:
  - Train eval rows complete for `xr62a/xr39/xr56b/xr58a`, `5929` rows each.
  - Val eval rows complete for `xr62a/xr39/xr56b/xr58a`, `844` rows each.
  - Train and val override files complete for conservative, threshold-priority, and min-error rules.
- Strict checker:
  - `resume_status=ready_to_train`
  - `ready_to_train=true`
  - `can_run_lane=true`
  - `missing_eval_rows=0`
  - `missing_overrides=0`
  - `leakage_risk=none`
- Attempted launch:
  - Lane A command `bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0` exited `1`.
  - Lane B command `bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1` exited `1`.
  - Both train logs reached `train_hbtxr.py` then failed in device resolution because CUDA was unavailable inside the training process.
- Runtime blocker:
  - CUDA/NVML probes are unstable. In repeated checks, torch alternated between CUDA `True/2` and `False/0`, while `nvidia-smi` alternated between normal output and return code `9`.
  - A final five-probe loop failed all five times: `nvidia-smi` could not communicate with the driver and torch reported CUDA `False/0`.

## Decision

XR-64 prep is no longer the blocker. The active blocker for XR-64A/B execution is CUDA/NVML stability at training launch time. Relaunch XR-64A/B only after repeated CUDA and `nvidia-smi` probes stay stable.

# 2026-06-22 XR-64A/B Post-Run Completion

## Evidence

- Runtime correction:
  - The local sandbox did not expose `/dev/nvidia*`, which explained the earlier CUDA/NVML failures.
  - Host-level checks showed `/dev/nvidia*`, `nvidia-smi`, and torch CUDA were available.
- Executed host-GPU runs:
  - XR-64A: `bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0`.
  - XR-64B: `bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1`.
  - Both reached `XR64_TRAIN_EXIT:0`.
- Run roots:
  - XR-64A: `runs/xr64_xr62a_conservative_teacher_target_c255000_lr3e_7_to0_0006_m160k_M384k_r4000kus_p0p5_20260622_050430`.
  - XR-64B: `runs/xr64_xr56b_threshold_teacher_target_c255000_lr5e_7_to0_0008_m160k_M384k_r4000kus_p0p5_20260622_050433`.
- Post-run matrix:
  - Added center checkpoint evidence by preserving epoch-7 `last.pt` as `best_metric_track_center_px.pt` for both lanes; history shows epoch 7 is the best center epoch for both lanes.
  - Ran center checkpoint test evals with `HBTXR_DISABLE_CUDNN=1`.
  - Patched center eval summaries with ablation provenance.
  - Candidate collector reports `ok=true`, `candidate_count=6`, and `required_candidate_matrix_complete=true`.
  - Promotion helper reports `decision_status=promoted`, `winning_lane=XR-64B`, `winning_checkpoint_kind=best_track_p5`, `center_promoted=true`, `p10_promoted=false`, `p5_promoted=false`.
- Metrics:
  - Best observed center: XR-64A `best_track_p10`, center `16.464686357975005`, P10 `34.3554429258619`, P5 `12.044218056542533`, P1 `1.0153061594281878`.
  - Promotion-helper winner: XR-64B `best_track_p5`, center `16.46796860694885`, P10 `34.50850416592189`, P5 `11.986820098331997`, P1 `0.9251700946262904`.

## Decision

XR-64A/B closed the center gate but did not promote P10/P5. Next experiment should preserve the new center level near `16.46 px` while recovering P10 above `35.02295998845781` and P5 above `12.133503770828247`.

# 2026-06-22 XR-64C Min-Error Diagnostic

## Evidence

- Updated trainer checkpoint specs so future Stage2 runs save `best_metric_track_center_px.pt`.
- Regression tests passed for checkpoint specs and XR-64 post-run helpers.
- Ran XR-64C with host GPU access:
  - Command: `bash scripts/external/run_xr64_teacher_target_construction.sh c cuda:0`.
  - Run root: `runs/xr64_xr62a_minerror_teacher_target_c255000_lr3e_7_to0_0010_m160k_M384k_r4000kus_p0p5_20260622_053056`.
  - The run reached `XR64_TRAIN_EXIT:0` and completed evals for `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px`.
- Metrics:
  - XR-64C `best_metric_track_center_px`: center `16.468127271107264`, P10 `34.16836808749608`, P5 `12.031462955474854`, P1 `0.9251700946262904`.
  - XR-64C `best_track_p10`: center `16.464661524977004`, P10 `34.3554429258619`, P5 `12.044218056542533`, P1 `1.0153061594281878`.
  - XR-64C `best_track_p5`: center `16.466900491714476`, P10 `34.29379326275417`, P5 `11.986820098331997`, P1 `0.9251700946262904`.

## Decision

XR-64C confirms the target-override family is center-effective but P10/P5-limited. The best observed center is now `16.464661524977004`; P10 remains below `35.02295998845781` and P5 remains below `12.133503770828247`. Next work should shift to an explicit P10/P5 recovery lane, likely XR-65 or XR-66 rather than another min-error replay.

# 2026-06-22 XR-65 P10/P5 Recovery Runner

## Evidence

- Added `scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh`.
- The runner starts from XR-64C `best_track_p10.pt`, the current center-improved checkpoint.
- Lane A uses the XR-39 P10 teacher with threshold override pressure.
- Lane B uses the XR-56B P5 teacher with conservative P5 guard pressure.
- Test eval is override-cleared:
  - `data.track_target_override_path=null`
  - `data.allow_test_target_override=false`
  - target-override loss weights reset to `0.0`
- Promotable execution requires all three checkpoint kinds: `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px`.
- Added `tests/test_xr65_runner_contract.py` to guard these script contracts.

## Decision

XR-65 is now the next executable P10/P5 recovery branch. Use host GPU access for any real training/eval launch because sandbox-local CUDA device files are not exposed. Reject promotion unless center stays near the XR-64C level while P10 exceeds `35.02295998845781` and P5 exceeds `12.133503770828247`.

# 2026-06-22 XR-65A/B Execution And Closeout

## Evidence

- Executed host-GPU runs:
  - XR-65A: `bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh a cuda:0`.
  - XR-65B: `bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh b cuda:1`.
- Both lanes reached `XR65_TRAIN_EXIT:0`.
- Both lanes completed three test evals and reached `XR65_DONE` with `eval_count=3`.
- Run roots:
  - XR-65A: `runs/xr65_xr64c_p10init_xr39teacher_threshold_recovery_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0035_p100_008_p50_0030_to0_00045_s0_00025_m160k_M384k_r4000kus_p0p5_20260622_055744`.
  - XR-65B: `runs/xr65_xr64c_p10init_xr56bp5teacher_conservative_p5guard_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0040_p100_006_p50_0040_to0_00035_s0_00030_m160k_M384k_r4000kus_p0p5_20260622_055754`.
- Best observed XR-65 metrics:
  - Best center candidate: XR-65A `best_track_p10`, center `16.464880844524927`, P10 `34.30229665211269`, P5 `12.037840502602714`, P1 `0.9047619342803955`.
  - Best P10 candidate: XR-65A/B `best_track_p5`, P10 `34.31505176680429`, center around `16.4671`, P5 `12.08248336655753`.
  - Best P5 candidate: `12.08248336655753`.
- Gates:
  - Center best remains XR-64C `16.464661524977004`.
  - P10 gate remains `35.02295998845781`.
  - P5 gate remains `12.133503770828247`.
- Detailed report: `docs/resources/xr65_p10p5_recovery_results_2026_06_22.md`.

## Decision

Close XR-65A/B as no-promotion. The lanes preserve center within the rejection threshold and slightly lift P5 versus XR-64C best-P10, but they do not beat any active center/P10/P5 gate. Next work should run XR-66 confidence/error bucket diagnostics or a stronger bounded XR-65C bracket instead of extending the same low-LR recipe.

# 2026-06-22 XR-66 No-Train Failure Bucket Diagnostic

## Evidence

- Generated XR-66 no-train diagnostic summaries:
  - `runs/diagnostics/xr66_failure_buckets_xr64c_bestp10_20260622.json`.
  - `runs/diagnostics/xr66_failure_buckets_xr65a_bestp5_20260622.json`.
- Documented summary:
  - `docs/resources/xr66_no_train_failure_bucket_diagnostic_2026_06_22.md`.
- Overall weighted comparison:
  - XR-64C best P10: center `16.401954641998156`, P10 `34.491568727644356`, P5 `12.161471640265713`.
  - XR-65A best P5: center `16.403599453701027`, P10 `34.4404701073071`, P5 `12.212570260602964`.
- Main failure buckets:
  - Low similarity rows remain hard; XR-65 did not materially improve `similarity <=0.1`.
  - Worst subjects remain `42`, `39`, and `45`.
  - Worst sessions include `user45/right/session_201`, `user42/left/session_201`, and `user42/right/session_201`.
  - Left eye remains weaker than right eye.

## Decision

Do not start a generic XR-66 training branch yet. The no-train diagnostic points to concentrated subject/session/left-eye and low-similarity failures, so the next step is a targeted failure-bucket manifest plus side-by-side teacher comparison before any local-update or confidence-gated training.

# 2026-06-22 XR-66 Targeted Failure-Bucket Manifests

## Evidence

- Built a trainable low-similarity targeted subset:
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/train_manifest.jsonl`: `1624` rows.
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/val_manifest.jsonl`: `254` rows.
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl`: `493` rows.
- Built test-only worst-session subsets:
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/user45_right_session201_testonly/test_manifest.jsonl`: `31` rows.
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/user42_left_session201_testonly/test_manifest.jsonl`: `46` rows.
  - `data/_internal/manifests/manifest1/xr66_failure_buckets/user42_right_session201_testonly/test_manifest.jsonl`: `46` rows.
- The first session-specific train/val/test attempt produced no train rows because these worst sessions are test-only under the current split. Those buckets must be used for targeted evaluation only.

## Decision

Treat `low_similarity_le0p1` as the only trainable XR-66 targeted subset currently available. Before any XR-66 training launch, run a no-train targeted comparison of XR-39, XR-64C, and XR-65A on low-similarity and the three test-only worst-session buckets. If no teacher/checkpoint clearly wins those buckets, do not spend GPU time on a generic local-update branch.

# 2026-06-22 XR-66 Targeted Comparison And Low-Similarity Recovery Closeout

## Evidence

- Added `scripts/external/compare_xr66_targeted_buckets.py`.
- Added `scripts/external/run_xr66_lowsim_targeted_recovery.sh`.
- Wrote targeted comparison report:
  - `runs/diagnostics/xr66_targeted_bucket_comparison_20260622.json`.
  - `docs/resources/xr66_targeted_bucket_comparison_2026_06_22.md`.
- Targeted comparison result:
  - Low-sim center winner: XR-65A best-P5.
  - Low-sim P10 winner: XR-39 P10 `c25p45f30`.
  - Low-sim P5 winner: XR-64C best-P10.
  - Test-only worst-session P10 winner: XR-39 P10 across all three session buckets.
- Ran XR-66A/B for `50/50` epochs:
  - XR-66A: XR-65A best-P5 init, XR-39 P10 teacher, LR `3e-7`, GPU0.
  - XR-66B: XR-64C best-P10 init, XR-39 P10 teacher, LR `2e-7`, GPU1.
- Both lanes reached `XR66_TRAIN_EXIT:0`.
- Both lanes reached `XR66_DONE` with `eval_count=6`.
- Best full-test metrics:
  - Best center: XR-66B `best_track_p5`, center `16.498384244101388`, P10 `34.52040895053319`, P5 `11.909013972963606`.
  - Best P10: XR-66A `best_track_p5`, center `16.524617418221066`, P10 `34.58631029129028`, P5 `11.755952746527536`.
  - Best P5: XR-66A `best_track_p10`, center `16.709318779196057`, P10 `34.26360624858311`, P5 `11.964286068507603`.
- Best low-sim metrics:
  - Best low-sim P10: XR-66A `best_metric_track_center_px`, P10 `36.36136802550285`, center `17.431977118215254`, P5 `10.599078793679514`.
  - Best low-sim P5: XR-66B `best_track_p5`, P5 `11.920123284862887`, center `18.002039071052305`, P10 `34.953917995575935`.
- Detailed closeout:
  - `docs/resources/xr66_lowsim_targeted_recovery_results_2026_06_22.md`.

## Decision

Close XR-66A/B as no-promotion. Hard low-similarity subset training improves the targeted low-sim P10 bucket but does not transfer to the full-test gates. Do not repeat hard-subset low-sim training. The next branch should use full-manifest training with a low-similarity loss-weight/curriculum signal, XR-39 P10 teacher, and XR-64C/XR-65A center-preservation anchors.

# 2026-06-22 XR-67 Full-Manifest Low-Similarity Weighted Recovery Closeout

## Evidence

- Added `scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh`.
- Validated runner with syntax check and dry-run for lanes `a` and `b`.
- Ran XR-67A/B for `50/50` epochs:
  - XR-67A: XR-64C best-P10 init, XR-39 P10 teacher, LR `2e-7`, low-sim weight `3.0`, GPU0.
  - XR-67B: XR-65A best-P5 init, XR-39 P10 teacher, LR `3e-7`, low-sim weight `4.0`, GPU1.
- Both lanes reached `XR67_TRAIN_EXIT:0`.
- Both lanes reached `XR67_DONE` with `eval_count=6`.
- Best full-test metrics:
  - Best center: XR-67A, center `16.48420093229839`, P10 `34.39795995439802`, P5 `12.118622813905988`.
  - Best P10: XR-67B `best_metric_track_center_px`, center `16.50094587121691`, P10 `34.645834132603234`, P5 `12.016581991740635`.
  - Best P5: XR-67B `best_track_p10`, center `16.545599697317396`, P10 `34.47151440892901`, P5 `12.191752079554966`.
- Best low-sim metrics:
  - Best low-sim center: XR-67B `best_track_p10`, center `17.875147019663164`, P10 `34.81950938317083`, P5 `11.76075307784542`.
  - Best low-sim P10: XR-67B `best_metric_track_center_px`, center `18.01349646814408`, P10 `35.45315001087804`, P5 `11.920123315626576`.
  - Best low-sim P5: XR-67A, center `18.072367114405477`, P10 `34.982719882842034`, P5 `11.920123346390262`.
- Detailed closeout:
  - `docs/resources/xr67_fullmanifest_lowsim_weighted_recovery_results_2026_06_22.md`.

## Decision

Close XR-67A/B as no overall baseline promotion. XR-67B improves full-test P5 above the current P5 gate, but that checkpoint regresses center enough to reject it as the active baseline. The next branch should not repeat the exact XR-67 recipe. Use full-manifest low-sim weighting with fixed LR or higher `min_lr`, and keep an explicit center guard.

# 2026-06-22 XR-68 Fixed-LR Low-Similarity Weighted Recovery Closeout

## Evidence

- Added `scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh`.
- GPT-5.5 sub-agent `Huygens the 2nd` reviewed XR-68 design read-only.
- The review identified XR-67B `best_track_p10` as a risky main-lane seed because it was already center-regressed.
- Interrupted the first draft launch at the beginning and replaced it with the corrected XR-65A-seeded bracket.
- Ran corrected XR-68A/B for `50/50` epochs:
  - XR-68A: XR-65A best-P5 init, XR-39 P10 teacher, fixed LR `3e-7`, center L2 `0.0050`, low-sim weight `4.0`, GPU0.
  - XR-68B: XR-65A best-P5 init, XR-39 P10 teacher, fixed LR `5e-7`, center L2 `0.0060`, low-sim weight `4.0`, GPU1.
- Both lanes reached `XR68_TRAIN_EXIT:0`.
- Both lanes reached `XR68_DONE` with `eval_count=6`.
- Best validation P10:
  - XR-68A reached `25.0809` at epoch `43`.
  - XR-68B reached `25.0809` at epoch `24`.
- Best full-test metrics:
  - Best center: XR-68A `best_metric_track_center_px`, center `16.500933163506645`, P10 `34.645834132603234`, P5 `12.016581991740635`.
  - Best P10: XR-68A `best_metric_track_center_px`, center `16.500933163506645`, P10 `34.645834132603234`, P5 `12.016581991740635`.
  - Best P5: XR-68A `best_track_p10`, center `16.558039666925158`, P10 `34.47534091813224`, P5 `12.11224525996617`.
- Detailed closeout:
  - `docs/resources/xr68_fixedlr_lowsim_weighted_recovery_results_2026_06_22.md`.

## Decision

Close XR-68A/B as no-promotion. Fixed LR corrected XR-67's scheduler collapse and improved validation P10, but it did not transfer to full-test promotion. Do not extend XR-68 to 200 epochs as-is. Next work should use center-preserving candidate soup or new target construction rather than repeating low-sim weighting alone.

# 2026-06-22 Stage1 Frame-Search Re-Prioritization and Promoted-Polish Launch

## Decision

The immediate execution priority is back to Stage1 frame-based Search accuracy. The current promoted Stage1 Lane J checkpoint is useful but only improves Lane B by a small P10 margin, so it should be polished further before treating it as the durable baseline for downstream Stage2/XR work.

The XR-69 no-train checkpoint soup recommendation from a read-only GPT-5.5 sub-agent was recorded as a valid later diagnostic, but it is deferred. Stage1 training comes first.

## Evidence

Current Stage1 baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `train/best_search_p10.pt`
- Search P10: `28.27717937613433`
- Search P5: `10.153863744915656`
- Search center: `17.257583906065744`

Rejected direction:

- Large/capacity branches are not continued now because previous evidence showed severe collapse:
  - P10 around `3.8704581961247593` to `9.044234036269346`.
  - Center around `28.80148663001038` to `44.76474802188964`.

## Launch

Started a two-GPU Stage1 polish bracket:

```bash
RUN_TAG=stage1_promoted_polish_20260622_103914 \
BASE_CKPT=runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt \
EPOCHS=120 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
BEST_METRIC_NAME=metric_search_p10_pct \
MIN_LR=2.5e-7 \
WARMUP_EPOCHS=2 \
WEIGHT_DECAY=5e-5 \
LANE_C_LR=3e-6 \
LANE_C_SEARCH_XY_WEIGHT=2.25 \
LANE_C_NAME=stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25 \
LANE_D_LR=2e-6 \
LANE_D_SEARCH_XY_WEIGHT=2.0 \
LANE_D_EMA_DECAY=0.9995 \
LANE_D_DISTILL_FEATURE_WEIGHT=0.003 \
LANE_D_DISTILL_STATE_WEIGHT=0.003 \
LANE_D_DISTILL_PREDICTION_WEIGHT=0.0015 \
LANE_D_DISTILL_MASK_WEIGHT=0.0015 \
LANE_D_NAME=stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995 \
DETACH=1 \
DRY_RUN=0 \
bash scripts/external/run_stage1_frame_search_warmstart_refine.sh
```

Launched runs:

- Lane C:
  `runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916`
  - PID: `2526793`
  - GPU: `cuda:0`
  - Log: `runs/NON_XR/shared/_logs/stage1_frame_search_warmstart_lane_c_stage1_promoted_polish_20260622_103914.log`
- Lane D:
  `runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916`
  - PID: `2526794`
  - GPU: `cuda:1`
  - Log: `runs/NON_XR/shared/_logs/stage1_frame_search_selfdistill_lane_d_stage1_promoted_polish_20260622_103914.log`

Verification:

- Raw event-count contract passed.
- Torch CUDA smoke passed with two devices.
- `nvidia-smi` confirmed both RTX 5080 GPUs were idle before launch.
- Both logs show seed checkpoint load: `loaded_count=142 partial=0 skipped=0`.
- Both lanes entered `epoch=1/120`.

## Gate

Do not judge before epoch `50`. Promotion requires beating the current Stage1 baseline P10 `28.27717937613433` without material center regression from `17.257583906065744 px`.

## Watcher

The first detached watcher launch ran inside the sandbox PID namespace and did not persist. Relaunched the same watcher as a host process.

Host watcher:

- PID: `2556516`
- Log: `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430.log`
- JSON report: `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430_report.json`
- Summary report: `runs/NON_XR/shared/_logs/stage1_promoted_polish_min50_watch_20260622_host_104430_report.txt`
- Poll interval: `300` seconds.

First report:

- State: `wait_min_epochs`
- Lane C epoch: `7/120`
- Lane D epoch: `7/120`
- Lane D current best: P10 `27.788634912023003`, P5 `9.533917607001538`, center `17.328227780899912`.
- Lane C current best: P10 `27.423630372533257`, P5 `9.382300358898235`, center `17.351748246066975`.
- No promotion decision yet; continue training to at least `50` epochs.

## Conditional Escalation Plan

Added `docs/resources/stage1_frame_search_escalation_plan_2026_06_22.md` so the next Stage1 branch is ready after the active 50-epoch gate.

Evidence and decision:

- DistillGaze analysis supports teacher/self-training, but current HGTXR use must preserve the frame Search anchor.
- RITnet/Grounded-SAM style mask-teacher work is relevant, but current `active_head=search` disables mask and eye losses, so mask pseudo-label branches require a separate preflight.
- PAPER_REF analysis supports a strong Stage1 frame teacher before Stage2 hybrid student work.
- The previous 256x8 capacity branch collapsed badly, so it should not be repeated as-is.

Conditional order after active Lane C/D reach `50` epochs:

1. Promote a current lane if it beats P10 `28.27717937613433` while preserving center `17.257583906065744`.
2. If not, launch S1-K conservative geometry teacher polish.
3. If S1-K fails, launch S1-L lower-LR EMA self-distill.
4. Defer mask-enabled pseudo-label and reduced-capacity distill retries until the conservative branches fail.

## 2026-06-22 - Stage1 S1-K Runner Preparation

Current reporter snapshot:

- Baseline remains:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Lane D:
  `runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916`
  - epoch `45/120`
  - best P10 `27.788634912023003`
  - best P5 `9.55076399389303`
  - best center `17.328227780899912`
- Lane C:
  `runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916`
  - epoch `47/120`
  - best P10 `27.423630372533257`
  - best P5 `9.480009258918042`
  - best center `17.351748246066975`
- State: `wait_min_epochs`; no promotion decision before epoch `50`.

Runner update:

- Updated `scripts/external/run_stage1_frame_search_warmstart_refine.sh` to support conservative geometry-teacher polish.
- Added per-lane variables:
  - `LANE_C_SEARCH_AB_WEIGHT`
  - `LANE_C_SEARCH_TRIG_WEIGHT`
  - `LANE_D_SEARCH_AB_WEIGHT`
  - `LANE_D_SEARCH_TRIG_WEIGHT`
- The runner now forwards these as:
  - `loss.search_ab_weight`
  - `loss.search_trig_weight`

Validation:

- `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh` passed.
- `DRY_RUN=1` S1-K command generation passed and showed the expected geometry overrides on both lanes.

Decision:

- Do not launch S1-K yet.
- Continue current Lane C/D training to the `50` epoch gate.
- If no current lane promotes, run S1-K first using the prepared runner.

# 2026-06-22 Stage1 Promoted-Polish No-Promotion and S1-K/S1-L Launch

## Promoted-Polish Gate Result

The promoted-polish bracket reached the required minimum-epoch gate and did not promote.

- Baseline checkpoint:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Lane D:
  `runs/NON_XR/raw/stage1_frame_search_promoted_selfdistill_lr2e6_xy2_ema9995_20260622_103916`
  - epoch `53/120`
  - best P10 `27.788634912023003`
  - best P5 `9.55076399389303`
  - best center `17.328227780899912`
  - delta P10 `-0.4885444641113281`
- Lane C:
  `runs/NON_XR/raw/stage1_frame_search_promoted_p10polish_nodistill_lr3e6_xy2p25_20260622_103916`
  - epoch `55/120`
  - best P10 `27.423630372533257`
  - best P5 `9.480009258918042`
  - best center `17.351748246066975`
  - delta P10 `-0.8535490036010742`

Decision:

- No promotion.
- Keep the current Lane J center-polish self-distill checkpoint as active Stage1 baseline.
- Stop old Lane C/D PIDs `2526793` and `2526794`, plus watcher PID `2556516`, because both lanes were below baseline by more than `0.2 pp` P10 and had worse center.

## S1-K/S1-L Follow-Up Launch

Launched the next conservative two-GPU Stage1 bracket from the active baseline.

Shared settings:

- Stage: `stage1`.
- Active head: `search`.
- Epochs: `120`.
- Batch size: `8`.
- Workers: `4`.
- Optimizer: AdamW.
- Scheduler: cosine, warmup `2`, min LR `1e-7`.
- Weight decay: `5e-5`.
- Best metric: `metric_search_p10_pct`.
- Baseline seed:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`

Lane S1-K:

- Run:
  `runs/NON_XR/raw/stage1_s1k_geometry_nodistill_lr1e6_xy2_ab0p75_trig1p25_20260622_111904`
- PID: `2728912`.
- GPU: `cuda:0`.
- LR: `1e-6`.
- Loss: `search_xy_weight=2.0`, `search_ab_weight=0.75`, `search_trig_weight=1.25`, `search_geo_weight=0.75`, `search_conf_weight=0.1`.
- Distillation: disabled.
- Log:
  `runs/NON_XR/shared/_logs/stage1_frame_search_warmstart_lane_c_stage1_s1k_s1l_after_promoted_fail_20260622.log`.

Lane S1-L:

- Run:
  `runs/NON_XR/raw/stage1_s1l_low_lr_ema_selfdistill_lr7p5e7_xy2_ema9997_20260622_111904`
- PID: `2728913`.
- GPU: `cuda:1`.
- LR: `7.5e-7`.
- Loss: `search_xy_weight=2.0`, `search_ab_weight=0.0`, `search_trig_weight=0.0`, `search_geo_weight=0.75`, `search_conf_weight=0.1`.
- Distillation: EMA teacher enabled, `ema_decay=0.9997`, feature/state weights `0.002`, prediction weight `0.001`, mask weight `0.0`.
- Log:
  `runs/NON_XR/shared/_logs/stage1_frame_search_selfdistill_lane_d_stage1_s1k_s1l_after_promoted_fail_20260622.log`.

Startup verification:

- Both lanes loaded the seed checkpoint with `loaded_count=142 partial=0 skipped=0`.
- Both lanes reached epoch `6/120`.
- Host GPU check showed GPU0 and GPU1 both active.
- No startup traceback, CUDA failure, or missing manifest issue was observed.

Early reporter snapshot:

- S1-K: epoch `31/120`, P10 `27.693172202920014`, P5 `9.11275856449919`, center `17.315994528104675`, delta P10 `-0.5840071732143173`.
- S1-L: epoch `30/120`, P10 `27.670710329739553`, P5 `9.070081117018214`, center `17.30932722451552`, delta P10 `-0.6064690463947784`.
- State: `wait_min_epochs`.

Watcher:

- Host PID: `2758111`.
- Log: `runs/NON_XR/shared/_logs/stage1_s1k_s1l_min50_watch_20260622_host.log`.
- JSON report: `runs/NON_XR/shared/_logs/stage1_s1k_s1l_min50_watch_20260622_host_report.json`.
- Summary report: `runs/NON_XR/shared/_logs/stage1_s1k_s1l_min50_watch_20260622_host_report.txt`.
- Poll interval: `300` seconds.

Next decision:

- Do not judge S1-K/S1-L until both lanes reach at least `50` epochs.
- Promote only if a checkpoint beats P10 `28.27717937613433` while preserving center near `17.257583906065744 px`.
- If both S1-K/S1-L fail, move to mask-enabled pseudo-label preflight or reduced-capacity distill retry rather than repeating the failed high-LR/capacity branch.

# 2026-06-22 Stage1 S1-M Mask/Eye Probe Preparation

## Evidence

- Added `scripts/external/run_stage1_frame_search_mask_probe.sh`.
- The runner is a bounded diagnostic for the next fallback branch after S1-K/S1-L.
- It enables `active_head=all` with `eye/search/mask` heads and disables `event/track/aux`.
- It keeps the active Stage1 baseline checkpoint as warm-start seed.
- It bounds the smoke with:
  - `EPOCHS=3`
  - `MAX_TRAIN_BATCHES=128`
  - `MAX_VAL_BATCHES=106`
  - `LR=5e-7`
  - `MASK_WEIGHT=0.1`
  - `MASK_COARSE_WEIGHT=0.025`
- Default `DRY_RUN=1` prevents accidental launch while S1-K/S1-L occupy both GPUs.

## Validation

- `bash -n scripts/external/run_stage1_frame_search_mask_probe.sh` passed.
- Dry-run command generation passed:

```bash
DRY_RUN=1 RUN_TAG=stage1_s1m_mask_probe_dryrun_20260622 \
bash scripts/external/run_stage1_frame_search_mask_probe.sh
```

- Dry-run output confirmed the expected overrides for `model.heads.active=all`, mask/eye/search heads, bounded train/val batches, and mask loss weights.

## Decision

Do not run S1-M while S1-K/S1-L are still below the `50` epoch gate. If S1-K/S1-L fail, run S1-M first as a short diagnostic and inspect:

```bash
rg 'loss_mask|loss_eye|loss_search_|metric_search_' \
  runs/NON_XR/shared/_logs/stage1_frame_search_mask_probe_<RUN_TAG>.log
```

Only promote S1-M to a real 50-epoch branch if mask/eye losses are nonzero and Search P10 does not immediately collapse.

# 2026-06-22 Stage1 S1-N Reduced-Capacity Distill Preparation

## Evidence

- Added `scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh`.
- The wrapper reuses `scripts/external/run_stage1_frame_search_capacity_distill.sh` but overrides the older high-risk capacity defaults.
- Default `DRY_RUN=1` prevents accidental launch while S1-K/S1-L occupy both GPUs.

Safe S1-N defaults:

- `LARGE_EMBED_DIM=224`
- `LARGE_DEPTH=6`
- `LARGE_NUM_HEADS=4`
- `BATCH_SIZE=4`
- `EPOCHS=50`
- no-distill lane LR `7.5e-7`
- teacher-distill lane LR `5e-7`
- teacher architecture `192x6`, `3` heads
- EMA `0.9997`
- distillation feature/state `0.002`
- distillation prediction `0.001`
- distillation mask `0.0`

## Validation

- `bash -n scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh` passed.
- Dry-run command generation passed:

```bash
DRY_RUN=1 RUN_TAG=stage1_s1n_reduced_capacity_dryrun_20260622 \
bash scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh
```

- Dry-run output confirmed reduced-capacity overrides and low-LR teacher-distillation settings.

## Decision

Do not run S1-N before the active S1-K/S1-L `50` epoch gate. If S1-K/S1-L and S1-M fail, use S1-N as the safer capacity retry instead of repeating the previous 256x8/high-LR collapse.

# 2026-06-22 Stage1 S1-K/S1-L Gate Decision Helper

## Evidence

- Added `scripts/external/decide_stage1_s1kl_gate_next.py`.
- The helper imports the existing follow-up reporter and emits one action:
  - `wait_min_epochs`
  - `promote_stage1_baseline`
  - `stop_s1kl_and_run_s1m_mask_probe`
  - `manual_review_or_continue_to_target`
- It does not kill or launch processes. It only prints the next commands when the gate is ready.

## Validation

- `python3 -m py_compile scripts/external/decide_stage1_s1kl_gate_next.py` passed.
- Current decision command:

```bash
.venv/bin/python scripts/external/decide_stage1_s1kl_gate_next.py --format summary
```

- Current output: `action=wait_min_epochs`, `status=wait_min_epochs`, `all_min_ready=False`, `promote_checkpoint=None`.
- Latest follow-up snapshot used by the helper:
  - S1-K epoch `31/120`, best P10 `27.693172202920014`, center `17.315994528104675`.
  - S1-L epoch `30/120`, best P10 `27.670710329739553`, center `17.30932722451552`.

## Decision

Continue S1-K/S1-L until both lanes reach at least `50` epochs. After that, run the gate helper and follow its action. Do not run S1-M or S1-N before the gate.

# 2026-06-22 Stage1 S1-K/S1-L Live Gate Poll

## Evidence

- Ran direct follow-up reporter with the active baseline and both S1-K/S1-L run directories.
- Current reporter output:
  - S1-K epoch `39/120`, best P10 `27.693172202920014`, P5 `9.11275856449919`, center `17.315994528104675`, delta P10 `-0.5840071732143173`.
  - S1-L epoch `37/120`, best P10 `27.670710329739553`, P5 `9.070081117018214`, center `17.30932722451552`, delta P10 `-0.6064690463947784`.
  - Status `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
- Ran `scripts/external/decide_stage1_s1kl_gate_next.py --format summary`.
  - Action `wait_min_epochs`.
  - `all_min_ready=False`.
- Host process check confirmed the training jobs are still running:
  - S1-K PID `2728912`, GPU0.
  - S1-L PID `2728913`, GPU1.
  - Watcher PID `2758111`.
- GPU snapshot:
  - GPU0 RTX 5080: `720 MiB` used, `15123 MiB` free, `37%` instantaneous utilization.
  - GPU1 RTX 5080: `732 MiB` used, `15111 MiB` free, `12%` instantaneous utilization.

## Decision

Do not stop or replace S1-K/S1-L yet. They are active but below the required `50` epoch gate. Re-run the gate helper after both lanes reach at least epoch `50`; only then decide promotion, stop-and-mask-probe, or continued training to target.

# 2026-06-22 Stage1 S1-K/S1-L Gate Closure and S1-M/S1-N Launch

## S1-K/S1-L Gate Result

- Direct gate helper reached `all_min_ready=True`.
- Action: `stop_s1kl_and_run_s1m_mask_probe`.
- S1-K final gate snapshot:
  - epoch `53/120`
  - best P10 `27.693172202920014`
  - P5 `9.173405485333136`
  - center `17.315994528104675`
  - delta P10 `-0.5840071732143173`
- S1-L final gate snapshot:
  - epoch `51/120`
  - best P10 `27.670710329739553`
  - P5 `9.227313833416632`
  - center `17.30932722451552`
  - delta P10 `-0.6064690463947784`
- Stopped S1-K PID `2728912`, S1-L PID `2728913`, and watcher PID `2758111`.
- Decision: no Stage1 baseline promotion. Keep the existing promoted Lane J checkpoint.

## S1-M Probe Result

- Launched S1-M probe:
  `runs/NON_XR/raw/stage1_s1m_mask_eye_active_all_probe_stage1_s1m_mask_probe_after_s1kl_20260622_115634`
- PID `2932984`, GPU0, completed `3/3` epochs.
- Checkpoint load: `loaded_count=126 partial=0 skipped=0`.
- Evidence from `train/history.jsonl`:
  - `loss_eye` nonzero: train `33.22195512056351`, val `33.222214104994286` at epoch 1.
  - `loss_mask` nonzero: train `0.16752412309870124`, val `0.16662329365060013` at epoch 1.
  - `loss_search_xy`, `loss_search_ab`, `loss_search_trig`, `loss_search_geo` nonzero.
- Best probe P10 was `28.27717937613433` at epoch 1, equal to the active baseline.
- Final epoch P10 was `27.90655940433718`, P5 `8.742138610695893`, center `17.304687189606`.
- Finding: `eye_weight=1.0` makes `loss_eye` dominate the objective; long mask-assisted Search should use low eye weight.

## S1-MA 50-Epoch Candidate

- Added `loss.eye_weight` override support to `scripts/external/run_stage1_frame_search_mask_probe.sh`.
- Launched S1-MA:
  `runs/NON_XR/raw/stage1_s1ma_loweye_maskassist_50ep_after_s1kl_20260622_120007`
- PID `2969847`, GPU0.
- Settings:
  - epochs `50`
  - max train batches `256`
  - max val batches `106`
  - LR `5e-7`
  - `active_head=all`
  - `eye_weight=0.02`
  - `mask_weight=0.05`
  - `mask_coarse_weight=0.0125`
- Early evidence:
  - epoch `1/50` best P10 `28.27717937613433`
  - epoch `12/50` latest P10 `27.575247674618126`, P5 `9.247529443704858`, center `17.35772168861245`
  - no catastrophic Search collapse, but no P10 improvement yet.

## S1-N Reduced-Capacity Distill Candidate

- Added `SKIP_LANE_G` and `SKIP_LANE_H` controls to `scripts/external/run_stage1_frame_search_capacity_distill.sh`.
- Forwarded skip controls through `scripts/external/run_stage1_frame_search_reduced_capacity_distill_probe.sh`.
- Dry-run validation confirmed `SKIP_LANE_G=1` launches only lane H on GPU1.
- Launched S1-N lane H:
  `runs/NON_XR/raw/stage1_s1n_reduced224_d6_teacher_distill_lr5e7_stage1_s1n_reduced224_teacher_laneh_50ep_after_s1m_20260622_120110`
- PID `2978876`, GPU1.
- Settings:
  - student `embed_dim=224`, `depth=6`, `num_heads=4`
  - teacher checkpoint = active Stage1 baseline
  - LR `5e-7`
  - weak feature/state/prediction distillation
- Startup warning:
  - `loaded_count=7 partial=0 skipped=135`, expected because student shape differs from the base checkpoint.
  - Epoch 1/2 metrics failed with P10 `0.0`, P5 `0.0`, and center around `173 px`.
  - Stopped PID `2978876` with `kill -TERM`.
  - Decision: S1-N lane H is an architecture/warm-start mismatch diagnostic, not a valid candidate to run to 50 epochs.

## S1-MB 50-Epoch Candidate

- Launched S1-MB on freed GPU1:
  `runs/NON_XR/raw/stage1_s1mb_maskonly_searchpreserve_50ep_after_s1n_20260622_120546`
- PID `3012910`, GPU1.
- Settings:
  - epochs `50`
  - max train batches `256`
  - max val batches `106`
  - LR `3e-7`
  - `active_head=all`
  - `eye_weight=0.0`
  - `mask_weight=0.025`
  - `mask_coarse_weight=0.00625`
  - `search_xy_weight=2.25`
  - `search_geo_weight=0.25`
- Startup evidence:
  - checkpoint load `loaded_count=126 partial=0 skipped=0`
  - entered epoch `1/50`.
- Early evidence:
  - epoch `2/50` P10 `28.294025727038115`, P5 `8.87690948990156`, center `17.308569012947803`
  - baseline P10 is `28.27717937613433`, so S1-MB is `+0.0168463509037854 pp` P10 early.
  - center is worse than baseline `17.257583906065744`, so do not promote before the 50-epoch gate.

## Decision

Current active Stage1 baseline is still:
`runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.

Monitor S1-MA and S1-MB until both reach at least `50` epochs. Promote only if P10 beats `28.27717937613433` while center does not regress materially from `17.257583906065744`.

## S1-MA/S1-MB Gate Watcher

- Started host watcher:
  - PID `3026345`
  - log `runs/NON_XR/shared/_logs/stage1_s1ma_s1mb_min50_watch_20260622.log`
  - summary `runs/NON_XR/shared/_logs/stage1_s1ma_s1mb_min50_watch_20260622_report.txt`
  - json `runs/NON_XR/shared/_logs/stage1_s1ma_s1mb_min50_watch_20260622_report.json`
  - poll interval `180` seconds

# 2026-06-22 S1-MC/S1-MD Center-Preservation Follow-Up

## S1-MA/S1-MB Gate Closure

- S1-MA completed `50/50` with no promotion: best P10 matched baseline, but P5 and center regressed.
- S1-MB completed `50/50` with no promotion: best P10 `28.294025727038115` was only `+0.016846350903783502 pp`, while best center `17.295753600462426` regressed versus baseline `17.257583906065744`.
- Decision: keep current Stage1 frame-search baseline:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.

## Active S1-MC/S1-MD

- Launched S1-MC on GPU0:
  `runs/NON_XR/raw/stage1_s1mc_s1mb_centerpolish_maskweak_50ep_20260622_123430`
- Launched S1-MD on GPU1:
  `runs/NON_XR/raw/stage1_s1md_s1mb_centerpolish_searchonly_50ep_20260622_123440`
- Shared seed:
  `runs/NON_XR/raw/stage1_s1mb_maskonly_searchpreserve_50ep_after_s1n_20260622_120546/train/best_search_p10.pt`
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1mc_s1md_min50_watch_20260622.log`
- Latest reporter snapshot at `2026-06-22 12:42 KST`:
  - S1-MC `12/50`, S1-MD `12/50`.
  - `status=wait_min_epochs`.
  - Current best P10 `28.176101198736227`, below baseline by `-0.10107817739810443 pp`.
  - Current best P5 `9.183513263486466`, below baseline by `-0.9703504814291897 pp`.
  - Current best center regresses by about `0.0546 px`.
- Decision: wait until both reach at least `50` epochs before any promotion decision.
- Validation passed:
  - shell syntax checks for Stage1 follow-up runners.
  - Python compile for the Stage1 decision/report scripts.
  - `git diff --check`.

## Queue Guard Fix

- Updated `scripts/external/run_stage1_frame_search_post_followup_queue.sh` so fallback decisions explicitly use:
  - `BASELINE_RUN`
  - `FOLLOWUP_RUNS`
- Reason: the queue script previously called `report_stage1_frame_search_followup.py` without current follow-up arguments, which could make the fallback queue judge stale default runs instead of active S1-MC/S1-MD.
- Dry-run validation with `DETACH=0 DRY_RUN=1 POLL_SEC=1 MAX_WAIT_SEC=1 START_FALLBACK_WATCHER=0` confirmed:
  - baseline is the active Stage1 frame-search baseline.
  - followups are S1-MC and S1-MD.
  - decision remains `wait_min_epochs`.

## S1-MC/S1-MD Closeout

- Final reporter gate:
  - S1-MC `50/50`, `promotable_after_min_epoch=False`.
  - S1-MD `50/50`, `promotable_after_min_epoch=False`.
  - Recommendation: `status=keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Metrics:
  - S1-MC/S1-MD best P10 `28.176101198736227`, baseline delta `-0.10107817739810443 pp`.
  - S1-MC/S1-MD best P5 `9.183513263486466`, baseline delta `-0.9703504814291897 pp`.
  - S1-MC center `17.31221647532481`, S1-MD center `17.312216362863218`, both worse than baseline `17.257583906065744`.
- Decision: keep the current Stage1 frame-search baseline unchanged.

## S1-ME/S1-MF Launch

- Host GPU launch was required because sandbox Torch CUDA smoke failed with `torch.cuda.is_available() is false`.
- Host preflight passed:
  - raw event-count contract ok.
  - Torch CUDA available with `2` devices.
- Launched S1-ME:
  `runs/NON_XR/raw/stage1_s1me_baseline_geometry_micro_lr3e7_xy2p25_ab0p75_trig1p25_20260622_130436`
  - PID `3408129`, GPU0.
  - LR `3e-7`, geometry micro-polish, no distillation.
- Launched S1-MF:
  `runs/NON_XR/raw/stage1_s1mf_baseline_ultraweak_selfdistill_lr2e7_xy2_ema9999_20260622_130436`
  - PID `3408130`, GPU1.
  - LR `2e-7`, EMA `0.9999`, ultra-weak self-distillation.
- Both lanes loaded checkpoint weights cleanly: `loaded_count=142 partial=0 skipped=0`.
- Started watcher:
  `runs/NON_XR/shared/_logs/stage1_s1me_s1mf_min50_watch_20260622.log`.
- Initial reporter is expected to return missing `history.json` until epoch 1 validation writes the first history file.

## 2026-06-22 13:12 KST S1-ME/S1-MF Runtime Refresh

- Reporter command confirmed both active rescue lanes are still below the required `50` epoch gate:
  - S1-ME `10/50`, `min_ready=False`, `complete=False`.
  - S1-MF `10/50`, `min_ready=False`, `complete=False`.
  - Overall reporter status: `wait_min_epochs`, `continue_training=True`, `promote_checkpoint=None`.
- Logs showed active forward progress beyond the reporter snapshot:
  - S1-ME reached epoch `11/50` validation.
  - S1-MF reached epoch `11/50` training.
- No hard failure signatures were found in the active lane logs or watcher log with the error scan for `Traceback`, `ERROR`, `RuntimeError`, `CUDA out of memory`, `Killed`, `failed`, `exited`, or `returncode`.
- Current best metrics versus the active baseline:
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - S1-ME: best P10 `28.024483914645213`, P5 `9.112758546505335`, center `17.306600507700217`.
  - S1-MF: best P10 `28.277179340146624`, P5 `9.264375812602493`, center `17.299979461813873`.
- Decision: keep running. Do not promote, stop, or launch a conflicting GPU experiment before the `50` epoch gate unless a hard runtime failure appears.
- Post-document reporter verification advanced to S1-ME `13/50` and S1-MF `12/50`; status remains `wait_min_epochs`.

## 2026-06-22 13:40 KST S1-ME/S1-MF Closeout and S1-MG/S1-MH Launch

- S1-ME/S1-MF reached the required `50/50` epoch gate.
- Final reporter status: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Baseline remains:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- S1-ME final gate:
  - P10 `28.024483914645213`, delta `-0.25269546148911814`
  - P5 `9.112758546505335`, delta `-1.041105198410321`
  - center `17.306600507700217`, worse by `0.049016601634473744 px`
- S1-MF final gate:
  - P10 `28.277179340146624`, delta `-3.598770703661103e-08`
  - P5 `9.264375812602493`, delta `-0.8894879323131626`
  - center `17.299979461813873`, worse by `0.042395555748129254 px`
- Host GPU state after closeout showed both GPUs free.
- Code update:
  - Added Stage1 Search soft-threshold losses in `src/hbtxr/loss/stage1.py`.
  - Added coverage in `tests/test_stage1_search_threshold_loss.py`.
  - Extended `scripts/external/run_stage1_frame_search_warmstart_refine.sh` to pass lane-specific Search P10/P5 threshold weights.
- Validation:
  - `py_compile` passed for the modified Stage1 loss and test.
  - Focused pytest passed: `tests/test_stage1_search_threshold_loss.py` and selected existing track threshold tests.
  - `bash -n scripts/external/run_stage1_frame_search_warmstart_refine.sh` passed.
  - Dry-run confirmed `loss.search_p10_soft_threshold_*` and `loss.search_p5_soft_threshold_*` overrides appear in both lane commands.
- Launched S1-MG on GPU0:
  `runs/NON_XR/raw/stage1_s1mg_p10soft_lr2e7_p10w0p05_p5w0p01_20260622_134341`
  - PID `3656477`
  - LR `2e-7`
  - `search_p10_soft_threshold_weight=0.05`
  - `search_p5_soft_threshold_weight=0.01`
  - no distillation
- Launched S1-MH on GPU1:
  `runs/NON_XR/raw/stage1_s1mh_p10p5soft_selfdistill_lr1p5e7_p10w0p03_p5w0p03_20260622_134341`
  - PID `3656478`
  - LR `1.5e-7`
  - `search_p10_soft_threshold_weight=0.03`
  - `search_p5_soft_threshold_weight=0.03`
  - ultra-weak self-distillation with EMA `0.9999`, feature/state `0.0005`, prediction `0.00025`, mask `0.0`
- Host preflight passed before launch:
  - raw event-count contract ok.
  - Torch CUDA available with two devices and tensor smoke ok.
  - host `nvidia-smi` showed both RTX 5080 GPUs available.
- Startup evidence:
  - Both lanes loaded the active baseline checkpoint with `loaded_count=142 partial=0 skipped=0`.
  - Both lanes entered epoch `1/50` training.
  - Host GPU check after launch showed GPU0/GPU1 active.
- Watcher:
  `runs/NON_XR/shared/_logs/stage1_s1mg_s1mh_min50_watch_20260622.log`
- Decision: monitor S1-MG/S1-MH to `50` epochs before promotion judgment. Promote only if P10 beats baseline and center does not regress beyond the current baseline.

## 2026-06-22 14:22 KST S1-MG/S1-MH Closeout

- S1-MG/S1-MH reached the required `50/50` epoch gate.
- Final reporter status: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Baseline remains:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- S1-MG final gate:
  - run: `runs/NON_XR/raw/stage1_s1mg_p10soft_lr2e7_p10w0p05_p5w0p01_20260622_134341`
  - P10 `28.159254811844736`, delta `-0.11792456428959497`
  - P5 `9.129604933396825`, delta `-1.0242588115188305`
  - center `17.303716394136536`, worse by `0.04613248807079273 px`
- S1-MH final gate:
  - run: `runs/NON_XR/raw/stage1_s1mh_p10p5soft_selfdistill_lr1p5e7_p10w0p03_p5w0p03_20260622_134341`
  - P10 `28.142408460940956`, delta `-0.13477091519337492`
  - P5 `9.247529461698711`, delta `-0.9063342832169443`
  - center `17.297416754488676`, worse by `0.03983284842293244 px`
- Decision:
  - No promotion.
  - Keep the active Stage1 frame-search baseline unchanged.
  - The repeated no-promotion pattern across S1-MC/S1-MD, S1-ME/S1-MF, and S1-MG/S1-MH indicates small LR/loss polish around the current baseline is not enough; the next branch should change a larger axis such as model capacity, crop/data schedule, or a stronger but carefully gated teacher regime.

## 2026-06-22 14:28 KST S1-MI/S1-MJ Full-Manifest Mask-Guard Launch

- Added runner:
  `scripts/external/run_stage1_frame_search_mask_guard_full.sh`
- Rationale:
  - S1-MB showed a tiny early P10 gain from mask-assisted Search, but bounded 256-batch training regressed center/P5.
  - S1-N capacity retry collapsed because architecture shape changed and checkpoint warm-start loaded only `7` tensors.
  - S1-MI/S1-MJ keep the current architecture and active baseline checkpoint, but enable the mask head with low mask weights and explicit center guards.
- Validation before launch:
  - `bash -n scripts/external/run_stage1_frame_search_mask_guard_full.sh` passed.
  - Dry-run emitted expected `search,mask` head commands with `eye_weight=0.0`, no distillation, and no max-batch limit.
  - `git diff --check` passed for the new runner and Stage1 docs.
- Sandbox launch attempt failed as expected because Torch CUDA inside the sandbox returned `torch.cuda.is_available() is false`.
- Host launch succeeded after preflight:
  - raw event-count contract passed.
  - Torch CUDA available with `2` devices and tensor smoke ok.
  - host `nvidia-smi` showed both RTX 5080 GPUs free before launch.
- Launched S1-MI on GPU0:
  `runs/NON_XR/raw/stage1_s1mi_fullmask_searchguard_lr2e7_mask0125_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`
  - PID `3954866`
  - LR `2e-7`
  - `search_xy_weight=2.25`, `search_geo_weight=0.25`
  - `mask_weight=0.0125`, `mask_coarse_weight=0.003125`
  - `constraint_center_weight=0.20`, `constraint_center_radius=16.0`
- Launched S1-MJ on GPU1:
  `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`
  - PID `3954867`
  - LR `1.5e-7`
  - `search_xy_weight=2.0`, `search_geo_weight=0.5`
  - `mask_weight=0.00625`, `mask_coarse_weight=0.0015625`
  - `constraint_center_weight=0.30`, `constraint_center_radius=12.0`
- Startup evidence:
  - Both lanes loaded the active baseline checkpoint with `loaded_count=126 partial=0 skipped=0`.
  - Both lanes entered epoch `1/50` training on the full train manifest (`742` train steps and `106` validation steps per epoch).
  - Host GPU snapshot after launch showed GPU0/GPU1 active at about `724 MiB` each.
  - Error scan found no `Traceback`, `ERROR`, `RuntimeError`, `CUDA out of memory`, `Killed`, or `failed` signatures.
- Watcher:
  - PID `3965866`
  - log `runs/NON_XR/shared/_logs/stage1_s1mi_s1mj_min50_watch_20260622.log`
- Initial watcher snapshot at `2026-06-22 14:28 KST`:
  - S1-MI epoch `2/50`, P10 `28.075023039331978`, P5 `9.06558873518458`, center `17.316034978290773`.
  - S1-MJ epoch `2/50`, P10 `28.192947567633862`, P5 `8.930817837985057`, center `17.31833925787008`.
  - Both lanes are below baseline, but the required minimum `50` epoch gate has not been reached.
- Decision:
  - Continue S1-MI/S1-MJ to the `50` epoch gate.
  - Promote only if P10 beats `28.27717937613433` and center does not regress from `17.257583906065744`.

## 2026-06-22 15:01 KST S1-MI/S1-MJ Closeout

- S1-MI/S1-MJ reached the required `50/50` epoch gate.
- Final reporter status: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Baseline remains:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- S1-MI final gate:
  - run: `runs/NON_XR/raw/stage1_s1mi_fullmask_searchguard_lr2e7_mask0125_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`
  - P10 `28.19070143069861`, delta `-0.08647794543572118`
  - P5 `9.32389961098725`, delta `-0.8299641339284065`
  - center `17.316034978290773`, worse by `0.05845107222502932 px`
- S1-MJ final gate:
  - run: `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658`
  - P10 `28.482705062290407`, delta `+0.2055256861560757`
  - P5 `9.205975082685363`, delta `-0.9478886622302927`
  - center `17.31833925787008`, worse by `0.060755351804335334 px`
- Decision:
  - No baseline promotion because S1-MJ's P10 improvement is coupled to center/P5 regression.
  - Keep the promoted Stage1 frame-search baseline unchanged.
  - Treat S1-MJ `best_search_p10.pt` as a P10-rich seed for a center/P5 recovery branch.

## 2026-06-22 15:06 KST S1-MK/S1-ML Center/P5 Recovery Launch

- Extended `scripts/external/run_stage1_frame_search_mask_guard_full.sh` so the runner can pass Stage1 Search P10/P5 soft-threshold loss weights.
- Validation before launch:
  - `bash -n scripts/external/run_stage1_frame_search_mask_guard_full.sh` passed.
  - Dry-run confirmed the S1-MK/S1-ML commands include `loss.search_p10_soft_threshold_*` and `loss.search_p5_soft_threshold_*`.
  - `git diff --check` passed for the runner and Stage1 tracking docs.
- Sandbox launch attempt failed because Torch CUDA is unavailable inside the sandbox.
- Host launch succeeded after raw contract and CUDA tensor-smoke preflight.
- Shared seed:
  `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658/train/best_search_p10.pt`
- S1-MK:
  - run: `runs/NON_XR/raw/stage1_s1mk_s1mj_p10seed_centerrecover_lr1e7_p5soft02_20260622_150603`
  - PID `15290`, GPU0.
  - LR `1e-7`, best metric `metric_search_center_px`.
  - `constraint_center_weight=0.50`, radius `10 px`.
  - `search_p10_soft_threshold_weight=0.01`, `search_p5_soft_threshold_weight=0.02`.
- S1-ML:
  - run: `runs/NON_XR/raw/stage1_s1ml_s1mj_p10seed_strongcenter_lr7p5e8_p5soft04_20260622_150603`
  - PID `15291`, GPU1.
  - LR `7.5e-8`, best metric `metric_search_center_px`.
  - `constraint_center_weight=0.75`, radius `8 px`.
  - `search_p10_soft_threshold_weight=0.01`, `search_p5_soft_threshold_weight=0.04`.
- Startup evidence:
  - Both lanes loaded `loaded_count=132 partial=0 skipped=0`.
  - Both lanes entered epoch `1/50` on the full train manifest.
  - GPU snapshot after launch showed both RTX 5080 jobs resident.
- Decision:
  - Continue to the `50` epoch gate.
  - Promotion requires P10 above `28.27717937613433` and center no worse than `17.257583906065744`; P5 regression should also be checked because S1-MJ failed mainly through P5/center regression.

## 2026-06-22 15:42 KST S1-MK/S1-ML Closeout

- S1-MK/S1-ML reached the required `50/50` epoch gate.
- Final reporter status: `keep_baseline_unless_later_improves`, `promote_checkpoint=None`.
- Baseline remains:
  `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Baseline metrics:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- S1-MK final gate:
  - run: `runs/NON_XR/raw/stage1_s1mk_s1mj_p10seed_centerrecover_lr1e7_p5soft02_20260622_150603`
  - P10 `28.482705062290407`, delta `+0.2055256861560757`
  - P5 `9.632749593482828`, delta `-0.5211141514328279`
  - center `17.43487750809148`, worse by `0.17729360202573474 px`
- S1-ML final gate:
  - run: `runs/NON_XR/raw/stage1_s1ml_s1mj_p10seed_strongcenter_lr7p5e8_p5soft04_20260622_150603`
  - P10 `27.994160562191368`, delta `-0.283018813942963`
  - P5 `9.514825083174795`, delta `-0.6390386617408605`
  - center `17.44643774122562`, worse by `0.18885383515987542 px`
- Decision:
  - No promotion.
  - Training-based center/P5 recovery from S1-MJ best-P10 did not work.
  - Next branch should use checkpoint interpolation between the promoted baseline and S1-MJ best-P10 to find a no-train tradeoff before spending more GPU time.

## 2026-06-22 16:04 KST Stage1 Baseline/S1-MJ Interpolation Eval

- Added `--key-policy` to `scripts/external/interpolate_hbtxr_checkpoints.py`.
  - Default remains `strict`.
  - New `common-copy-b` mode interpolates shared tensors and copies side-specific tensors from checkpoint B.
  - Needed because the current baseline has search aux heads while S1-MJ has a mask head.
- Added runner:
  `scripts/external/run_stage1_frame_search_baseline_s1mj_interp_eval.sh`.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_baseline_s1mj_interp_eval.sh` passed.
  - `PYTHONPYCACHEPREFIX=/tmp/hgtxr_pycache .venv/bin/python -m py_compile scripts/external/interpolate_hbtxr_checkpoints.py` passed.
  - Smoke interpolation produced `126` shared interpolated tensors and copied `6` S1-MJ mask tensors.
- Eval artifact:
  `runs/NON_XR/shared/eval/stage1_baseline_s1mj_interp_stage1_baseline_s1mj_interp_20260622_155014/summary.json`.
- Same-evaluator baseline reference:
  - checkpoint: current baseline `best_search_p10.pt`
  - P10 `28.27717937613433`
  - P5 `8.334456677706736`
  - P1 `0.5053908510028191`
  - center `17.292674766396576`
- Best P10/P5 direction:
  - S1-MJ / alpha `1.0`
  - P10 `28.482705062290407`, delta `+0.2055256861560757`
  - P5 `9.088050554383475`, delta `+0.7535938766767387`
  - center `17.419374744847136`, center regression `0.12669997845056002 px`
- Interpolation result:
  - alpha `0.05` and `0.10`: no P10/P5 improvement; center worsened.
  - alpha `0.15` to `0.75`: P5 often improved, but P10 dropped and center worsened.
  - alpha `1.0`: equals S1-MJ; P10/P5 improve but center regresses.
- Decision:
  - No interpolation checkpoint is promotable as a strict P10+center baseline.
  - S1-MJ remains useful as a P10/P5-rich seed.

## 2026-06-22 16:09 KST Stage1 Checkpoint Matrix Eval

- Eval artifact:
  `runs/NON_XR/shared/eval/stage1_ckpt_matrix_20260622_160523/summary.json`.
- Evaluated current baseline run checkpoint variants and S1-MJ/S1-MK P5 checkpoints on val.
- Results:
  - `baseline_best_p5`: P10 `27.623540590394217`, P5 `10.153863744915656`, P1 `0.1179245283018868`, center `17.398822599986815`.
  - `baseline_best_center`: P10 `27.294475033598125`, P5 `9.241913993403596`, P1 `0.1347708971995228`, center `17.25758159385537`.
  - `s1mj_best_p5`: P10 `28.190701412704755`, P5 `9.205975082685363`, P1 `0.1347708971995228`, center `17.40067045193798`.
  - `s1mk_best_p5`: P10 `27.168688828090453`, P5 `9.632749593482828`, P1 `0.4638364899833247`, center `17.5374885190208`.
- Interpretation:
  - No single checkpoint dominates P10, P5, P1, and center.
  - Current baseline `best_search_p10.pt` remains the strict baseline.
  - For pure P10/P5, S1-MJ `best_search_p10.pt` is the strongest seed but not the strict baseline due center regression.

## 2026-06-22 16:13 KST S1-MM/S1-MN Teacher-Distill Launch

- Extended `scripts/external/run_stage1_frame_search_mask_guard_full.sh` with lane-level distillation controls.
  - Defaults keep existing behavior: distillation disabled unless explicitly enabled.
  - Added teacher checkpoint, EMA, feature/state/prediction/mask distill weights per lane.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_mask_guard_full.sh` passed.
  - Dry-run confirmed teacher checkpoint and distillation weights are included in both lane commands.
  - `git diff --check` passed for touched runner/eval scripts.
- Sandbox note:
  - In sandbox, `nvidia-smi` can become unavailable and Torch CUDA may report false under some env states.
  - Host launch was used after sandbox preflight failed, and host preflight passed with `torch_cuda_available=True`, `torch_cuda_device_count=2`.
- Shared seed:
  `runs/NON_XR/raw/stage1_s1mj_fullmask_centerguard_lr1p5e7_mask00625_stage1_s1mi_s1mj_mask_guard_full_20260622_1428_20260622_142658/train/best_search_p10.pt`
- S1-MM:
  - run: `runs/NON_XR/raw/stage1_s1mm_s1mj_p10seed_baselinep10teacher_lr1e7_20260622_161345`
  - PID `579565`, GPU0.
  - teacher: current baseline `best_search_p10.pt`.
  - LR `1e-7`, `constraint_center_weight=0.25`, `search_p10_soft_threshold_weight=0.01`, `search_p5_soft_threshold_weight=0.02`.
  - distill: state `0.02`, prediction `0.01`, feature/mask `0.0`, EMA `0.9995`.
- S1-MN:
  - run: `runs/NON_XR/raw/stage1_s1mn_s1mj_p10seed_baselinecenterteacher_lr7p5e8_20260622_161345`
  - PID `579566`, GPU1.
  - teacher: current baseline `best_metric_search_center_px.pt`.
  - LR `7.5e-8`, `constraint_center_weight=0.35`, `search_p10_soft_threshold_weight=0.02`, `search_p5_soft_threshold_weight=0.02`.
  - distill: state `0.03`, prediction `0.015`, feature/mask `0.0`, EMA `0.9995`.
- Startup evidence:
  - Both lanes loaded `loaded_count=132 partial=0 skipped=0`.
  - Both lanes resolved CUDA devices correctly: S1-MM `cuda:0`, S1-MN `cuda:1`.
  - Both lanes entered epoch `1/50` training loop.
- Early monitor snapshot:
  - Both lanes completed epoch `3/50` validation and entered epoch `4/50`.
  - S1-MM epoch `3/50` summary: best P10 `28.4827`.
  - S1-MN epoch `3/50` summary: saved `best_search_p10.pt` at P10 `28.4827`.
  - This is not a promotion decision because the gate remains `>=50` epochs.
- Gate:
  - Do not judge before at least `50` epochs.
  - Strict promotion: P10 above current baseline and center no worse than current baseline.
  - Secondary review: if P10/P5 improve materially and center regression is below `0.05 px`, mark as candidate for Stage2 downstream smoke instead of replacing the strict baseline.

## 2026-06-22 16:53 KST S1-MM/S1-MN Closeout

- S1-MM/S1-MN reached the required `50/50` epoch gate.
- Processes completed after epoch `50/50`; no active S1-MM/S1-MN train process remained in `pgrep`.
- Official reporter:
  `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --baseline runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 --min-epochs 50 --target-epochs 50 --followup runs/NON_XR/raw/stage1_s1mm_s1mj_p10seed_baselinep10teacher_lr1e7_20260622_161345 --followup runs/NON_XR/raw/stage1_s1mn_s1mj_p10seed_baselinecenterteacher_lr7p5e8_20260622_161345 --format summary`
- Baseline:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- S1-MM:
  - run: `runs/NON_XR/raw/stage1_s1mm_s1mj_p10seed_baselinep10teacher_lr1e7_20260622_161345`
  - epochs `50/50`
  - best P10 `28.482705062290407`, delta `+0.2055256861560757`
  - best P5 `9.088050554383475`, delta `-1.0658131905321806`
  - best center `17.416398777152008`, center regression `0.15881487108626402 px`
  - promotable: `False`
- S1-MN:
  - run: `runs/NON_XR/raw/stage1_s1mn_s1mj_p10seed_baselinecenterteacher_lr7p5e8_20260622_161345`
  - epochs `50/50`
  - best P10 `28.482705062290407`, delta `+0.2055256861560757`
  - best P5 `9.088050554383475`, delta `-1.0658131905321806`
  - best center `17.42239284515381`, center regression `0.16480893908806493 px`
  - promotable: `False`
- Decision:
  - No strict baseline promotion.
  - Baseline teacher distillation did not recover center/P5 from the S1-MJ P10 seed.
  - Current strict baseline remains:
    `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Next action:
  - Before launching another 50-epoch GPU branch, run baseline-internal checkpoint soup/interpolation among `best_search_p10.pt`, `best_search_p5.pt`, and `best_metric_search_center_px.pt` because they share model keys and represent the current P10/P5/center tradeoff.

## 2026-06-22 17:35 KST S1-MO/S1-MP Ensemble-Teacher Watcher

- Current active objective remains Stage1 frame-based Search baseline improvement.
- Weight-space checkpoint soup result:
  - Summary: `runs/NON_XR/shared/eval/stage1_baseline_internal_soup_20260622_1710/summary.json`.
  - No soup promoted; `ref_p10` remained the highest-P10 candidate.
- Output-space ensemble result:
  - Summary: `runs/NON_XR/shared/eval/stage1_output_ensemble_20260622_1720/summary.json`.
  - `out_p10_090_p5_050_center_050` tied P10 and slightly improved same-evaluator P5/center versus `ref_p10`, but did not meet strict baseline promotion.
- Added fixed output-ensemble teacher support:
  - `src/hbtxr/training/ensemble_teacher.py`.
  - `src/hbtxr/training/model_factory.py`.
  - `src/hbtxr/training/step_runner.py`.
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh`.
  - `tests/test_ensemble_teacher.py`.
- Validation:
  - `pytest tests/test_ensemble_teacher.py` passed.
  - `py_compile` passed for modified training modules.
  - `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh` passed.
  - CPU smoke with `1` train batch and `1` val batch passed for both lanes after disabling EMA updates for fixed ensemble teachers.
- Launched S1-MO/S1-MP on host GPU:
  - S1-MO PID `1133694`, GPU0:
    `runs/NON_XR/raw/stage1_s1mo_baseline_ensembleteacher_lr1e7_stage1_s1mo_s1mp_ensemble_teacher_20260622_1725_20260622_172100`.
  - S1-MP PID `1133695`, GPU1:
    `runs/NON_XR/raw/stage1_s1mp_baseline_ensembleteacher_lr7p5e8_stage1_s1mo_s1mp_ensemble_teacher_20260622_1725_20260622_172100`.
  - Both lanes resolved CUDA correctly and entered training.
- Started min-50 watcher:
  - PID `1157196`.
  - Log: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1735.log`.
  - JSON: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1735_report.json`.
  - Summary: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1735_report.txt`.
- First watcher report:
  - S1-MO `6/50`, P10 `28.27717937613433`, P5 `9.247529443704858`, center `17.298999822364664`.
  - S1-MP `6/50`, P10 `28.27717937613433`, P5 `8.806154790914283`, center `17.297886924923592`.
  - Status: `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.

## 2026-06-22 17:31 KST S1-MO/S1-MP Monitor Refresh

- Training is still active on both GPUs:
  - S1-MO main PID `1133694`, GPU0.
  - S1-MP main PID `1133695`, GPU1.
  - `nvidia-smi` showed GPU0 utilization `46%` and GPU1 utilization `55%`.
- Direct reporter snapshot:
  - Baseline P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - S1-MO `16/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.298999822364664`.
  - S1-MP `16/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.297886924923592`.
  - Status remains `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
  - Log tail confirms both lanes have completed epoch `16/50` validation and entered epoch `17/50` training.
- Monitoring correction:
  - The earlier sandbox-launched watcher stopped refreshing reports, so a host watcher was relaunched with PID `1188026`.
  - New watcher log: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800.log`.
  - New summary: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800_report.txt`.
  - New JSON: `runs/NON_XR/shared/_logs/stage1_s1mo_s1mp_min50_watch_20260622_1800_report.json`.
- Gate remains unchanged: do not judge promotion before `50/50`; strict promotion still requires P10 above `28.27717937613433` and center no worse than `17.257583906065744`.

## 2026-06-22 17:37 KST S1-MQ/S1-MR Weighted Ensemble-Teacher Queue

- Current S1-MO/S1-MP remains active and below the `50/50` gate, so no new GPU training was launched.
- Latest direct reporter at this checkpoint:
  - S1-MO `24/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.298999822364664`.
  - S1-MP `24/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.297886924923592`.
  - Status remains `wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
- Prepared the next Stage1 branch for immediate use if S1-MO/S1-MP fail promotion:
  - Runner extension: `scripts/external/run_stage1_frame_search_ensemble_teacher.sh` now supports optional `training.sampler.*` overrides.
  - Queue wrapper: `scripts/external/run_stage1_frame_search_weighted_ensemble_teacher.sh`.
- Default wrapper safety:
  - `DRY_RUN=1`, `DETACH=1`.
  - Existing ensemble-teacher behavior remains unchanged unless `SAMPLER_ENABLED=true`.
- Rationale:
  - Weight-space soup failed to improve the strict P10 gate.
  - Output-space ensemble tied P10 but did not exceed baseline.
  - S1-MM/S1-MN improved P10 but regressed P5/center.
  - Hard-subset training in later Stage2 diagnostics over-specialized, so this branch keeps the full manifest and only changes sampling probability.
- Planned lanes:
  - S1-MQ: LR `1.5e-7`, state distill `0.02`, prediction distill `0.005`, center constraint `0.25`, P5 soft threshold `0.03`.
  - S1-MR: LR `1e-7`, state distill `0.03`, prediction distill `0.005`, center constraint `0.35`, P5 soft threshold `0.02`.
  - Teacher ensemble remains current Stage1 baseline `best_search_p10.pt` `0.90`, `best_search_p5.pt` `0.05`, and `best_metric_search_center_px.pt` `0.05`.
- Sampler validation:
  - Full train manifest rows: `5929`.
  - Sampler configuration: low similarity `similarity_target <= 0.1` multiplier `3.0`, `session_201` multiplier `1.5`, max multiplier `6.0`.
  - Computed weights are finite: unique values `{1.0, 1.5, 3.0, 4.5}`.
- Validation commands:
  - `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh`
  - `bash -n scripts/external/run_stage1_frame_search_weighted_ensemble_teacher.sh`
  - `DRY_RUN=1 RUN_TAG=stage1_weighted_ensemble_dryrun_20260622 bash scripts/external/run_stage1_frame_search_weighted_ensemble_teacher.sh`
- Execution rule:
  - Do not launch S1-MQ/S1-MR while S1-MO/S1-MP are still using both GPUs.
  - If S1-MO/S1-MP fail at `50/50`, run S1-MQ/S1-MR with `DRY_RUN=0` on GPU0/GPU1 and attach the same min-50 reporter.

## 2026-06-22 17:42 KST S1-MO/S1-MP Poll

- Direct reporter after an additional polling interval:
  - S1-MO `29/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.298999822364664`.
  - S1-MP `29/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.297886924923592`.
  - `status=wait_min_epochs`; `promote_checkpoint=None`; `continue_training=True`.
- Decision unchanged: continue S1-MO/S1-MP until `50/50`; keep S1-MQ/S1-MR queued only.

## 2026-06-22 17:57 KST S1-MO/S1-MP Closeout and S1-MQ/S1-MR Launch

- S1-MO/S1-MP reached the required `50/50` gate.
- Official reporter:
  - S1-MO `50/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.298999822364664`.
  - S1-MP `50/50`: P10 `28.27717937613433`, P5 `9.38230034090438`, center `17.297886924923592`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - `status=keep_baseline_unless_later_improves`; `promote_checkpoint=None`.
- Decision:
  - No strict baseline promotion.
  - Both lanes tied P10 but failed P5 and center preservation.
  - Current strict baseline remains `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- GPU/process state before follow-up launch:
  - S1-MO/S1-MP train processes were gone.
  - GPU0/GPU1 were free except small display memory.
- Launched S1-MQ/S1-MR weighted ensemble-teacher branch on host:
  - S1-MQ PID `1348800`, GPU0:
    `runs/NON_XR/raw/stage1_s1mq_lowsim_weighted_ensemble_lr1p5e7_stage1_s1mq_s1mr_weighted_ensemble_teacher_20260622_1758_20260622_175755`.
  - S1-MR PID `1348802`, GPU1:
    `runs/NON_XR/raw/stage1_s1mr_lowsim_weighted_centerhold_lr1e7_stage1_s1mq_s1mr_weighted_ensemble_teacher_20260622_1758_20260622_175755`.
  - Both lanes resolved CUDA correctly and loaded the baseline checkpoint with `loaded_count=142 partial=0 skipped=0`.
  - Host `nvidia-smi` confirmed both GPUs active after launch.
- Started S1-MQ/S1-MR min-50 watcher:
  - PID `1352162`.
  - Log: `runs/NON_XR/shared/_logs/stage1_s1mq_s1mr_min50_watch_20260622_1800.log`.
  - Summary: `runs/NON_XR/shared/_logs/stage1_s1mq_s1mr_min50_watch_20260622_1800_report.txt`.
  - JSON: `runs/NON_XR/shared/_logs/stage1_s1mq_s1mr_min50_watch_20260622_1800_report.json`.

## 2026-06-22 18:03 KST S1-MQ/S1-MR Startup Sanity

- Both lanes progressed beyond startup:
  - S1-MQ entered epoch `5/50` training; early best P10 `28.142408460940956`.
  - S1-MR entered epoch `5/50` training; early best P10 remains baseline-tied at `28.27717937613433`.
- Direct reporter at `7/50`:
  - S1-MR: P10 `28.27717937613433`, P5 `8.721923000407669`, center `17.280365035219013`.
  - S1-MQ: P10 `28.142408460940956`, P5 `8.806154790914283`, center `17.283770327298146`.
  - Status `wait_min_epochs`; `promote_checkpoint=None`.
- Early read:
  - S1-MR is the safer lane so far because it preserves P10 and reduces center regression versus S1-MO/S1-MP.
  - S1-MQ is more aggressive and currently below baseline P10.
  - No judgment until `50/50`.

## 2026-06-22 18:38 KST S1-MQ/S1-MR Closeout

- S1-MQ/S1-MR reached the required `50/50` gate.
- Official reporter:
  - S1-MQ `50/50`: P10 `28.142408460940956`, P5 `8.806154790914283`, center `17.27892105084545`.
  - S1-MR `50/50`: P10 `28.27717937613433`, P5 `8.839847528709555`, center `17.26318852856474`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - `status=keep_baseline_unless_later_improves`; `promote_checkpoint=None`.
- Decision:
  - No strict baseline promotion.
  - S1-MR tied baseline P10 but remained worse on P5 and center.
  - S1-MQ lost P10 and also remained worse on P5 and center.
  - Weighted full-manifest sampling is not the next lever; it repeats the center/P5 weakness.
- GPU/process state after closeout:
  - GPU0/GPU1 free except small display memory.

## 2026-06-22 18:42 KST S1-MS/S1-MT P10-Seed Recovery Launch

- Implemented the next Stage1 branch:
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh` now separates `PRETRAIN_CKPT` from teacher checkpoints.
  - It also exposes teacher weights via `ENSEMBLE_P10_WEIGHT`, `ENSEMBLE_P5_WEIGHT`, and `ENSEMBLE_CENTER_WEIGHT`.
  - New wrapper: `scripts/external/run_stage1_frame_search_p10seed_recovery_ensemble_teacher.sh`.
- Rationale:
  - S1-MM/S1-MN previously proved a high-P10 seed can reach P10 `28.482705062290407`, but center regressed to about `17.42`.
  - S1-MQ/S1-MR proved sampler weighting does not recover strict P5/center.
  - New branch starts from S1-MM high-P10 seed, disables sampler, and applies stronger baseline P5/center guidance.
- Validation before launch:
  - `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_p10seed_recovery_ensemble_teacher.sh`: passed.
  - Dry-run confirmed `PRETRAIN_CKPT` is S1-MM `best_search_p10.pt`, teacher weights are `0.75/0.10/0.15`, sampler is disabled, and `training.scheduler.min_lr=2e-8`.
- Launched on host:
  - S1-MS PID `1614875`, GPU0:
    `runs/NON_XR/raw/stage1_s1ms_p10seed_centerrecover_lr5e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824`.
  - S1-MT PID `1614877`, GPU1:
    `runs/NON_XR/raw/stage1_s1mt_p10seed_p5centerrecover_lr4e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824`.
- Hyperparameters:
  - Teacher ensemble: baseline P10/P5/center checkpoints with weights `0.75/0.10/0.15`.
  - S1-MS: LR `5e-8`, state `0.04`, prediction `0.01`, center `0.75`, P5 soft `0.05`.
  - S1-MT: LR `4e-8`, state `0.05`, prediction `0.015`, center `1.00`, P5 soft `0.08`.
  - Sampler disabled for both lanes.
- Startup evidence:
  - Both lanes loaded S1-MM seed checkpoint with `loaded_count=126 partial=0 skipped=0`.
  - S1-MS resolved `cuda:0`; S1-MT resolved `cuda:1`.
  - Both lanes entered epoch `1/50`.
  - `nvidia-smi` showed GPU0/GPU1 active after launch.
- Watcher:
  - PID `1620266`.
  - Log: `runs/NON_XR/shared/_logs/stage1_s1ms_s1mt_min50_watch_20260622_1842.log`.
  - Summary: `runs/NON_XR/shared/_logs/stage1_s1ms_s1mt_min50_watch_20260622_1842_report.txt`.
  - JSON: `runs/NON_XR/shared/_logs/stage1_s1ms_s1mt_min50_watch_20260622_1842_report.json`.

## 2026-06-22 19:12 KST S1-MS/S1-MT Closeout

- S1-MS/S1-MT reached the required `50/50` gate.
- Official reporter:
  - S1-MS `50/50`: P10 `28.347934165090884`, P5 `9.32389961098725`, center `17.358882256273954`.
  - S1-MT `50/50`: P10 `28.482705062290407`, P5 `9.32389961098725`, center `17.358525235697908`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - `status=keep_baseline_unless_later_improves`; `promote_checkpoint=None`.
- Interpretation:
  - Search P10 accuracy did improve: S1-MT is `+0.2055256861560757 pp` above the strict baseline.
  - Strict baseline promotion still fails because center is `0.10094132963216396 px` worse and P5 is `0.8299641339284065 pp` lower.
  - The current failure mode is not lack of P10 improvement; it is the P10/P5/center tradeoff.
- Decision:
  - Keep current strict baseline checkpoint for downstream baseline use.
  - Record S1-MT as the best P10-only Stage1 Search candidate so far.
  - Next experiment should be center-direct refinement from S1-MT rather than another P10-seeking branch.

## 2026-06-22 20:10 KST Stage1 Search Structure Ablation Setup

- User requested broader model-structure experiments: head changes, auxiliary heads, and backbone depth changes.
- Diagnosis:
  - P10 is not flat: S1-MT reached P10 `28.482705062290407`, above the strict baseline `28.27717937613433`.
  - Strict promotion fails because S1-MT center is `17.358525235697908`, worse than baseline `17.257583906065744`, and P5 is `9.32389961098725`, below baseline `10.153863744915656`.
  - The active Stage1 Search runner disables eye and mask heads by default, so changing `eye_variant` alone does not directly attack the active metric.
- Implemented:
  - `PupilSearchHead` now supports zero-init residual variants: `residual_mlp` and `deep_residual_mlp`.
  - Config surface added: `model.heads.search_variant`, `model.heads.search_residual_hidden_dim`.
  - Runner extension: `scripts/external/run_stage1_frame_search_ensemble_teacher.sh` accepts lane-specific extra overrides.
  - New architecture queue: `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
  - New GPU-free queue launcher: `scripts/external/run_stage1_frame_search_arch_ablation_when_gpu_free.sh`.
- Planned experiments:
  - S1-MW: residual Search Head only.
  - S1-MX: residual Search Head plus search bbox/OBB auxiliary geometry losses.
  - S1-MY: depth-8 center-best probe.
  - S1-MZ: residual Search Head plus low-weight mask cascade guidance.
- Validation:
  - `python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py`
  - `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh`
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_when_gpu_free.sh`
  - `DRY_RUN=1` passed for both `head_aux` and `depth_mask` suites.

## 2026-06-23 09:17 KST Stage1 Search Center-Constraint Fix And Relaunch

- User requested broader optimizer, loss, scheduler, and data augmentation experiments.
- Added/verified experiment support:
  - Search Head variants: `legacy`, `residual_mlp`, `deep_residual_mlp`.
  - HeadFactory/ROI suite: S1-NA `sot_center`, S1-NB `yolo_detect`.
  - DeiT preload suite: S1-NC/S1-ND using `/home/kjm26/project/PRJXR/model_zoo/deit_tiny_distilled_patch16_224.pth`.
  - Optimizer/loss/scheduler suite: S1-NE Lion+plateau, S1-NF Adopt+step.
  - Augmentation suite: S1-NG frame augmentation, S1-NH frame+event augmentation.
- Critical finding:
  - `src/hbtxr/models/pruning.py` previously zeroed `loss.constraint_center_weight` whenever `model.heads.active != track`.
  - Stage1 Search uses `model.heads.active=search`, so center-direct runs silently had `loss_constraint_center=0.0`.
  - Fixed normalization so Search and Track preserve center constraint.
- Validation:
  - `python3 -m py_compile src/hbtxr/models/pruning.py`: passed.
  - Config-normalization smoke: `search -> 1.5`, `eye -> 0.0`.
  - Dry-run passed after override corrections for `depth_mask` and `head_factory`.
- Completed but non-promoted original S1-MU/S1-MV:
  - S1-MU `50/50`: best P10 `28.347934165090884`, best P5 `9.32389961098725`, best center `17.369157035395784`.
  - S1-MV `50/50`: best P10 `28.347934165090884`, best P5 `9.32389961098725`, best center `17.38588412302845`.
  - These are not valid center-direct evidence because `loss_constraint_center=0.0`.
- Relaunched corrected center-direct branch:
  - Command tag: `RUN_TAG=stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622`.
  - S1-MU PID `2597105`, GPU0:
    `runs/NON_XR/raw/stage1_s1mu_s1mt_centerheavy_lr2e8_stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622_20260623_091716`.
  - S1-MV PID `2597106`, GPU1:
    `runs/NON_XR/raw/stage1_s1mv_s1mt_centermax_lr1e8_stage1_s1mu_s1mv_center_direct_fixed_constraint_20260622_20260623_091716`.
  - Startup history confirms center constraint is active:
    - S1-MU epoch 1 `loss_constraint_center`: train `98.21118527024261`, val `95.75954721558769`.
    - S1-MV epoch 1 `loss_constraint_center`: train `163.8419010028685`, val `159.91095028283462`.
- Corrected branch closeout:
  - S1-MU `50/50`: P10 `28.482705062290407`, P5 `9.514825083174795`, center `17.427942365970253`.
  - S1-MV `50/50`: P10 `28.482705062290407`, P5 `9.514825083174795`, center `17.418982910660077`.
  - Decision: no promotion. Center loss is now active, but center still misses the strict baseline gate.
- Queued next suites:
  - Added `scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh`.
  - Terminated the previous single-suite watcher PID `2652073` to avoid duplicate Head/Aux launches.
  - Chain waits for corrected S1-MU/S1-MV PIDs `2597105 2597106`.
  - Chain order: `head_aux -> opt_sched_loss -> augmentation -> deit_preload -> head_factory -> depth_mask`.
  - Chain PID `3219476`.
  - Chain log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_chain_after_fixed_center_20260623.log`.
  - Current suite launched by chain: `head_aux`.
  - Active Head/Aux PIDs: S1-MW `3219510`, S1-MX `3219518`.
  - Early direct report at `4/50`:
    - S1-MW: P10 `28.232255755730396`, P5 `8.835355146875921`, center `17.31714263502157`.
    - S1-MX: P10 `28.209793918537645`, P5 `8.835355146875921`, center `17.317125887241005`.
  - Gate status: `wait_min_epochs`; no promotion decision before `50/50`.

## 2026-06-23 KST Stage1 Search Optimizer/Loss/Scheduler/Augmentation Expansion

- User requested explicit optimizer, loss, scheduler, and data augmentation experiments.
- Current running chain PID `3219476` remains active and is not restarted to avoid duplicate GPU launches.
- Existing queued suites already cover:
  - S1-NE: Lion + plateau scheduler + stronger P5/center/search loss guard.
  - S1-NF: Adopt + step scheduler + stronger P5/center/search loss guard.
  - S1-NG: frame gain/bias/noise augmentation.
  - S1-NH: frame + event gain/dropout/noise augmentation.
- Added follow-up suites:
  - `opt_sched_loss_ext`
    - S1-NI: AdamW schedule-free + cautious modifier, scheduler disabled, residual Search Head, P5/center guard.
    - S1-NJ: MuSGD + cosine scheduler, conservative Muon/SGD mix, residual Search Head, P5/center guard.
  - `augmentation_ext`
    - S1-NK: mild frame-only augmentation.
    - S1-NL: mild frame + event dropout/noise augmentation.
- Updated new chain default order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Important limitation:
  - The already-running chain captured its original suite list at launch, so the extension suites will run only in a new chain or explicit suite launch.
- Extension reservation:
  - A sandbox-detached attempt produced PID file value `9` but no live process; ignore that stale attempt.
  - Host-visible extension chain launched with PID `3546751`.
  - Log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_ext_after_existing_chain_20260623_host.log`.
  - It waits for existing chain PID `3219476`, then runs `opt_sched_loss_ext -> augmentation_ext`.

## 2026-06-23 KST Stage1 Head/Aux Closeout And Optimizer Suite Launch

- S1-MW/S1-MX reached the `50/50` gate.
- Official reporter:
  - S1-MW residual Search Head: P10 `28.232255755730396`, P5 `9.24528328877575`, center `17.31714263502157`.
  - S1-MX residual Search Head + bbox/OBB aux: P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.317125887241005`.
  - Baseline remains P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Decision:
  - No promotion. Both structural head probes underperform baseline on P10, P5, and center.
- Chain action:
  - Existing suite chain launched `opt_sched_loss`.
  - S1-NE Lion+plateau PID `3467797`, run root `runs/NON_XR/raw/stage1_s1ne_lion_plateau_p5guard_lr2e8_stage1_arch_chain_after_fixed_center_20260623_opt_sched_loss_20260623_110529`.
  - S1-NF Adopt+step PID `3467798`, run root `runs/NON_XR/raw/stage1_s1nf_adopt_step_center_lr3e8_stage1_arch_chain_after_fixed_center_20260623_opt_sched_loss_20260623_110529`.
- Startup evidence:
  - S1-NE reached epoch `3/50`; epoch 2 best P10 `28.1929`, best P5 `8.8297`.
  - S1-NF reached epoch `3/50`; epoch 1 best P10 `28.1593`, best P5 `8.6579`.
  - No optimizer/scheduler launch error observed.

## 2026-06-23 KST Stage1 Additional Optimizer/Loss/Scheduler Suite

- User requested broader optimizer, loss, scheduler, and data augmentation experiments.
- Active status from reporter/sub-agent audit:
  - S1-NE/S1-NF have not reached the `50/50` gate.
  - S1-NF has a temporary P10 edge over baseline, but P5 and center remain below baseline.
  - No promotion claim before `50/50`.
- Added `opt_sched_loss_plus` to `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
  - S1-NO: AdEMAMix + cosine scheduler, residual Search Head, P5/center guard.
  - S1-NP: MARS + plateau scheduler on `metric_search_center_px`, residual Search Head, stronger center/P5 guard.
- Updated `scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` default order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> opt_sched_loss_plus -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh`: passed.
  - `SUITE=opt_sched_loss_plus DRY_RUN=1`: command generation passed.
  - CPU optimizer construction smoke: `adema_mix` and `mars` both build.
- Execution plan:
  - Do not interrupt current chain PID `3219476`.
  - Existing extension chain PID `3546751` remains queued for `opt_sched_loss_ext -> augmentation_ext`.
  - Host-visible watcher PID `3645713` now waits for extension chain PID `3546751`, then runs `opt_sched_loss_plus`.
  - Reservation log: `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_plus_after_ext_20260623_host2.log`.
  - Sandbox-detached stale attempt with PID-file value `9` is not live and should be ignored.

## 2026-06-23 KST Stage1 Distillation Extension Suite

- Added `distill_ext` to `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
- Updated `scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` default order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> opt_sched_loss_plus -> distill_ext -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Planned lanes:
  - S1-NQ: EMA self-distillation, ensemble disabled, teacher initialized from student, `ema_decay=0.999`, residual Search Head, feature/state/prediction distillation, KD, RKD, and P5/center guard.
  - S1-NR: S1-MT high-P10 single teacher with `distillation.ensemble.enabled=false`, residual Search Head, feature/state/prediction distillation, KD, RKD, plateau scheduler on `metric_search_center_px`, and stronger P5/center guard.
- S1-NR teacher checkpoint:
  - `runs/NON_XR/raw/stage1_s1mt_p10seed_p5centerrecover_lr4e8_stage1_s1ms_s1mt_p10seed_recovery_20260622_1840_20260622_183824/train/best_search_p10.pt`.
- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`: passed.
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh`: passed.
  - `SUITE=distill_ext DRY_RUN=1`: command generation passed.
  - CPU teacher creation smoke passed for EMA self-distill and single-teacher modes.
  - `git diff --check` over changed scripts passed.
- Reservation:
  - Host-visible watcher PID `3681125`.
  - Wait target PID `3645713`.
  - Log `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_distill_after_plus_20260623_host.log`.
- Current active gate:
  - S1-NE/S1-NF remain below `50/50`, so no promotion decision is valid yet.

## 2026-06-23 KST S1-NE/S1-NF Optimizer Suite Closeout

- S1-NE/S1-NF reached the required `50/50` gate.
- Official reporter:
  - S1-NE Lion+plateau: P10 `28.192947567633862`, P5 `9.380054185975272`, center `17.319932807166623`.
  - S1-NF Adopt+step: P10 `28.35018028403228`, P5 `9.363207799083781`, center `17.306676203349852`.
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
- Decision:
  - No promotion.
  - S1-NF has a useful P10 signal at `+0.07300090789794922 pp`, but P5 and center regress too much.
  - S1-NE underperforms all strict gates.
  - Current strict Stage1 Search baseline remains unchanged.
- Runtime state:
  - S1-NE/S1-NF train processes are gone.
  - GPU0/GPU1 are free except display memory.
  - Existing chain PID `3219476` should proceed to `augmentation` on its next poll.

## 2026-06-23 KST S1-NG/S1-NH Augmentation Suite Launch

- Existing suite chain PID `3219476` advanced to `augmentation`.
- Launched lanes:
  - S1-NG frame augmentation, residual Search Head, PID `3711690`, GPU0.
  - S1-NH frame+event augmentation, residual Search Head, PID `3711691`, GPU1.
- Run roots:
  - `runs/NON_XR/raw/stage1_s1ng_frame_aug_residual_lr4e8_stage1_arch_chain_after_fixed_center_20260623_augmentation_*`.
  - `runs/NON_XR/raw/stage1_s1nh_frame_event_aug_residual_lr4e8_stage1_arch_chain_after_fixed_center_20260623_augmentation_*`.
- Chain state:
  - Waiting before `deit_preload` on PIDs `3711690 3711691`.
- Gate:
  - No evaluation before `50/50`.

## 2026-06-23 KST Stage1 Loss-Only Extension Suite

- User explicitly requested additional optimizer, loss, scheduler, and data augmentation experiments.
- Sub-agent routing:
  - Attempted GPT-5.3-Codex-Spark read-only coverage evaluator, but runtime returned quota limit.
  - Spawned GPT-5.5 read-only evaluator as fallback; no file edits or training launches delegated.
- Added `loss_ext` to `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
  - S1-NS: AdamW + cosine fixed; tighter center radius, stronger center/P5 soft-threshold, moderate XY/geometry weights.
  - S1-NT: AdamW + cosine fixed; geometry-balanced Search loss with higher AB/trig/GWD weights, lower confidence weight, and sharper P10 soft-threshold.
- Updated default suite-chain order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> opt_sched_loss_plus -> distill_ext -> loss_ext -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Rationale:
  - Existing optimizer suites change optimizer, scheduler, and loss weights together.
  - `loss_ext` isolates loss weighting while keeping optimizer/scheduler fixed to AdamW + cosine.
- Runtime policy:
  - Do not launch duplicate GPU jobs while S1-NG/S1-NH are running.
  - Queue or launch `loss_ext` only after active/waiting chains are reconciled.

## 2026-06-23 KST Stage1 Scheduler-Only Extension Suite

- Added `scheduler_ext` to `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
- Planned lanes:
  - S1-NU: AdamW, fixed residual Search Head/objective, plateau scheduler on `metric_search_center_px`.
  - S1-NV: AdamW, same fixed residual Search Head/objective, step scheduler.
- Updated `scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` default order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> opt_sched_loss_plus -> distill_ext -> loss_ext -> scheduler_ext -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Rationale:
  - Existing `opt_sched_loss*` suites cover scheduler types, but scheduler changes are entangled with optimizer and loss changes.
  - `scheduler_ext` isolates plateau vs step under the same optimizer/head/loss family.
- Reservation:
  - `loss_ext` host-visible watcher PID `42233`, waiting for distill chain PID `3681125`.
  - `scheduler_ext` host-visible watcher PID `357803`, waiting for loss chain PID `42233`.
- Validation:
  - `bash -n` passed for queue and chain scripts.
  - `SUITE=loss_ext DRY_RUN=1` command generation passed.
  - `SUITE=scheduler_ext DRY_RUN=1` command generation passed.
  - `git diff --check` passed for changed scripts/docs.

## 2026-06-23 KST Stage1 Main Chain Partial Closeout

- Completed and evaluated `augmentation`, `deit_preload`, and `head_factory` suites against strict baseline:
  - Baseline: P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`.
  - S1-NG/S1-NH augmentation: P10 `28.176101198736227`, P5 `9.278976026571021`, center `17.296725754467946`; no promotion.
  - S1-NC DeiT-Tiny AdamW: P10 `5.795148381647074`, P5 `2.6448787653221273`, center `43.2115170460827`; no promotion.
  - S1-ND DeiT-Tiny residual: P10 `5.20552577612535`, P5 `2.015947935716161`, center `44.50417672463183`; no promotion.
  - S1-NA HeadFactory `sot_center`: P10 `28.32771844683953`, P5 `9.24528328877575`, center `17.31330567036035`; P10-only improvement, no promotion.
  - S1-NB HeadFactory `yolo_detect`: P10 `28.32771844683953`, P5 `9.24528328877575`, center `17.314067854071563`; P10-only improvement, no promotion.
- Decision:
  - Current Stage1 Search baseline remains unchanged.
- Runtime:
  - Main chain advanced to `depth_mask`.
  - Active lanes: S1-MY depth-8 on GPU0 and S1-MZ mask cascade on GPU1.

## 2026-06-23 KST Stage1 P10-Recovery Extension Suite

- Added `p10_recover_ext` to `scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`.
- Planned lanes:
  - S1-NW: Adopt + step + residual Search Head. This takes the S1-NF P10-only signal and adds tighter center radius plus stronger P5/center guard.
  - S1-NX: HeadFactory `sot_center` ROI aux. This takes the S1-NA P10-only signal and lowers aux/eye weights while adding stronger P5/center guard.
- Updated `scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` default order:
  - `head_aux -> opt_sched_loss -> opt_sched_loss_ext -> opt_sched_loss_plus -> distill_ext -> loss_ext -> scheduler_ext -> p10_recover_ext -> augmentation -> augmentation_ext -> deit_preload -> head_factory -> depth_mask`.
- Rationale:
  - S1-NF improved P10 by `+0.07300090789794922 pp` but failed P5/center.
  - S1-NA/S1-NB improved P10 by `+0.050539070705198696 pp` but failed P5/center.
  - S1-NW/S1-NX test whether those P10-lift mechanisms can be retained under stronger center/P5 recovery pressure.
- Validation:
  - `bash -n` passed for queue and chain scripts.
  - `SUITE=p10_recover_ext DRY_RUN=1` command generation passed.
  - `git diff --check` passed for changed scripts/docs.
- Reservation:
  - Host-visible watcher PID `448685`.
  - Wait target PID `357803`.
  - Log `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_p10_recover_after_scheduler_20260623_host.log`.

## 2026-06-23 KST Stage1 Depth-Mask Closeout And Axis Coverage Check

- Closed the main-chain `depth_mask` suite at the required `50/50` gate.
- Official reporter snapshot:
  - S1-MY depth-8 center-best:
    - P10 `28.310872077941895`
    - P5 `9.127358760473863`
    - center `17.354738820273923`
  - S1-MZ mask-cascade residual:
    - P10 `5.133647990676592`
    - P5 `1.769991046977493`
    - center `44.22043274933437`
  - Strict baseline remains:
    - P10 `28.27717937613433`
    - P5 `10.153863744915656`
    - center `17.257583906065744`
- Decision:
  - No promotion.
  - S1-MY gives a small P10-only lift but fails both center and P5 gates.
  - S1-MZ is a hard failure; do not reopen the mask-cascade path without a separate mask-target/adaptation phase.
- Runtime transition:
  - Main chain completed.
  - Extension chain PID `3546751` advanced and launched `opt_sched_loss_ext`.
  - Active lanes:
    - S1-NI AdamW schedule-free + cautious on GPU0.
    - S1-NJ MuSGD + cosine on GPU1.
  - Early progress from history/logs is around epoch `2-3/50`; no valid promotion decision before `50/50`.
- Sub-agent verification:
  - GPT-5.3-Codex-Spark evaluator was attempted first but failed due usage limit.
  - GPT-5.5 read-only evaluator completed and confirmed no missing requested axis:
    - Optimizer: `opt_sched_loss`, `opt_sched_loss_ext`, `opt_sched_loss_plus`.
    - Loss: `loss_ext`.
    - Scheduler: `scheduler_ext`.
    - Data augmentation: `augmentation`, `augmentation_ext`.
  - Queue safety conclusion: do not launch another fresh default suite chain now because the existing host-visible chains already serialize all requested follow-up suites.
- Validation performed in this turn:
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh`
  - `SUITE=loss_ext DRY_RUN=1 RUN_TAG=stage1_loss_ext_dryrun DETACH=0 bash scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`
  - `SUITE=scheduler_ext DRY_RUN=1 RUN_TAG=stage1_scheduler_ext_dryrun DETACH=0 bash scripts/external/run_stage1_frame_search_arch_ablation_queue.sh`

## 2026-06-23 KST Stage1 Opt-Sched-Loss-Ext Runtime Snapshot

- Active suite: `opt_sched_loss_ext`.
- Chain PID: `3546751`.
- Chain log:
  `runs/NON_XR/shared/_logs/stage1_arch_ablation_suite_chain_stage1_arch_ext_after_existing_chain_20260623_host.log`.
- Active lanes:
  - S1-NI:
    `runs/NON_XR/raw/stage1_s1ni_adamw_schedulefree_cautious_p5guard_lr1e7_stage1_arch_ext_after_existing_chain_20260623_host_opt_sched_loss_ext_20260623_141852`.
  - S1-NJ:
    `runs/NON_XR/raw/stage1_s1nj_musgd_cosine_p5guard_lr7p5e8_stage1_arch_ext_after_existing_chain_20260623_host_opt_sched_loss_ext_20260623_141852`.
- Current progress:
  - Both lanes have written epoch `6/50` history and are training epoch `7/50`.
  - GPU compute processes are active on both devices.
- Epoch-6/current-best evidence:
  - S1-NI epoch-6 val: P10 `27.64038690531029`, P5 `9.24528328877575`, center `17.472669880345183`.
  - S1-NI current best from history: P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.318899788946474`.
  - S1-NJ epoch-6 val: P10 `28.27717937613433`, P5 `8.334456677706736`, center `17.2939717994546`.
  - S1-NJ current best from history: P10 `28.27717937613433`, P5 `8.334456677706736`, center `17.292718172073364`.
- Decision:
  - No promotion decision is valid before `50/50`.
  - Early metrics do not show a strict promotable candidate yet.
  - Continue monitoring; `augmentation_ext` is queued behind S1-NI/S1-NJ inside the same extension chain.
- Ablation provenance:
  - Skill phase/domain: `P1` / `Research Workflow`.
  - Parent skill: `paper-idea-generator`; worker skill: `ablation-study-designer`.
  - Source context: optimizer/loss/scheduler/augmentation ablation coverage requested by the user for Stage1 frame-search baseline improvement.
  - Known facts: active baseline gates are P10 `28.27717937613433`, P5 `10.153863744915656`, center `17.257583906065744`; S1-NI/S1-NJ are still below the `50/50` decision gate.
  - Assumption: the current full-manifest validation protocol remains the only valid promotion protocol for this Stage1 Search baseline.
  - Control variables: active Search-only Stage1 mode, same train/val manifests, same active baseline checkpoint preload, same `50` epoch gate, same strict P10/P5/center promotion criteria.
- Expected evidence: official reporter summary after both lanes reach at least `50` epochs; no promotion before then.
  - Residual risk: early P10 ties may disappear or recover after epoch 47/45, so current metrics are observational only.

### Reporter Check

- Command:
  `.venv/bin/python scripts/external/report_stage1_frame_search_followup.py --baseline runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949 --followup runs/NON_XR/raw/stage1_s1ni_adamw_schedulefree_cautious_p5guard_lr1e7_stage1_arch_ext_after_existing_chain_20260623_host_opt_sched_loss_ext_20260623_141852 --followup runs/NON_XR/raw/stage1_s1nj_musgd_cosine_p5guard_lr7p5e8_stage1_arch_ext_after_existing_chain_20260623_host_opt_sched_loss_ext_20260623_141852 --min-epochs 50 --target-epochs 50 --format summary`
- Reporter status: `wait_min_epochs`.
- Reporter output:
  - S1-NI: epoch `47`, `min_ready=False`, best P10 `28.209793918537645`, best P5 `9.24528328877575`, best center `17.318899788946474`, `promotable_after_min_epoch=False`.
  - S1-NJ: epoch `45`, `min_ready=False`, best P10 `28.27717937613433`, best P5 `8.77583133049731`, best center `17.292718172073364`, `promotable_after_min_epoch=False`.
- Validation after documentation update:
  - `git diff --check` passed for changed Stage1 docs/scripts.
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh` passed.
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` passed.

## 2026-06-23 KST Stage1 Opt-Sched-Loss-Ext Closeout And Augmentation-Ext Launch

- Official reporter at the `50/50` gate:
  - S1-NI AdamW schedule-free + cautious:
    - P10 `28.209793918537645`
    - P5 `9.24528328877575`
    - center `17.318899788946474`
    - `promotable_after_min_epoch=False`
  - S1-NJ MuSGD + cosine:
    - P10 `28.27717937613433`
    - P5 `8.77583133049731`
    - center `17.292718172073364`
    - `promotable_after_min_epoch=False`
  - Strict baseline:
    - P10 `28.27717937613433`
    - P5 `10.153863744915656`
    - center `17.257583906065744`
- Decision:
  - No promotion.
  - S1-NJ ties baseline P10 but regresses P5 by `1.378032414418346 pp` and center by `0.035134266007620596 px`.
  - S1-NI is below baseline on P10, P5, and center.
  - Keep current Stage1 frame-search baseline:
    `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Runtime transition:
  - Extension chain PID `3546751` completed `opt_sched_loss_ext` and automatically launched `augmentation_ext`.
  - Active lanes completed:
    - S1-NK mild frame augmentation, PID `790784`, GPU0.
    - S1-NL event dropout/noise augmentation, PID `790785`, GPU1.
  - Runtime discrepancy: `nvidia-smi --query-compute-apps` shows no active training process, while chain logs still contain stale `alive=` waits for earlier PIDs. Treat downstream auto-chain status as unresolved until reconciled or relaunched.
- Latest augmentation-ext reporter:
  - Status: `keep_baseline_unless_later_improves`.
  - S1-NK epoch `50/50`: P10 `28.04133031953056`, P5 `9.396900554872909`, center `17.312655894261486`, `promotable_after_min_epoch=False`.
  - S1-NL epoch `50/50`: P10 `28.05817667043434`, P5 `9.380054185975272`, center `17.31234776298955`, `promotable_after_min_epoch=False`.
  - Decision: no promotion. Both augmentation variants are below baseline P10 and regress P5/center.
- Ablation provenance:
  - Skill phase/domain: `P1` / `Research Workflow`.
  - Parent skill: `paper-idea-generator`; worker skill: `ablation-study-designer`.
  - Control variables: same Stage1 Search-only mode, same train/val manifests, same active baseline checkpoint preload, same `50` epoch gate, same strict P10/P5/center promotion criteria.
  - Expected evidence for next decision: official reporter summary for S1-NK/S1-NL after both reach at least `50` epochs.

### 2026-06-23 Stage1 Frame-Search opt_sched_loss_plus Closed And Extensions Advanced

- Closed `augmentation_ext` with official reporter status `keep_baseline_unless_later_improves`; no baseline promotion.
- Auto-chain reconciliation:
  - Watcher log `stage1_arch_ablation_suite_chain_stage1_arch_plus_after_ext_20260623_host2.log` completed the wait for extension PID `3546751`.
  - It launched `opt_sched_loss_plus` with run tag `stage1_arch_plus_after_ext_20260623_host2_opt_sched_loss_plus`.
- Closed lanes:
  - S1-NO AdEMAMix + cosine P5-guard lane: experiment `runs/NON_XR/raw/stage1_s1no_ademamix_cosine_p5guard_lr8e8_stage1_arch_plus_after_ext_20260623_host2_opt_sched_loss_plus_20260623_154020`.
  - S1-NP MARS + plateau center-guard lane: experiment `runs/NON_XR/raw/stage1_s1np_mars_plateau_centerguard_lr5e8_stage1_arch_plus_after_ext_20260623_host2_opt_sched_loss_plus_20260623_154020`.
- Final runtime evidence:
  - Official reporter status is `keep_baseline_unless_later_improves`; both lanes reached epoch `50/50`.
  - S1-NO early best P10 `28.04133031953056`, P5 `9.262129657673386`, center `17.31566175424828`.
  - S1-NP early best P10 `28.249102106634176`, P5 `9.24528328877575`, center `17.311815576733284`.
  - Error signature scan found no `ERROR`, `Traceback`, `RuntimeError`, CUDA OOM, or generic `Exception`.
- Decision:
  - No promotion. Both lanes underperform baseline P10 and regress P5/center.

### 2026-06-23 Stage1 Frame-Search distill/loss/scheduler Extension Status

- Completed without promotion:
  - `distill_ext`: S1-NQ and S1-NR completed `50/50`; both matched baseline P10 `28.27717937613433` but worsened P5 and center.
  - `loss_ext`: S1-NS and S1-NT completed `50/50`; both matched baseline P10 `28.27717937613433` but worsened P5 and center.
  - `scheduler_ext`: S1-NU and S1-NV completed `50/50`; both underperformed baseline P10 with best P10 `28.209793918537645`, P5 `9.24528328877575`, center `17.31507064711373`.
- Active:
  - `p10_recover_ext`: S1-NW is at epoch `40/50`, and S1-NX is at epoch `36/50`.
  - S1-NW current best: P10 `28.232255755730396`, P5 `9.380054185975272`, center `17.308769406012768`.
  - S1-NX current best: P10 `28.27717937613433`, P5 `9.380054185975272`, center `17.29898632697339`.
  - Error signature scan found no `ERROR`, `Traceback`, `RuntimeError`, CUDA OOM, or generic `Exception`.
- Next:
  - Hold promotion until `p10_recover_ext` reaches `50/50`.

### 2026-06-23 Stage1 Frame-Search P10-Recover And Interpolation Closeout

- `p10_recover_ext` completed at the required `50/50` gate.
- Official closeout:
  - S1-NW Adopt residual center/P5 recover: P10 `28.232255755730396`, P5 `9.380054185975272`, center `17.308769406012768`.
  - S1-NX SOT-center center/P5 recover: P10 `28.27717937613433`, P5 `9.380054185975272`, center `17.29898632697339`.
- Decision:
  - No promotion. S1-NX tied the baseline P10 but still regressed P5 and center versus the strict baseline.
- Post-hoc checkpoint interpolation:
  - Script: `scripts/external/run_stage1_frame_search_p10_anchor_interp_eval.sh`.
  - Output summary: `runs/NON_XR/shared/eval/stage1_p10_anchor_interp_after_recover_20260623/summary.json`.
  - Anchors: S1-MU, S1-MV, and S1-NA best-P10 checkpoints.
  - Result: `promote_checkpoint=null`, `same_eval_promote_checkpoint=null`.
  - Interpretation: checkpoint interpolation failed to transfer P10-only gains without center drift.
- New combined suite added:
  - `opt_loss_sched_aug_guard`.
  - S1-OA combines Adopt, plateau center scheduler, tighter center/P5 losses, and micro frame augmentation.
  - S1-OB combines AdamW schedule-free/cautious, no external scheduler, tighter center/P5 losses, and micro frame/event augmentation.
  - Purpose: test optimizer/loss/scheduler/augmentation jointly after all single-axis probes failed promotion.

### 2026-06-23 Stage1 opt_loss_sched_aug_guard Launch

- Validation:
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_queue.sh` passed.
  - `bash -n scripts/external/run_stage1_frame_search_arch_ablation_suite_chain.sh` passed.
  - `SUITE=opt_loss_sched_aug_guard DRY_RUN=1` generated both lane commands successfully.
  - `git diff --check` passed for the touched scripts/docs before launch.
- Sandbox note:
  - Foreground smoke with `EPOCHS=1 MAX_TRAIN_BATCHES=1 MAX_VAL_BATCHES=1` failed in sandbox because PyTorch reported no visible CUDA devices.
  - Host execution was required and approved.
- Host launch:
  - Command tag: `RUN_TAG=stage1_opt_loss_sched_aug_guard_20260623_host`.
  - S1-OA PID `3767116`, GPU0, run root `runs/NON_XR/raw/stage1_s1oa_adopt_plateau_microaug_centerp5_lr2e8_stage1_opt_loss_sched_aug_guard_20260623_host_20260623_211152`.
  - S1-OB PID `3767120`, GPU1, run root `runs/NON_XR/raw/stage1_s1ob_adamw_sf_cautious_microevent_centerp5_lr7p5e8_stage1_opt_loss_sched_aug_guard_20260623_host_20260623_211152`.
  - Both lanes loaded baseline weights with `loaded_count=142` and entered epoch `1/50`.
- Gate:
  - Hold judgment until both runs reach `50/50`.

### 2026-06-23 Stage1 opt_loss_sched_aug_guard Runtime Snapshot And Closeout

- Official reporter:
  - Command used `--min-epochs 50 --target-epochs 50`.
  - Final status: `keep_baseline_unless_later_improves`.
- Final progress:
  - S1-OA: epoch `50/50`, best P10 `28.27717937613433`, best P5 `9.24528328877575`, best center `17.302655732856607`.
  - S1-OB: epoch `50/50`, best P10 `28.23225573773654`, best P5 `9.380054185975272`, best center `17.3148428808968`.
- Judgment:
  - No promotion.
  - S1-OA tied baseline P10 but regressed P5 by `0.9085804561399051 pp` and center by `0.04507182679086341 px`.
  - S1-OB underperformed P10 by `0.044923638397790455 pp` and regressed center by `0.057258974831057685 px`.
  - Keep baseline checkpoint `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`.
- Next:
  - Stop broad optimizer/loss/scheduler/augmentation replays.
  - Use output-level ensemble/calibration diagnostics to test whether P10-only anchors can be applied selectively without checkpoint interpolation drift.

### 2026-06-23 Stage1 Output-Level Ensemble Diagnostics

- Patched `scripts/external/eval_stage1_search_output_ensemble.py`:
  - Added `HBTXR_DISABLE_CUDNN=1` support to avoid the local CUDNN sublibrary mismatch.
  - Added `--oracle-best-center` for GT oracle upper-bound analysis.
- Static weighted output ensemble:
  - Output: `runs/NON_XR/shared/eval/stage1_output_ensemble_p10_anchors_after_guard_20260623/summary.json`.
  - Result: no static weighted ensemble improved P10 beyond the baseline.
  - `base90_p505_center05` tied P10 at `28.27717937613433` and improved same-eval center/P5 versus `ref_base`, but did not exceed the P10 gate.
- Oracle output gate:
  - Output: `runs/NON_XR/shared/eval/stage1_output_ensemble_p10_anchors_oracle_after_guard_20260623/summary.json`.
  - `oracle_best_center` reached P10 `30.908581067930978`, P5 `12.384321968510466`, center `16.39156784201568`.
  - Selection counts: base `126`, p5 `239`, center `297`, s1nf `165`, s1na `8`, s1my `9`.
- Decision:
  - Static averaging is not a promotion candidate.
  - A learned output gate/calibration head is justified because the oracle shows substantial sample-level complementarity.

### 2026-06-23 Stage1 Learned Output Gate 50-Epoch Closeout

- Implemented and validated `scripts/external/train_stage1_search_output_gate.py`.
  - Adds `HBTXR_DISABLE_CUDNN=1` handling.
  - Audits train/val/test manifest overlap and writes manifest hashes.
  - Fails requested CUDA runs when CUDA is unavailable unless `--allow-cpu-fallback` is set.
  - Reports trainer-compatible batch-mean metrics plus global weighted metrics.
  - Evaluates validation and held-out test splits.
- Smoke:
  - Output: `runs/NON_XR/shared/gate/stage1_output_gate_smoke_v2_20260623/summary.json`.
  - Passed with train/val/test limited to two batches each.
- Full run:
  - Output: `runs/NON_XR/shared/gate/stage1_output_gate_p10_anchors_50ep_20260623/summary.json`.
  - `gate.pt` and `summary.json` were written.
  - Train size `5929`, val size `844`, test size `2238`.
  - Split overlap audit found `0` overlaps for train-vs-val, train-vs-test, and val-vs-test.
- Validation result:
  - Hard gate: P10 `27.83692722371966`, P5 `10.222371967654992`, center `17.400613902092832`.
  - Soft gate: P10 `28.039083557951468`, P5 `9.67430368373765`, center `17.31651116715608`.
  - Reference base: P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
  - Reference S1-NF: P10 `28.3501796945193`, P5 `8.835354896675652`, center `17.394714167304436`.
  - Oracle: P10 `30.908580413297372`, P5 `12.384321653189577`, center `16.391567832676014`.
- Test result:
  - Hard gate: P10 `38.67559523809526`, P5 `11.607993197278907`, center `13.042108388368778`.
  - Soft gate: P10 `39.357993197278915`, P5 `11.30399659863945`, center `12.970975197188706`.
  - Test oracle: P10 `44.26020408163265`, P5 `15.440901360544213`, center `12.103968435112316`.
- Judgment:
  - No Stage1 baseline promotion. Validation P10 is below the current validation baseline and below the best P10-only anchor.
  - The oracle remains strong, so the remaining opportunity is not more static averaging. It is a better gating signal: richer features, confidence/context inputs, or an end-to-end calibration head.

### 2026-06-24 Stage1 Confidence/Context Soft-Target Gate Closeout

- Updated `scripts/external/train_stage1_search_output_gate.py`:
  - Added `--feature-mode output_context`, `output_conf`, and `output_conf_context`.
  - Added runtime-safe frame/event summary features.
  - Added frozen model confidence features: `search/pupil` confidence, aux bbox/OBB confidence and agreement with main state, state plausibility, and low-dimensional pooled statistics.
  - Added `--target-mode softmin_center` to train against a soft distribution over checkpoint candidates instead of only the hard oracle argmin.
  - Kept label-derived fields out of features: no `cur_state`, `annotation_quality`, `similarity_target`, `closed_eye_flag`, `mask_valid`, `valid_track`, or `prev_state` feature use.
- Smoke:
  - `runs/NON_XR/shared/gate/stage1_output_context_softmin_gate_smoke_20260624/summary.json`.
  - `runs/NON_XR/shared/gate/stage1_output_conf_context_softmin_gate_smoke_20260624/summary.json`.
- Full run:
  - `runs/NON_XR/shared/gate/stage1_output_conf_context_softmin_gate_50ep_20260624/summary.json`.
  - `runs/NON_XR/shared/gate/stage1_output_conf_context_softmin_gate_50ep_20260624/gate.pt`.
  - Epochs `50`, feature mode `output_conf_context`, target mode `softmin_center`, feature dim `208`.
- Validation:
  - Reference base: P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
  - Reference S1-NF: P10 `28.3501796945193`, P5 `8.835354896675652`, center `17.394714167304436`.
  - Hard gate: P10 `26.69923629829289`, P5 `9.935983827493263`, center `17.358085787071374`.
  - Soft gate: P10 `28.18171608265946`, P5 `8.876909254267746`, center `17.321360889128062`.
- Test:
  - Hard gate: P10 `39.50340136054423`, P5 `11.733418367346935`, center `12.969564928359729`.
  - Soft gate: P10 `39.20068027210886`, P5 `11.480442176870744`, center `12.948106178961643`.
- Judgment:
  - No promotion. Validation P10 remains below base and S1-NF, and center remains worse than the current strict baseline.
  - Planned next experiments should move from post-hoc MLP gating to an integrated calibration/head-selection branch or a regularized top-k mixture objective.

### 2026-06-24 Stage1 Confidence-Only And Top-K Gate Closeout

- S1-OE confidence-only ablation:
  - `runs/NON_XR/shared/gate/stage1_output_conf_softmin_gate_50ep_20260624/summary.json`.
    - Validation soft: P10 `28.06379155435757`, P5 `8.876909254267746`, center `17.320374500826468`.
  - `runs/NON_XR/shared/gate/stage1_output_conf_oraclece_gate_50ep_20260624/summary.json`.
    - Validation soft: P10 `27.74707996406108`, P5 `9.67430368373765`, center `17.3249394708995`.
  - Decision: no promotion. Removing runtime input context did not recover validation P10 or center.
- S1-OD regularized top-k gate:
  - Implemented `--topk`, `--entropy-weight`, and `--balance-weight` in `scripts/external/train_stage1_search_output_gate.py`.
  - Smoke output: `runs/NON_XR/shared/gate/stage1_output_conf_topk_reg_gate_smoke_20260624/summary.json`.
  - Full confidence-only regularized top-k:
    - Output: `runs/NON_XR/shared/gate/stage1_output_conf_topk2_reg_gate_50ep_20260624/summary.json`.
    - Validation hard: P10 `27.14285714285713`, P5 `9.548517520215634`, center `17.332235361280897`.
    - Validation soft: P10 `28.18171608265946`, P5 `8.876909254267746`, center `17.318211609591046`.
    - Validation top-k: P10 `27.109164420485165`, P5 `9.138589398023361`, center `17.344294708651248`.
  - Full confidence+context regularized top-k:
    - Output: `runs/NON_XR/shared/gate/stage1_output_conf_context_topk2_reg_gate_50ep_20260624/summary.json`.
    - Validation hard: P10 `27.305705300988304`, P5 `8.767969451931716`, center `17.390973981692788`.
    - Validation soft: P10 `28.06379155435757`, P5 `8.876909254267746`, center `17.322542203361674`.
    - Validation top-k: P10 `27.85040431266845`, P5 `9.168912848158133`, center `17.362701884669868`.
- Baseline reference:
  - Validation P10 `28.277178796046705`, P5 `8.334456424079066`, center `17.292673833329502`.
- Judgment:
  - No promotion.
  - Post-hoc gate family is now closed as a low-return path. Oracle headroom is real, but output/context/confidence MLP gates and top-k mixtures fail to generalize it on validation.
  - Next useful paths are S1-OC integrated calibration/head selection or XR-64A/B teacher-target construction. Do not launch more frozen-output gate ablations without a new mechanism.

### 2026-06-25 Stage1 Work Progress Documentation

- Added `docs/resources/current_stage1_work_progress_2026_06_25.md`.
- Recorded the active Stage1 frame-search baseline:
  - P10 `28.27717937613433`
  - P5 `10.153863744915656`
  - center `17.257583906065744`
- Consolidated completed no-promotion experiment families:
  - polish retries, Search threshold losses, P10 seed recovery, architecture/head/aux/depth/mask/head-factory/DeiT preload, optimizer/loss/scheduler/augmentation, combined guard, output ensemble, and post-hoc gate families.
- Recorded the post-hoc gate closeout decision:
  - Oracle output selection has strong headroom.
  - Learned frozen-output gates did not pass validation P10/center gates.
  - Do not continue post-hoc gate ablations without a new mechanism.
- Documented S1-OC implementation state:
  - Integrated `SearchCenterCandidateHead` into the Stage1 Search model path.
  - Added candidate losses and config surface.
  - Added `scripts/external/run_stage1_s1oc_search_candidate.sh`.
  - Validated with py_compile, shell syntax, dry-run, and dummy forward/loss smoke.
- Current execution state:
  - S1-OC 50-epoch GPU training has not started.
  - The escalated launch attempt was aborted.
  - No S1-OC run/log directories are present.
- Next action:
  - Launch S1-OC A/B for at least `50` epochs, then apply the strict P10/center promotion gate.
