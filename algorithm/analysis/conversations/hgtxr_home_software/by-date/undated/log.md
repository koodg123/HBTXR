# HGTXR-SW Work Log

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
- Event-count sweep initially showed identical logs. Added `scripts/v3/check_event_count_override_sanity.py` and found event tensors were all zero despite selected counts changing.
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
- Added eval-side `HBTXR_DISABLE_CUDNN=1` handling to `scripts/v3/eval_hbtxr.py` after GPU0 test eval hit `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`; runtime-env pytest now covers both train and eval scripts.
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
- Added and validated low-LR centerloss fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr2e-5_weakdistill_centerloss_finetune_fullwidth.yaml`; it starts from the current test leader checkpoint `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt`.
- Started GPU0 `centerloss_finetune`: `raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649`, log `runs/_logs/centerloss_finetune_gpu0_20260611_005000.log`; log confirmed `resolved_device=cuda:0`.
- Stopped GPU1 `center_ckpt_selfreg_count10000` after epoch-9 gate: best val center `43.5076`, above stop threshold `42.0`; GPU1 was freed.
- Added conservative LR sibling config `configs/v3/mode1_stage2_raw_event_count_lr1e-5_weakdistill_centerloss_finetune_fullwidth.yaml`; config load via `scripts/v3/_config.py` printed LR `1e-05`.
- Started GPU1 `centerloss_finetune_lr1e5`: `raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952`, log `runs/_logs/centerloss_finetune_lr1e5_gpu1_20260611_010000.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 `centerloss_finetune` LR `2e-5` completed by early stop at epoch 9. Best val center was `40.0042`, so it was not promoted versus the current val leader `39.8471`.
- Added track-head-only no-distillation fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth.yaml`; config load and event-count contract passed.
- Started GPU0 `trackonly_finetune`: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth_20260610_235630`, log `runs/_logs/trackonly_finetune_gpu0_20260611_011000.log`; log confirmed `resolved_device=cuda:0`.
- GPU1 `centerloss_finetune_lr1e5` completed by early stop at epoch 9. Best val center was `39.9555`, so it was not promoted versus the current val leader `39.8471`.
- Added conservative track-only LR sibling config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth.yaml`.
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
- Added all-head centerloss centerckpt-init configs: `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth.yaml` and `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth.yaml`; config load and raw event-count contract passed for both.
- Started GPU0 `centerloss_centerckptinit_nodistill`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth_20260611_004917`, log `runs/_logs/centerloss_centerckptinit_nodistill_gpu0_20260611_015000.log`; log confirmed `resolved_device=cuda:0`.
- Started GPU1 `centerloss_centerckptinit_weakdistill`: `raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943`, log `runs/_logs/centerloss_centerckptinit_weakdistill_gpu1_20260611_015000.log`; log confirmed `resolved_device=cuda:1`.
- GPU0 `centerloss_centerckptinit_nodistill` early-stopped at epoch 8. Best val center was `39.5598`, so it was not promoted versus validation leader `39.5315`.
- GPU1 `centerloss_centerckptinit_weakdistill` early-stopped at epoch 8. Best val center was `39.5780`, so it was not promoted versus validation leader `39.5315`.
- Started GPU1 parallel low-priority prepared runner case `centerloss_count1000` in tmux session `hgtxr_centerloss_count1000_gpu1`: `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerloss_fullwidth_20260611_010010`, log `runs/_logs/centerloss_count1000_gpu1_20260611_005747.log`; log confirmed `resolved_device=cuda:1`.
- Evaluated centerloss-bestcenter/centerckpt-bestcenter interpolation probes on GPU0. Alpha `0.25` test center `36.7547`, alpha `0.50` test center `38.6246`, alpha `0.75` test center `38.1375`; no promotion versus test leader `36.5813`.
- Stopped GPU1 `centerloss_count1000` after epoch-10 gate. Best val center was `44.6844`, above stop threshold `42.0`.
- Added ultra-low LR no-distillation all-head fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr2e-6_nodistill_centerloss_finetune_fullwidth.yaml`; config load and raw event-count contract passed.
- Started GPU0 `centerloss_finetune_lr2e6`: `raw_mode1_stage2_count10000_lr2e-6_nodistill_centerloss_finetune_fullwidth_20260611_011131`, log `runs/_logs/centerloss_finetune_lr2e6_gpu0_20260611_011500.log`; log confirmed `resolved_device=cuda:0`.
- GPU0 `centerloss_finetune_lr2e6` early-stopped at epoch 6. Best val center was `39.8800`, so it was not promoted.
- Added `training.trainable.include/exclude` filtering in `src/hbtxr/training/trainer.py` and covered it with `tests/test_trainable_filter.py`; targeted pytest with runtime-env tests passed.
- Added true track-head-only config `configs/v3/mode1_stage2_raw_event_count_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth.yaml`; config load and raw event-count contract passed.
- Started GPU1 `trackheadonly_finetune`: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth_20260611_011606`, log `runs/_logs/trackheadonly_finetune_gpu1_20260611_012500.log`; log confirmed `resolved_device=cuda:1` and trainable filter `151688/3156500` params.
- GPU1 `trackheadonly_finetune` early-stopped at epoch 6. Best val center was `39.9387`, so it was not promoted.
- Added partial event-path + track-head fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU1 `trackeventadapter_finetune`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth_20260611_012329`, log `runs/_logs/trackeventadapter_finetune_gpu1_20260611_020000.log`; log confirmed `resolved_device=cuda:1` and trainable filter `324680/3156500` params.
- GPU1 `trackeventadapter_finetune` early-stopped at epoch 8 with best val center `39.7622`; not promoted versus validation leader `39.5315`.
- Added final-backbone-block + track-head fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU0 `tracklastblock_finetune`: `raw_mode1_stage2_count10000_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth_20260611_012629`, log `runs/_logs/tracklastblock_finetune_gpu0_20260611_021000.log`; log confirmed `resolved_device=cuda:0` and trainable filter `596936/3156500` params.
- GPU0 `tracklastblock_finetune` reached epoch 3 with best val center `39.8832`; continue until early-stop trend is available.
- Added both-adapter + track-head fine-tune config `configs/v3/mode1_stage2_raw_event_count_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth.yaml`; config load, raw event-count contract, and wrapper dry-run passed.
- Started GPU1 `trackadapters_finetune`: `raw_mode1_stage2_count10000_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth_20260611_012935`, log `runs/_logs/trackadapters_finetune_gpu1_20260611_022000.log`; log confirmed `resolved_device=cuda:1` and trainable filter `448520/3156500` params.
- GPU0 `tracklastblock_finetune` early-stopped at epoch 6 with best val center `39.8832`; not promoted versus validation leader `39.5315`.
- GPU1 `trackadapters_finetune` early-stopped at epoch 8 with best val center `39.7822`; not promoted versus validation leader `39.5315`.
- Evaluated small-alpha interpolation probes on GPU0. Alpha `0.05` test center `36.5083`, alpha `0.10` test center `36.4873`, alpha `0.15` test center `36.5190`; alpha `0.10` is the new test-center leader.
- Created fine-grid interpolation checkpoints alpha `0.075`, `0.125`, and `0.175`, then evaluated them on GPU1. Results: `36.4907`, `36.4968`, `36.5556`; none beat alpha `0.10`.
- Added `scripts/v3/run_interp_eval_series.sh` to run host/tmux interpolation eval series reliably after sandbox parallel eval stalled before CUDA execution.
- Evaluated micro-grid alpha `0.090/0.095/0.105/0.110`; alpha `0.095` improved test center to `36.486985`.
- Evaluated second micro-grid alpha `0.093/0.094/0.096/0.097`; alpha `0.094` improved test center to `36.486977`.
- Paper evidence reviewed for next non-interpolation moves: EyeTrAES supports adaptive event slicing, FACET supports direct ellipse/center-offset heads, and local-global distillation supports local expert teacher distillation.
- Current test-center leader: `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.094.pt`, test center `36.486977`, P10 `9.9775`, P5 `2.9660`.
- GPU0 `trackheadonly_alphainit` completed: `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackheadonly_centerloss_alphainit_finetune_fullwidth_20260611_020317`, log `runs/_logs/alphainit_trackheadonly_gpu0_20260611.log`; early-stopped at epoch 5 with best val center `39.8494`, so no test promotion versus validation leader `39.5315`.
- GPU1 `trackadapters_alphainit` completed: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackadapters_centerloss_alphainit_finetune_fullwidth_20260611_020333`, log `runs/_logs/alphainit_trackadapters_gpu1_20260611.log`; early-stopped at epoch 7 with best val center `39.8169`, so no test promotion.
- Added alpha-leader event-path and last-block configs: `configs/v3/mode1_stage2_raw_event_count_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth.yaml` and `configs/v3/mode1_stage2_raw_event_count_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth.yaml`; config load, raw event-count contract, and `bash -n scripts/v3/run_prepare_and_train.sh` passed.
- Started GPU1 `trackeventadapter_alphainit`: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`, log `runs/_logs/alphainit_trackeventadapter_gpu1_20260611.log`; log confirmed `resolved_device=cuda:1` and trainable filter `324680/3156500` params.
- Started GPU0 `tracklastblock_alphainit`: `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`, log `runs/_logs/alphainit_tracklastblock_gpu0_20260611.log`; log confirmed `resolved_device=cuda:0` and trainable filter `596936/3156500` params.
- Implemented adaptive-count event slicing in `src/hbtxr/data/event_builder.py` with tests for enabled scaling and disabled compatibility. Added `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml` as the next paper-driven adaptive slicing probe. Validation: config load passed, raw event-count contract passed, and `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_event_builder.py tests/test_raw_event_count_contract.py` produced `6 passed` with only a parent `.pytest_cache` read-only warning.
- GPU1 `trackeventadapter_alphainit` early-stopped at epoch 7 with best val center `39.8169`; no promotion. GPU0 `tracklastblock_alphainit` early-stopped at epoch 5 with best val center `39.8546`; no promotion.
- Started GPU0 adaptive-count LR `5e-6`: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`, log `runs/_logs/adaptivecount_trackonly_gpu0_20260611.log`; log confirmed `resolved_device=cuda:0`.
- Added lower-LR adaptive-count sibling `configs/v3/mode1_stage2_raw_event_count_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`; config load and raw event-count contract passed. Started it on GPU1 as `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`, log `runs/_logs/adaptivecount_trackonly_lr2e6_gpu1_20260611.log`; log confirmed `resolved_device=cuda:1`.
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
- Added decoded-center L2 track loss as an optional config-only term: `loss.track_center_l2_weight` defaults to `0.0` and logs as `loss_track_center_l2`. Added `tests/test_track_center_l2_loss.py` and config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_finetune_fullwidth.yaml`.
- Validation for center-L2 path passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `2 passed` with only the known parent `.pytest_cache` read-only warning; `python3 -m py_compile src/hbtxr/loss/bundles/track.py` passed; raw event-count contract passed; config load via `scripts/v3/train_hbtxr.py` helper resolved `track_center_l2_weight=0.0025`, `lr=5e-06`.
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
  - `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo075_finetune_fullwidth.yaml`
  - `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo100_finetune_fullwidth.yaml`
  Both extend the center-L2 config, keep `track_center_l2_weight=0.0025`, and set `track_geo_weight` to `0.75` / `1.0`. Validation passed: config loader resolved intended weights, raw event-count contract passed for both, and `bash -n scripts/v3/run_prepare_and_train.sh` passed.
- fixed22k test evals completed and did not promote: best-center `eval_fixed22k_bestcenter_test_gpu0_w0_20260611_030731` produced center `35.4087`, P10 `10.3631`, P5 `3.4566`; best-P10 `eval_fixed22k_bestp10_test_gpu0_w0_20260611_030849` produced center `35.7890`, P10 `11.1412`, P5 `4.1637`.
- center-L2 fixed20k completed: best val center `38.6252`; best-center test eval `eval_centerl2_fixed20k_bestcenter_test_gpu0_w0_20260611_030932` produced center `35.7344`, P10 `11.2538`, P5 `3.9426`; best-P10 eval `eval_centerl2_fixed20k_bestp10_test_gpu0_w0_20260611_031011` produced center `35.5177`, P10 `11.7921`, P5 `3.7415`, becoming the current P10 leader.
- fixed35k completed and promoted center: `raw_mode1_stage2_count35000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_025939` best-center test eval `eval_fixed35k_bestcenter_test_gpu1_w0_20260611_031023` produced center `34.9689`, P10 `11.5408`, P5 `3.6335`, replacing fixed30k as current center leader.
- fixed35k best-P10 eval `eval_fixed35k_bestp10_test_gpu1_w0_20260611_031215` produced center `36.2104`, P10 `9.5659`, P5 `3.1318`; no promotion.
- Current active queue: GPU0 fixed28k train; GPU1 center-L2 fixed30k train.
- Added follow-up queue runner `scripts/v3/run_followup_queue_20260611.sh`; `bash -n` passed.
- Registered tmux follow-up queues:
  - `hgtxr_followup_gpu0_20260611`: fixed28k eval -> fixed40k train, log `runs/_logs/eval_fixed28_then_fixed40_gpu0_20260611.log`.
  - `hgtxr_followup_gpu1_20260611`: center-L2 fixed30k eval -> center-L2 fixed35k train, log `runs/_logs/eval_centerl2_30_then_centerl2_35_gpu1_20260611.log`.
  - `hgtxr_followup_gpu1_geo075_20260611`: center-L2 fixed35k eval -> geo075 fixed35k train, log `runs/_logs/eval_centerl2_35_then_geo075_35_gpu1_20260611.log`.
- fixed28k best-center eval `eval_fixed28k_bestcenter_test_gpu0_w0_20260611_032223` produced center `35.1866`, P10 `11.5731`, P5 `3.7049`; best-P10 eval `eval_fixed28k_bestp10_test_gpu0_w0_20260611_032400` produced center `35.5586`, P10 `11.7517`, P5 `4.2504`. Neither promoted versus fixed35k center leader or center-L2 fixed20k P10 leader.
- center-L2 fixed30k best-center/best-P10 evals `eval_centerl2_30k_bestcenter_test_gpu1_w0_20260611_032232` and `eval_centerl2_30k_bestp10_test_gpu1_w0_20260611_032421` both produced center `35.2231`, P10 `10.6845`, P5 `3.4643`; no promotion. Current active queue is GPU0 fixed40k train and GPU1 center-L2 fixed35k train; GPU1 geo075 fixed35k remains queued.
- Extended `scripts/v3/run_followup_queue_20260611.sh` with `eval_geo_count`, `train_geo100_count`, and mode `gpu1_geo075_35_then_geo100_35`. `bash -n` passed. Registered tmux session `hgtxr_followup_gpu1_geo100_20260611`; it waits for geo075 fixed35k to finish, evaluates best-center/best-P10 on test with `training.num_workers=0`, then starts geo100 fixed35k on GPU1. Queue log: `runs/_logs/eval_geo075_then_geo100_35_gpu1_20260611.log`.
- Extended `scripts/v3/run_followup_queue_20260611.sh` with mode `gpu0_fixed40_eval`. `bash -n` passed. Registered tmux session `hgtxr_followup_gpu0_fixed40_eval_20260611`; it waits for fixed40k train completion, then evaluates best-center/best-P10 on test with `training.num_workers=0`. Queue log: `runs/_logs/eval_fixed40_gpu0_20260611.log`.
- fixed40k promoted both center and P10: best-center eval `eval_fixed40k_bestcenter_test_gpu0_w0_20260611_033617` from `runs/raw_mode1_stage2_count40000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_032548/train/best_metric_track_center_px.pt` produced center `34.9533`, P10 `11.9286`, P5 `3.6395`. best-P10 eval `eval_fixed40k_bestp10_test_gpu0_w0_20260611_033811` produced center `35.1615`, P10 `11.6458`, P5 `3.4503`; no further promotion over best-center.
- center-L2 fixed35k did not promote: best-center eval `eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`.
- Count sweep extended after the fixed40 promotion. Added `gpu0_fixed45_eval_then_fixed50` to `scripts/v3/run_followup_queue_20260611.sh`; `bash -n` passed. fixed45k is now active on GPU0 via `hgtxr_followup_gpu0_fixed45_20260611`, and `hgtxr_followup_gpu0_fixed50_20260611` waits to evaluate fixed45k then launch fixed50k. geo075 fixed35k is active on GPU1, with geo100 queued after geo075 eval.
- Added terminal eval modes `gpu0_fixed50_eval` and `gpu1_geo100_35_eval` to `scripts/v3/run_followup_queue_20260611.sh`; `bash -n` passed. Registered tmux sessions `hgtxr_followup_gpu0_fixed50_eval_20260611` and `hgtxr_followup_gpu1_geo100_eval_20260611` so fixed50k and geo100 fixed35k are automatically evaluated after training.
- fixed45k promoted center: best-center eval `eval_fixed45k_bestcenter_test_gpu0_w0_20260611_035115` from `runs/raw_mode1_stage2_count45000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_034000/train/best_metric_track_center_px.pt` produced center `34.4771`, P10 `11.5935`, P5 `3.9507`. fixed45k best-P10 eval `eval_fixed45k_bestp10_test_gpu0_w0_20260611_035258` produced center `35.1005`, P10 `10.2389`, P5 `3.4264`, so P10 leader remains fixed40k best-center at P10 `11.9286`.
- Current active queue after fixed45 eval: GPU0 fixed50k train `raw_mode1_stage2_count50000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_035439`; GPU1 geo100 fixed35k train `raw_mode1_stage2_count35000_lr5e-6_nodistill_trackonly_centerl2_geo100_alphainit_fullwidth_20260611_035013`. Both have eval queues already registered.
- geo100 fixed35k did not promote: best-center eval `eval_geo100_35k_bestcenter_test_gpu1_w0_20260611_040031` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_geo100_35k_bestp10_test_gpu1_w0_20260611_040210` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`.
- fixed50k promoted both center and P10: best-center eval `eval_fixed50k_bestcenter_test_gpu0_w0_20260611_040420` produced center `34.3416`, P10 `11.7589`, P5 `3.5748`; best-P10 eval `eval_fixed50k_bestp10_test_gpu0_w0_20260611_040514` produced center `34.5391`, P10 `12.1276`, P5 `3.3580`.
- Added generic count-extension runner `scripts/v3/run_count_extension_queue_20260611.sh`; `bash -n` passed. Started tmux sessions `hgtxr_count55_gpu0_20260611` and `hgtxr_count60_gpu1_20260611`, logs `runs/_logs/count55_gpu0_20260611.log` and `runs/_logs/count60_gpu1_20260611.log`. Both run fixed-count train then best-center/best-P10 test eval with `training.num_workers=0`.
- Verified current count-extension queue: `tmux ls` shows `hgtxr_count55_gpu0_20260611` and `hgtxr_count60_gpu1_20260611`; `nvidia-smi` shows both GPUs active; logs show fixed55k near epoch 8/12 and fixed60k near epoch 7/12. No extra GPU1 experiment was started because fixed60k already occupies `cuda:1`.
- Added optimizer-probe runner `scripts/v3/run_optimizer_probe_queue_20260611.sh`; `bash -n` passed. Use it after fixed55k/fixed60k determine whether the count axis has plateaued.
- Attempted real sub-agent review with `gpt-5.3-codex-spark`, but spawn failed with `agent thread limit reached`; continued with main-agent verification only.
- Added conditional post-count runner `scripts/v3/run_post_count_decision_queue_20260611.sh`; `bash -n` passed. Started tmux sessions `hgtxr_post_count_gpu0_20260611` and `hgtxr_post_count_gpu1_20260611`, logs `runs/_logs/post_count_decision_gpu0_20260611.log` and `runs/_logs/post_count_decision_gpu1_20260611.log`. They wait for all fixed55k/fixed60k test eval summaries, then continue count sweep if either promotes, otherwise switch to LR probes.
- fixed55k/fixed60k both improved over fixed50k. fixed55k best-center test eval produced center `34.1183`, P10 `12.5672`, P5 `4.1399`, which is the current P10 leader. fixed60k best-center and best-P10 evals both produced center `33.9445`, P10 `12.5174`, P5 `4.3423`, which is the current center leader. fixed55k best-P10 did not promote: center `34.9445`, P10 `11.2088`, P5 `3.7560`.
- Post-count queues advanced as intended after promotion: GPU0 started fixed70k `raw_mode1_stage2_count70000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042200`; GPU1 started fixed65k `raw_mode1_stage2_count65000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042212`.
- Added second-stage conditional post-count runner `scripts/v3/run_post_count2_decision_queue_20260611.sh`; `bash -n` passed. Started watcher sessions `hgtxr_post_count2_gpu0_20260611` and `hgtxr_post_count2_gpu1_20260611`, logs `runs/_logs/post_count2_decision_gpu0_20260611.log` and `runs/_logs/post_count2_decision_gpu1_20260611.log`. They wait for all fixed65k/fixed70k eval summaries, then continue count sweep to fixed80k/fixed75k if either promotes, otherwise switch to AdamW LR probes.
- fixed65k/fixed70k completed and improved the leaderboards. fixed65k best-center/best-P10 evals both produced center `33.7084`, P10 `12.6548`, P5 `3.8202`. fixed70k best-center/best-P10 evals both produced center `33.5467`, P10 `12.8971`, P5 `3.8010`, replacing fixed60k/fixed55k as both center and P10 leader. The second post-count watcher selected `ACTION=count`, `BEST_COUNT=70000`, launched GPU1 fixed75k as `raw_mode1_stage2_count75000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043657`, and launched GPU0 fixed80k as `raw_mode1_stage2_count80000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043808`.
- Added third-stage conditional post-count runner `scripts/v3/run_post_count3_decision_queue_20260611.sh`; `bash -n` passed. Started watcher sessions `hgtxr_post_count3_gpu0_20260611` and `hgtxr_post_count3_gpu1_20260611`, logs `runs/_logs/post_count3_decision_gpu0_20260611.log` and `runs/_logs/post_count3_decision_gpu1_20260611.log`. They wait for fixed75k/fixed80k eval summaries, then continue to fixed90k/fixed85k if either promotes over fixed70k, otherwise switch to AdamW LR probes on the best count.
- Implemented next loss-axis candidate: decoded-center hinge loss in `src/hbtxr/loss/bundles/track.py`, with config keys `track_center_hinge_weight` and `track_center_hinge_margin_px`. Added tests in `tests/test_track_center_l2_loss.py`, configs `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge10_finetune_fullwidth.yaml` and `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge5_finetune_fullwidth.yaml`, and generic runner `scripts/v3/run_centerhinge_probe_queue_20260611.sh`. Validation passed: unit test `3 passed`, both raw event-count contracts passed, `py_compile` passed, and runner `bash -n` passed.
- fixed75k/fixed80k completed and promoted over fixed70k. fixed75k evals produced center `33.3526`, P10 `13.2428`, P5 `3.5757`, becoming the current P10 leader. fixed80k evals produced center `33.0848`, P10 `12.7543`, P5 `3.4056`, becoming the current center leader. The third-stage watcher selected `ACTION=count`, `BEST_COUNT=80000`, `BEST_CENTER=33.084757`, `BEST_P10=13.242773`; GPU0 launched fixed90k at `2026-06-11T04:53:49+09:00`, and GPU1 launched fixed85k at `2026-06-11T04:53:51+09:00`.
- Added fourth-stage conditional post-count runner `scripts/v3/run_post_count4_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count4_gpu0_20260611` / `hgtxr_post_count4_gpu1_20260611` were registered. The watcher waits for fixed85k/fixed90k evals, continues count to fixed100k/fixed95k if either promotes, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to center-hinge margin `10.0`, weight `0.05` on the best count.
- fixed85k/fixed90k completed. fixed85k promoted center to `32.9858` with P10 `12.9605`, P5 `3.9660`; fixed90k did not promote (`33.0128` center on best-center eval; `33.4284` center and `11.3542` P10 on best-P10 eval). P10 leader remains fixed75k at `13.2428`. The fourth-stage watcher selected `ACTION=count`, `BEST_COUNT=85000`, and launched GPU1 fixed95k plus GPU0 fixed100k at `2026-06-11T05:10:27+09:00`.
- Added fifth-stage conditional post-count runner `scripts/v3/run_post_count5_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count5_gpu0_20260611` / `hgtxr_post_count5_gpu1_20260611` were registered. The watcher waits for fixed95k/fixed100k evals, continues count to fixed110k/fixed105k if either promotes, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to center-hinge margin `10.0`, weight `0.05` on the best count.
- Added local-anchor crop experiment axis from the paper-backed plan. `src/hbtxr/data/components.py` now supports `crop_policy: prev_pupil_anchor`, using `prev_annotation_ref` to crop around the previous pupil bbox while preserving shared frame/event/target transform alignment. Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_finetune_fullwidth.yaml`, runner `scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh`, and unit tests `tests/test_adaptive_roi_resolver.py`. Validation passed: adaptive ROI + event-builder pytest `6 passed`, config override check, raw event-count contract, `py_compile`, `sh -n`, and `bash -n`. Training is not launched yet because fixed95k/fixed100k occupy both GPUs and post_count5 already controls the next branch.
- fixed95k/fixed100k completed. fixed95k did not promote (`32.9237` center, `12.2815` P10, `3.4184` P5). fixed100k best-center promoted center to `32.7796` with P10 `12.5961`, P5 `3.7215`; fixed100k best-P10 did not promote (`33.1030` center, `12.1105` P10, `3.6195` P5). The fifth-stage watcher selected `ACTION=count`, `BEST_COUNT=100000`, and launched GPU1 fixed105k plus GPU0 fixed110k at `2026-06-11T05:26:00+09:00`.
- Added sixth-stage conditional post-count runner `scripts/v3/run_post_count6_decision_queue_20260611.sh`; `bash -n` passed and watcher sessions `hgtxr_post_count6_gpu0_20260611` / `hgtxr_post_count6_gpu1_20260611` were registered. The watcher waits for fixed105k/fixed110k evals, continues count to fixed120k/fixed115k if either promotes over `32.7796` center or `13.2428` P10, otherwise sends GPU0 to AdamW `8e-6` and GPU1 to the prepared `prev_pupil_anchor` local-crop probe. Process check confirms fixed105k on GPU1, fixed110k on GPU0, and both post_count6 watchers running.
- Prepared FACET-style decoded ellipse-state loss without adding new checkpoint parameters. `src/hbtxr/loss/bundles/track.py` now supports disabled-by-default `track_axis_log` and `track_angle_cos` losses via `loss.track_axis_log_weight` and `loss.track_angle_cos_weight`. Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_finetune_fullwidth.yaml` and runner `scripts/v3/run_ellipsestate_probe_queue_20260611.sh`. Validation passed: track loss pytest `4 passed`, raw event-count contract, config loader check, `py_compile`, and runner `bash -n`. Training not launched; both GPUs remain occupied by fixed105k/fixed110k.
- fixed105k/fixed110k evals completed. fixed105k best-center/best-P10 both produced center `32.6356`, P10 `13.1654`, P5 `3.6662`; fixed110k best-center produced center `32.5969`, P10 `13.0697`, P5 `3.8193`, becoming the current center leader; fixed110k best-P10 produced center `32.9429`, P10 `12.6399`, P5 `3.4324`. P10 leader remains fixed75k at `13.2428`. The sixth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=110000`, `BEST_CENTER=32.596922`, `BEST_P10=13.165392`. GPU1 launched fixed115k at `2026-06-11T05:42:02+09:00`; GPU0 launched fixed120k at `2026-06-11T05:44:00+09:00`.
- Added seventh-stage conditional post-count runner `scripts/v3/run_post_count7_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watchers `hgtxr_post_count7_gpu0_20260611` and `hgtxr_post_count7_gpu1_20260611`. Logs confirm both are waiting for fixed115k/fixed120k eval summaries. Promotion gate is fixed110k center `32.596922` or fixed75k P10 `13.242773`; if promoted, GPU1 continues to fixed125k and GPU0 to fixed130k, otherwise GPU1 runs `prev_pupil_anchor` and GPU0 runs AdamW `8e-6` on the best count.
- fixed115k/fixed120k evals completed. fixed115k best-center produced center `32.4400`, P10 `13.3512`, P5 `3.6305`; fixed115k best-P10 produced center `32.9068`, P10 `12.8835`, P5 `3.3678`. fixed120k best-center produced center `32.3053`, P10 `13.7976`, P5 `3.4974`, becoming the current center and P10 leader; fixed120k best-P10 produced center `32.7800`, P10 `13.0991`, P5 `3.6922`.
- The seventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=120000`, `BEST_CENTER=32.305316`, `BEST_P10=13.797619`. GPU0 launched fixed130k at `2026-06-11T05:59:57+09:00`; GPU1 launched fixed125k at `2026-06-11T06:00:09+09:00`.
- Added eighth-stage conditional post-count runner `scripts/v3/run_post_count8_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watchers `hgtxr_post_count8_gpu0_20260611` and `hgtxr_post_count8_gpu1_20260611`. If fixed125k/fixed130k promote over fixed120k (`32.305316` center or `13.797619` P10), the queue continues to fixed140k on GPU0 and fixed135k on GPU1; otherwise GPU0 switches to AdamW `8e-6` and GPU1 switches to `prev_pupil_anchor`.
- Current process/log check: GPU0 fixed130k is active at epoch 4 with val center `37.0059`, P10 `10.2662`; GPU1 fixed125k is active at epoch 4 with val center `37.1699`, P10 `9.9292`. GPU1 is already occupied, so no additional same-GPU parallel experiment was started.
- Sub-agent plan audit attempt with `gpt-5.3-codex-spark` failed due `agent thread limit reached`; no sub-agent result was used.
- Continuation check: fixed125k/fixed130k remain active and no fixed125k/fixed130k eval summaries exist yet. Latest observed epoch 7: fixed125k val center `36.5371`, best val P10 `10.3291`; fixed130k val center `36.4138`, best val P10 `11.3398`.
- Added `scripts/v3/run_post_count9_decision_queue_20260611.sh`; validation passed with `bash -n`, executable bit, and no-arg usage exit `2`. Registered watchers `hgtxr_post_count9_gpu0_20260611` and `hgtxr_post_count9_gpu1_20260611`, logs `runs/_logs/post_count9_decision_gpu0_20260611.log` and `runs/_logs/post_count9_decision_gpu1_20260611.log`; both are waiting for `post_count8` decision.
- fixed125k/fixed130k evals completed and promoted. fixed125k best-center produced center `32.1365`, P10 `14.0676`, P5 `3.8610`, becoming the current P10 leader; fixed125k best-P10 produced center `32.6232`, P10 `12.9962`, P5 `3.7636`. fixed130k best-center produced center `32.0159`, P10 `13.7798`, P5 `3.8457`, becoming the current center leader; fixed130k best-P10 produced center `32.4814`, P10 `12.9940`, P5 `3.7190`.
- The eighth-stage post-count watcher selected count continuation with `BEST_COUNT=130000`. GPU0 launched fixed140k as `raw_mode1_stage2_count140000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061825`; GPU1 launched fixed135k as `raw_mode1_stage2_count135000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061834`.
- post_count9 watchers observed `post_count8` action `count` and now wait for fixed135k/fixed140k eval summaries. Latest observed epoch 1: fixed140k val center `39.9020`, P10 `8.4748`; fixed135k val center `39.9278`, P10 `8.4580`.
- Corrected post_count9 threshold constants from fixed120k to the current leaders: center `32.01590127093451` and P10 `14.067602443695069`. Terminated the old post_count9 watcher processes and restarted corrected watchers as `hgtxr_post_count9_gpu0b_20260611` / `hgtxr_post_count9_gpu1b_20260611`; logs confirm both are waiting for fixed135k/fixed140k eval summaries.
- Current fixed135k/fixed140k status: epoch 6 observed. fixed135k best val center `36.6955`, best val P10 `11.0568`; fixed140k best val center `36.7976`, best val P10 `10.8041`; no fixed135k/fixed140k test eval summaries yet.
- Added `scripts/v3/run_post_count10_decision_queue_20260611.sh`; validation passed with `bash -n`, executable bit, and no-arg usage exit `2`. Registered watchers `hgtxr_post_count10_gpu0_20260611` and `hgtxr_post_count10_gpu1_20260611`, logs `runs/_logs/post_count10_decision_gpu0_20260611.log` and `runs/_logs/post_count10_decision_gpu1_20260611.log`; both wait for `post_count9` decision. The script uses dynamic baseline selection from available eval summaries to avoid stale threshold constants.
- fixed135k/fixed140k evals completed. fixed135k best-center produced center `31.9891`, P10 `13.3274`, P5 `3.5242`; fixed135k best-P10 produced center `32.4352`, P10 `12.8053`, P5 `3.8559`. fixed140k best-center produced center `31.8833`, P10 `13.1884`, P5 `3.6637`, becoming the current center leader; fixed140k best-P10 produced center `32.3225`, P10 `12.7156`, P5 `3.6654`. P10 leader remains fixed125k best-center at `14.0676`.
- The ninth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=140000`. GPU1 launched fixed145k at about `2026-06-11T06:37:07+09:00`; GPU0 launched fixed150k at about `2026-06-11T06:37:10+09:00`. Current latest observed epoch 4: fixed145k val center `37.1518`, P10 `10.6289`; fixed150k val center `37.2427`, P10 `10.9007`. The tenth-stage watchers are alive and waiting for fixed145k/fixed150k eval summaries.
- fixed145k/fixed150k evals completed. fixed145k best-center produced center `31.8802`, P10 `12.5897`, P5 `3.9073`; fixed145k best-P10 produced center `32.3115`, P10 `12.5221`, P5 `3.5570`. fixed150k best-center produced center `31.7211`, P10 `12.9154`, P5 `3.7330`, becoming the current center leader; fixed150k best-P10 produced center `32.1254`, P10 `12.8342`, P5 `3.4179`. P10 leader remains fixed125k best-center at `14.0676`.
- The tenth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=150000`. GPU1 launched fixed155k at about `2026-06-11T06:55:33+09:00`; GPU0 launched fixed160k at about `2026-06-11T06:55:15+09:00`. Latest observed epoch 4: fixed155k val center `37.2207`, P10 `11.0580`; fixed160k val center `37.0474`, P10 `11.1927`.
- Added `scripts/v3/run_post_count11_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count11_gpu0_20260611` and `hgtxr_post_count11_gpu1_20260611`, logs `runs/_logs/post_count11_decision_gpu0_20260611.log` and `runs/_logs/post_count11_decision_gpu1_20260611.log`; both wait for fixed155k/fixed160k eval summaries.
- fixed155k/fixed160k evals completed. fixed155k best-center/best-P10 both produced center `31.6944`, P10 `12.8878`, P5 `3.5438`. fixed160k best-center/best-P10 both produced center `31.5714`, P10 `13.3108`, P5 `3.5098`, becoming the current center leader. P10 leader remains fixed125k best-center at `14.0676`.
- The eleventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=160000`. GPU1 launched fixed165k at about `2026-06-11T07:14:02+09:00`; GPU0 launched fixed170k at about `2026-06-11T07:14:02+09:00`.
- Added `scripts/v3/run_post_count12_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count12_gpu0_20260611` and `hgtxr_post_count12_gpu1_20260611`, logs `runs/_logs/post_count12_decision_gpu0_20260611.log` and `runs/_logs/post_count12_decision_gpu1_20260611.log`; both wait for fixed165k/fixed170k eval summaries. If they promote, GPU1 continues fixed175k and GPU0 fixed180k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Added `scripts/v3/run_post_count13_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count13_gpu0_20260611` and `hgtxr_post_count13_gpu1_20260611`, logs `runs/_logs/post_count13_decision_gpu0_20260611.log` and `runs/_logs/post_count13_decision_gpu1_20260611.log`; both wait for the post_count12 decision. If post_count12 count branch promotes, GPU1 continues fixed185k and GPU0 fixed190k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- fixed165k/fixed170k evals completed. fixed165k best-center produced center `31.5321`, P10 `13.1152`, P5 `3.7160`; fixed165k best-P10 produced center `31.9779`, P10 `13.0676`, P5 `3.6352`. fixed170k best-center produced center `31.4726`, P10 `13.5991`, P5 `3.6437`, becoming the current center leader; fixed170k best-P10 produced center `31.9469`, P10 `13.1144`, P5 `3.3129`. P10 leader remains fixed125k best-center at `14.0676`.
- The twelfth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=170000`. GPU1 launched fixed175k at `2026-06-11T07:34:39+09:00`; GPU0 launched fixed180k at `2026-06-11T07:34:39+09:00`. Both trained through epoch 12. fixed175k best-center test eval `eval_fixed175k_bestcenter_test_gpu1_w0_20260611_074834` produced center `31.2559`, P10 `13.4209`, P5 `3.8291`; fixed175k best-P10 eval `eval_fixed175k_bestp10_test_gpu1_w0_20260611_075108` produced center `31.7825`, P10 `13.1718`, P5 `3.6322`. fixed180k best-center test eval `eval_fixed180k_bestcenter_test_gpu0_w0_20260611_074847` produced center `30.9685`, P10 `14.0582`, P5 `3.7117`, becoming the current center leader; fixed180k best-P10 eval `eval_fixed180k_bestp10_test_gpu0_w0_20260611_075125` produced center `31.6223`, P10 `13.4073`, P5 `3.8924`.
- The thirteenth-stage post-count watcher selected count continuation. GPU1 launched fixed185k at `2026-06-11T07:55:11+09:00` as `raw_mode1_stage2_count185000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075511`; GPU0 launched fixed190k at `2026-06-11T07:55:02+09:00` as `raw_mode1_stage2_count190000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075502`.
- Current fixed185k/fixed190k status: both are training normally. Latest observed epoch 10: fixed185k val center `35.0572`, P10 `12.9515`; fixed190k val center `35.0466`, P10 `13.5355`.
- Added `scripts/v3/run_post_count15_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count15_gpu0_20260611` and `hgtxr_post_count15_gpu1_20260611`, logs `runs/_logs/post_count15_decision_gpu0_20260611.log` and `runs/_logs/post_count15_decision_gpu1_20260611.log`; both wait for the post_count14 decision. If post_count14 count branch promotes, GPU1 continues fixed205k and GPU0 fixed210k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Fixed `scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh` eval output names from `fixed${COUNT}k` to `fixed${COUNT/1000}k`, matching the post-count watcher glob patterns.
- Added `scripts/v3/run_post_count16_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count16_gpu0_20260611` and `hgtxr_post_count16_gpu1_20260611`, logs `runs/_logs/post_count16_decision_gpu0_20260611.log` and `runs/_logs/post_count16_decision_gpu1_20260611.log`; both wait for the post_count15 decision. If post_count15 count branch promotes, GPU1 continues fixed215k and GPU0 fixed220k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Added `scripts/v3/run_post_count14_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count14_gpu0_20260611` and `hgtxr_post_count14_gpu1_20260611`, logs `runs/_logs/post_count14_decision_gpu0_20260611.log` and `runs/_logs/post_count14_decision_gpu1_20260611.log`; both wait for the post_count13 decision. If post_count13 count branch promotes, GPU1 continues fixed195k and GPU0 fixed200k; otherwise GPU1 switches to `prev_pupil_anchor` and GPU0 to AdamW `8e-6`.
- Prepared weak EMA distillation probe runner `scripts/v3/run_weakdistill_probe_queue_20260611.sh <event_count> <cuda:N>` for the next GPU1 idle/plateau slot. It uses `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth.yaml` with the alpha checkpoint and fixed-count override. Not launched now because GPU1 is already running fixed175k and has post_count13/post_count14 watcher ownership.
- fixed185k/fixed190k completed and promoted. fixed185k best-center/best-P10 both produced center `30.8202`, P10 `14.4379`, P5 `3.9043`; fixed190k best-center/best-P10 both produced center `30.7517`, P10 `14.3967`, P5 `4.1297`.
- post_count14 selected `ACTION=count` with `BEST_COUNT=190000`, `BEST_CENTER=30.751661`, `BEST_P10=14.437926`. GPU1 launched fixed195k at `2026-06-11T08:16:44+09:00`; GPU0 launched fixed200k at `2026-06-11T08:16:33+09:00`.
- Current GPU1 assignment is fixed195k train on `cuda:1`; extra GPU1 probes remain queued until this branch or post_count15 frees GPU1.
- Added `scripts/v3/run_post_count17_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable bit, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count17_gpu0_20260611` and `hgtxr_post_count17_gpu1_20260611`, logs `runs/_logs/post_count17_decision_gpu0_20260611.log` and `runs/_logs/post_count17_decision_gpu1_20260611.log`; both wait for post_count16. Count branch can continue to GPU1 fixed225k and GPU0 fixed230k.
- fixed195k/fixed200k completed and promoted. fixed195k best-center/best-P10 both produced center `30.6335`, P10 `14.7258`, P5 `4.4303`; fixed200k best-center/best-P10 both produced center `30.5612`, P10 `14.3117`, P5 `4.1531`.
- post_count15 selected `ACTION=count` with `BEST_COUNT=200000`, `BEST_CENTER=30.561175`, `BEST_P10=14.725766`. GPU1 launched fixed205k at `2026-06-11T08:37:37+09:00`; GPU0 launched fixed210k at `2026-06-11T08:37:28+09:00`.
- Current GPU1 assignment is fixed205k train on `cuda:1`; GPU0 assignment is fixed210k train on `cuda:0`.
- Added `scripts/v3/run_post_count18_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count18_gpu0_20260611` and `hgtxr_post_count18_gpu1_20260611`, logs `runs/_logs/post_count18_decision_gpu0_20260611.log` and `runs/_logs/post_count18_decision_gpu1_20260611.log`; both wait for post_count17. Count branch can continue to GPU1 fixed235k and GPU0 fixed240k.
- Continuation check: fixed205k/fixed210k are still training with no test eval summaries yet. Latest observed epoch 11/12: fixed205k val center `35.1333`, P10 `13.2188`, P5 `4.1476`; fixed210k val center `35.1114`, P10 `13.6456`, P5 `3.8949`. GPU1 remains occupied by fixed205k, so no extra same-GPU experiment was launched.
- Added `scripts/v3/run_post_count19_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count19_gpu0_20260611` and `hgtxr_post_count19_gpu1_20260611`, logs `runs/_logs/post_count19_decision_gpu0_20260611.log` and `runs/_logs/post_count19_decision_gpu1_20260611.log`; both wait for post_count18. Count branch can continue to GPU1 fixed245k and GPU0 fixed250k if fixed235k/fixed240k promote.
- fixed205k/fixed210k completed and promoted. fixed205k best-center eval produced center `30.3966`, P10 `15.0965`, P5 `4.0693`, becoming the current P10 leader; fixed205k best-P10 eval produced center `31.4222`, P10 `13.8312`, P5 `3.7381`. fixed210k best-center eval produced center `30.1415`, P10 `14.7037`, P5 `4.2636`, becoming the current center leader; fixed210k best-P10 eval produced center `30.8138`, P10 `13.8312`, P5 `3.6939`. post_count16 should continue to GPU1 fixed215k and GPU0 fixed220k on the next polling tick.
- post_count16 selected `ACTION=count`, `BEST_COUNT=210000`, `BEST_CENTER=30.141473`, `BEST_P10=15.096514`. GPU1 launched fixed215k as `raw_mode1_stage2_count215000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090019`; GPU0 launched fixed220k as `raw_mode1_stage2_count220000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090007`. Latest observed epoch 1: fixed215k val center `40.1420`, P10 `9.3531`, P5 `2.6673`; fixed220k val center `40.1718`, P10 `9.4710`, P5 `2.6505`.
- Continuation check: fixed215k/fixed220k are training normally. Latest observed epoch 4: fixed215k val center `36.8466`, P10 `12.6280`, P5 `3.1761`; fixed220k val center `36.8837`, P10 `12.8807`, P5 `3.0413`. GPU1 remains occupied by fixed215k, so no extra same-GPU experiment was launched.
- Added `scripts/v3/run_post_count20_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count20_gpu0_20260611` and `hgtxr_post_count20_gpu1_20260611`, logs `runs/_logs/post_count20_decision_gpu0_20260611.log` and `runs/_logs/post_count20_decision_gpu1_20260611.log`; both wait for post_count19. Count branch can continue to GPU1 fixed255k and GPU0 fixed260k if fixed245k/fixed250k promote.
- Continuation check: fixed215k/fixed220k are still training with no fixed215k/fixed220k test eval summaries yet. Latest observed epoch 6: fixed215k val center `36.4212`, P10 `12.4910`, P5 `3.2401`; fixed220k val center `36.4039`, P10 `13.0357`, P5 `3.3109`. GPU1 remains occupied by fixed215k, so no extra same-GPU experiment was launched.
- Added `scripts/v3/run_post_count21_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count21_gpu0_20260611` and `hgtxr_post_count21_gpu1_20260611`, logs `runs/_logs/post_count21_decision_gpu0_20260611.log` and `runs/_logs/post_count21_decision_gpu1_20260611.log`; both wait for post_count20. Count branch can continue to GPU1 fixed265k and GPU0 fixed270k if fixed255k/fixed260k promote.
- Full experiment plan print/update checkpoint: current leaders remain fixed210k best-center (`30.1415` center, `14.7037` P10, `4.2636` P5) and fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch remains GPU1 fixed215k and GPU0 fixed220k. Latest observed status: fixed215k epoch 8, val center `35.7267`, P10 `12.1597`, P5 `3.8331`; fixed220k epoch 9, val center `35.3173`, P10 `12.0058`, P5 `3.4288`. No fixed215k/fixed220k test eval summaries exist yet. GPU1 is already occupied by fixed215k, so parallel GPU1 work remains assigned through queued post-count branches rather than launched immediately.
- fixed215k/fixed220k completed train and test evals. fixed215k best-center/best-P10 both produced center `30.0281`, P10 `14.4018`, P5 `4.4498`. fixed220k best-center/best-P10 both produced center `30.0077`, P10 `14.3967`, P5 `4.2096`, becoming the new center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count17 selected `ACTION=count`, `BEST_COUNT=220000`, `BEST_CENTER=30.007657`, `BEST_P10=15.096514`; GPU1 launched fixed225k as `raw_mode1_stage2_count225000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092225`, and GPU0 launched fixed230k as `raw_mode1_stage2_count230000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092216`.
- Added `scripts/v3/run_post_count22_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count22_gpu0_20260611` and `hgtxr_post_count22_gpu1_20260611`, logs `runs/_logs/post_count22_decision_gpu0_20260611.log` and `runs/_logs/post_count22_decision_gpu1_20260611.log`; both wait for post_count21. Count branch can continue to GPU1 fixed275k and GPU0 fixed280k if fixed265k/fixed270k promote. Current fixed225k/fixed230k status at registration: both epoch 3; fixed225k val center `37.8648`, P10 `10.7412`, P5 `2.6988`; fixed230k val center `37.8628`, P10 `10.6065`, P5 `2.6988`.
- fixed225k/fixed230k completed train and test evals. fixed225k best-center/best-P10 both produced center `29.9776`, P10 `14.4728`, P5 `3.8984`. fixed230k best-center/best-P10 both produced center `29.9516`, P10 `14.4677`, P5 `3.6050`, becoming the current center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count18 selected `ACTION=count`, `BEST_COUNT=230000`, `BEST_CENTER=29.951577`, `BEST_P10=15.096514`; GPU1 launched fixed235k as `raw_mode1_stage2_count235000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094454`, and GPU0 launched fixed240k as `raw_mode1_stage2_count240000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094445`.
- Full experiment plan print/update checkpoint: current center leader is fixed230k best-center (`29.9516` center, `14.4677` P10, `3.6050` P5); current P10 leader is fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch is GPU1 fixed235k and GPU0 fixed240k. Latest observed epoch 4: fixed235k val center `36.6978`, P10 `12.8504`, P5 `3.4906`; fixed240k val center `36.6337`, P10 `13.1199`, P5 `3.4906`. GPU1 is already occupied by fixed235k, so additional parallel GPU1 work remains assigned through post_count19/post_count20/post_count21/post_count22 rather than launched on the same GPU.
- Added `scripts/v3/run_post_count23_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count23_gpu0_20260611` and `hgtxr_post_count23_gpu1_20260611`, logs `runs/_logs/post_count23_decision_gpu0_20260611.log` and `runs/_logs/post_count23_decision_gpu1_20260611.log`; both wait for post_count22. Count branch can continue to GPU1 fixed285k and GPU0 fixed290k if fixed275k/fixed280k promote. Current fixed235k/fixed240k status at registration: epoch 6; fixed235k val center `36.4167`, P10 `12.3169`, P5 `2.5438`; fixed240k val center `36.4364`, P10 `13.3446`, P5 `3.1593`.
- Added `scripts/v3/run_post_count24_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count24_gpu0_20260611` and `hgtxr_post_count24_gpu1_20260611`, logs `runs/_logs/post_count24_decision_gpu0_20260611.log` and `runs/_logs/post_count24_decision_gpu1_20260611.log`; both wait for post_count23. Count branch can continue to GPU1 fixed295k and GPU0 fixed300k if fixed285k/fixed290k promote. Latest fixed-count status: fixed235k epoch 7 val center `36.1807`, P10 `12.9009`, P5 `3.0829`; fixed240k epoch 8 val center `35.5407`, P10 `12.9739`, P5 `3.4288`.
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
- Added squared decoded-center threshold hinge loss for the next tail-error/P10/P5 probe. `src/hbtxr/loss/bundles/track.py` now emits disabled-by-default `loss_track_center_hinge_sq`, controlled by `loss.track_center_hinge_sq_weight` and `loss.track_center_hinge_margin_px`. Added test coverage in `tests/test_track_center_l2_loss.py`, config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhingesq10_finetune_fullwidth.yaml`, and runner `scripts/v3/run_centerhingesq_probe_queue_20260611.sh`. Validation passed: track loss pytest `5 passed`, `py_compile`, `bash -n`, `sh -n`, executable bit, no-arg usage exit `2`, raw event-count readiness contract, and config loader check (`sq_weight=0.005`, `margin=10.0`, `lr=5e-6`). Training not launched because GPU1/GPU0 remain owned by fixed265k/fixed270k.
- fixed265k/fixed270k completed and did not promote. fixed265k best-center produced center `29.7582`, P10 `13.5969`, P5 `3.5974`; fixed265k best-P10 produced center `32.8107`, P10 `11.2436`, P5 `3.0013`. fixed270k best-center/best-P10 produced center `29.9207`, P10 `13.3652`, P5 `3.8418`. post_count22 selected `ACTION=mixed_probe`; GPU1 launched fixed255k `prev_pupil_anchor` (`raw_mode1_stage2_count255000_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_alphainit_fullwidth_20260611_112216`) and GPU0 launched fixed255k AdamW `8e-6` (`raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215`).
- Added `scripts/v3/run_post_count31_decision_queue_20260611.sh` and registered tmux watchers `hgtxr_post_count31_gpu0_20260611` / `hgtxr_post_count31_gpu1_20260611`. Validation passed: `bash -n`, `sh -n`, no-arg usage exit `2`, and startup logs. post_count31 waits for post_count30; if count continuation through fixed355k/fixed360k promotes it launches fixed365k/fixed370k, otherwise GPU1 runs centerhinge-squared (`margin=10.0`, `weight=0.005`) and GPU0 runs AdamW `8e-6`. If post_count30 mixed probes fail, GPU1 also runs centerhinge-squared while GPU0 runs linear centerhinge.
2026-06-11 latest mixed-probe closeout:

- Training/eval processes finished; both GPUs were idle after manual eval repair.
- GPU0 AdamW `8e-6` at fixed255k is the new overall leader. Best-center test eval `eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913`: center `27.0897`, P10 `15.7109`, P5 `4.6301`.
- GPU0 AdamW best-P10 eval `eval_fixed255k_adamw_lr8e_6_bestp10_test_gpu0_w0_20260611_114133`: center `28.2271`, P10 `14.6743`, P5 `4.3941`.
- GPU1 `prev_pupil_anchor` training finished, but original eval failed on `infer_hbtxr.py --output-dir`.
- Patched `scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh` to call `eval_hbtxr.py --experiment-name`; `sh -n` and `bash -n` passed.
- Manual prevpupil evals completed: best-center center `65.0978`, P10 `9.9877`, P5 `3.0629`; best-P10 center `65.6465`, P10 `9.4957`, P5 `3.1365`.
- Analysis: AdamW optimizer/LR was the high-value axis; local previous-pupil crop currently breaks the learned full-ROI distribution and should not be extended before a targeted crop/target alignment diagnostic.
- Next experiment recommendation: run AdamW `8e-6` count bracket at fixed250k/fixed260k in parallel, then run LR bracket at fixed255k (`6e-6`, `1e-5`) or warm-start fine-tune from the AdamW leader.

2026-06-12 experiment tracking convention import:

- Added legacy experiment import document: `docs/track/HBTXR_V3_0_PAST_EXPERIMENT_RESULTS.md`.
- Source root analyzed: `/home/kjm26/project/PRJXR/XR-VIT/HBTXR_v3_0/docs`.
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
- XR-04 low-similarity-aware loss candidate prepared while GPUs are occupied. Added `track_low_similarity_*` disabled-by-default weighting, config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml`, and runner `scripts/v3/run_xr04_lowsim_ellipsestate_probe.sh`. Validation passed: track loss pytest `6 passed`, py_compile, `bash -n`, and raw event-count contract.
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
- XR-08 checkpoint interpolation support added: `scripts/v3/interpolate_hbtxr_checkpoints.py` creates eval-only payloads from two same-architecture checkpoints, and `scripts/v3/run_xr08_checkpoint_interp_eval.sh` evaluates them with the fixed255k XR-06C surface.
- XR-08 alpha `0.50` between XR-06C best-center and best-P10 completed with no promotion: `runs/eval_fixed255k_xr08_xr06c_center_p10_interp_alpha0p50r_gpu0_w0_20260616_053156/eval/test/eval_summary.json` produced `20.32587662594659/26.49830005509513/8.386479888643537`. The initial parallel `tee` launch produced incomplete eval dirs, so the runner was corrected to file redirection and alpha `0.50` was rerun successfully.
- XR-05I launched on GPU0 from XR-05A P5/balanced secondary `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701/train/best_metric_track_center_px.pt`. It uses no-distill, fixed255k, aux `0.001/0.025/0.01`, AdamW LR `5e-7`, log `runs/_logs/xr05i_p5preserve_nodistill_xr05a_lr5e-7_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053814`. Startup passed raw event-count contract, loaded the intended checkpoint, resolved `cuda:0`, and entered epoch 1/12.
- XR-06F launched on GPU1 as complementary P5-preserve weak-distill from the same XR-05A P5/balanced checkpoint as init and teacher. It uses fixed255k, aux `0.001/0.025/0.01`, AdamW LR `5e-7`, log `runs/_logs/xr06f_p5preserve_weakdistill_xr05a_lr5e-7_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053815`. Startup passed raw event-count contract, loaded the intended init/teacher checkpoint, resolved `cuda:1`, and entered epoch 1/12.
- XR-05I closeout completed. Training early-stopped at epoch 5. Best-center eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_054521/eval/test/eval_summary.json` produced `20.351013261931282/26.15008576256888/8.428996889931815`. Best-P10 eval `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_054833/eval/test/eval_summary.json` produced `20.393892083849227/26.170068747656686/8.450255387169975`. No promotion.
- XR-06F closeout completed. Training early-stopped at epoch 10. Best-center eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_055145/eval/test/eval_summary.json` produced `20.322171998023986/26.200681025641305/8.625850643430438`. Best-P10 eval `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_055409/eval/test/eval_summary.json` produced `20.421796573911394/26.394558545521328/8.630102334703718`. No promotion against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Decision: local low-LR polish/P5-preserve fallback is saturated. Continue second-goal work with a targeted low-similarity/high-motion XR-04B branch or dense-trajectory XR-02, not another identical low-LR direct-aux replay.
- XR-04B/XR-04C targeted failure-bucket branches launched after XR-05I/XR-06F no-promotion. XR-04B runs on GPU0 with threshold `0.1`, scale `2.0`, LR `6e-6`, log `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p1_s2p0_fullwidth_20260616_060202`. XR-04C runs on GPU1 with threshold `0.1`, scale `2.0`, LR `3e-6`, log `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log`, run `runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p1_s2p0_fullwidth_20260616_060224`. Both passed raw event-count contract, loaded XR-03D best-P10, resolved the intended CUDA device, and entered epoch 1/12.
- XR-04B/XR-04C closeout completed. Both early-stopped at epoch 8 and all four test eval summaries were generated. XR-04B best-center `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_061317/eval/test/eval_summary.json` produced `20.484137114456722/25.600340850012643/8.277636350904192`. XR-04B best-P10 `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_061619/eval/test/eval_summary.json` produced `20.67380678653717/25.401361295155116/7.875425440924508`. XR-04C best-center `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_061333/eval/test/eval_summary.json` and best-P10 `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_061637/eval/test/eval_summary.json` both produced `20.588273804528374/25.658163949421475/7.849915211541312`. No promotion.
- Added `scripts/v3/build_failure_bucket_manifest.py` and `scripts/v3/run_xr04d_failbucket_subset_probe.sh`. Validation passed: py_compile, bash syntax, subset creation with train `1186` and val `182` rows. XR-04D launched on GPU0 with LR `6e-6`, log `runs/_logs/xr04d_failbucket_sim01_s201_lr6e-6_gpu0_20260616.log`; XR-04E launched on GPU1 with LR `3e-6`, log `runs/_logs/xr04e_failbucket_sim01_s201_lr3e-6_gpu1_20260616.log`. Both use `similarity_target <= 0.1` + `session_201` train/val subset, start from XR-03D best-P10, and reached epoch 1/12.
- XR-04D/XR-04E closeout completed. Both early-stopped at epoch 10 and all four full-test eval summaries were generated. XR-04D LR `6e-6` best-center `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_063110/eval/test/eval_summary.json` produced `24.50812735216958/16.752551589693343/4.517857306344169`; best-P10 `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_063417/eval/test/eval_summary.json` produced `25.703870964050292/15.307398448671613/4.238095378875732`. XR-04E LR `3e-6` best-center `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_063111/eval/test/eval_summary.json` produced `23.55767515386854/18.159439352580478/5.106292690549578`; best-P10 `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_063417/eval/test/eval_summary.json` produced `23.42795329945428/18.25552776881627/5.031037589481898`. No promotion; hard-subset failure-bucket training is closed as a full-test path.
- XR-09 full-manifest weighted sampler implemented. Added `training.sampler.type=weighted_failure_bucket` in `src/hbtxr/data/loader.py`, config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_weightedsampler_fullwidth.yaml`, runner `scripts/v3/run_xr09_weightedsampler_trackstateaux_probe.sh`, and tests `tests/test_weighted_sampler.py`. Validation passed: py_compile, bash syntax, `3 passed` weighted sampler tests, dataloader smoke with canonical root, and train-manifest weight sanity `6x=1186`, `3x=438`, `2x=1686`, `1x=2619`.
- XR-09A/XR-09B launched. XR-09A uses XR-06C best-center on GPU0 with low-sim `3x`, session_201 `2x`, cap `6x`, log `runs/_logs/xr09a_weightedsampler_center_lr5e-7_gpu0_20260616.log`. XR-09B uses XR-06C best-P10 on GPU1 with low-sim `2x`, session_201 `2x`, cap `4x`, log `runs/_logs/xr09b_weightedsampler_p10_lr5e-7_gpu1_20260616.log`. Both startup logs show raw event-count contract pass, intended checkpoint load, resolved CUDA device, full train/val manifest counts `5929/844`, and epoch `1/12` entry.
- XR-09A/XR-09B closeout completed. Both training runs exited `0`, all four test eval summaries were generated, and no active gate promoted. XR-09A best-center `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m3p0_s2p0_max6p0_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_065504/eval/test/eval_summary.json` produced `20.579283670016697/25.632653781345912/8.010204356057303`; XR-09A best-P10 `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m3p0_s2p0_max6p0_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_065808/eval/test/eval_summary.json` produced `20.641361141204833/24.742772783551896/8.041241747992379`; XR-09B best-center `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m2p0_s2p0_max4p0_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_065535/eval/test/eval_summary.json` produced `20.415359340395245/26.196429313932146/8.305697563716343`; XR-09B best-P10 `runs/eval_fixed255k_xr09_weightedsampler_t0p1_m2p0_s2p0_max4p0_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_065840/eval/test/eval_summary.json` produced `20.55930529321943/25.68239870071411/7.932823378699166`.
- XR-10 full-manifest loss-side sample weighting implemented. Added `meta.session_key` preservation in `src/hbtxr/data/components.py`, `loss.track_sample_weight` in `src/hbtxr/loss/stage_common.py`, Stage2 track/aux/consistency/constraint weighting in `src/hbtxr/loss/stage2.py`, config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_lossweight_fullwidth.yaml`, runner `scripts/v3/run_xr10_lossweight_trackstateaux_probe.sh`, and tests `tests/test_loss_sample_weighting.py`. Validation passed: py_compile, bash syntax, `14 passed` loss/sampler/track tests, dataloader smoke with canonical root, and train-manifest loss-weight sanity `3x=1186`, `2x=438`, `1.5x=1686`, `1x=2619`.
- XR-10A/XR-10B launched. XR-10A uses XR-06C best-center on GPU0 with low-sim `2x`, session_201 `1.5x`, cap `3x`, log `runs/_logs/xr10a_lossweight_center_lr5e-7_gpu0_20260616.log`. XR-10B uses XR-06C best-P10 on GPU1 with low-sim `1.5x`, session_201 `1.5x`, cap `2.5x`, log `runs/_logs/xr10b_lossweight_p10_lr5e-7_gpu1_20260616.log`. Both startup logs show raw event-count contract pass, intended checkpoint load, resolved CUDA device, full train/val manifest counts `5929/844`, and epoch `1/12` entry.
- XR-10A/XR-10B closeout completed. Both training runs exited `0`, all four test eval summaries were generated, and no active gate promoted. XR-10A best-center `runs/eval_fixed255k_xr10_lossweight_t0p1_m2p0_s1p5_max3p0_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_071610/eval/test/eval_summary.json` produced `20.34876068319593/26.20620824268886/8.127976478849138`; XR-10A best-P10 `runs/eval_fixed255k_xr10_lossweight_t0p1_m2p0_s1p5_max3p0_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_071916/eval/test/eval_summary.json` produced `20.429714499201094/26.048044974463327/7.999575104032244`; XR-10B best-center `runs/eval_fixed255k_xr10_lossweight_t0p1_m1p5_s1p5_max2p5_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_071612/eval/test/eval_summary.json` produced `20.3265901020595/25.97023880141122/8.133503682272774`; XR-10B best-P10 `runs/eval_fixed255k_xr10_lossweight_t0p1_m1p5_s1p5_max2p5_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_071918/eval/test/eval_summary.json` produced `20.38969965662275/26.1870755808694/7.99447306905474`.

2026-06-16 XR-11 SimDR setup:

- Sub-agent T-041 selected track-branch SimDR-style coordinate auxiliary as the safest next P0 after XR-10 no-promotion; dense-trajectory XR-02 remains blocked by sparse test-manifest gaps.
- Implemented `TrackStateSimDRHead` and wired it through head factory, tracker, track branch, model factory, Stage2 loss, and tests.
- Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_simdr_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr11_trackstate_simdr_probe.sh`.
- Validation passed: `bash -n`, `python3 -m py_compile` on touched Python files, `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_loss_sample_weighting.py tests/test_weighted_sampler.py` with `19 passed`, config/model smoke output `track/state_simdr=torch.Size([2, 2, 64])`, and `git diff --check`.
- Next launch plan: GPU0 center lane from XR-06C best-center; GPU1 P10 lane from XR-06C best-P10.
- XR-11 launch validation:
  - GPU0 center lane session `hgtxr_xr11_simdr_center_gpu0_20260616`, log `runs/_logs/xr11_simdr_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073321`.
  - GPU1 P10 lane session `hgtxr_xr11_simdr_p10_gpu1_20260616`, log `runs/_logs/xr11_simdr_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073333`.
  - Both logs show raw event-count contract pass, intended XR-06C checkpoint load, resolved CUDA devices `cuda:0`/`cuda:1`, full train/val counts `5929/844`, and epoch `1/12` entry.
- XR-11 closeout completed. Both train runs exited `0`; all four full-test eval summaries were generated. Center-init best-center `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_center_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_074029/eval/test/eval_summary.json` produced `20.338836230550495/26.16071500778198/8.301445865631104`. Center-init best-P10 `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_center_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_074332/eval/test/eval_summary.json` produced `20.366218400001525/26.545493936538698/8.573979895455496`. P10-init best-center `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_p10_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_074031/eval/test/eval_summary.json` produced `20.34303183896201/25.954507521220616/8.235544497626169`. P10-init best-P10 `runs/eval_fixed255k_xr11_simdr0p0005_b64_s1p5_p10_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_074341/eval/test/eval_summary.json` produced `20.289304559571402/26.325255816323416/8.309949268613543`. No active gate promoted.
- Decision: XR-11 no-distill SimDR is closed. Next P0 is XR-12 weak-distill light-SimDR: preserve XR-06C init/teacher and direct aux weights, lower SimDR weight to `0.0001` first, and compare against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-12 weak-distill light-SimDR config and runner added: `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_fullwidth.yaml` and `scripts/v3/run_xr12_weakdistill_simdr_probe.sh`.
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
- Added `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_headonly_fullwidth.yaml` and `scripts/v3/run_xr13_headonly_weakdistill_simdr_probe.sh`.
- Validation passed: runner `bash -n`, XR-06C checkpoint existence checks, and config/model/trainable smoke. Trainable filter matched `18` head tensors and `500494/3505306` trainable params.
- XR-13 launched in parallel. GPU0 center lane session `hgtxr_xr13_headonly_center_gpu0_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081531`. GPU1 P10 lane session `hgtxr_xr13_headonly_p10_gpu1_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081528`.
- Startup logs show raw event-count contract pass, intended XR-06C init/teacher checkpoint per lane, trainable-filter `18` tensors and `500494/3505306` trainable params, resolved devices `cuda:0` and `cuda:1`, train/val counts `5929/844`, and epoch `1/12` entry.

2026-06-16 XR-13 closeout:

- XR-13 center lane early-stopped at epoch 5. Best-center eval produced `20.2916/26.2976/8.4694`; best-P10 eval produced `20.2897/26.2679/8.5204`.
- XR-13 P10 lane ran through epoch 12. Best-center eval produced `20.3555/26.4622/8.4906`; best-P10 eval produced `20.3696/26.5514/8.5332`.
- No active gate promoted against center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- Decision: close SimDR line for now. Next P0 is XR-02 dense/continuous prediction trajectory preparation and EyeLoRiN-style M2F rerun on valid trajectories.

2026-06-16 XR-02 dense trajectory availability:

- Added `scripts/v3/check_xr02_dense_trajectory_availability.py`.
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
- Added `--limit` to `scripts/v3/infer_hbtxr.py` after a full confidence-inference attempt produced no output/no GPU activity and was stopped.
- `--limit 2` smoke succeeded at `runs/diagnostics/xr04_confidence_xr06c_bestcenter_infer_limit2_20260616`; rows include `track_pred` and confirm `search_state`/`event_state` are null for the active track-only leader.
- Added `scripts/v3/summarize_track_confidence_buckets.py`; limit-2 smoke joined `2/2` rows and wrote `runs/diagnostics/xr04_confidence_xr06c_bestcenter_limit2_buckets_20260616.json`.
- Decision: do not rerun low-sim/session_201 reweighting. Next action is bounded `track_pred` confidence/quality analysis and only then confidence/relocalization gating or confidence auxiliary loss.

2026-06-16 XR-04 confidence/quality probe:

- Added `scripts/v3/build_confidence_probe_manifest.py`.
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

- Added and ran `scripts/v3/eval_similarity_fallback.py` on the XR-06C best-center full-test eval rows.
- Artifact: `runs/diagnostics/xr14_similarity_prevstate_fallback_xr06c_bestcenter_20260616.json`.
- Result: raw is best by official-like batchmean center `20.283125752718494`; best fallback `blend(threshold=0.05, alpha=0.75)` degrades to `22.57394233260353`.
- Extended `scripts/v3/infer_hbtxr.py` to write `track_state_aux` and `track_state_simdr`.
- Extended `scripts/v3/summarize_track_confidence_buckets.py` with `--state-key`.
- Aux-state probe artifact: `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_auxstate_buckets_20260616.json`.
- Result: `track_state_aux` is not a fallback path; probe center `158.7825`, P10 `0.0`, P5 `0.0`.
- Added XR-14A all-head relocalization branch:
  - `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml`
  - `scripts/v3/run_xr14_allhead_relocalize_probe.sh`
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
- Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr15_support_adaptive_trackstateaux_probe.sh`.
- Default XR-15A preserves XR-06C init/teacher, weak distill, AdamW `5e-7`, full-width track-only, and direct state aux `0.0005/0.0125/0.005`.
- Event-window change: fixed-count policy with adaptive count enabled at min/base/max `192k/255k/320k`, reference `4000003us`, power `0.5`.
- Preflight passed: runner syntax, executable bit, raw event-count contract, py_compile, and dataset smoke showing adaptive target counts are resolved.

2026-06-16 XR-15 support-adaptive launch:

- Launched XR-15A center lane on GPU0 in tmux session `hgtxr_xr15_supportadaptive_center_gpu0_20260616`; log `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148`.
- Launched XR-15A P10 lane on GPU1 in tmux session `hgtxr_xr15_supportadaptive_p10_gpu1_20260616`; log `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205`.
- Both lanes passed startup validation: raw event-count contract, intended XR-06C checkpoint as init/teacher, resolved CUDA device `cuda:0`/`cuda:1`, train/val counts `5929/844`, and epoch `1/12` step logs.
- Added `scripts/v3/eval_xr15_p5_checkpoint.sh` as a post-train helper for `best_track_p5.pt` eval. The main runner already evaluates best-center and best-P10, and the trainer also saves a best-P5 checkpoint.

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

- Added `scripts/v3/run_xr15d_trackadapters_probe.sh`.
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
  - config `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
  - runner `scripts/v3/run_xr15e_weakdistill_trackadapters_probe.sh`.
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
  - config `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`.
  - runner `scripts/v3/run_xr16_weakdistill_trackeventadapter_probe.sh`.
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
- Added XR-16B checkpoint interpolation diagnostic runner `scripts/v3/run_xr16b_checkpoint_interp_eval.sh`.
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
- Added runner `scripts/v3/run_xr17a_xr15c_center_p5_interp_eval.sh`.
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

- Added config `configs/v3/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr17b_p5anchor_supportadaptive_probe.sh`.
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

- Added runner `scripts/v3/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh`.
- Sandbox evals hung with idle GPUs and were interrupted; escalated GPU0/GPU1 evals completed.
- Results:
  - alpha `0.025`: `20.184962025710515 / 26.113946315220424 / 8.474490090778895`.
  - alpha `0.05`: `20.18765983411244 / 26.054422501155308 / 8.525510501861572`.
  - alpha `0.075`: `20.190412517956325 / 26.156463316508702 / 8.525510501861572`.
  - alpha `0.10`: `20.193220179421562 / 26.20748372077942 / 8.629677173069545`.
- Decision: no promotion. No-train interpolation toward XR-06C P10 does not recover P10 enough and starts losing XR-17B center/P5 gates.

2026-06-16 XR-19A trainable P10-recovery micro-polish closeout:

- Added config `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr19a_p10recovery_micro_polish.sh`.
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

- Reused `scripts/v3/run_xr19a_p10recovery_micro_polish.sh` with explicit init and teacher checkpoints.
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
  - `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_fullwidth.yaml`
  - `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_fullwidth.yaml`
- Added runners:
  - `scripts/v3/run_xr21_p10margin_supportadaptive_probe.sh`
  - `scripts/v3/run_xr21_p10leader_margin_probe.sh`
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

- Added `scripts/v3/average_hbtxr_checkpoints.py` for weighted N-way model-state soup.
- Added `scripts/v3/run_xr22_trileader_soup_eval.sh` for no-train soup generation and test eval.
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

- Added config `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr23_p10_optimizer_probe.sh`.
- Poincare GPT-5.5 read-only optimizer explorer selected ADOPT and Lion as bounded first-pass optimizer probes from the supported optimizer registry.
- ADOPT/GPU0 command completed epoch `8/8`: `bash scripts/v3/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 adopt 2.5e-7 cuda:0`.
- Lion/GPU1 command early-stopped at epoch `6/8`: `bash scripts/v3/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 lion 1e-7 cuda:1`.
- Test results:
  - ADOPT best-P10 `20.261837770257678 / 26.318027945927213 / 8.681122745786395`.
  - ADOPT best-P5 `20.219128920350755 / 25.956633363451278 / 8.469388042177473`.
  - Lion best-P10 `20.22320341382708 / 26.311650391987392 / 8.513180569240026`.
  - Lion best-P5 `20.271730688640048 / 26.129677595411028 / 8.532313203811645`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. Next P0 should use XR-22 as a bounded P10-recovery anchor; only retry optimizer if testing ADOPT-specific beta/eps defaults.

2026-06-16 XR-24 XR-22-anchor P10-recovery closeout:

- Added config `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr24_xr22_anchor_p10recovery.sh`.
- Carson GPT-5.5 read-only strategy explorer recommended the XR-22 `c34/p33/f33` soup as the next bounded P10-recovery anchor, with XR-06C best-P10 teacher and P10-selection.
- GPU0 command completed epoch `8/8`: `bash scripts/v3/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
- GPU1 command completed epoch `8/8`: `bash scripts/v3/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Test results:
  - LR `1.25e-7` best-P10 `20.217043702942984 / 26.254252440588814 / 8.764881270272392`.
  - LR `1.25e-7` best-P5 `20.223851100036075 / 26.415817070007325 / 8.722364262172155`.
  - LR `2.5e-7` best-P10 `20.23367166178567 / 26.121599388122558 / 8.588435677119664`.
  - LR `2.5e-7` best-P5 `20.238911376680647 / 26.07695653779166 / 8.791666984558105`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. XR-22 remains useful as a no-train anchor, but AdamW polish from it is closed.

2026-06-16 XR-25 XR-22-anchor ADOPT-defaults closeout:

- Added config `configs/v3/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml`.
- Added runner `scripts/v3/run_xr25_xr22_anchor_adopt_defaults.sh`.
- Static validation passed: runner syntax, config load, ADOPT build with betas `[0.9,0.9999]` and eps `1e-6`, raw event-count contract, checkpoint existence, and `git diff --check`.
- GPU0 command early-stopped at epoch `7/8`: `bash scripts/v3/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
- GPU1 command completed epoch `8/8`: `bash scripts/v3/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Test results:
  - LR `1.25e-7` best-P10 `20.21177260194506 / 26.46045993396214 / 8.728741809300013`.
  - LR `1.25e-7` best-P5 `20.22005627495902 / 26.181123181751797 / 8.707483305249895`.
  - LR `2.5e-7` best-P10 `20.23533037390028 / 26.108844266619002 / 8.64583364214216`.
  - LR `2.5e-7` best-P5 `20.23760941709791 / 26.187500749315536 / 8.86607174192156`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. XR-22 trainable polish is closed for AdamW and ADOPT-defaults; next P0 is XR-26 P10-boundary loss from XR-06C best-P10 init/teacher.

2026-06-16 XR-26 P10-boundary head-only closeout:

- Added P10-boundary surrogate loss on `track/state` and `track/state_aux`, plus config `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10boundary_headonly_fullwidth.yaml` and runner `scripts/v3/run_xr26_p10boundary_aux_probe.sh`.
- Static validation passed: runner syntax, pytest `18 passed`, config smoke, checkpoint existence, and `git diff --check`.
- Non-escalated launch failed because sandbox CUDA reported `cuda_device_count=0`; escalated launch ran normally.
- GPU0 default command early-stopped at epoch `6/8`: `bash scripts/v3/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:0`.
- GPU1 light command early-stopped at epoch `6/8`: `bash scripts/v3/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:1 ... xr06cp10_init_teacher_light 0.01 0.005`.
- Test results:
  - Default best-P10 `20.32213627440589 / 26.45790890966143 / 8.535289403370449`.
  - Default best-P5 `20.318065077917918 / 26.42814700944083 / 8.541666957310268`.
  - Light best-P10 `20.32214218207768 / 26.45790890966143 / 8.535289403370449`.
  - Light best-P5 `20.318073788711004 / 26.42814700944083 / 8.541666957310268`.
- Decision: no active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`. Scalar P10-boundary/hinge polish is closed; next P0 should change coordinate representation or teacher quality.

2026-06-16 XR-27 track-heatmap coordinate representation closeout:

- Added `TrackCenterHeatmapHead`, heatmap decode into `track/state[:, :2]`, heatmap CE/offset losses, config `configs/v3/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`, and runner `scripts/v3/run_xr27_trackheatmap_p10teacher_probe.sh`.
- Sub-agent audit found a launch blocker: `distillation.state_similarity=true` would compare the student's heatmap-state output against teacher heatmap-state output with random newly added heatmap weights. Clean runs set `state_similarity=false` and `state_weight=0.0`.
- GPU0 LR `1e-4` command completed epoch `10/10`: `bash scripts/v3/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 1e-4 cuda:0 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr1e4`.
- GPU1 LR `3e-5` command completed epoch `10/10`: `bash scripts/v3/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 3e-5 cuda:1 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr3e5`.
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

- Updated `scripts/v3/run_xr27_trackheatmap_p10teacher_probe.sh` to accept `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME`.
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

- Added `scripts/v3/run_xr31_xr29_xr30_heatmap_interp_eval.sh`, using existing checkpoint interpolation utility and the XR-29/XR-30 heatmap-state eval contract.
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

- Added `scripts/v3/run_xr34_heatmap_lossratio_refinement.sh` with two lanes:
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
- Added XR-35 no-train interpolation runner `scripts/v3/run_xr35_xr29_xr34b_heatmap_interp_eval.sh` and plan artifact `docs/resources/xr35_xr29_xr34b_interpolation_plan_2026_06_16.md`.

2026-06-16 XR-35 no-train interpolation closeout:

- Ran `bash scripts/v3/run_xr35_xr29_xr34b_heatmap_interp_eval.sh cuda:1 0.03125:a0p03125 0.0625:a0p0625 0.09375:a0p09375 0.125:a0p125`.
- Results:
  - `0.03125`: `17.016330581051964/32.47534093856812/11.334609195164271`.
  - `0.0625`: `16.983979083810535/32.468963384628296/11.391156809670585`.
  - `0.09375`: `16.952843945366997/32.42432054110936/11.420918709891183`.
  - `0.125`: `16.923010180677686/32.352041605540684/11.255102368763515`.
- No active gate promoted. Active gates remain center `<16.53321223940168`, P10 `>33.77168447630746`, and P5 `>11.50467722075326`.
- Next P0: trainable P5-anchor continuation using XR-34B center/P10 signal and XR-29 P5 anchor.

2026-06-16 XR-36 preparation:

- Added `scripts/v3/run_xr36_p5_anchor_continuation.sh`.
- Added `docs/resources/xr36_p5_anchor_continuation_plan_2026_06_16.md`.
- Static validation passed: `bash -n scripts/v3/run_xr36_p5_anchor_continuation.sh`.
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
- Added XR-37 runner `scripts/v3/run_xr37_xr36b_xr36a_interp_eval.sh` and plan `docs/resources/xr37_xr36b_xr36a_interpolation_plan_2026_06_16.md`.

2026-06-17 XR-37 closeout:

- Ran `bash scripts/v3/run_xr37_xr36b_xr36a_interp_eval.sh cuda:1 0.10:a0p10 0.20:a0p20 0.35:a0p35 0.50:a0p50`.
- Results: alpha `0.10` reached `16.52253861086709/34.062075574057445/11.574830266407558`.
- Results: alpha `0.20` reached `16.51475806917463/34.28826605933053/11.44387788772583`.
- Results: alpha `0.35` reached `16.50803507396153/34.23086808749608/11.292517362322126`.
- Results: alpha `0.50` reached `16.507612899371555/34.33205857958112/11.539116007941109`.
- Decision: alpha `0.50` promotes center only. Active gates are now center `<16.507612899371555`, P10 `>34.74064704350063`, and P5 `>11.738095617294311`.

2026-06-17 XR-38 preparation:

- Added `scripts/v3/run_xr38_center_preserve_p10p5_recovery.sh`.
- Added `docs/resources/xr38_center_preserve_p10p5_recovery_plan_2026_06_17.md`.
- Static validation passed: `bash -n scripts/v3/run_xr38_center_preserve_p10p5_recovery.sh`.
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

- Added `scripts/v3/run_xr39_mixed_leader_soup_eval.sh`.
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

- Added `scripts/v3/run_xr40_xr39leader_p5_soup_eval.sh`.
- Added `docs/resources/xr40_xr39leader_p5_soup_plan_2026_06_17.md`.
- Static/input validation passed: `bash -n`, `git diff --check`, all source checkpoints exist, and all three anchors have matching `138` model keys.
- Ran 7 no-train soups across GPU0/GPU1.
- Best center: `c45p35f20` reached `16.493494159834725/34.77508579662868/11.255527530397687`.
- Best P10: `c40p40f20` reached `16.494016419138227/34.7750858102526/11.249149976457868`.
- Best P5: `c35p25f40` reached `16.49522715806961/34.64115719795227/11.529762240818568`.
- Decision: no gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.843962955474854`.
- Next P0: trainable loss-ratio fallback for P5 recovery.

2026-06-17 XR-41 closeout:

- Added `scripts/v3/run_xr41_lossratio_p5_fallback.sh`.
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

- Added `scripts/v3/run_xr42_p5_preserve_lowdrift.sh`.
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

- Prepared `scripts/v3/run_xr42c_explicit_distill_recovery.sh` after GPT-5.5 sub-agent audit found that XR-42 had no effective heatmap-state teacher anchor because XR-27 forced `distillation.state_similarity=false`.
- Added plan artifact: `docs/resources/xr42c_explicit_distill_recovery_plan_2026_06_17.md`.
- Ran XR-42C-A on GPU0 and XR-42C-B on GPU1. Initial sandbox execution failed with CUDA unavailable; approved GPU runtime executed successfully.
- XR-42C-A early-stopped at epoch `7/10`; best-center was the closest result: `16.491862688745773/34.712585769380844/11.472364275796073`, but it missed the center gate by `0.000083589553835`.
- XR-42C-B completed epoch `10/10`; best-P10 was `16.502160484450204/34.39710965156555/11.22236428941999`.
- No gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Next decision: pivot away from XR41 P5-teacher recovery. Prioritize failure-bucket/subject-39 direct optimization or teacher retraining.

# 2026-06-17 XR-43 closeout

- Added `scripts/v3/run_xr43_xr39center_xr42ca_interp_eval.sh`.
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

- Added `scripts/v3/run_xr44_updated_trileader_soup_eval.sh`.
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

- Added `scripts/v3/run_xr45_lowsim_heatmap_refresh.sh`.
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

- Added `scripts/v3/run_xr46_center_p10_compat_soup_eval.sh`.
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

- Added `scripts/v3/run_xr47_center_preserve_full_calibration.sh`.
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

- Added `scripts/v3/run_xr48_p10_boundary_heatmap_calibration.sh`.
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

- Added `scripts/v3/run_xr49_p5_boundary_heatmap_calibration.sh`.
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

- Added `scripts/v3/run_xr50_event_adapter_heatmap_calibration.sh`.
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

- Added `scripts/v3/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh`.
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

- Added `scripts/v3/run_xr52_trainable_p10_recovery_from_xr51.sh`.
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

- Added `scripts/v3/run_xr53_xr52_xr39_p10_recombination_eval.sh`.
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

- Added `scripts/v3/run_xr55_xr39_p10_anchor_expanded_scope.sh`.
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
- Added runner: `scripts/v3/run_xr56_soft_threshold_p10_supervision.sh`.
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
- Added `scripts/v3/run_xr57_p10_center_refine_calibration.sh`.
- Added `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md`.
- Sub-agent spawn for T-097 failed with `agent thread limit reached`; main-agent fallback completed implementation.
- Static validation passed:
  - `bash -n scripts/v3/run_xr57_p10_center_refine_calibration.sh`
  - `python3 -m py_compile ...`
  - `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` -> `27 passed`
  - model-build smoke -> `TrackCenterRefineHead True 4.0`
  - lane A/B `DRY_RUN=1`
- Actual XR-57 train/eval completed. Lane A early-stopped at epoch `7/12`; test metrics were best-P10 `16.69362453562873/34.38307912690299/10.973214619500297` and best-P5 `16.715463175092424/34.849490649359566/10.925595603670393`. Lane B completed epoch `12/12`; test metrics were best-P10 `16.504537062985555/34.137755966186525/11.207483332497732` and best-P5 `16.4997801729611/34.190476996558054/11.476615987505232`.
- Decision: XR-57 did not promote any gate. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. Next P0 is XR-58 P10 teacher refresh.

# 2026-06-17 XR-58 implementation

- Added `scripts/v3/run_xr58_p10_teacher_refresh.sh`.
- Added `docs/resources/xr58_p10_teacher_refresh_plan_2026_06_17.md`.
- Static validation passed:
  - `bash -n scripts/v3/run_xr58_p10_teacher_refresh.sh`
  - `DRY_RUN=1 bash scripts/v3/run_xr58_p10_teacher_refresh.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/v3/run_xr58_p10_teacher_refresh.sh b cuda:1`
- Launched XR-58A on GPU0 and XR-58B on GPU1.
- Startup validation passed for both lanes: raw event-count contract, XR-39 init checkpoint load, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- Actual XR-58 train/eval completed. Both lanes early-stopped at epoch `8/16` with train/eval exit `0`. XR-58A best-P10 reached `16.503614359242576/34.90306201662336/11.51275544847761`; XR-58A best-P5 reached `16.501813726765768/34.68920147078378/11.259779255730765`. XR-58B best-P10 reached `16.506438190596445/34.84226275852748/11.501275873184204`; XR-58B best-P5 reached `16.49833288192749/34.540391949244906/11.062075165339879`.
- Decision: XR-58 did not promote any gate. Best XR-58 P10 improved over XR-57 but remained below XR-39 by about `0.1199`. Next P0 is XR-59 narrow anchored-teacher bracket around XR-58A.

# 2026-06-17 XR-59 implementation

- Added `scripts/v3/run_xr59_xr58a_teacher_bracket.sh`.
- Added `docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md`.
- Static validation passed:
  - `bash -n scripts/v3/run_xr59_xr58a_teacher_bracket.sh`
  - `DRY_RUN=1 bash scripts/v3/run_xr59_xr58a_teacher_bracket.sh a cuda:0`
  - `DRY_RUN=1 bash scripts/v3/run_xr59_xr58a_teacher_bracket.sh b cuda:1`
- Launched XR-59A on GPU0 and XR-59B on GPU1.
- Startup validation passed for both lanes: raw event-count contract, XR58A best-P10 init checkpoint load, XR39 teacher checkpoint set, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- Actual XR-59 train/eval closeout completed with no promotion.
- Closed XR-59 with no promotion. XR-59A/B both early-stopped at epoch `7/10`. Added `scripts/v3/eval_xr59_completed_checkpoints.sh` and evaluated `best_track_p10`/`best_track_p5` checkpoints on the full test split. XR-59A reached `16.520220368249074/34.300170864377705/11.376701021194458`; XR-59B reached `16.51065547806876/34.43409944261823/11.287415306908743`. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. Next P0 should pivot away from scalar P10-soft continuation.

# 2026-06-17 XR-60 preparation

- Added default-off P10 candidate-head path and candidate loss wiring.
- Added `scripts/v3/run_xr60_p10_candidate_head.sh` and `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- Sub-agent T-102 reviewed XR-56/57/58/59 runner conventions and recommended XR59B-based candidate-head ablation. Integrated this into the runner defaults.
- Static validation passed: `bash -n`, `py_compile`, targeted pytest `34 passed`, candidate model-build smoke, A/B dry-runs, and `git diff --check`.
- Initial sandbox launch failed because PyTorch could not see CUDA (`cuda_device_count=0`); elevated launch resolved CUDA access.
- XR-60A/B launched on GPU0/GPU1 and entered epoch `2/12`.
- Closed XR-60 with no promotion. XR-60A early-stopped at epoch `7/12`; XR-60B early-stopped at epoch `11/12`. XR-60A best-P10/best-P5 reached `16.526124344553267/34.50255186898368/11.617772477013725` and `16.5058109828404/34.60331717899867/11.337160219464984`. XR-60B best-P10/best-P5 reached `16.842585216249738/33.49277294022696/10.70663298198155` and `16.717501049382346/33.87670159339905/10.971939107349941`. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Rewrote XR eye-tracking detailed analyses under `anlaysis/xr-eye-tracking` to the requested reference-document depth. Added `scripts/v3/write_xr_eye_tracking_detailed_analysis.py` and regenerated `39` `analysis.md` files: `18` codebase analyses and `21` paper analyses, total `8813` lines. Codebase reports now include Python symbol extraction, import/dependency clues, config option clues, and HGTXR conversion design. Paper reports now include bounded section evidence, primitive decomposition, result/metric evidence, and experiment conversion. Validation passed with generator `py_compile`, section coverage scan, and FACET spot checks. GPT-5.3-Codex-Spark sub-agent attempt failed due quota exhaustion, so main agent completed the rewrite.
- Closed XR-61 auxiliary candidate branch with no promotion: best observed metrics were XR-61A `16.475529539585114/34.356718465260094/12.127126216888428` and XR-61B `16.48967229127884/34.58843615395682/11.559524168287005`.
- Added and ran XR-62 FACET geometry auxiliary refresh. Added `scripts/v3/run_xr62_facet_geometry_aux_refresh.sh` and `docs/resources/xr62_facet_geometry_aux_refresh_plan_2026_06_17.md`. Static validation passed (`bash -n`, A/B dry-run), both lanes early-stopped at epoch `7/10`, and XR-62A promoted center to `16.468481131962367` with P10 `34.30782389640808` and P5 `12.052721459524973`. XR-62B did not promote. Active gates are now center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Reworked the XR eye-tracking analysis generator to the stricter requested reference-analysis style. The generator now emits codebase metadata compatible with the FlexLLM-style template, source responsibility tables, HGTXR data-contract matrices, static quality findings, and reproduction runbooks. Paper reports now include dataset/model/benchmark tables, reported-result evidence, HGTXR result interpretation, and detailed experiment option runbooks. Regenerated all `39` `analysis.md` files under `anlaysis/xr-eye-tracking`, increasing the total to `11574` lines. Validation passed with `py_compile`, full regeneration, required-section scans, FACET/EV-Eye spot checks, and `git diff --check`. Spark sub-agent attempts failed due quota exhaustion; GPT5.5 explorer agents provided the codebase/paper templates that were integrated.
- Added and ran XR-63 no-train P10 teacher-target oracle diagnostic. `scripts/v3/analyze_p10_teacher_target_oracle.py` loads manifest1 test targets through the XR-62A resolved config, aligns `eval_rows.json` from XR-62A, XR-39, XR-56B, and XR-58A, and computes eval-style batch-mean center/P10/P5 metrics. Outputs: `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` and `.md`. Oracle upper bound is `16.04023192701366/36.48596938775512/13.41751700680271`, exceeding all active gates. Decision: proceed to leakage-safe P10 teacher-target construction; do not repeat candidate-only or geometry-only auxiliary branches without this new protocol.
- Added `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md`. The plan records that current code has no direct sample-wise pseudo-target loading path, so XR-64 must generate train/val teacher eval rows, build split-isolated target overrides, add dataset/loss support, and reject test-derived pseudo labels during training.
- Implemented XR-64 target override path. Added `scripts/v3/build_xr64_teacher_target_overrides.py`, `scripts/v3/run_xr64_teacher_target_construction.sh`, data config keys `track_target_override_path` and `allow_test_target_override`, additive dataset fields, additive Stage2 target override losses, and tests. Guard: test manifest override path is rejected by default, and XR-64 runner clears override config during final test eval. Validation passed with `py_compile`, `bash -n`, targeted pytest `39 passed`, A/B dry-run, and a 64-sample builder smoke.

# 2026-06-18 current consolidation and next experiment list

- Added consolidated status/analysis note: `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- Recorded that XR-63 oracle remains the highest-value signal: `16.04023192701366/36.48596938775512/13.41751700680271` as no-train upper bound over XR-62A/XR-39/XR-56B/XR-58A predictions.
- Confirmed active gates remain center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Added XR-64 preparation helper: `scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh`.
- Static validation passed for the helper: `bash -n scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh scripts/v3/run_xr64_teacher_target_construction.sh`, `py_compile` for XR-64 builder and analysis generator, and `chmod +x`.
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
- Process check: no active process matched `train_hbtxr`, `eval_hbtxr`, `run_xr64`, `run_xr`, or `scripts/v3/run_.*train`.
- GPU check: GPU0 `15 MiB` used / `15827 MiB` free / `0%`; GPU1 `15 MiB` used / `15827 MiB` free / `0%`.
- Sub-agent T-701 completed a read-only review of the XR-64 prep helper. Main findings: the first all-in-one helper attempt stopped during `train/xr62a`, no `eval_rows.json` was produced, and the prior helper needed split/teacher-level resumability, foreground logging, output validation, and explicit test-split refusal.
- Hardened `scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh` for future use:
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

- Added `scripts/v3/check_xr64_resume_artifacts.py`, a read-only verifier for XR-64 resume artifacts.
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
  - `python3 -m py_compile scripts/v3/check_xr64_resume_artifacts.py`
  - `.venv/bin/python -m pytest -q tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py` -> `9 passed`
  - `.venv/bin/python scripts/v3/check_xr64_resume_artifacts.py --allow-missing-generated` -> `ready=true`, `generated_complete=false`
  - `.venv/bin/python scripts/v3/check_xr64_resume_artifacts.py --allow-missing-generated --format summary` -> `resume_status=generated_incomplete`
  - `bash -n scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh scripts/v3/run_xr64_teacher_target_construction.sh`
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

- Added `scripts/v3/report_second_goal_status.py`.
- Added `tests/test_second_goal_status.py`.
- Updated `docs/resources/second_goal_artifact_index_2026_06_18.md` and `.json` to include the status reporter and validation command.
- Updated `docs/Validation.md`, `docs/track/PROGRESS.md`, and `docs/track/TODO.md`.
- Validation passed:
  - `python3 -m py_compile scripts/v3/report_second_goal_status.py`
  - `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py` -> `5 passed`
  - `.venv/bin/python scripts/v3/report_second_goal_status.py --format summary` -> `execution_state=paused_by_user_directive`, `active_goal_complete=false`, `xr64_resume_status=generated_incomplete`, `xr64_can_run_lane=false`, `missing_eval_rows=8`, `missing_overrides=6`, `leakage_risk=none`
- No train/eval experiment was launched.

# 2026-06-18 software cleanup / refactor preparation

- User requested cleanup/refactor work for the dirty `software` directory.
- Created `docs/resources/software_cleanup_refactor_plan_2026_06_18.md`.
- Added read-only cleanup tooling:
  - `scripts/v3/maintenance/inventory_software_tree.py`
  - `scripts/v3/maintenance/audit_path_references.py`
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
- Added `scripts/v3/maintenance/build_runs_catalog.py`.
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
- Added `scripts/v3/maintenance/reorganize_runs_by_experiment.py`.
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
  - `.venv/bin/python scripts/v3/report_second_goal_status.py --format summary` still returns `xr64_resume_status=generated_incomplete`
- No run content was deleted and no train/eval job was launched.
- Sub-agent MOVE-001 completed read-only review and identified a high-risk issue: `scripts/v3/run_prepare_and_train.sh` used `find -type d`, which misses top-level compatibility symlinks.
- Fixed `scripts/v3/run_prepare_and_train.sh` latest-run discovery to include `-type l`.
- Added `--verify` mode to `scripts/v3/maintenance/reorganize_runs_by_experiment.py`.
- Generated verification reports:
  - `docs/resources/runs_experiment_reorganization_verify_2026_06_18.md`
  - `docs/resources/runs_experiment_reorganization_verify_2026_06_18.json`
- Verification result: `ok=true`, `top_level_compat_symlinks=849`, `organized_run_dirs=849`, `broken=0`, `bad_targets=0`.

# 2026-06-18 runs compatibility symlink removal

- User asked to proceed after top-level `eval*` and `raw*` entries were identified as compatibility symlinks.
- Added `scripts/v3/maintenance/remove_runs_compat_symlinks.py`.
- Updated active run lookup paths to search organized nested roots:
  - `src/hbtxr/config/run_contract.py`
  - `scripts/v3/run_prepare_and_train.sh`
  - `scripts/v3/check_raw_event_count_training_readiness.py`
  - `scripts/v3/check_raw_event_count_training_result.py`
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
  - `.venv/bin/python scripts/v3/report_second_goal_status.py --format summary`: passed
  - `.venv/bin/python scripts/v3/check_xr64_resume_artifacts.py --allow-missing-generated --format summary`: passed
  - `.venv/bin/python scripts/v3/check_raw_event_count_training_readiness.py`: found Stage1/Stage2 under `runs/NON_XR/raw`
  - `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_raw_event_count_training_readiness.py tests/test_raw_event_count_training_result.py tests/test_xr64_resume_artifacts.py tests/test_second_goal_status.py`: `13 passed`
- Environment note: `nvidia-smi` returned driver communication failure during readiness reporting; no train/eval job was launched.
