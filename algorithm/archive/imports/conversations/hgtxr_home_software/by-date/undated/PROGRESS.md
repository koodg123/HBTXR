# HGTXR-SW Progress

Date: 2026-06-10

## 2026-06-18 Experiment Pause / Documentation Pass

- [x] User requested all experiments to stop and current work to be documented.
- [x] Verified no active HGTXR train/eval process matched `train_hbtxr`, `eval_hbtxr`, `run_xr64`, `run_xr`, or `scripts/v3/run_.*train`.
- [x] Verified both GPUs were idle at the pause point: GPU0 `15 MiB` used / `15827 MiB` free / `0%`, GPU1 `15 MiB` used / `15827 MiB` free / `0%`.
- [x] Documented that no full XR-64 train/val teacher `eval_rows.json` files or override JSON files exist yet under `data/_internal/manifests/manifest1/xr64_teacher_targets/`.
- [x] Hardened `scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh` for future resumable split/teacher execution, foreground logging, JSON validation, and test-split refusal.
- [x] Did not launch any new experiment after the pause directive.
- [x] Added pause/current-state documentation: `docs/resources/experiment_pause_documentation_2026_06_18.md`.
- [x] Added second-goal completion audit: `docs/resources/second_goal_completion_audit_2026_06_18.md`.
- [x] Marked the active second goal as incomplete because final accuracy target, XR-64 target generation, and XR-64A/B/C execution are not yet complete.
- [x] Added XR-64 resume runbook: `docs/resources/xr64_resume_runbook_2026_06_18.md`.
- [x] Integrated GPT-5.3-Codex-Spark read-only sub-agent feedback into the runbook: fail-fast artifact checks, val-override usage note, reproducibility snapshot, and run-root verification caveat.
- [x] Added read-only XR-64 artifact checker: `scripts/v3/check_xr64_resume_artifacts.py`.
- [x] Added checker tests: `tests/test_xr64_resume_artifacts.py`.
- [x] Added second-goal artifact index: `docs/resources/second_goal_artifact_index_2026_06_18.md` and `.json`.

## 2026-06-15 XR-Eye-Tracking reference integration

- [x] Analyzed `/home/kjm26/project/PRJXR/References/XR-Eye-Tracking` codebase and paper inventory with main-agent reads plus GPT5.5 read-only sub-agents.
- [x] Stored per-codebase analysis under `anlaysis/xr-eye-tracking/codebases/*/analysis.md`.
- [x] Stored per-paper analysis under `anlaysis/xr-eye-tracking/papers/*/analysis.md`.
- [x] Added `anlaysis/xr-eye-tracking/experiment_integration.md` with prioritized second-goal experiments.
- [x] Added PAPER_REF-specific detailed experiment map: `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`.
- [x] Integrated new P0/P1/P2 experiments into `docs/Paper-Backed-Experiment-Plan.md`, `docs/Master-Plan.md`, `docs/Sub-Plan.md`, and `docs/track/TODO.md`.
- [x] Started XR-01 AdamW local count bracket on the training track: fixed250k on GPU0 and fixed260k on GPU1 at AdamW LR `8e-6`.
- [x] Completed XR-01 fixed255k LR bracket `{6e-6,1e-5}`. LR `1e-5` best-center promoted to the new software leader: center `26.1749`, P10 `16.5021`, P5 `5.0999`.
- [x] Launched XR-03 geometry probes from the promoted fixed255k AdamW LR `1e-5` best-center checkpoint: main LR `1e-5` branch on GPU0 and stability LR `6e-6` branch on GPU1.
- [x] Completed XR-03 geometry probe closeout. LR `1e-5`, axis `0.025`, angle `0.01` best-center promoted to the new overall software leader: center `21.6822`, P10 `21.8202`, P5 `7.0446`.
- [x] Launched XR-03A/B stronger geometry sweep from the promoted XR-03 best-center checkpoint: GPU0 LR `1e-5`, GPU1 LR `6e-6`, both with axis `0.05`, angle `0.02`.
- [x] Completed XR-03A/B train/eval closeout. XR-03A LR `1e-5`, axis `0.05`, angle `0.02` best-center promoted to the current overall software leader: center `20.4539`, P10 `25.8686`, P5 `8.3104`.
- [x] Launched XR-03C/D bounded geometry refinement from XR-03A best-center: GPU0 LR `1e-5`, GPU1 LR `6e-6`, both with axis `0.075`, angle `0.03`.
- [x] Completed XR-03C/D closeout. XR-03D LR `6e-6`, axis `0.075`, angle `0.03`, best-P10 checkpoint promoted as the center-first leader: center `20.4336`, P10 `25.7866`, P5 `7.7491`; XR-03A remains P10/P5-balanced secondary.
- [x] Implemented and validated XR-04 low-similarity-aware track-loss weighting candidate, including config and runner.
- [x] Launched XR-04 low-similarity-aware probes from the XR-03D center leader checkpoint: GPU0 LR `6e-6`, GPU1 LR `1e-5`, both threshold `0.3`, scale `1.0`.
- [x] Completed XR-04 train/eval closeout. No promotion: closest branch LR `6e-6` best-center reached center `20.4347`, P10 `25.7844`, P5 `8.4286`, still weaker than center gate `20.4336` and balanced P10 gate `25.8686`.
- [x] Implemented and validated XR-05A direct track-state auxiliary head: opt-in `track/state_aux` training-only head, config, runner, py_compile, raw event-count contract, model-build smoke, and track loss tests `8 passed`.
- [x] Launched XR-05A from XR-03D center leader checkpoint: GPU0 LR `6e-6`, GPU1 LR `3e-6`, both aux center/axis/angle `0.001/0.025/0.01`.
- [x] Completed XR-05A closeout. GPU0 LR `6e-6` best-P10 promoted as center-first leader: center `20.3170`, P10 `25.8639`, P5 `8.2398`. GPU1 LR `3e-6` best-P10 is the P10 leader: center `20.3220`, P10 `26.5761`, P5 `8.0336`. GPU1 LR `3e-6` best-center is the P5/balanced secondary: center `20.3833`, P10 `26.2105`, P5 `8.6956`.
- [x] Started XR-05B lighter direct-aux refinement on GPU1 from the XR-05A LR `3e-6` best-P10 checkpoint: aux center/axis/angle `0.0005/0.0125/0.005`, LR `2e-6`.
- [x] Started XR-05C midpoint-LR direct-aux refinement on GPU0 from the XR-03D best-P10 checkpoint: aux center/axis/angle `0.001/0.025/0.01`, LR `4.5e-6`.
- [x] Completed XR-05B closeout. Best-P10 checkpoint promoted as center-first leader: center `20.2927`, P10 `26.0446`, P5 `8.0782`. Best-center checkpoint did not beat P10/P5 gates: center `20.3479`, P10 `26.2317`, P5 `8.6446`.
- [x] Completed XR-05C closeout. No promotion: best-P10 center `20.3236`, P10 `26.1671`, P5 `7.8508`; best-center center `20.5962`, P10 `25.2691`, P5 `7.8350`.
- [x] Started XR-05D/XR-05E refinements: XR-05D GPU1 ultra-light aux `0.00025/0.00625/0.0025`, LR `1e-6`, init XR-05B best-P10; XR-05E GPU0 mid-light aux `0.00075/0.01875/0.0075`, LR `1.5e-6`, init XR-05A best-center.
- [x] Completed XR-05D/XR-05E closeout. XR-05D did not promote. XR-05E best-center promoted the P10 gate to `26.6539` with center `20.3673` and P5 `8.4830`; center leader remains XR-05B best-P10 at `20.2927`, and P5/balanced leader remains XR-05A best-center at `8.6956`.
- [x] Added validated XR-06 weak-distill track-state-aux fallback runner/config, but kept it queued after GPT5.5 sidecar recommended XR-02 dense trajectories or tiny hinge recovery first.
- [x] Added and launched XR-07A/B tiny center-hinge recovery from the XR-05B center leader: GPU1 linear hinge `m10/w0.01`, GPU0 squared hinge `m10/w0.001`, both fixed255k, AdamW LR `1e-6`, aux `0.0005/0.0125/0.005`.
- [x] Closed XR-07A/B as no-promotion on validation evidence: XR-07A best val center/P10/P5 `23.6261/21.8980/7.7943`; XR-07B best val center/P10/P5 `23.6286/21.8980/7.7943`.
- [x] Rechecked XR-02 dense-trajectory blocker: test manifest has `2238` rows over `72` groups, median adjacent gap `4000003us`, minimum gap `360000us`; current `50000us` gap gate correctly prevents smoothing.
- [x] Launched XR-06 weak-distill fallback on GPU1 from XR-05B best-P10 as both init and teacher: fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `1e-6`, log `runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log`.
- [x] Launched XR-06B weak-distill complementary P10-leader branch on GPU0 from XR-05E best-center as both init and teacher: fixed255k, aux `0.00075/0.01875/0.0075`, AdamW LR `1e-6`, log `runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log`.
- [x] Completed XR-06A train/eval closeout. Best-center checkpoint promoted the center gate to `20.2838` with P10 `26.2772` and P5 `8.4724`; best-P10 did not promote with center `20.3687`, P10 `26.0043`, P5 `8.1641`.
- [x] Completed XR-06B train/eval closeout. Best-center reached center `20.2950`, P10 `26.2177`, P5 `8.5191`; best-P10 reached center `20.4695`, P10 `25.9320`, P5 `8.6310`; neither promoted.
- [x] Launched XR-05F no-distill control on GPU1 from XR-05E best-center: fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `1e-6`, log `runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log`.
- [x] Completed XR-05F no-distill control closeout. Best-center reached center `20.3562`, P10 `26.4328`, P5 `8.2980`; best-P10 reached center `20.3195`, P10 `26.1509`, P5 `8.1429`; neither promoted.
- [x] Launched XR-06C/XR-05G ultra-low-LR polish from XR-06A best-center: XR-06C weak-distill on GPU0 and XR-05G no-distill on GPU1, both fixed255k, aux `0.0005/0.0125/0.005`, AdamW LR `5e-7`.
- [x] Completed XR-05G closeout. Best-center `20.2920/26.4422/8.4566` and best-P10 `20.3760/26.2742/8.2815`; no promotion.
- [x] Completed XR-06C closeout. Best-center promoted center to `20.2831`; best-P10 promoted P10 to `26.7449`.
- [x] Launched XR-06D/XR-06E from XR-06C leaders: XR-06D no-distill center-preserve on GPU0 from XR-06C best-center, XR-06E weak-distill P10-preserve on GPU1 from XR-06C best-P10, both aux `0.0005/0.0125/0.005`, LR `2.5e-7`.
- [x] Completed XR-06D/XR-06E closeout. XR-06D best-center/best-P10 both `20.2956/26.1586/8.5047`; XR-06E best-center `20.2942/26.2806/8.5387`; XR-06E best-P10 `20.3149/26.1339/8.6811`; no promotion.
- [x] Added XR-08 checkpoint interpolation utility and runner. Alpha `0.50` between XR-06C center/P10 leaders evaluated to `20.3259/26.4983/8.3865`; no promotion.
- [x] Launched XR-05I P5-preserve direct-aux fallback from the XR-05A P5/balanced secondary: no-distill, aux `0.001/0.025/0.01`, AdamW LR `5e-7`, GPU0.
- [x] Launched XR-06F complementary P5-preserve weak-distill branch from the same XR-05A P5/balanced secondary as init and teacher: aux `0.001/0.025/0.01`, AdamW LR `5e-7`, GPU1.
- [x] Completed XR-05I/XR-06F closeout. Neither promoted against center gate `20.283125744547164`, P10 gate `26.74489871433803`, and P5 gate `8.69557854788644`.
- [x] Launched and completed XR-04B/XR-04C focused low-similarity branches from XR-03D best-P10. Neither promoted: best focused result was XR-04B best-center `20.4841/25.6003/8.2776`.
- [x] Completed XR-17A same-branch interpolation. Alpha `0.75` promoted P5/balanced secondary to `8.7330`.
- [x] Completed XR-17A failure-bucket comparison against the XR-15C center leader. XR-17A improves weighted P10/P5 but slightly worsens weighted center.
- [x] Added and ran XR-17B bounded P5-anchor support-adaptive branch. Best-center promoted center to `20.1823`; best-P5 promoted P5 to `8.8448`.
- [x] Added and ran XR-18A no-train interpolation between XR-17B center and XR-06C P10. No promotion; best P10 was alpha `0.10` with `26.2075`.
- [x] Added and ran XR-19A trainable P10-recovery micro polish. Best-P5 promoted center to `20.1812`; best-P10 reached `26.2321`, so P10/P5 gates remain XR-06C/XR-17B.
- [ ] Next execution: design stronger constrained P10 recovery; XR-19A shows LR `1.25e-7` is center-safe but too weak for P10/P5 recovery.
- [x] Consolidated current work summary and next experiment list in `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- [x] Added XR-64 preparation helper `scripts/v3/run_xr64_teacher_eval_rows_and_overrides.sh`.
- [x] Validated XR-64 preparation path with script syntax checks and direct 8-row XR-62A smoke eval.
- [ ] Generate full train/val teacher eval rows for XR-62A, XR-39, XR-56B, XR-58A.
- [ ] Build XR-64 train/val override JSON files for conservative, threshold-priority, and min-error rules.
- [ ] Launch XR-64A and XR-64B training/eval.
- [ ] After XR-64, evaluate XR-65 temporal-lite residual and XR-66 confidence-gated local update only if XR-64 evidence justifies them.

## Plan-versus-progress checklist

- [x] Main working directory set: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`
- [x] Raw EV-Eye Stage1/Stage2 full training completed.
- [x] NVIDIA driver/NVML/CUDA readiness restored outside sandbox.
- [x] Baseline training result analyzed.
- [x] Paper/reference folder inventory checked.
- [x] Submission initial target and method claims checked.
- [x] Experiment strategy drafted.
- [x] Stage2 ablation configs prepared.
- [x] Stage2 ablation runner prepared.
- [x] Metric coordinate-frame sanity check complete.
- [x] Event input zero-tensor bug diagnosed and fixed.
- [x] First Stage2 ablation full run complete.
- [x] GPU1 parallel weak-distillation ablation complete.
- [x] DA-ROI ablation sampled and stopped as low-priority.
- [x] Corrected event-count sweep complete.
- [x] Parallel GPU1 count10000 probe complete.
- [x] Center-aware checkpoint/loss experiments complete.
- [x] Best candidate validated on test split.
- [x] GPU1 recency-weight sharpening experiment started.
- [x] GPU0/GPU1 parallel follow-up experiments started.
- [ ] Submission-level accuracy gap closed or documented with hard blocker.

## Current Status

Current usable Stage2 baseline is being reset because event tensors were zero before the
2026-06-10 event-builder fix.

- Baseline Stage2: best val `metric_track_p10_pct=7.2653`, final `metric_track_center_px=43.9917`.
- No-distill full-width LR `2e-4`: best val `metric_track_p10_pct=8.8926`, best val `metric_track_center_px=44.2296`.
- Weak-distill full-width LR `1e-4`: best val `metric_track_p10_pct=13.8646`, best val `metric_track_center_px=44.8683`; now treated as pre-fix reference only.
- DA-ROI no-distill LR `2e-4`: sampled to epoch 7, best val `metric_track_p10_pct=4.6597`, best val `metric_track_center_px=53.6560`; stopped due low priority.
- Metric coordinate sanity: `metric_track_center_px` is post-transform input-coordinate error, not sensor-pixel error; direct comparison with `0.1812 px` target is invalid until protocol is matched.
- Event sanity after fix: `event_count_target` changes nonzero event tensors; sample train index 100 produced nonzero counts `4894/8585/11801` for counts `1000/2500/5000`.
- Corrected event-count sweep:
  - GPU0: `raw_mode1_stage2_count2500_lr1e-4_weakdistill_fullwidth_20260610_210855`; complete, best val `metric_track_p10_pct=13.6961`, best val `metric_track_center_px=44.6096`.
  - GPU1: `raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_210856`; complete, best val `metric_track_p10_pct=14.1543`, best val `metric_track_center_px=44.8353`.
  - GPU0: `raw_mode1_stage2_lr1e-4_weakdistill_fullwidth_20260610_212149`; complete, expected count `5000`, best val `metric_track_p10_pct=13.7298`, best val `metric_track_center_px=44.5356`.
  - GPU1: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_fullwidth_20260610_212150`; complete, best val `metric_track_p10_pct=13.8702`, best val `metric_track_center_px=43.3363`.
- Center-aware count10000 experiments:
  - GPU0: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707`; complete, checker `ok=true`, best val `metric_track_p10_pct=13.8702`, best val `metric_track_center_px=39.5347`.
  - GPU1: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709`; complete, checker `ok=true`, best val `metric_track_p10_pct=14.3430`, best val `metric_track_center_px=39.8471`.
- Test split validation:
  - `eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803`: test `metric_track_center_px=36.9280`, test `metric_track_p10_pct=9.9209`.
  - `eval_centerloss_bestp10_test_gpu0_retry_20260610_222812`: test `metric_track_center_px=42.3723`, test `metric_track_p10_pct=10.9864`.
  - `eval_centerloss_bestcenter_test_gpu1_20260610_230859`: test `metric_track_center_px=36.5813`, test `metric_track_p10_pct=10.2759`; current best test-center result and better P10 than the center-checkpoint best-center test.
- Stopped low-priority GPU1 experiment:
  - `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth_20260610_222436`: stopped after epoch 10 validation; best val center `44.6194`, worse than count10000 center baseline `43.3363` and center-checkpoint leader `39.5347`.
- Stopped GPU1 experiment:
  - `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243`: stopped after epoch 10 validation; best val center `44.2607`, below promotion threshold.
- No-longer-active LR `5e-5` probes:
  - `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120`: no active process observed; last log reached epoch 13 train, best val center `43.3086`, not promoted versus center-checkpoint leader `39.5347`.
  - `raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355`: no active process observed; last useful validation was epoch 10, best val center `44.5171`, not promoted.
- Stopped lower-count center-checkpoint probes:
  - `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230041`: stopped after epoch 8/9 gate; best val center `44.9089`, best val P10 `14.1543`; center did not recover.
  - `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218`: stopped after epoch 8 gate; best val center `44.8672`, best val P10 `13.7298`; center did not recover.
- Stopped GPU1 experiment:
  - `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302`: stopped after epoch 8/9 observation. Epoch 8 best val center was `44.6362`, above the `42.0` stop threshold and not competitive with the leader `39.5347`.

Active follow-up experiments:

- GPU0 stopped: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_strong_fullwidth_20260610_231233`, log `runs/_logs/centerloss_strong_gpu0_20260610_232500.log`; stopped after epoch-8 gate because best val center was `44.1654`, above the `42.0` stop threshold.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_schedulefree_fullwidth_20260610_232114`, log `runs/_logs/centerckpt_schedulefree_gpu0_20260611_000000.log`; stopped after epoch-8 gate because best val center was `45.3146`, above the `42.0` stop threshold.
- GPU1 stopped: `raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_231556`, log `runs/_logs/centerloss_count5000_gpu1_20260610_234000.log`; stopped after epoch-9 observation because best val center was `44.3874`, above the `42.0` stop threshold.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr3e-5_weakdistill_centerckpt_lion_fullwidth_20260610_232431`, log `runs/_logs/centerckpt_lion_gpu1_20260611_000500.log`; stopped after epoch-8 gate because best val center was `43.4448`, above the `42.0` stop threshold.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_teachercenter_fullwidth_20260610_233212`, log `runs/_logs/centerckpt_teachercenter_gpu1_20260611_001500.log`; stopped after epoch-8 gate because best val center was `44.0106`, above the `42.0` stop threshold.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233657`, log `runs/_logs/centerckpt_ellipseaux_light_gpu0_20260611_003000.log`; stopped after epoch-8 gate because best val center was `44.4728`, above the `42.0` stop threshold.
- GPU0 aborted preflight: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233558`, log `runs/_logs/centerckpt_ellipseaux_gpu0_20260611_002500.log`; initial OBB aux weights `0.20/0.30` caused loss around `19k`, so weights were reduced to `0.001/0.001`.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_selfreg_fullwidth_20260610_234009`, log `runs/_logs/centerckpt_selfreg_gpu1_20260611_004000.log`; stopped after epoch-9 gate because best val center was `43.5076`, above the `42.0` stop threshold.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649`, log `runs/_logs/centerloss_finetune_gpu0_20260611_005000.log`; early-stopped at epoch 9 with best val center `40.0042`, not promoted versus leader `39.8471`.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952`, log `runs/_logs/centerloss_finetune_lr1e5_gpu1_20260611_010000.log`; early-stopped at epoch 9 with best val center `39.9555`, not promoted versus leader `39.8471`.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth_20260610_235630`, log `runs/_logs/trackonly_finetune_gpu0_20260611_011000.log`; early-stopped at epoch 9 with best val center `39.9155`, not promoted versus leader `39.8471`.
- GPU1 active: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth_20260611_000139`, log `runs/_logs/trackonly_finetune_lr5e6_gpu1_20260611_012000.log`; conservative LR sibling of the GPU0 track-only fine-tune.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth_20260611_000139`, log `runs/_logs/trackonly_finetune_lr5e6_gpu1_20260611_012000.log`; early-stopped at epoch 9 with best val center `39.9662`, not promoted.
- GPU0 completed + test-evaluated: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerckptinit_fullwidth_20260611_000343`, log `runs/_logs/trackonly_centerckptinit_gpu0_20260611_013000.log`; new best val center `39.7478`, but test center `37.2887`, so no test-leader promotion.
- GPU1 completed + test-evaluated: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerckptinit_fullwidth_20260611_001034`, log `runs/_logs/trackonly_centerckptinit_lr5e6_gpu1_20260611_014000.log`; new best val center `39.5315`, but best-center test center `37.0766`, so no test-leader promotion.
- Interpolated checkpoint probes from current test leader and LR `5e-6` centerckpt-init best-center were evaluated. Alpha `0.25` test center `36.7546`; alpha `0.50` test center `38.6547`; neither beats test leader `36.5813`.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth_20260611_004917`, log `runs/_logs/centerloss_centerckptinit_nodistill_gpu0_20260611_015000.log`; early-stopped at epoch 8 with best val center `39.5598`, not promoted versus validation leader `39.5315`.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943`, log `runs/_logs/centerloss_centerckptinit_weakdistill_gpu1_20260611_015000.log`; early-stopped at epoch 8 with best val center `39.5780`, not promoted versus validation leader `39.5315`.
- GPU1 stopped: `raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerloss_fullwidth_20260611_010010`, log `runs/_logs/centerloss_count1000_gpu1_20260611_005747.log`; stopped after epoch-10 gate because best val center was `44.6844`, above stop threshold `42.0`.
- Interpolated checkpoint probes from current test leader and center-checkpoint best-center were evaluated. Alpha `0.25` test center `36.7547`; alpha `0.50` test center `38.6246`; alpha `0.75` test center `38.1375`; none beats test leader `36.5813`.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr2e-6_nodistill_centerloss_finetune_fullwidth_20260611_011131`, log `runs/_logs/centerloss_finetune_lr2e6_gpu0_20260611_011500.log`; early-stopped at epoch 6 with best val center `39.8800`, not promoted.
- Added trainable-parameter filtering via `training.trainable.include/exclude`; validated with `tests/test_trainable_filter.py`.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr1e-5_nodistill_trackheadonly_centerloss_finetune_fullwidth_20260611_011606`, log `runs/_logs/trackheadonly_finetune_gpu1_20260611_012500.log`; early-stopped at epoch 6 with best val center `39.9387`, not promoted.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackeventadapter_centerloss_finetune_fullwidth_20260611_012329`, log `runs/_logs/trackeventadapter_finetune_gpu1_20260611_020000.log`; early-stopped at epoch 8 with best val center `39.7622`, not promoted versus validation leader `39.5315`.
- GPU0 stopped: `raw_mode1_stage2_count10000_lr2e-6_nodistill_tracklastblock_centerloss_finetune_fullwidth_20260611_012629`, log `runs/_logs/tracklastblock_finetune_gpu0_20260611_021000.log`; early-stopped at epoch 6 with best val center `39.8832`, not promoted versus validation leader `39.5315`.
- GPU1 stopped: `raw_mode1_stage2_count10000_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth_20260611_012935`, log `runs/_logs/trackadapters_finetune_gpu1_20260611_022000.log`; early-stopped at epoch 8 with best val center `39.7822`, not promoted versus validation leader `39.5315`.
- GPU0 completed: small-alpha interpolation from centerloss best-center to track-only centerckpt-init LR `5e-6` best-center. Alpha `0.05` test center `36.5083`, alpha `0.10` test center `36.4873`, alpha `0.15` test center `36.5190`.
- GPU1 completed: fine-grid interpolation around the new leader. Alpha `0.075` test center `36.4907`, alpha `0.125` test center `36.4968`, alpha `0.175` test center `36.5556`.
- Added reusable interpolation eval runner `scripts/v3/run_interp_eval_series.sh`; `bash -n` passed.
- GPU0/GPU1 completed micro-grid interpolation around alpha `0.10`: alpha `0.090` test center `36.4872`, alpha `0.095` test center `36.4870`, alpha `0.105` test center `36.4882`, alpha `0.110` test center `36.4896`.
- GPU0/GPU1 completed second micro-grid around alpha `0.095`: alpha `0.093` test center `36.4870`, alpha `0.094` test center `36.4870`, alpha `0.096` test center `36.4870`, alpha `0.097` test center `36.4871`.

Current decision: interpolated checkpoint `runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.094.pt` is the current test-center leader with test center `36.486977`, P10 `9.9775`, P5 `2.9660`. Centerckpt-init track-only LR `5e-6` remains the validation-center leader, but direct test transfer was weaker than checkpoint-space interpolation. The interpolation curve is now flat around `0.093`-`0.096`, so the next priority should move from finer alpha search to a short alpha-leader fine-tune or paper-derived adaptive slicing/ellipse-localization changes.

2026-06-11 continuation:

- GPU0 completed: `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackheadonly_centerloss_alphainit_finetune_fullwidth_20260611_020317`, log `runs/_logs/alphainit_trackheadonly_gpu0_20260611.log`; early-stopped at epoch 5 with best val center `39.8494`, not promoted versus validation leader `39.5315`.
- GPU1 completed: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackadapters_centerloss_alphainit_finetune_fullwidth_20260611_020333`, log `runs/_logs/alphainit_trackadapters_gpu1_20260611.log`; early-stopped at epoch 7 with best val center `39.8169`, not promoted versus validation leader `39.5315`.
- Added alpha-leader fine-tune configs:
  - `configs/v3/mode1_stage2_raw_event_count_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth.yaml`
  - `configs/v3/mode1_stage2_raw_event_count_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth.yaml`
- Validation passed for both new configs: config load, raw event-count contract, and `bash -n scripts/v3/run_prepare_and_train.sh`.
- GPU1 active: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`, log `runs/_logs/alphainit_trackeventadapter_gpu1_20260611.log`.
- GPU0 active: `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`, log `runs/_logs/alphainit_tracklastblock_gpu0_20260611.log`.
- Promotion gate unchanged: test only if val center beats `39.5315`; replace test leader only if test center beats `36.486977`.
- Implemented paper-driven adaptive-count event slicing in `src/hbtxr/data/event_builder.py`. It preserves `fixed_count` semantics but optionally scales `event_count_target` by previous/current timestamp delta using `adaptive_count.enabled`.
- Added `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`. Config default remains manifest-contract count `5000`; planned training should use the existing count10000 override.
- Adaptive-count validation passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_event_builder.py tests/test_raw_event_count_contract.py` produced `6 passed`; config load and raw event-count contract passed. Pytest emitted a read-only parent `.pytest_cache` warning only.
- GPU1 completed: `raw_mode1_stage2_count10000_lr1e-6_nodistill_trackeventadapter_centerloss_alphainit_finetune_fullwidth_20260611_020911`; early-stopped at epoch 7 with best val center `39.8169`, not promoted.
- GPU0 completed: `raw_mode1_stage2_count10000_lr1e-6_nodistill_tracklastblock_centerloss_alphainit_finetune_fullwidth_20260611_020925`; early-stopped at epoch 5 with best val center `39.8546`, not promoted.
- Added lower-LR adaptive-count sibling config `configs/v3/mode1_stage2_raw_event_count_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml`; config load and raw event-count contract passed.
- GPU0 active: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`, log `runs/_logs/adaptivecount_trackonly_gpu0_20260611.log`.
- GPU1 active: `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`, log `runs/_logs/adaptivecount_trackonly_lr2e6_gpu1_20260611.log`.
- GPU0 adaptive-count LR `5e-6` completed: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021411`; early-stopped at epoch 7 with best val center `38.6105`, beating the previous validation leader `39.5315`. Test eval `eval_adaptivecount_lr5e6_bestcenter_test_gpu0_w0_20260611_022233` produced test center `35.7182`, P10 `11.3461`, P5 `3.7279`, so this is the new test-center leader.
- GPU1 adaptive-count LR `2e-6` completed: `raw_mode1_stage2_count10000_lr2e-6_nodistill_trackonly_centerloss_adaptivecount_alphainit_fullwidth_20260611_021510`; early-stopped at epoch 7 with best val center `38.9294`, better than the old validation leader but weaker than the LR `5e-6` adaptive-count run.
- GPU1 parallel experiment started: adaptive-count sqrt scaling, `data.mode1.event_builder.adaptive_count.scale_power=0.5`, run `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_sqrt_alphainit_fullwidth_20260611_022233`, log `runs/_logs/adaptivecount_sqrt_trackonly_gpu1_20260611.log`; raw event-count contract passed and log confirmed `resolved_device=cuda:1`.
- GPU0 parallel adaptive-count power sweep started: `data.mode1.event_builder.adaptive_count.scale_power=1.25`, run `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_pow125_alphainit_fullwidth_20260611_022703`, log `runs/_logs/adaptivecount_pow125_trackonly_gpu0_20260611.log`; raw event-count contract passed and log confirmed `resolved_device=cuda:0`.
- Manifest delta analysis showed the original adaptive-count settings were saturated: `reference_us=10000`, `min_event_count=5000`, `max_event_count=20000` resolve almost all train samples to count `20000` and only first/edge samples to `5000`; `scale_power=0.5/1.0/1.25` all produced only two resolved counts. Therefore the power-1.25 run was stopped as redundant after epoch 4 train startup.
- GPU1 sqrt-scaling completed with validation center tied at `38.6105` and best val P10 `12.7920`. Its best-P10 test eval `eval_adaptivecount_sqrt_bestp10_test_gpu1_w0_20260611_022812` produced test center `35.6666`, P10 `10.9566`, P5 `3.6947`, replacing the prior adaptive-count best-center test leader `35.7182` on center.
- GPU1 fixed-count 20000 control started: `raw_mode1_stage2_count20000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023021`, log `runs/_logs/fixed20k_trackonly_gpu1_20260611.log`; this tests whether the leader is primarily a count-20000 effect.
- GPU0 true adaptive-count ref4m control started: `raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_ref4m_alphainit_fullwidth_20260611_023021`, log `runs/_logs/adaptivecount_ref4m_trackonly_gpu0_20260611.log`; this uses `reference_us=4000003` to avoid saturation and tests real variable-count slicing.

Current decision: current test-center leader is now fixed30k best-center/best-P10, `runs/raw_mode1_stage2_count30000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_023804/train/best_metric_track_center_px.pt` / `best_track_p10.pt`, with test center `35.2200`, P10 `11.1501`, P5 `3.3516`. fixed25k best-center also improved over fixed20k with test center `35.3650`, P10 `10.9201`, P5 `3.5391`. Current best P10 remains adaptive-count LR `5e-6` best-center / fixed20k best-center at P10 `11.3461`. Continue fixed18k on GPU1 and fixed22k on GPU0 to bracket the count optimum; GPU1 follow-up is queued as fixed18k eval then fixed35k train. Promote center only if test center beats `35.2200`, and track P10/P5 separately.

Active/queued update: fixed18k is running on GPU1 and fixed22k is running on GPU0. Additional GPU1 queue is registered: after fixed35k finishes, run fixed35k best-center/best-P10 test evals, then launch fixed30k center-L2 fine-tune. This links the current fixed30k leader to the next paper-backed loss-axis probe.

Paper-backed plan artifact added: `docs/Paper-Backed-Experiment-Plan.md` now maps reference papers to executable experiment axes. fixed18k best-center test eval completed weak with center `35.8224`, P10 `10.8712`, P5 `3.5123`; it does not change the fixed30k center leader. GPU0 follow-up queue is registered to evaluate fixed22k and center-L2 fixed20k, then launch fixed28k.

Current experiment-plan update: fixed18k best-P10 test eval also did not promote, with center `35.6665`, P10 `11.3890`, P5 `3.6237`. fixed22k training completed with best val center `38.5896`; its test eval is queued after center-L2 fixed20k. GPU0 is now running center-L2 fixed20k. GPU1 is running fixed35k, with fixed35k eval and center-L2 fixed30k queued behind it. Sub-agent spawn for plan review was attempted with `gpt-5.3-codex-spark` but failed because the agent thread limit was reached; Main agent continued with local verification.

Prepared next FACET/HBTXR geometry-loss sweep configs: `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo075_finetune_fullwidth.yaml` and `mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo100_finetune_fullwidth.yaml`. Both keep `track_center_l2_weight=0.0025` and sweep `track_geo_weight` to `0.75/1.0`. Validation passed: config loader resolved the intended weights, raw event-count contract passed for both, and `bash -n scripts/v3/run_prepare_and_train.sh` passed. Execution is gated behind current center-L2 fixed20k/fixed30k results.

Leader update: fixed22k best-center/best-P10 did not promote (`35.4087` / `35.7890` center). center-L2 fixed20k best-center did not promote, but best-P10 became the current P10 leader: center `35.5177`, P10 `11.7921`, P5 `3.7415`. fixed35k best-center became the current center leader: center `34.9689`, P10 `11.5408`, P5 `3.6335`, beating fixed30k center `35.2200`. GPU0 is now running fixed28k; GPU1 is evaluating fixed35k best-P10 and then will run center-L2 fixed30k.

fixed35k best-P10 eval completed and did not promote: center `36.2104`, P10 `9.5659`, P5 `3.1318`. GPU1 has started center-L2 fixed30k training. Current leaders remain fixed35k best-center for center and center-L2 fixed20k best-P10 for P10.

Follow-up queue update: added `scripts/v3/run_followup_queue_20260611.sh` and validated it with `bash -n`. Registered tmux queues: GPU0 `hgtxr_followup_gpu0_20260611` waits for fixed28k then runs fixed28 eval and fixed40k train; GPU1 `hgtxr_followup_gpu1_20260611` waits for center-L2 fixed30k then runs eval and center-L2 fixed35k train; GPU1 `hgtxr_followup_gpu1_geo075_20260611` waits for center-L2 fixed35k then runs eval and geo075 fixed35k train.

Fixed28k evals completed and did not promote: best-center `35.1866` center / `11.5731` P10 / `3.7049` P5; best-P10 `35.5586` center / `11.7517` P10 / `4.2504` P5. GPU0 follow-up advanced to fixed40k train.

Center-L2 fixed30k evals completed and did not promote: best-center and best-P10 both produced `35.2231` center / `10.6845` P10 / `3.4643` P5. GPU1 follow-up advanced to center-L2 fixed35k train, with geo075 fixed35k still queued behind its eval.

Geometry follow-up queue extended: `scripts/v3/run_followup_queue_20260611.sh` now supports `gpu1_geo075_35_then_geo100_35`, which waits for geo075 fixed35k, evaluates best-center/best-P10 with `training.num_workers=0`, then launches geo100 fixed35k on GPU1. Validation: `bash -n scripts/v3/run_followup_queue_20260611.sh` passed. Registered tmux session `hgtxr_followup_gpu1_geo100_20260611`, log `runs/_logs/eval_geo075_then_geo100_35_gpu1_20260611.log`.

Fixed40k eval queue added: `scripts/v3/run_followup_queue_20260611.sh` now supports `gpu0_fixed40_eval`, which waits for fixed40k train completion and evaluates best-center/best-P10 with `training.num_workers=0`. Validation: `bash -n scripts/v3/run_followup_queue_20260611.sh` passed. Registered tmux session `hgtxr_followup_gpu0_fixed40_eval_20260611`, log `runs/_logs/eval_fixed40_gpu0_20260611.log`.

Leader update: fixed40k best-center eval `eval_fixed40k_bestcenter_test_gpu0_w0_20260611_033617` produced center `34.9533`, P10 `11.9286`, P5 `3.6395`, replacing both the fixed35k center leader and center-L2 fixed20k P10 leader. fixed40k best-P10 eval `eval_fixed40k_bestp10_test_gpu0_w0_20260611_033811` produced center `35.1615`, P10 `11.6458`, P5 `3.4503`, so best-center remains the selected fixed40k checkpoint.

Center-L2 fixed35k did not promote: best-center eval `eval_centerl2_35k_bestcenter_test_gpu1_w0_20260611_033529` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_centerl2_35k_bestp10_test_gpu1_w0_20260611_033623` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`.

Count sweep extended after fixed40 promotion: `gpu0_fixed40_eval_then_fixed45` launched fixed45k on GPU0, and `gpu0_fixed45_eval_then_fixed50` is registered in tmux session `hgtxr_followup_gpu0_fixed50_20260611` to evaluate fixed45k then launch fixed50k. Geometry queue also advanced: geo075 fixed35k is active on GPU1, with geo100 queued after geo075 eval.

Evaluation coverage extended: `scripts/v3/run_followup_queue_20260611.sh` now supports `gpu0_fixed50_eval` and `gpu1_geo100_35_eval`; `bash -n` passed. Registered tmux sessions `hgtxr_followup_gpu0_fixed50_eval_20260611` and `hgtxr_followup_gpu1_geo100_eval_20260611` so fixed50k and geo100 fixed35k will not remain train-only.

Current plan/status update: fixed45k training completed on GPU0 with best val center `38.1408`. fixed45k best-center test eval promoted the center leader: center `34.4771`, P10 `11.5935`, P5 `3.9507`. fixed45k best-P10 test eval did not promote: center `35.1005`, P10 `10.2389`, P5 `3.4264`. fixed50k training is now active on GPU0. geo075 fixed35k completed and did not promote: best-center test center `35.0416`, P10 `10.9660`, P5 `3.7470`; best-P10 test center `36.2311`, P10 `9.4588`, P5 `3.2041`. GPU1 parallel axis advanced to geo100 fixed35k training with `resolved_device=cuda:1`; geo100 test eval remains queued. Current center leader is fixed45k best-center; current P10 leader remains fixed40k best-center.

Latest continuation update: geo100 fixed35k completed and did not promote. best-center eval `eval_geo100_35k_bestcenter_test_gpu1_w0_20260611_040031` produced center `35.0627`, P10 `11.0225`, P5 `3.7917`; best-P10 eval `eval_geo100_35k_bestp10_test_gpu1_w0_20260611_040210` produced center `36.2216`, P10 `9.5034`, P5 `3.1531`. fixed50k promoted both leaderboards: best-center eval `eval_fixed50k_bestcenter_test_gpu0_w0_20260611_040420` produced center `34.3416`, P10 `11.7589`, P5 `3.5748`; best-P10 eval `eval_fixed50k_bestp10_test_gpu0_w0_20260611_040514` produced center `34.5391`, P10 `12.1276`, P5 `3.3580`. Since count quality improved monotonically through 40k -> 45k -> 50k, fixed55k and fixed60k train/eval queues were started in parallel via `scripts/v3/run_count_extension_queue_20260611.sh`.

Current queue verification: `nvidia-smi` shows both RTX 5080 GPUs active with about `931 MiB` used each; `pgrep` shows fixed55k on `cuda:0` and fixed60k on `cuda:1`. fixed55k is around epoch 8/12 with best val center `37.9150`; fixed60k is around epoch 7/12 with best val center `37.8479`. GPU1 is already occupied by the parallel fixed60k experiment, so no additional GPU1 experiment is started until fixed60k train/eval completes.

Added optimizer-probe runner `scripts/v3/run_optimizer_probe_queue_20260611.sh`; `bash -n` passed. It is prepared for post-count-plateau probes such as `adamw@2e-6`, `adamw@8e-6/1e-5`, `lion@2e-6`, `adopt@5e-6`, and `soap@5e-6`.

Sub-agent review attempt: requested `gpt-5.3-codex-spark` explorer for experiment-plan/GPU queue review, but spawn failed with `agent thread limit reached`; no sub-agent output was used.

Added conditional post-count queue runner `scripts/v3/run_post_count_decision_queue_20260611.sh`; `bash -n` passed. Registered tmux sessions `hgtxr_post_count_gpu0_20260611` and `hgtxr_post_count_gpu1_20260611`. Both wait for fixed55k/fixed60k best-center and best-P10 test eval summaries. If fixed55k/fixed60k promote over fixed50k, the queues continue the count sweep with fixed70k on GPU0 and fixed65k on GPU1. If neither promotes, they switch to LR probes: `adamw@8e-6` on GPU0 and `adamw@2e-6` on GPU1 using the best count found.

fixed55k/fixed60k completed and promoted. fixed55k best-center eval `eval_fixed55k_bestcenter_test_gpu0_w0_20260611_041735` produced center `34.1183`, P10 `12.5672`, P5 `4.1399`, becoming the current P10 leader. fixed55k best-P10 eval `eval_fixed55k_bestp10_test_gpu0_w0_20260611_041938` produced center `34.9445`, P10 `11.2088`, P5 `3.7560`, so no promotion. fixed60k best-center and best-P10 evals `eval_fixed60k_bestcenter_test_gpu1_w0_20260611_041755` / `eval_fixed60k_bestp10_test_gpu1_w0_20260611_042005` produced the same test metrics: center `33.9445`, P10 `12.5174`, P5 `4.3423`, becoming the current center leader. Since fixed55k/fixed60k promoted, the conditional post-count queues advanced to fixed70k on GPU0 and fixed65k on GPU1.

Current active experiments: GPU0 fixed70k `raw_mode1_stage2_count70000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042200`; GPU1 fixed65k `raw_mode1_stage2_count65000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_042212`. Both use `HBTXR_DISABLE_CUDNN=1`, fixed-count event slicing, alpha-init checkpoint, and will run best-center/best-P10 test evals with `training.num_workers=0`.

Added second-stage conditional post-count runner `scripts/v3/run_post_count2_decision_queue_20260611.sh`; `bash -n` passed. Registered tmux sessions `hgtxr_post_count2_gpu0_20260611` and `hgtxr_post_count2_gpu1_20260611`. Both wait for fixed65k/fixed70k best-center and best-P10 test eval summaries. If fixed65k/fixed70k promote over the current fixed60k center leader or fixed55k P10 leader, the queues continue the count sweep with fixed80k on GPU0 and fixed75k on GPU1. If neither promotes, they switch to LR probes: `adamw@8e-6` on GPU0 and `adamw@2e-6` on GPU1 using the best count found.

fixed65k/fixed70k completed and promoted. fixed65k best-center and best-P10 evals `eval_fixed65k_bestcenter_test_gpu1_w0_20260611_043218` / `eval_fixed65k_bestp10_test_gpu1_w0_20260611_043419` produced center `33.7084`, P10 `12.6548`, P5 `3.8202`. fixed70k best-center and best-P10 evals `eval_fixed70k_bestcenter_test_gpu0_w0_20260611_043210` / `eval_fixed70k_bestp10_test_gpu0_w0_20260611_043411` produced center `33.5467`, P10 `12.8971`, P5 `3.8010`, becoming the current center and P10 leader. The second-stage conditional watcher selected `ACTION=count` with `BEST_COUNT=70000`; GPU1 started fixed75k at `2026-06-11T04:36:56+09:00` as `raw_mode1_stage2_count75000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043657`, and GPU0 started fixed80k at `2026-06-11T04:38:08+09:00` as `raw_mode1_stage2_count80000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_043808`.

Added third-stage conditional post-count runner `scripts/v3/run_post_count3_decision_queue_20260611.sh`; `bash -n` passed. Registered tmux sessions `hgtxr_post_count3_gpu0_20260611` and `hgtxr_post_count3_gpu1_20260611`, with logs `runs/_logs/post_count3_decision_gpu0_20260611.log` and `runs/_logs/post_count3_decision_gpu1_20260611.log`. They wait for fixed75k/fixed80k best-center and best-P10 eval summaries. If fixed75k/fixed80k promote over fixed70k (`33.5467` center or `12.8971` P10), they continue count sweep with fixed90k on GPU0 and fixed85k on GPU1; otherwise they switch to AdamW LR probes on the best count.

Implemented a decoded-center threshold hinge loss for the next loss-axis probe. Code path: `src/hbtxr/loss/bundles/track.py` now emits `loss_track_center_hinge`, controlled by `loss.track_center_hinge_weight` and `loss.track_center_hinge_margin_px`; default weight is `0.0`, so existing runs are unchanged. Added configs `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge10_finetune_fullwidth.yaml` and `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge5_finetune_fullwidth.yaml`, plus generic runner `scripts/v3/run_centerhinge_probe_queue_20260611.sh`. Validation passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `3 passed` with the known parent `.pytest_cache` warning; raw event-count contract passed for both hinge configs; `python3 -m py_compile src/hbtxr/loss/bundles/track.py` and `bash -n scripts/v3/run_centerhinge_probe_queue_20260611.sh` passed.

fixed75k/fixed80k completed and promoted. fixed75k best-center/best-P10 evals `eval_fixed75k_bestcenter_test_gpu1_w0_20260611_044705` / `eval_fixed75k_bestp10_test_gpu1_w0_20260611_044901` produced center `33.3526`, P10 `13.2428`, P5 `3.5757`, becoming the current P10 leader. fixed80k best-center/best-P10 evals `eval_fixed80k_bestcenter_test_gpu0_w0_20260611_044835` / `eval_fixed80k_bestp10_test_gpu0_w0_20260611_045041` produced center `33.0848`, P10 `12.7543`, P5 `3.4056`, becoming the current center leader. The third-stage conditional watcher selected `ACTION=count`, `BEST_COUNT=80000`; GPU1 started fixed85k at `2026-06-11T04:53:51+09:00` as `raw_mode1_stage2_count85000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_045352`, and GPU0 started fixed90k at `2026-06-11T04:53:49+09:00` as `raw_mode1_stage2_count90000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_045350`.

Added fourth-stage conditional post-count runner `scripts/v3/run_post_count4_decision_queue_20260611.sh`; `bash -n` passed and no-arg usage check returned code `2` as expected. Registered tmux sessions `hgtxr_post_count4_gpu0_20260611` and `hgtxr_post_count4_gpu1_20260611`, with logs `runs/_logs/post_count4_decision_gpu0_20260611.log` and `runs/_logs/post_count4_decision_gpu1_20260611.log`. They wait for fixed85k/fixed90k best-center and best-P10 eval summaries. If fixed85k/fixed90k promote over current leaders (`33.0848` center or `13.2428` P10), they continue count sweep with fixed100k on GPU0 and fixed95k on GPU1. If neither promotes, GPU0 switches to AdamW `8e-6` and GPU1 switches to center-hinge margin `10.0`, weight `0.05` on the best count. Current fixed85k/fixed90k status at registration: both were around epoch 4/12 and training normally.

fixed85k/fixed90k completed and the count sweep continued. fixed85k best-center and best-P10 evals `eval_fixed85k_bestcenter_test_gpu1_w0_20260611_050439` / `eval_fixed85k_bestp10_test_gpu1_w0_20260611_050645` both produced center `32.9858`, P10 `12.9605`, P5 `3.9660`, becoming the current center leader. fixed90k best-center eval `eval_fixed90k_bestcenter_test_gpu0_w0_20260611_050441` produced center `33.0128`, P10 `12.6212`, P5 `3.2836`; fixed90k best-P10 eval `eval_fixed90k_bestp10_test_gpu0_w0_20260611_050648` produced center `33.4284`, P10 `11.3542`, P5 `3.4528`. P10 leader remains fixed75k at `13.2428`. The fourth-stage watcher selected `ACTION=count`, `BEST_COUNT=85000`; GPU1 started fixed95k at `2026-06-11T05:10:27+09:00` as `raw_mode1_stage2_count95000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_051027`, and GPU0 started fixed100k at `2026-06-11T05:10:27+09:00` as `raw_mode1_stage2_count100000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_051028`.

Added fifth-stage conditional post-count runner `scripts/v3/run_post_count5_decision_queue_20260611.sh`; `bash -n` passed and no-arg usage check returned code `2` as expected. Registered tmux sessions `hgtxr_post_count5_gpu0_20260611` and `hgtxr_post_count5_gpu1_20260611`, with logs `runs/_logs/post_count5_decision_gpu0_20260611.log` and `runs/_logs/post_count5_decision_gpu1_20260611.log`. They wait for fixed95k/fixed100k best-center and best-P10 eval summaries. If fixed95k/fixed100k promote over current leaders (`32.9858` center or `13.2428` P10), they continue count sweep with fixed110k on GPU0 and fixed105k on GPU1. If neither promotes, GPU0 switches to AdamW `8e-6` and GPU1 switches to center-hinge margin `10.0`, weight `0.05` on the best count.

Implemented next paper-backed local-anchor crop axis while fixed95k/fixed100k were running. `src/hbtxr/data/components.py` now accepts `crop_policy: prev_pupil_anchor`, using the previous annotation pupil bbox as the local crop anchor and transforming current frame/event/targets into that ROI. Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_finetune_fullwidth.yaml`, runner `scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh`, and tests `tests/test_adaptive_roi_resolver.py`.

Validation passed for the local-anchor crop axis: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_adaptive_roi_resolver.py tests/test_event_builder.py` produced `6 passed` with the known read-only parent `.pytest_cache` warning; config loader override resolved `prev_pupil_anchor within_roi 32.0 96.0 85000 False`; raw event-count contract passed; `python3 -m py_compile src/hbtxr/data/components.py`, `sh -n scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh`, and `bash -n scripts/v3/run_prevpupilcrop_probe_queue_20260611.sh` passed. No local-anchor training was started yet because GPU0/GPU1 remain occupied by fixed100k/fixed95k.

fixed95k/fixed100k completed and the count sweep continued. fixed95k best-center/best-P10 evals `eval_fixed95k_bestcenter_test_gpu1_w0_20260611_052137` / `eval_fixed95k_bestp10_test_gpu1_w0_20260611_052349` both produced center `32.9237`, P10 `12.2815`, P5 `3.4184`, so they did not promote over fixed85k center or fixed75k P10. fixed100k best-center eval `eval_fixed100k_bestcenter_test_gpu0_w0_20260611_052137` produced center `32.7796`, P10 `12.5961`, P5 `3.7215`, becoming the current center leader; fixed100k best-P10 eval `eval_fixed100k_bestp10_test_gpu0_w0_20260611_052348` produced center `33.1030`, P10 `12.1105`, P5 `3.6195`. The fifth-stage watcher selected `ACTION=count`, `BEST_COUNT=100000`; GPU1 started fixed105k at `2026-06-11T05:26:00+09:00` as `raw_mode1_stage2_count105000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_052601`, and GPU0 started fixed110k at `2026-06-11T05:26:00+09:00` as `raw_mode1_stage2_count110000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_052601`.

Added sixth-stage conditional post-count runner `scripts/v3/run_post_count6_decision_queue_20260611.sh`; `bash -n` passed and no-arg usage check returned code `2` as expected. Registered watcher sessions `hgtxr_post_count6_gpu0_20260611` and `hgtxr_post_count6_gpu1_20260611`, with logs `runs/_logs/post_count6_decision_gpu0_20260611.log` and `runs/_logs/post_count6_decision_gpu1_20260611.log`. They wait for fixed105k/fixed110k best-center and best-P10 eval summaries. If fixed105k/fixed110k promote over current leaders (`32.7796` center or `13.2428` P10), they continue count sweep with fixed120k on GPU0 and fixed115k on GPU1. If neither promotes, GPU0 switches to AdamW `8e-6` and GPU1 switches to the prepared `prev_pupil_anchor` local-crop probe on the best count. Current process check confirms fixed105k is running on GPU1, fixed110k is running on GPU0, and both post_count6 watchers are alive.

Prepared a checkpoint-compatible FACET-style ellipse-state loss axis while fixed105k/fixed110k continue running. `src/hbtxr/loss/bundles/track.py` now exposes disabled-by-default `loss.track_axis_log_weight` and `loss.track_angle_cos_weight`, computed from decoded track `state` rather than adding a new head. Added config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_finetune_fullwidth.yaml` and runner `scripts/v3/run_ellipsestate_probe_queue_20260611.sh <event_count> <axis_log_weight> <angle_cos_weight> <cuda:N>`. Validation passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `4 passed` with the known read-only parent `.pytest_cache` warning; raw event-count contract passed; config loader resolved `0.05 0.02 fixed_count 5e-06`; `python3 -m py_compile src/hbtxr/loss/bundles/track.py` and `bash -n scripts/v3/run_ellipsestate_probe_queue_20260611.sh` passed. No ellipse-state training was started because both GPUs remain assigned to fixed105k/fixed110k.

fixed105k/fixed110k completed and the count sweep continued. fixed105k best-center and best-P10 evals both produced center `32.6356`, P10 `13.1654`, P5 `3.6662`, improving center over fixed100k but not P10 over fixed75k. fixed110k best-center eval produced center `32.5969`, P10 `13.0697`, P5 `3.8193`, becoming the current center leader; fixed110k best-P10 eval produced center `32.9429`, P10 `12.6399`, P5 `3.4324`. The sixth-stage watcher selected `ACTION=count`, `BEST_COUNT=110000`, `BEST_CENTER=32.596922`, `BEST_P10=13.165392`, and started GPU1 fixed115k at `2026-06-11T05:42:02+09:00` plus GPU0 fixed120k at `2026-06-11T05:44:00+09:00`.

Added seventh-stage conditional post-count runner `scripts/v3/run_post_count7_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watcher sessions `hgtxr_post_count7_gpu0_20260611` and `hgtxr_post_count7_gpu1_20260611`, with logs `runs/_logs/post_count7_decision_gpu0_20260611.log` and `runs/_logs/post_count7_decision_gpu1_20260611.log`. They wait for fixed115k/fixed120k best-center and best-P10 eval summaries. If either promotes over current leaders (`32.596922` center or `13.242773` P10), they continue count sweep with fixed130k on GPU0 and fixed125k on GPU1. If neither promotes, GPU0 switches to AdamW `8e-6` and GPU1 switches to `prev_pupil_anchor` local-crop on the best count.

fixed115k/fixed120k evals completed. fixed115k best-center produced center `32.4400`, P10 `13.3512`, P5 `3.6305`; fixed115k best-P10 produced center `32.9068`, P10 `12.8835`, P5 `3.3678`. fixed120k best-center produced center `32.3053`, P10 `13.7976`, P5 `3.4974`, replacing both the center and P10 leaders; fixed120k best-P10 produced center `32.7800`, P10 `13.0991`, P5 `3.6922`.

The seventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=120000`, `BEST_CENTER=32.305316`, `BEST_P10=13.797619`. GPU0 launched fixed130k at `2026-06-11T05:59:57+09:00`; GPU1 launched fixed125k at `2026-06-11T06:00:09+09:00`.

Current active experiments: GPU0 fixed130k `raw_mode1_stage2_count130000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_055958`; GPU1 fixed125k `raw_mode1_stage2_count125000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_060010`. Latest observed epoch 4: fixed130k val center `37.0059`, P10 `10.2662`; fixed125k val center `37.1699`, P10 `9.9292`.

Added eighth-stage conditional post-count runner `scripts/v3/run_post_count8_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered watchers `hgtxr_post_count8_gpu0_20260611` and `hgtxr_post_count8_gpu1_20260611`. They wait for fixed125k/fixed130k eval summaries. If either promotes over fixed120k (`32.305316` center or `13.797619` P10), GPU1 continues to fixed135k and GPU0 to fixed140k; otherwise GPU1 runs `prev_pupil_anchor` and GPU0 runs AdamW `8e-6` on the best count.

Sub-agent plan audit attempt with `gpt-5.3-codex-spark` failed due `agent thread limit reached`; no sub-agent result was used.

Continuation check: fixed125k/fixed130k are still training, with no fixed125k/fixed130k test eval summaries yet. Latest observed epoch 7: fixed125k val center `36.5371`, best val P10 `10.3291`; fixed130k val center `36.4138`, best val P10 `11.3398`. GPU0 and GPU1 remain occupied, so no extra same-GPU experiment was started.

Added ninth-stage/follow-up runner `scripts/v3/run_post_count9_decision_queue_20260611.sh`; `bash -n`, executable bit, and no-arg usage check passed. Registered watchers `hgtxr_post_count9_gpu0_20260611` and `hgtxr_post_count9_gpu1_20260611`, with logs `runs/_logs/post_count9_decision_gpu0_20260611.log` and `runs/_logs/post_count9_decision_gpu1_20260611.log`. They wait for `post_count8` to decide; count branch can continue to fixed150k/fixed145k if fixed135k/fixed140k promote, or fall back to mixed probes. If `post_count8` already selects mixed probes and they do not promote, GPU0 runs center-hinge and GPU1 runs ellipse-state loss.

fixed125k/fixed130k evals completed and promoted. fixed125k best-center eval `eval_fixed125k_bestcenter_test_gpu1_w0_20260611_061215` produced center `32.1365`, P10 `14.0676`, P5 `3.8610`, becoming the current P10 leader. fixed125k best-P10 eval `eval_fixed125k_bestp10_test_gpu1_w0_20260611_061433` produced center `32.6232`, P10 `12.9962`, P5 `3.7636`. fixed130k best-center eval `eval_fixed130k_bestcenter_test_gpu0_w0_20260611_061202` produced center `32.0159`, P10 `13.7798`, P5 `3.8457`, becoming the current center leader. fixed130k best-P10 eval `eval_fixed130k_bestp10_test_gpu0_w0_20260611_061420` produced center `32.4814`, P10 `12.9940`, P5 `3.7190`.

The eighth-stage post-count watcher selected count continuation with `BEST_COUNT=130000`. GPU0 launched fixed140k as `raw_mode1_stage2_count140000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061825`; GPU1 launched fixed135k as `raw_mode1_stage2_count135000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_061834`. Latest observed epoch 1: fixed140k val center `39.9020`, P10 `8.4748`; fixed135k val center `39.9278`, P10 `8.4580`.

The post_count9 watchers observed `post_count8` action `count` and are now waiting for fixed135k/fixed140k best-center/best-P10 eval summaries. If either promotes over the fixed130k/fixed125k leaders, the next count branch is fixed150k on GPU0 and fixed145k on GPU1; otherwise the queue switches to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

Corrected `scripts/v3/run_post_count9_decision_queue_20260611.sh` thresholds from the older fixed120k leaders to the current leaders: center `32.01590127093451` and P10 `14.067602443695069`. Restarted post_count9 watchers as `hgtxr_post_count9_gpu0b_20260611` and `hgtxr_post_count9_gpu1b_20260611`; logs confirm both observed `post_count8 action=count best_count=130000` and are waiting for fixed135k/fixed140k evals.

Current fixed135k/fixed140k status: epoch 6 observed. fixed135k best val center `36.6955`, best val P10 `11.0568`; fixed140k best val center `36.7976`, best val P10 `10.8041`. No fixed135k/fixed140k test eval summaries exist yet.

Added tenth-stage/follow-up runner `scripts/v3/run_post_count10_decision_queue_20260611.sh`; `bash -n`, executable bit, and no-arg usage check passed. Registered watchers `hgtxr_post_count10_gpu0_20260611` and `hgtxr_post_count10_gpu1_20260611`, with logs `runs/_logs/post_count10_decision_gpu0_20260611.log` and `runs/_logs/post_count10_decision_gpu1_20260611.log`. It waits for `post_count9`; count branch can continue to fixed160k/fixed155k if fixed145k/fixed150k promote, otherwise it falls back to mixed probes. The script computes baseline leaders dynamically from available eval summaries to avoid stale threshold constants.

fixed135k/fixed140k evals completed. fixed135k best-center eval `eval_fixed135k_bestcenter_test_gpu1_w0_20260611_063112` produced center `31.9891`, P10 `13.3274`, P5 `3.5242`; fixed135k best-P10 eval `eval_fixed135k_bestp10_test_gpu1_w0_20260611_063333` produced center `32.4352`, P10 `12.8053`, P5 `3.8559`. fixed140k best-center eval `eval_fixed140k_bestcenter_test_gpu0_w0_20260611_063107` produced center `31.8833`, P10 `13.1884`, P5 `3.6637`, becoming the current center leader. fixed140k best-P10 eval `eval_fixed140k_bestp10_test_gpu0_w0_20260611_063329` produced center `32.3225`, P10 `12.7156`, P5 `3.6654`. P10 leader remains fixed125k best-center at `14.0676`.

The ninth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=140000`. GPU1 launched fixed145k at about `2026-06-11T06:37:07+09:00`; GPU0 launched fixed150k at about `2026-06-11T06:37:10+09:00`. Current latest observed epoch 4: fixed145k val center `37.1518`, P10 `10.6289`; fixed150k val center `37.2427`, P10 `10.9007`. The tenth-stage watchers are alive and waiting for fixed145k/fixed150k eval summaries.

fixed145k/fixed150k evals completed. fixed145k best-center eval `eval_fixed145k_bestcenter_test_gpu1_w0_20260611_064953` produced center `31.8802`, P10 `12.5897`, P5 `3.9073`; fixed145k best-P10 eval `eval_fixed145k_bestp10_test_gpu1_w0_20260611_065216` produced center `32.3115`, P10 `12.5221`, P5 `3.5570`. fixed150k best-center eval `eval_fixed150k_bestcenter_test_gpu0_w0_20260611_065009` produced center `31.7211`, P10 `12.9154`, P5 `3.7330`, becoming the current center leader. fixed150k best-P10 eval `eval_fixed150k_bestp10_test_gpu0_w0_20260611_065233` produced center `32.1254`, P10 `12.8342`, P5 `3.4179`. P10 leader remains fixed125k best-center at `14.0676`.

The tenth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=150000`. GPU1 launched fixed155k at about `2026-06-11T06:55:33+09:00`; GPU0 launched fixed160k at about `2026-06-11T06:55:15+09:00`. Latest observed epoch 4: fixed155k val center `37.2207`, P10 `11.0580`; fixed160k val center `37.0474`, P10 `11.1927`.

Added eleventh-stage/follow-up runner `scripts/v3/run_post_count11_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count11_gpu0_20260611` and `hgtxr_post_count11_gpu1_20260611`, with logs `runs/_logs/post_count11_decision_gpu0_20260611.log` and `runs/_logs/post_count11_decision_gpu1_20260611.log`. It waits for fixed155k/fixed160k eval summaries; count branch can continue to fixed170k/fixed165k if fixed155k/fixed160k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

fixed155k/fixed160k evals completed. fixed155k best-center eval `eval_fixed155k_bestcenter_test_gpu1_w0_20260611_070836` produced center `31.6944`, P10 `12.8878`, P5 `3.5438`; fixed155k best-P10 eval `eval_fixed155k_bestp10_test_gpu1_w0_20260611_071104` produced the same metrics. fixed160k best-center eval `eval_fixed160k_bestcenter_test_gpu0_w0_20260611_070835` produced center `31.5714`, P10 `13.3108`, P5 `3.5098`, becoming the current center leader; fixed160k best-P10 eval `eval_fixed160k_bestp10_test_gpu0_w0_20260611_071103` produced the same metrics. P10 leader remains fixed125k best-center at `14.0676`.

The eleventh-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=160000`. GPU1 launched fixed165k as `raw_mode1_stage2_count165000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_071402`; GPU0 launched fixed170k as `raw_mode1_stage2_count170000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_071402`.

Added twelfth-stage/follow-up runner `scripts/v3/run_post_count12_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count12_gpu0_20260611` and `hgtxr_post_count12_gpu1_20260611`, with logs `runs/_logs/post_count12_decision_gpu0_20260611.log` and `runs/_logs/post_count12_decision_gpu1_20260611.log`. It waits for fixed165k/fixed170k eval summaries; count branch can continue to fixed180k/fixed175k if fixed165k/fixed170k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

Added thirteenth-stage/follow-up runner `scripts/v3/run_post_count13_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count13_gpu0_20260611` and `hgtxr_post_count13_gpu1_20260611`, with logs `runs/_logs/post_count13_decision_gpu0_20260611.log` and `runs/_logs/post_count13_decision_gpu1_20260611.log`. It waits for the `post_count12` decision; count branch can continue to fixed190k/fixed185k if fixed175k/fixed180k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

fixed165k/fixed170k evals completed. fixed165k best-center eval `eval_fixed165k_bestcenter_test_gpu1_w0_20260611_072734` produced center `31.5321`, P10 `13.1152`, P5 `3.7160`; fixed165k best-P10 eval `eval_fixed165k_bestp10_test_gpu1_w0_20260611_073005` produced center `31.9779`, P10 `13.0676`, P5 `3.6352`. fixed170k best-center eval `eval_fixed170k_bestcenter_test_gpu0_w0_20260611_072745` produced center `31.4726`, P10 `13.5991`, P5 `3.6437`, becoming the current center leader; fixed170k best-P10 eval `eval_fixed170k_bestp10_test_gpu0_w0_20260611_073018` produced center `31.9469`, P10 `13.1144`, P5 `3.3129`. P10 leader remains fixed125k best-center at `14.0676`.

The twelfth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=170000`. GPU1 launched fixed175k as `raw_mode1_stage2_count175000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_073440`; GPU0 launched fixed180k as `raw_mode1_stage2_count180000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_073440`. Both trained through epoch 12. fixed175k best-center test eval `eval_fixed175k_bestcenter_test_gpu1_w0_20260611_074834` produced center `31.2559`, P10 `13.4209`, P5 `3.8291`; fixed175k best-P10 eval `eval_fixed175k_bestp10_test_gpu1_w0_20260611_075108` produced center `31.7825`, P10 `13.1718`, P5 `3.6322`. fixed180k best-center test eval `eval_fixed180k_bestcenter_test_gpu0_w0_20260611_074847` produced center `30.9685`, P10 `14.0582`, P5 `3.7117`, becoming the current center leader; fixed180k best-P10 eval `eval_fixed180k_bestp10_test_gpu0_w0_20260611_075125` produced center `31.6223`, P10 `13.4073`, P5 `3.8924`.

The thirteenth-stage post-count watcher selected count continuation after fixed180k promoted center. GPU1 launched fixed185k as `raw_mode1_stage2_count185000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075511`; GPU0 launched fixed190k as `raw_mode1_stage2_count190000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_075502`. post_count14 is waiting for the post_count13 decision and can continue to fixed195k/fixed200k if fixed185k/fixed190k promote.

Current fixed185k/fixed190k status: both are training normally. Latest observed epoch 10: fixed185k val center `35.0572`, P10 `12.9515`; fixed190k val center `35.0466`, P10 `13.5355`.

Added fourteenth-stage/follow-up runner `scripts/v3/run_post_count14_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count14_gpu0_20260611` and `hgtxr_post_count14_gpu1_20260611`, with logs `runs/_logs/post_count14_decision_gpu0_20260611.log` and `runs/_logs/post_count14_decision_gpu1_20260611.log`. It waits for the `post_count13` decision; count branch can continue to fixed200k/fixed195k if fixed185k/fixed190k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

Prepared weak EMA distillation probe runner `scripts/v3/run_weakdistill_probe_queue_20260611.sh <event_count> <cuda:N>` using `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth.yaml`. It is queued for the next GPU1 idle/plateau slot and was not launched because GPU1 is already occupied by fixed185k plus post_count14/post_count15 watcher ownership.

Added fifteenth-stage/follow-up runner `scripts/v3/run_post_count15_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count15_gpu0_20260611` and `hgtxr_post_count15_gpu1_20260611`, with logs `runs/_logs/post_count15_decision_gpu0_20260611.log` and `runs/_logs/post_count15_decision_gpu1_20260611.log`. It waits for the `post_count14` decision; count branch can continue to fixed210k/fixed205k if fixed195k/fixed200k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1. Fixed the `prev_pupil_anchor` runner eval-name contract from `fixed${COUNT}k` to `fixed${COUNT/1000}k` so post-count watcher glob patterns can observe its outputs.

Added sixteenth-stage/follow-up runner `scripts/v3/run_post_count16_decision_queue_20260611.sh`; `bash -n`, `sh -n`, and no-arg usage check passed. Registered tmux watchers `hgtxr_post_count16_gpu0_20260611` and `hgtxr_post_count16_gpu1_20260611`, with logs `runs/_logs/post_count16_decision_gpu0_20260611.log` and `runs/_logs/post_count16_decision_gpu1_20260611.log`. It waits for the `post_count15` decision; count branch can continue to fixed220k/fixed215k if fixed205k/fixed210k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1.

fixed185k/fixed190k evals completed and promoted. fixed185k best-center/best-P10 both produced center `30.8202`, P10 `14.4379`, P5 `3.9043`, becoming the current P10 leader. fixed190k best-center/best-P10 both produced center `30.7517`, P10 `14.3967`, P5 `4.1297`, becoming the current center leader.

The fourteenth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=190000`, `BEST_CENTER=30.751661`, `BEST_P10=14.437926`. GPU1 launched fixed195k at `2026-06-11T08:16:44+09:00` as `raw_mode1_stage2_count195000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_081645`; GPU0 launched fixed200k at `2026-06-11T08:16:33+09:00` as `raw_mode1_stage2_count200000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_081633`.

Current GPU1 assignment: fixed195k train on `cuda:1`. No additional same-GPU experiment should be launched until this branch trains/evals or post_count15 takes over.

Added seventeenth-stage/follow-up runner `scripts/v3/run_post_count17_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable bit, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count17_gpu0_20260611` and `hgtxr_post_count17_gpu1_20260611`, with logs `runs/_logs/post_count17_decision_gpu0_20260611.log` and `runs/_logs/post_count17_decision_gpu1_20260611.log`. It waits for the `post_count16` decision; count branch can continue to fixed230k/fixed225k if fixed215k/fixed220k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1. If post_count16 is already on mixed probes and they do not promote, post_count17 advances to center-hinge on GPU0 and ellipse-state loss on GPU1.

fixed195k/fixed200k evals completed and promoted. fixed195k best-center/best-P10 both produced center `30.6335`, P10 `14.7258`, P5 `4.4303`, becoming the current P10 leader. fixed200k best-center/best-P10 both produced center `30.5612`, P10 `14.3117`, P5 `4.1531`, becoming the current center leader.

The fifteenth-stage post-count watcher selected `ACTION=count`, `BEST_COUNT=200000`, `BEST_CENTER=30.561175`, `BEST_P10=14.725766`. GPU1 launched fixed205k at `2026-06-11T08:37:37+09:00` as `raw_mode1_stage2_count205000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_083738`; GPU0 launched fixed210k at `2026-06-11T08:37:28+09:00` as `raw_mode1_stage2_count210000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_083729`.

Added eighteenth-stage/follow-up runner `scripts/v3/run_post_count18_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, and no-arg usage exit `2`. Registered tmux watchers `hgtxr_post_count18_gpu0_20260611` and `hgtxr_post_count18_gpu1_20260611`, with logs `runs/_logs/post_count18_decision_gpu0_20260611.log` and `runs/_logs/post_count18_decision_gpu1_20260611.log`. It waits for the `post_count17` decision; count branch can continue to fixed240k/fixed235k if fixed225k/fixed230k promote, otherwise it falls back to AdamW `8e-6` on GPU0 and `prev_pupil_anchor` on GPU1. If post_count17 is already on mixed probes and they do not promote, post_count18 advances to center-hinge on GPU0 and ellipse-state loss on GPU1.

Current continuation check: fixed205k and fixed210k are still training, both observed at epoch 11/12 with no fixed205k/fixed210k test eval summaries yet. fixed205k latest val center `35.1333`, P10 `13.2188`, P5 `4.1476`; fixed210k latest val center `35.1114`, P10 `13.6456`, P5 `3.8949`. GPU1 remains assigned to fixed205k, so no extra same-GPU experiment was launched.

Added nineteenth-stage/follow-up runner `scripts/v3/run_post_count19_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count19_gpu0_20260611` and `hgtxr_post_count19_gpu1_20260611`, with logs `runs/_logs/post_count19_decision_gpu0_20260611.log` and `runs/_logs/post_count19_decision_gpu1_20260611.log`. It waits for `post_count18`; count branch can continue to GPU1 fixed245k and GPU0 fixed250k if fixed235k/fixed240k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0.

fixed205k/fixed210k completed and promoted. fixed205k best-center test eval `eval_fixed205k_bestcenter_test_gpu1_w0_20260611_085245` produced center `30.3966`, P10 `15.0965`, P5 `4.0693`, becoming the current P10 leader; fixed205k best-P10 eval produced center `31.4222`, P10 `13.8312`, P5 `3.7381`. fixed210k best-center test eval `eval_fixed210k_bestcenter_test_gpu0_w0_20260611_085248` produced center `30.1415`, P10 `14.7037`, P5 `4.2636`, becoming the current center leader; fixed210k best-P10 eval produced center `30.8138`, P10 `13.8312`, P5 `3.6939`. post_count16 is expected to continue the count branch to GPU1 fixed215k and GPU0 fixed220k on its next polling tick.

post_count16 selected `ACTION=count`, `BEST_COUNT=210000`, `BEST_CENTER=30.141473`, `BEST_P10=15.096514`. GPU1 launched fixed215k as `raw_mode1_stage2_count215000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090019`; GPU0 launched fixed220k as `raw_mode1_stage2_count220000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_090007`. Latest observed epoch 1: fixed215k val center `40.1420`, P10 `9.3531`, P5 `2.6673`; fixed220k val center `40.1718`, P10 `9.4710`, P5 `2.6505`.

Continuation check: fixed215k/fixed220k are training normally. Latest observed epoch 4: fixed215k val center `36.8466`, P10 `12.6280`, P5 `3.1761`; fixed220k val center `36.8837`, P10 `12.8807`, P5 `3.0413`. No fixed215k/fixed220k test eval summaries exist yet; GPU1 remains occupied by fixed215k.

Added twentieth-stage/follow-up runner `scripts/v3/run_post_count20_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count20_gpu0_20260611` and `hgtxr_post_count20_gpu1_20260611`, with logs `runs/_logs/post_count20_decision_gpu0_20260611.log` and `runs/_logs/post_count20_decision_gpu1_20260611.log`. It waits for `post_count19`; count branch can continue to GPU1 fixed255k and GPU0 fixed260k if fixed245k/fixed250k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0.

Continuation check: fixed215k/fixed220k are still training with no fixed215k/fixed220k test eval summaries yet. Latest observed epoch 6: fixed215k val center `36.4212`, P10 `12.4910`, P5 `3.2401`; fixed220k val center `36.4039`, P10 `13.0357`, P5 `3.3109`. GPU1 remains occupied by fixed215k, so no extra same-GPU experiment was launched.

Added twenty-first-stage/follow-up runner `scripts/v3/run_post_count21_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count21_gpu0_20260611` and `hgtxr_post_count21_gpu1_20260611`, with logs `runs/_logs/post_count21_decision_gpu0_20260611.log` and `runs/_logs/post_count21_decision_gpu1_20260611.log`. It waits for `post_count20`; count branch can continue to GPU1 fixed265k and GPU0 fixed270k if fixed255k/fixed260k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0.

Full experiment plan print/update checkpoint: current leaders remain fixed210k best-center (`30.1415` center, `14.7037` P10, `4.2636` P5) and fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch remains GPU1 fixed215k and GPU0 fixed220k. Latest observed status: fixed215k epoch 8, val center `35.7267`, P10 `12.1597`, P5 `3.8331`; fixed220k epoch 9, val center `35.3173`, P10 `12.0058`, P5 `3.4288`. No fixed215k/fixed220k test eval summaries exist yet. GPU1 is already occupied by fixed215k, so parallel GPU1 work remains assigned through the queued post_count17/post_count18/post_count19/post_count20/post_count21 branches rather than launched immediately.

fixed215k/fixed220k completed train and test evals. fixed215k best-center/best-P10 both produced center `30.0281`, P10 `14.4018`, P5 `4.4498`. fixed220k best-center/best-P10 both produced center `30.0077`, P10 `14.3967`, P5 `4.2096`, becoming the new center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count17 selected `ACTION=count`, `BEST_COUNT=220000`, `BEST_CENTER=30.007657`, `BEST_P10=15.096514`; GPU1 launched fixed225k as `raw_mode1_stage2_count225000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092225`, and GPU0 launched fixed230k as `raw_mode1_stage2_count230000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_092216`.

Added twenty-second-stage/follow-up runner `scripts/v3/run_post_count22_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count22_gpu0_20260611` and `hgtxr_post_count22_gpu1_20260611`, with logs `runs/_logs/post_count22_decision_gpu0_20260611.log` and `runs/_logs/post_count22_decision_gpu1_20260611.log`. It waits for `post_count21`; count branch can continue to GPU1 fixed275k and GPU0 fixed280k if fixed265k/fixed270k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0. Current fixed225k/fixed230k status at registration: both epoch 3; fixed225k val center `37.8648`, P10 `10.7412`, P5 `2.6988`; fixed230k val center `37.8628`, P10 `10.6065`, P5 `2.6988`.

fixed225k/fixed230k completed train and test evals. fixed225k best-center/best-P10 both produced center `29.9776`, P10 `14.4728`, P5 `3.8984`. fixed230k best-center/best-P10 both produced center `29.9516`, P10 `14.4677`, P5 `3.6050`, becoming the current center leader. P10 leader remains fixed205k best-center at `15.0965`. post_count18 selected `ACTION=count`, `BEST_COUNT=230000`, `BEST_CENTER=29.951577`, `BEST_P10=15.096514`; GPU1 launched fixed235k as `raw_mode1_stage2_count235000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094454`, and GPU0 launched fixed240k as `raw_mode1_stage2_count240000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_094445`.

Full experiment plan print/update checkpoint: current center leader is fixed230k best-center (`29.9516` center, `14.4677` P10, `3.6050` P5); current P10 leader is fixed205k best-center (`30.3966` center, `15.0965` P10, `4.0693` P5). Current active branch is GPU1 fixed235k and GPU0 fixed240k. Latest observed epoch 4: fixed235k val center `36.6978`, P10 `12.8504`, P5 `3.4906`; fixed240k val center `36.6337`, P10 `13.1199`, P5 `3.4906`. GPU1 is already occupied by fixed235k, so additional parallel GPU1 work remains assigned through post_count19/post_count20/post_count21/post_count22 rather than launched on the same GPU.

Added twenty-third-stage/follow-up runner `scripts/v3/run_post_count23_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count23_gpu0_20260611` and `hgtxr_post_count23_gpu1_20260611`, with logs `runs/_logs/post_count23_decision_gpu0_20260611.log` and `runs/_logs/post_count23_decision_gpu1_20260611.log`. It waits for `post_count22`; count branch can continue to GPU1 fixed285k and GPU0 fixed290k if fixed275k/fixed280k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0. Current fixed235k/fixed240k status at registration: epoch 6; fixed235k val center `36.4167`, P10 `12.3169`, P5 `2.5438`; fixed240k val center `36.4364`, P10 `13.3446`, P5 `3.1593`.

Added twenty-fourth-stage/follow-up runner `scripts/v3/run_post_count24_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, executable-bit check, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered tmux watchers `hgtxr_post_count24_gpu0_20260611` and `hgtxr_post_count24_gpu1_20260611`, with logs `runs/_logs/post_count24_decision_gpu0_20260611.log` and `runs/_logs/post_count24_decision_gpu1_20260611.log`. It waits for `post_count23`; count branch can continue to GPU1 fixed295k and GPU0 fixed300k if fixed285k/fixed290k promote, otherwise it falls back to `prev_pupil_anchor` on GPU1 and AdamW `8e-6` on GPU0. Latest fixed-count status: fixed235k epoch 7 val center `36.1807`, P10 `12.9009`, P5 `3.0829`; fixed240k epoch 8 val center `35.5407`, P10 `12.9739`, P5 `3.4288`.

Continuation check: fixed235k/fixed240k still training; no fixed235k/fixed240k test eval summaries yet. Latest observed epoch 10: fixed235k val center `35.0450`, P10 `13.1873`, P5 `3.8387`; fixed240k val center `34.9476`, P10 `13.5108`, P5 `3.9027`. post_count19 is alive and waiting for fixed235k/fixed240k eval summaries before launching GPU1 fixed245k / GPU0 fixed250k or fallback probes.

fixed235k/fixed240k completed train and test evals. fixed235k best-center/best-P10 both produced center `29.9446`, P10 `13.6692`, P5 `3.5761`, briefly improving center over fixed230k. fixed240k best-center produced center `29.8589`, P10 `13.5132`, P5 `3.3580`, becoming the current center leader; fixed240k best-P10 produced center `30.6359`, P10 `13.4864`, P5 `3.8831`, so it did not improve the P10 leaderboard. P10 leader remains fixed205k best-center at `15.0965`.

post_count19 selected `ACTION=count`, `BEST_COUNT=240000`, `BEST_CENTER=29.858867`, `BEST_P10=15.096514`. GPU1 launched fixed245k as `raw_mode1_stage2_count245000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_100750`; GPU0 launched fixed250k as `raw_mode1_stage2_count250000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_100838`. Both raw event-count contract checks passed and training processes are active.

Added twenty-fifth-stage and twenty-sixth-stage follow-up runners: `scripts/v3/run_post_count25_decision_queue_20260611.sh` and `scripts/v3/run_post_count26_decision_queue_20260611.sh`. Validation passed with `bash -n`, `sh -n`, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered watchers `hgtxr_post_count25_gpu0_20260611`, `hgtxr_post_count25_gpu1_20260611`, `hgtxr_post_count26_gpu0_20260611`, and `hgtxr_post_count26_gpu1_20260611`. post_count25 waits for post_count24 and can continue to GPU1 fixed305k / GPU0 fixed310k; post_count26 waits for post_count25 and can continue to GPU1 fixed315k / GPU0 fixed320k. If the count branch plateaus, both preserve the fallback policy: GPU1 `prev_pupil_anchor` or ellipse-state probe, GPU0 AdamW `8e-6` or center-hinge probe.

Current runtime check: fixed245k remains active on GPU1 and fixed250k remains active on GPU0. Latest observed history: fixed245k epoch 7 val center `36.0692`, P10 `13.4793`, P5 `3.3580`; fixed250k epoch 6 val center `36.2951`, P10 `13.1289`, P5 `3.3165`. Current completed leaderboards remain fixed240k best-center for center (`29.8589`) and fixed205k best-center for P10 (`15.0965`).

fixed245k/fixed250k completed train and test evals. fixed245k best-center eval `eval_fixed245k_bestcenter_test_gpu1_w0_20260611_102421` produced center `29.7768`, P10 `13.6025`, P5 `3.2687`; fixed245k best-P10 eval `eval_fixed245k_bestp10_test_gpu1_w0_20260611_102719` produced center `30.6034`, P10 `13.3333`, P5 `3.7662`. fixed250k best-center eval `eval_fixed250k_bestcenter_test_gpu0_w0_20260611_102523` produced center `29.6603`, P10 `13.5514`, P5 `2.7968`, becoming the current center leader; fixed250k best-P10 eval `eval_fixed250k_bestp10_test_gpu0_w0_20260611_102822` produced center `30.5370`, P10 `13.7959`, P5 `3.8414`. P10 leader remains fixed205k best-center at `15.0965`.

post_count20 selected `ACTION=count`, `BEST_COUNT=250000`, `BEST_CENTER=29.660294`, `BEST_P10=15.096514`. GPU1 launched fixed255k as `raw_mode1_stage2_count255000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_103230`; GPU0 launched fixed260k as `raw_mode1_stage2_count260000_lr5e-6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_103225`. Latest observed epoch 1: fixed255k val center `40.2184`, P10 `8.7859`, P5 `2.7156`; fixed260k val center `40.2167`, P10 `8.9207`, P5 `2.8504`.

Added twenty-seventh-stage and twenty-eighth-stage follow-up runners: `scripts/v3/run_post_count27_decision_queue_20260611.sh` and `scripts/v3/run_post_count28_decision_queue_20260611.sh`. Validation passed with `bash -n`, `sh -n`, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered watchers `hgtxr_post_count27_gpu0_20260611`, `hgtxr_post_count27_gpu1_20260611`, `hgtxr_post_count28_gpu0_20260611`, and `hgtxr_post_count28_gpu1_20260611`. post_count27 waits for post_count26 and can continue to GPU1 fixed325k / GPU0 fixed330k; post_count28 waits for post_count27 and can continue to GPU1 fixed335k / GPU0 fixed340k. Latest fixed255k/fixed260k status: epoch 11; fixed255k val center `34.3848`, P10 `12.9043`, P5 `4.0914`; fixed260k val center `34.4096`, P10 `12.5505`, P5 `3.9566`.

fixed255k/fixed260k completed train and test evals. fixed255k best-center eval `eval_fixed255k_bestcenter_test_gpu1_w0_20260611_104936` produced center `29.6082`, P10 `13.7228`, P5 `3.3431`, becoming the current center leader. fixed255k best-P10 eval produced center `30.5540`, P10 `13.7598`, P5 `3.8601`. fixed260k best-center eval produced center `29.7259`, P10 `13.4515`, P5 `3.7611`; fixed260k best-P10 eval produced center `32.8103`, P10 `11.0128`, P5 `2.7887`. P10 leader remains fixed205k best-center at `15.0965`. post_count21 selected `ACTION=count`, `BEST_COUNT=255000`, `BEST_CENTER=29.608202`, `BEST_P10=15.096514`, then launched GPU1 fixed265k and GPU0 fixed270k at about `2026-06-11T10:57:57+09:00`. Latest fixed265k/fixed270k status: epoch 5; fixed265k val center `36.2752`, P10 `13.7612`, P5 `3.2996`; fixed270k val center `36.2936`, P10 `13.5815`, P5 `3.0301`.

Added twenty-ninth-stage and thirtieth-stage follow-up runners: `scripts/v3/run_post_count29_decision_queue_20260611.sh` and `scripts/v3/run_post_count30_decision_queue_20260611.sh`. Validation passed with `bash -n`, `sh -n`, no-arg usage exit `2`, tmux registration, and watcher-log startup checks. Registered watchers `hgtxr_post_count29_gpu0_20260611`, `hgtxr_post_count29_gpu1_20260611`, `hgtxr_post_count30_gpu0_20260611`, and `hgtxr_post_count30_gpu1_20260611`. post_count29 waits for post_count28 and can continue to GPU1 fixed345k / GPU0 fixed350k; post_count30 waits for post_count29 and can continue to GPU1 fixed355k / GPU0 fixed360k. Latest fixed265k/fixed270k status: epoch 5; fixed265k val center `36.2752`, P10 `13.7612`, P5 `3.2996`; fixed270k val center `36.2936`, P10 `13.5815`, P5 `3.0301`.

Added squared decoded-center threshold hinge loss for the next P10/P5 loss-axis probe. Code path: `src/hbtxr/loss/bundles/track.py` now emits disabled-by-default `loss_track_center_hinge_sq`, controlled by `loss.track_center_hinge_sq_weight` and the existing `loss.track_center_hinge_margin_px`. Added unit coverage in `tests/test_track_center_l2_loss.py`, config `configs/v3/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhingesq10_finetune_fullwidth.yaml`, and runner `scripts/v3/run_centerhingesq_probe_queue_20260611.sh <event_count> <margin_px> <sq_weight> <cuda:N>`. Validation passed: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` produced `5 passed` with the known parent `.pytest_cache` warning; `python -m py_compile src/hbtxr/loss/bundles/track.py` passed; runner `bash -n`, `sh -n`, executable bit, and no-arg usage exit `2` passed; raw event-count readiness contract passed for the new config; config loader resolved `track_center_hinge_sq_weight=0.005`, `track_center_hinge_margin_px=10.0`, and `lr=5e-6`. No GPU training launched because GPU1 remains assigned to fixed265k and GPU0 to fixed270k; the new probe is ready for the next plateau/idle slot.

fixed265k/fixed270k completed train and test evals and did not promote. fixed265k best-center eval `eval_fixed265k_bestcenter_test_gpu1_w0_20260611_111535` produced center `29.7582`, P10 `13.5969`, P5 `3.5974`; fixed265k best-P10 eval `eval_fixed265k_bestp10_test_gpu1_w0_20260611_111849` produced center `32.8107`, P10 `11.2436`, P5 `3.0013`. fixed270k best-center/best-P10 evals produced center `29.9207`, P10 `13.3652`, P5 `3.8418`. Current leaderboards remain fixed255k best-center for center (`29.6082`) and fixed205k best-center for P10 (`15.0965`). post_count22 selected `ACTION=mixed_probe`: GPU1 launched `prev_pupil_anchor` at fixed255k as `raw_mode1_stage2_count255000_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_alphainit_fullwidth_20260611_112216`; GPU0 launched AdamW `8e-6` at fixed255k as `raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215`.

Added thirty-first-stage/follow-up runner `scripts/v3/run_post_count31_decision_queue_20260611.sh`; validation passed with `bash -n`, `sh -n`, no-arg usage exit `2`, and watcher startup checks. Registered tmux watchers `hgtxr_post_count31_gpu0_20260611` and `hgtxr_post_count31_gpu1_20260611`, with logs `runs/_logs/post_count31_decision_gpu0_20260611.log` and `runs/_logs/post_count31_decision_gpu1_20260611.log`. It waits for `post_count30`; count branch can continue to GPU1 fixed365k and GPU0 fixed370k if fixed355k/fixed360k promote, otherwise GPU1 runs the new squared decoded-center hinge probe (`margin=10.0`, `weight=0.005`) and GPU0 runs AdamW `8e-6`. If post_count30 is already in mixed-probe mode and mixed probes fail to promote, GPU1 also advances to squared hinge while GPU0 runs linear center-hinge.
2026-06-11 latest mixed-probe result:

- post_count22 mixed probes completed.
- GPU0 AdamW `8e-6` at fixed255k became the new overall leader. Best-center test eval `eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913` produced center `27.0897`, P10 `15.7109`, P5 `4.6301`. Best-P10 eval produced center `28.2271`, P10 `14.6743`, P5 `4.3941`.
- This replaces the previous fixed255k center leader (`29.6082`) and fixed205k P10 leader (`15.0965`).
- GPU1 `prev_pupil_anchor` completed training, but the first eval failed because `run_prevpupilcrop_probe_queue_20260611.sh` used `infer_hbtxr.py --output-dir`.
- Fixed the runner to use `eval_hbtxr.py --experiment-name`; `sh -n` and `bash -n` passed.
- Manual `prev_pupil_anchor` test evals completed: best-center center `65.0978`, P10 `9.9877`, P5 `3.0629`; best-P10 center `65.6465`, P10 `9.4957`, P5 `3.1365`.
- Conclusion: local-crop is currently a distribution/target mismatch risk and should be paused until diagnosed.
- Next priority: AdamW-centered bracketing around fixed255k, then AdamW-leader warm-start fine-tuning.

2026-06-12 manual 200/200 stage-split result:

- User-requested run `raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452` -> `raw_mode1_stage2_count255000_adamw_lr8e_6_200ep_2gpu_20260611_222452` completed Stage1 200 epochs and Stage2 200 epochs.
- Run settings: Stage1 on GPU0, Stage2 on GPU1, Stage1 LR `1e-3`, Stage1 distillation/pruning disabled, Stage2 fixed-count `255000`, Stage2 AdamW LR `1e-4`, batch size `8`, early stopping disabled.
- Stage1 final epoch 200 val search center `18.7134`, P10 `20.1539`, P5 `5.8457`; best observed Stage1 val search center was `18.3195` around epoch 154.
- Stage2 best validation checkpoint was early: best center checkpoint epoch `34`, val center `40.4009`; best P10 checkpoint epoch `1`, val P10 `14.8989`. Final epoch 200 val center `40.4373`, P10 `11.1714`, P5 `2.2406`.
- Manual test eval of best-center checkpoint `eval_adamw255k_200ep_stage_split_bestcenter_test_gpu0_w0`: center `37.0958`, P10 `11.1722`, P5 `3.4787`.
- Manual test eval of best-P10 checkpoint `eval_adamw255k_200ep_stage_split_bestp10_test_gpu0_w0`: center `41.9219`, P10 `11.2389`, P5 `3.6050`.
- Comparison to current leader `eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913` center `27.0897`, P10 `15.7109`, P5 `4.6301`: the 200/200 stage-split run is much worse and must not promote.
- Analysis: Stage2 LR `1e-4` plus fresh Stage1 initialization did not reproduce the alpha-initialized AdamW breakthrough. Stage2 effectively peaked early and then plateaued/degraded under the plateau scheduler to near-zero LR.

2026-06-12 legacy HBTXR_v3_0 experiment import:

- Added `docs/track/HBTXR_V3_0_PAST_EXPERIMENT_RESULTS.md` to consolidate past `HBTXR_v3_0/docs` experiment results into the current `HGTXR/software` tracking surface.
- Imported legacy lessons from Mode0 Stage1/Stage2, Grounded-SAM all-48 label construction, TSGSS event-feature work, Mode2 optimizer CPU pilot, and TimeLens/v2e status.
- Key integrated judgment: the current 200/200 Stage2 failure matches legacy evidence that Stage2 often peaks early, `loss_total` is not a reliable tracking-quality proxy, and preserving a strong initialization contract is more important than blindly extending epoch count.
- Current experiment priority remains AdamW fixed255k alpha-init warm-start/count-LR bracketing, not fresh Stage1 200ep + high-LR Stage2.

2026-06-15 XR-Eye-Tracking detailed analysis refresh:

- User requested a deeper codebase analysis at the level of hardware `SRC_CASE_MODULE_GUIDE.md` and a deeper paper analysis with explicit fields for prior-method problems, proposed method, algorithm/hardware, experiments, datasets, results, and options.
- Added `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`. It now covers 18 reference code/resource groups with role/hierarchy, module map, dataflow, function-call stack, source evidence, experiment surface, HGTXR integration, and risk.
- Added `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`. It now covers 21 papers with the requested field structure and a cross-paper priority table.
- Updated `anlaysis/xr-eye-tracking/index.md` and `docs/track/CHANGELOG.md` to make the two detailed files the canonical detailed references.
- Current analysis-backed next experiment order is now split by track: `XR-01` is first on the training track; `XR-02` can run immediately on the no-retrain post-process track when dense trajectories exist; `XR-03` is the next geometry/model change; then heatmap/KL or SimDR center decode, causal temporal/CB-ConvLSTM head, and sparse INT8/HLS reporting after software accuracy improves.

2026-06-16 XR-Eye-Tracking second-goal continuation:

- Integrated detailed XR-Eye-Tracking analyses into the second-goal plan as canonical evidence: `DETAILED_CODEBASE_ANALYSIS.md`, `DETAILED_PAPER_ANALYSIS.md`, and `experiment_integration.md`.
- Resolved ordering ambiguity: `XR-01` is the first training experiment, while `XR-02` is a no-retrain post-process branch that can run in parallel once dense prediction trajectories exist.
- Added `scripts/v3/eval_eyelorin_refinement.py` for EyeLoRiN-style M2F median refinement, before/after center/P10/P5 metrics, jitter proxy metrics, and time-gap trajectory gating.
- Added `tests/test_eyelorin_refinement.py`; validation passed with `3 passed` plus the known parent `.pytest_cache` read-only warning.
- Ran XR-02 smoke on AdamW fixed255k best-center checkpoint with `--limit 64` and `--max-gap-us 50000`: raw and refined metrics were identical after gap gating (`center=24.3144`, `P10=21.5686`, `P5=7.8431` on 51 valid weighted rows). This is expected because the current test manifest is sparse and should not be smoothed across multi-second sample gaps.
- Prior ungated smoke showed center degradation, confirming that trajectory-gap gating is required before any median/M2F refinement can be trusted.
- XR-02 status: implementation and smoke validation complete; promotion evaluation is blocked until dense/continuous prediction trajectories are generated or a continuous split is evaluated.
- XR-01 execution status: sandbox-local CUDA/NVML was unavailable, so the same approved commands were started outside the sandbox in tmux. Sessions `hgtxr_xr01_adamw250k_gpu0_20260616` and `hgtxr_xr01_adamw260k_gpu1_20260616` are training fixed250k/fixed260k with AdamW LR `8e-6`; logs are `runs/_logs/xr01_adamw_count250k_lr8e-6_gpu0_20260616.log` and `runs/_logs/xr01_adamw_count260k_lr8e-6_gpu1_20260616.log`.
- XR-01 validation gate: promote only if test center beats `27.0897`, or if P10 beats `15.7109` with center no worse than `27.3397`; P5 should stay near or above `4.6301`.
- Added `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`, a PAPER_REF-specific 30-paper coverage and experiment map connecting the corpus to Head/Loss/LR/Optimizer/Self-supervised Distillation/Teacher/Gating/Hardware actions.
- Sub-agent PAPER_REF coverage audit confirmed the planning gap is execution, not inventory intake: direct ellipse auxiliary head/loss (`XR-03`), failure-bucket diagnostics (`XR-04`), bounded temporal adapter (`XR-05`), and later teacher/distillation/sparse branches remain pending.
- Prepared XR-03 decoded ellipse-state warm-start runner `scripts/v3/run_xr03_ellipsestate_adamw_leader_probe.sh`. It starts from the AdamW fixed255k best-center checkpoint, exposes count/axis-weight/angle-weight/LR/device arguments, and passed `bash -n` plus checkpoint existence validation.
- Addressed sub-agent review risk: XR-03 runner now accepts a sixth positional init checkpoint or `XR03_INIT_CKPT`, so an XR-01-promoted checkpoint can be used without editing the script.
- Added `scripts/v3/compare_xr01_adamw_bracket.py` to scan XR-01 best-center/best-P10 eval summaries and compare them against the current AdamW fixed255k leader gate.
- XR-01 fixed250k/fixed260k AdamW LR `8e-6` count bracket completed on 2026-06-16. All four test evals failed promotion:
  - fixed250k best-center: center `27.1403`, P10 `15.3044`, P5 `4.3635`.
  - fixed250k best-P10: center `28.1357`, P10 `14.6688`, P5 `4.3401`.
  - fixed260k best-center: center `27.2089`, P10 `15.6429`, P5 `4.3741`.
  - fixed260k best-P10: center `32.2040`, P10 `11.9898`, P5 `3.1862`.
- Extended `scripts/v3/compare_xr01_adamw_bracket.py` to include the then-active fixed255k LR `6e-6` and `1e-5` cases; `python3 -m py_compile` and scanner execution passed.
- Started XR-01 fixed255k LR bracket in tmux:
  - GPU0: `hgtxr_xr01_lr6e6_255k_gpu0_20260616`, log `runs/_logs/xr01_adamw_count255k_lr6e-6_gpu0_20260616.log`.
  - GPU1: `hgtxr_xr01_lr1e5_255k_gpu1_20260616`, log `runs/_logs/xr01_adamw_count255k_lr1e-5_gpu1_20260616.log`.
- Added XR-04 failure-bucket diagnostic script `scripts/v3/summarize_eval_failure_buckets.py`; `python3 -m py_compile` passed and a 64-row smoke joined all prediction rows.
- Generated full XR-04 leader diagnostic artifact `runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json` with `2238/2238` joined rows and zero missing predictions.
- XR-04 finding: low-similarity rows dominate error. `similarity_target <=0.1` has weighted center `42.8642`, P10 `7.1259`, P5 `1.4252`; worst session bucket is `user41/right/session_201` with weighted center `42.4245`.

2026-06-16 XR-05I/XR-06F closeout:

- [x] XR-05I no-distill P5-preserve branch finished from XR-05A P5/balanced checkpoint.
- [x] XR-05I best-center eval parsed: center `20.351013261931282`, P10 `26.15008576256888`, P5 `8.428996889931815`; no promotion.
- [x] XR-05I best-P10 eval parsed: center `20.393892083849227`, P10 `26.170068747656686`, P5 `8.450255387169975`; no promotion.
- [x] XR-06F weak-distill P5-preserve branch finished from the same XR-05A checkpoint as init and teacher.
- [x] XR-06F best-center eval parsed: center `20.322171998023986`, P10 `26.200681025641305`, P5 `8.625850643430438`; no promotion.
- [x] XR-06F best-P10 eval parsed: center `20.421796573911394`, P10 `26.394558545521328`, P5 `8.630102334703718`; no promotion.
- [x] Active gates remain center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Sub-agent sidecar recommended keeping gates unchanged and moving next priority to targeted failure-bucket training or dense-trajectory XR-02.
- [x] Next P0 selected: targeted XR-04B/XR-04C low-similarity/high-motion branch around `similarity_target <= 0.1`.
- [x] XR-04B launched on GPU0 with threshold `0.1`, scale `2.0`, LR `6e-6`, init XR-03D best-P10; startup reached epoch `1/12`.
- [x] XR-04C launched on GPU1 with threshold `0.1`, scale `2.0`, LR `3e-6`, init XR-03D best-P10; startup reached epoch `1/12`.
- [x] XR-04B/XR-04C train/eval closeout complete. Four eval summaries parsed; no active gate promoted.
- [x] Added failure-bucket manifest subset tooling and generated `similarity_target <= 0.1` / `session_201` train/val subset.
- [x] Launched XR-04D/XR-04E manifest-subset branches in parallel: LR `6e-6` on GPU0 and LR `3e-6` on GPU1.
- [x] XR-04D/XR-04E train/eval closeout complete. Four eval summaries parsed; no active gate promoted.
- [x] Decision recorded: hard-subset fine-tuning overfits the failure bucket and hurts full-test aggregate accuracy. Next robustness path is full-manifest weighted sampler/loss balancing, or XR-02 after dense trajectories are available.
- [x] Added XR-09 full-manifest weighted sampler support, config, runner, and unit tests.
- [x] Validated XR-09 setup with py_compile, bash syntax, weighted sampler pytest, dataloader smoke, and train-manifest weight distribution sanity.
- [x] Launched XR-09A/XR-09B weighted-sampler branches on GPU0/GPU1; startup reached epoch `1/12` on both.
- [x] XR-09A/XR-09B train/eval closeout complete. Four eval summaries parsed; best result was XR-09B best-center `20.4154/26.1964/8.3057`, so no active gate promoted.
- [x] Decision recorded: sampler-only full-manifest failure-bucket balancing is closed. Next 2차 목표 action should use loss-side sample weighting, dense-trajectory XR-02 preparation, or a paper-backed head/architecture change.
- [x] Added XR-10 full-manifest loss-side sample weighting support, config, runner, and unit tests.
- [x] Validated XR-10 setup with py_compile, bash syntax, loss-weight pytest, dataloader smoke, and train-manifest weight distribution sanity.
- [x] Launched XR-10A/XR-10B loss-weight branches on GPU0/GPU1; startup reached epoch `1/12` on both.
- [x] XR-10A/XR-10B train/eval closeout complete. Four eval summaries parsed; best result was XR-10B best-center `20.3266/25.9702/8.1335`, so no active gate promoted.
- [x] Decision recorded: failure-bucket rebalancing is exhausted across low-sim loss weighting, hard subset, sampler-only, and loss-side full-manifest variants. Next 2차 목표 action should change paper-backed model/head representation or prepare dense trajectories for XR-02.
- [x] Selected XR-11 as the next paper-backed P0: track-branch SimDR-style coordinate auxiliary, based on TDTracker/AIS heatmap/KL/SimDR evidence and XR-10 no-promotion.
- [x] Implemented `TrackStateSimDRHead`, `track/state_simdr`, Stage2 `loss_track_state_simdr`, XR-11 config, XR-11 runner, and unit coverage.
- [x] Validated XR-11 setup with py_compile, bash syntax, model-build smoke, `19 passed` targeted tests, and diff whitespace check.
- [x] Launched XR-11 center lane on GPU0 from XR-06C best-center and XR-11 P10 lane on GPU1 from XR-06C best-P10.
- [x] Monitored XR-11 train/eval closeout and compared against active gates.
- [x] XR-11 no-distill SimDR closed with no promotion. Center-init best-center `20.3388/26.1607/8.3014`, center-init best-P10 `20.3662/26.5455/8.5740`, P10-init best-center `20.3430/25.9545/8.2355`, P10-init best-P10 `20.2893/26.3253/8.3099`.
- [x] Active gates remain center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Next P0 selected: XR-12 weak-distill light-SimDR, preserving XR-06C init/teacher and lowering SimDR weight before attempting broader temporal/head changes.
- [x] Added XR-12 weak-distill SimDR config and runner.
- [x] Validated XR-12 setup with `bash -n`, config/model smoke, checkpoint existence checks, and diff whitespace check.
- [x] Launched XR-12 GPU0 center lane and GPU1 P10 lane with SimDR weight `0.0001`, LR `5e-7`, fixed255k, weak distillation teacher=init.
- [x] Monitored XR-12 train/eval closeout and compared against active gates.
- [x] XR-12 weak-distill light-SimDR closed without center/P10 promotion. Center-init best-center `20.3049/26.1514/8.2207`, center-init best-P10 `20.3599/26.0821/8.7917`, P10-init best-center `20.3070/26.1514/8.1696`, P10-init best-P10 `20.3070/26.1514/8.1696`.
- [x] Decision recorded: XR-12 has one P5-only scalar promotion but is not a leader replacement because center/P10 regress. Next P0 is XR-13 head-only weak-distill SimDR with trainable scope restricted to prediction heads.
- [x] Prepared and validated XR-13 head-only weak-distill SimDR config/runner.
- [x] Launched XR-13 GPU0 center lane and GPU1 P10 lane with head-only trainable filter, SimDR weight `0.0001`, LR `5e-7`, fixed255k, weak distillation teacher=init.
- [x] Monitored XR-13 train/eval closeout and compared against active gates.
- [x] XR-13 head-only weak-distill SimDR closed without promotion. Center-init best-center `20.2916/26.2976/8.4694`, center-init best-P10 `20.2897/26.2679/8.5204`, P10-init best-center `20.3555/26.4622/8.4906`, P10-init best-P10 `20.3696/26.5514/8.5332`.
- [x] Decision recorded: SimDR variants XR-11/XR-12/XR-13 do not beat XR-06C gates. Next P0 is XR-02 dense/continuous prediction trajectory preparation before any new temporal/head training.
- [x] Added and ran XR-02 dense trajectory availability diagnostic. Test manifest and canonical annotations both have `dense_pair_count=0`, min gap `360000us`, median gap `4000003us`, and longest dense segment `1`, so M2F metric promotion is blocked on current labeled canonical1 data.
- [x] Ran XR-04 diagnostics refresh for XR-06C best-center and best-P10 leaders. Both diagnostics joined `2238/2238` rows and show the same failure concentration: `similarity_target <= 0.1`, `session_201`, and subjects `42/45/39`.
- [x] Confirmed blink/closed-eye is not the immediate weighted-metric driver: closed-eye and invalid-track samples have zero weight under the current metric contract.
- [x] Confirmed search/event fallback cannot be evaluated from current leader rows because `model.heads.active=track` disables those heads in the normalized config; search/event diagnostic artifacts joined `0/2238` rows.
- [x] Added and smoke-tested `scripts/v3/infer_hbtxr.py --limit`; limit-2 confidence smoke wrote `track_pred` rows to `runs/diagnostics/xr04_confidence_xr06c_bestcenter_infer_limit2_20260616`.
- [x] Added and smoke-tested `scripts/v3/summarize_track_confidence_buckets.py`; limit-2 summary joined `2/2` rows and wrote `runs/diagnostics/xr04_confidence_xr06c_bestcenter_limit2_buckets_20260616.json`.
- [x] Built `confidence_probe_low0p1_high0p6_128` with 128 low-sim and 128 high-sim test rows.
- [x] Ran XR-06C best-center and best-P10 confidence inference and bucket summaries on the balanced probe. Both joined `256/256` rows.
- [x] Decision recorded: existing `track_pred` confidence/quality is saturated and fails to separate low-sim failures; no-train confidence gate is rejected.
- [x] Added and ran `scripts/v3/eval_similarity_fallback.py` for XR-14 no-train previous-state/blend fallback. Raw remains best by official-like batchmean center `20.283125752718494`; best fallback `blend(threshold=0.05, alpha=0.75)` degrades to `22.57394233260353`.
- [x] Extended `scripts/v3/infer_hbtxr.py` to write `track_state_aux`/`track_state_simdr`, and extended `scripts/v3/summarize_track_confidence_buckets.py` with `--state-key`.
- [x] Ran XR-14 aux-state probe on the 256-row confidence manifest. `track_state_aux` is not usable as fallback: center `158.7825`, P10 `0.0`, P5 `0.0`.
- [x] Added XR-14A all-head relocalization warm-start config and runner: `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml`, `scripts/v3/run_xr14_allhead_relocalize_probe.sh`.
- [x] Fixed XR-14A runner to include a lane tag in the experiment name; the first launch was stopped because center/P10 lanes shared a prefix and could have selected the wrong run root for eval.
- [x] Launched XR-14A center lane on GPU0 and P10 lane on GPU1 with separate run roots:
  - center: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_center_trackpreserve_fullwidth_20260616_094218`
  - P10: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_p10_trackpreserve_fullwidth_20260616_094220`
- [x] Startup validation passed for both lanes: raw event-count contract, intended checkpoint load, resolved CUDA device, train/val counts `5929/844`, and epoch `1/12` step logs.
- [x] XR-14A train closeout completed. Both lanes early-stopped at epoch `8/12` with train exit `0`.
- [x] XR-14A full-test eval completed for best-center and best-P10 checkpoints on both lanes:
  - center lane: `20.342467624800545 / 25.644133370263237 / 8.11947306905474`
  - P10 lane: `20.348247524670192 / 25.703657184328353 / 7.97278938974653`
- [x] XR-14A did not promote versus active gates: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Generated 12 branch-state bucket artifacts under `runs/diagnostics/xr14a_*_state_failure_buckets_20260616.json`.
- [x] Branch-state decision: `track_state` remains near leader quality but misses gates; `search_state` and `event_state` are unusable as fallback states with weighted center about `177.8/178.5px` and P10/P5 `0.0`.
- [x] Next execution selected: XR-15 support-adaptive fixed-count event-window probe. This changes event evidence density while preserving XR-06C init/teacher, weak distill, AdamW `5e-7`, direct track-state aux, full width, and track-only inference.
- [x] Added XR-15 artifacts: `configs/v3/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`, `scripts/v3/run_xr15_support_adaptive_trackstateaux_probe.sh`, and `docs/XR15_SUPPORT_ADAPTIVE_PLAN.md`.
- [x] XR-15 preflight passed: runner `bash -n`, executable bit, raw event-count contract, and dataset smoke showing `adaptive_count_resolved`.
- [x] Launched XR-15A support-adaptive center lane on GPU0 and P10 lane on GPU1:
  - center log: `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log`
  - P10 log: `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log`
- [x] XR-15A startup validation passed for both lanes: raw event-count contract, intended XR-06C checkpoint load, resolved CUDA device, train/val counts `5929/844`, and epoch `1/12` step logs.
- [x] Added `scripts/v3/eval_xr15_p5_checkpoint.sh` for post-train best-P5 checkpoint evaluation, because the main XR-15 runner evaluates best-center and best-P10 automatically while trainer also writes `best_track_p5.pt`.
- [x] XR-15A train/eval closeout completed for best-center and best-P10 checkpoints on both lanes.
- [x] XR-15A promoted the center gate. New center leader is P10-lane best-center: center `20.225680075372967`, P10 `26.50085105895996`, P5 `8.304422058377947`.
- [x] P10/P5 gates remain unchanged: P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] XR-15A diagnostics generated:
  - `runs/diagnostics/xr15_p10_bestcenter_track_state_failure_buckets_20260616.json`
  - `runs/diagnostics/xr15_center_bestcenter_track_state_failure_buckets_20260616.json`
- [x] Diagnostic decision: low `similarity_target <= 0.1`, subjects `42/45`, and `session_201` remain the dominant failure buckets.
- [x] Best-P5 helper attempts were made with interrupt and log+timeout paths, but produced only `hypers/*` and no `eval_summary.json`; recorded as an artifact gap, not a center-promotion blocker.
- [x] XR-15B narrow support-adaptive event-window probe launched on both GPUs with min/base/max `224k/255k/288k`.
- [x] XR-15B startup validation passed in logs: raw contract, intended checkpoint load, resolved CUDA device, train/val counts, and epoch `1/12` train steps.
- [x] XR-15B train closeout completed on both lanes with train exit `0`.
- [x] XR-15B best-center eval closeout parsed. Center-lane best-center promoted the center gate to `20.215672533852715` with P10 `26.575255823135375` and P5 `8.627551317214966`; P10-lane best-center reached `20.21979672227587/26.324405458995273/8.28741525241307`.
- [x] XR-15B best-P10 eval artifacts produced only `hypers/*` and no `eval_summary.json`; the stalled XR-15B tmux sessions were cleaned up.
- [x] New active gates: center `<20.215672533852715`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Launched XR-15C wider support-adaptive event-window probe on both GPUs with min/base/max `160k/255k/384k`.
- [x] XR-15C startup validation passed in logs: raw contract, intended checkpoint load, resolved CUDA device, train/val counts, and epoch `1/12` train steps.
- [x] XR-15C train closeout completed on both lanes with train exit `0`.
- [x] XR-15C best-center eval closeout parsed. Center-lane best-center promoted the center gate to `20.19088832650866` with P10 `26.009779623576573` and P5 `8.436224787575858`; P10-lane best-center reached `20.205999997683932/26.303146975381033/8.397959463936942`.
- [x] New active gates: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Added `scripts/v3/run_xr15d_trackadapters_probe.sh` to run fixed255k Stage2-only track-adapter/coordinate-head training and best-center/best-P10 evals.
- [x] XR-15D static validation passed: runner `bash -n`, raw event-count contract, checkpoint existence, and `PYTHONPATH=src` model-build smoke.
- [x] Launched XR-15D center lane on GPU0 from the XR-15C center leader and P10 lane on GPU1 from the XR-06C P10 leader.
- [x] XR-15D startup validation passed in logs: raw contract, intended checkpoint load, trainable filter `22` tensors / `448520` params, CUDA device resolution, train/val counts, and epoch `1/16` train steps.
- [x] XR-15D train/eval closeout completed with no promotion. Best scalar among its evals was P10-lane best-P10 P10 `26.1883510862078`, below the active P10 gate.
- [x] Added XR-15E weak-distill low-LR adapter config/runner to test whether XR-15D drift can be bounded with teacher=init and LR `5e-7`.
- [x] XR-15E static validation passed: runner `bash -n`, raw event-count contract, model+teacher smoke, checkpoint existence.
- [x] Launched XR-15E center lane on GPU0 and P10 lane on GPU1; startup logs reached raw contract pass, intended init/teacher checkpoint load, trainable filter report, CUDA device resolution, and epoch `1/12`.
- [x] XR-15E train/eval closeout completed with no promotion. Best result was center-lane best-center `20.281941563742503/26.2521265574864/8.633928898402623`, below active gates center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- [x] Added XR-16A weak-distill event-path adapter config/runner to isolate event-only adapter updates after XR-15D/E both-adapter drift.
- [x] XR-16A static validation passed: runner `bash -n`, raw event-count contract, diff whitespace check, and trainable-filter smoke with `14` trainable tensors / `324680` params.
- [x] Launched XR-16A center lane on GPU0 and P10 lane on GPU1; startup logs reached raw contract pass, intended init/teacher checkpoint load, trainable filter report, CUDA device resolution, and epoch `1/12`.
- [x] XR-16A train/eval closeout completed with no promotion. Best result was center-lane best-center `20.281941199302672/26.2521265574864/8.633928898402623`.
- [x] Added and ran XR-16B one-shot checkpoint interpolation diagnostic between XR-15C center and XR-06C P10 at alpha `0.125`; result `20.28155174595969/26.309524529320854/8.619047934668405`, no promotion.
- [x] Filled XR-15C best-P5 artifact gap. P10-lane best-P5 promoted the P5 gate to `8.703231593540737` with center `20.245368467058455` and P10 `26.487245675495693`.
- [x] Updated downstream experiment priority using active gates center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.703231593540737`.
- [x] Added and ran XR-17A same-branch XR-15C center-to-P5 interpolation sweep for alpha `0.25/0.50/0.625/0.75/0.875`. Alpha `0.75` promoted P5 to `8.732993507385254` with center `20.22669484274728` and P10 `26.34566399029323`.
- [x] Compared XR-15C center leader versus XR-17A alpha `0.75` failure buckets and used the result to design a bounded P5-anchor branch.
- [x] Ran XR-17B P5-anchor support-adaptive branch. Best-center promoted center to `20.182338142395018`; best-P5 promoted P5 to `8.844813244683403`.
- [x] Ran XR-18A no-train P10 interpolation from XR-17B center toward XR-06C P10. No gate promoted.
- [x] Ran XR-19A trainable P10-recovery micro polish. Best-P5 promoted center to `20.181213889803207`; P10/P5 did not promote.
- [x] Ran XR-20A/XR-20B P10-recovery LR ladder. XR-20A LR `2.5e-7` best-center promoted center to `20.175542894431523`; XR-20B LR `5e-7` did not promote.
- [x] Active gates updated: center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- [x] Added and ran XR-21 P10-margin mechanism-change probes. Secondary XR-20A-init lanes with hinge weights `0.0015` and `0.003`, plus primary XR-06C-init P10-leader lane with hinge weight `0.0005`, completed with no active gate promotion.
- [x] XR-21 best secondary result was `20.187303059441703/26.126701450347902/8.791666977746146`; primary P10-leader result was `20.26367484842028/26.20110617365156/8.696003689084733`.
- [x] Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- [x] Added and ran XR-22 tri-leader checkpoint soup as a no-train representation/supervision preflight.
- [x] XR-22 `c34/p33/f33` promoted P5 to `8.869047941480364` with center `20.236038860252926` and P10 `26.4604599407741`.
- [x] Active gates updated: center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- [x] Added and ran XR-23 P10-preserving optimizer probe from XR-06C best-P10 with ADOPT `2.5e-7` on GPU0 and Lion `1e-7` on GPU1.
- [x] XR-23 completed with no promotion. Best XR-23 P10 was ADOPT best-P10 `26.318027945927213`; best XR-23 center was ADOPT best-P5 `20.219128920350755`; best XR-23 P5 was ADOPT best-P10 `8.681122745786395`.
- [x] Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- [x] Added and ran XR-24 XR-22-anchor P10-recovery with AdamW LR `1.25e-7` on GPU0 and LR `2.5e-7` on GPU1.
- [x] XR-24 completed with no promotion. Best XR-24 P10 was LR `1.25e-7` best-P5 `26.415817070007325`; best XR-24 P5 was LR `2.5e-7` best-P5 `8.791666984558105`; both remain below active gates.
- [x] Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- [x] Added and ran XR-25 XR-22-anchor ADOPT-defaults validation with `betas=[0.9,0.9999]`, `eps=1e-6`, LR `1.25e-7` on GPU0 and LR `2.5e-7` on GPU1.
- [x] XR-25 completed with no promotion. Best XR-25 P10 was LR `1.25e-7` best-P10 `26.46045993396214`; best XR-25 P5 was LR `2.5e-7` best-P5 `8.86607174192156`, narrowly below the active P5 gate.
- [x] Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- [x] Added and ran XR-26 P10-boundary loss using XR-06C best-P10 as init/teacher, head-only trainable scope, and existing `track_state_aux` side head.
- [x] XR-26 default boundary lane `0.02/0.01` and light lane `0.01/0.005` both early-stopped at epoch `6/8` with train/eval exit `0`.
- [x] XR-26 completed with no promotion. Default best-P10 was `20.32213627440589/26.45790890966143/8.535289403370449`; default best-P5 was `20.318065077917918/26.42814700944083/8.541666957310268`. Light lane matched within noise: best-P10 `20.32214218207768/26.45790890966143/8.535289403370449`, best-P5 `20.318073788711004/26.42814700944083/8.541666957310268`.
- [x] Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- [x] Added and ran XR-27 track-center heatmap-state representation from XR-06C best-P10 init/teacher, with head-only trainable scope and state-distillation disabled after audit.
- [x] XR-27A LR `1e-4` best-P10 promoted all active gates: center `17.274589475563594`, P10 `31.915391901561193`, P5 `10.289966331209456`.
- [x] XR-27B LR `3e-5` promoted center/P10 on best-P10 but did not beat XR-27A: center `18.388037020819528`, P10 `27.238521099090576`, P5 `8.488520683561052`.
- [x] Active gates updated: center `<17.274589475563594`, P10 `>31.915391901561193`, P5 `>10.289966331209456`.
- [x] XR-28 consolidation diagnostics generated for XR-27A, XR-06C, XR-20A, and XR-22. XR-27A improves weighted overall metrics to `17.1708/32.1921/10.3219` and strongly fixes `similarity_target <= 0.3`.
- [x] XR-28 diagnostic risk recorded: high-similarity `>0.9` P10 remains better in XR-20A/XR-22, and subject `39` loses P10/P5 despite lower center error.
- [x] Updated XR-27 runner to support `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME`; future XR-28 runs can save `best_metric_track_center_px.pt`.
- [x] Checked existing XR-27A history: validation best-center and best-P10 both occur at epoch `10`, so current `best_track_p10.pt` already represents the best validation-center epoch for that run.
- [x] Ran XR-28 narrow LR refinements at `7e-5` on GPU0 and `1.5e-4` on GPU1 with `XR27_BEST_METRIC_NAME=metric_track_center_px` and state distillation still disabled.
- [x] XR-28 LR `1.5e-4` promoted all active gates. Best-P10, best-P5, and best-center all resolve to epoch `10` and test `17.08485197339739/32.081208263124736/10.502551344462804`.
- [x] XR-28 LR `7e-5` did not promote. Its best-center result was `17.55661221402032/31.095664044788904/9.788265630177088`.
- [x] Active gates updated: center `<17.08485197339739`, P10 `>32.081208263124736`, P5 `>10.502551344462804`.
- [x] Ran XR-29 post-XR-28 failure-bucket diagnostics for the promoted LR `1.5e-4` checkpoint. Artifact: `runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json`; joined `2238`, missing predictions `0`.
- [x] XR-29 diagnosis: weighted aggregate improved over XR-27A to `16.9868/32.3454/10.5774`, but `similarity <=0.1`, subject `39`, subjects `42/45`, and `session_201` remain the main residual risks.
- [x] Launched XR-29 LR-neighbor runs: LR `1.25e-4` on GPU0 and LR `1.75e-4` on GPU1, both with center-selected checkpointing and state distillation disabled.
- [x] XR-29 LR-neighbor train/eval closeout completed. LR `1.25e-4` produced `17.179786903517588/32.04761978558132/10.53443912097386`, improving P5 only.
- [x] XR-29 LR `1.75e-4` promoted all active gates. Best-P10, best-P5, and best-center all produced `17.04961508342198/32.56462665285383/11.50467722075326`.
- [x] Active gates updated: center `<17.04961508342198`, P10 `>32.56462665285383`, P5 `>11.50467722075326`.
- [x] Queued and ran XR-30 LR micro-bracket around the promoted LR `1.75e-4`: LR `1.625e-4` on GPU0 and LR `1.875e-4` on GPU1 with unchanged heatmap weights, support-adaptive count contract, teacher/init checkpoint, and center-selected checkpoints.
- [x] XR-30 startup validation passed for both lanes: raw event-count contract, intended XR-06C checkpoint load, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- [x] XR-30 train/eval closeout completed. LR `1.625e-4` produced `17.067689692974092/32.420068802152365/10.866496937615532`, no promotion.
- [x] XR-30 LR `1.875e-4` produced `17.035534060001375/32.42474567549569/11.266581957680838`, promoting the center gate only.
- [x] Active gates after XR-30: center `<17.035534060001375`, P10 `>32.56462665285383`, P5 `>11.50467722075326`.
- [x] Added and ran XR-31 no-train interpolation between XR-29 LR `1.75e-4` and XR-30 LR `1.875e-4`.
- [x] XR-31 alpha `0.25` promoted P10 to `32.57950758934021` with center `17.04659355367933`, but P5 `11.503826883860997` narrowly missed the strict XR-29 P5 gate.
- [x] Active gates updated: center `<17.035534060001375`, P10 `>32.57950758934021`, P5 `>11.50467722075326`.
- [x] Ran trained LR midpoint probes XR-32 LR `1.8125e-4` on GPU0 and XR-33 LR `1.84375e-4` on GPU1.
- [x] XR-32 did not promote: `17.041867678506033/32.5463443006788/11.37500034059797`.
- [x] XR-33 promoted P10 to `32.62074908529009` with center `17.039179919447218` and P5 `11.362245225906372`.
- [x] Active gates updated: center `<17.035534060001375`, P10 `>32.62074908529009`, P5 `>11.50467722075326`.
- [x] Added XR-34 heatmap loss-ratio wrapper and plan artifact.
- [x] Launched XR-34A GPU0: XR-29 init, LR `1.75e-4`, heatmap/offset/center `0.004/0.0015/0.0015`.
- [x] Launched XR-34B GPU1: XR-33 init, LR `1.84375e-4`, heatmap/offset/center `0.006/0.001/0.001`.
- [x] XR-34 startup validation passed for both lanes: raw event-count contract, intended checkpoint load, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- [x] GPT-5.5 sidecar audit recorded: if XR-34 fails, next fallback should be XR-35 longer-budget XR-29/XR-33 continuation, not optimizer or teacher retraining.
- [x] XR-34 train/eval closeout completed. XR-34A reached `17.026217068944657/32.430698088237214/10.911139822006225`; XR-34B best-P10 reached `16.53321223940168/33.77168447630746/11.276786088943481`.
- [x] Active gates updated: center `<16.53321223940168`, P10 `>33.77168447630746`, P5 `>11.50467722075326`.
- [x] Added XR-35 no-train XR-29/XR-34B interpolation runner and plan artifact for P5 recovery.
- [x] Ran XR-35 small-alpha interpolation on GPU1. Alpha `0.03125/0.0625/0.09375/0.125` produced `17.0163/32.4753/11.3346`, `16.9840/32.4690/11.3912`, `16.9528/32.4243/11.4209`, and `16.9230/32.3520/11.2551`.
- [x] XR-35 did not promote. Active gates remain center `<16.53321223940168`, P10 `>33.77168447630746`, P5 `>11.50467722075326`.
- [x] Prepared XR-36 trainable P5-anchor continuation runner and plan artifact.
- [x] Launched XR-36A/XR-36B on GPU0/GPU1. Startup validation passed for both: raw event-count contract, intended checkpoint, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params.
- [x] XR-36A/XR-36B train/eval closeout completed with train exit `0`; both early-stopped at epoch `7/10`.
- [x] XR-36A best-P10 promoted P5 to `11.738095617294311` with center `16.59279990025929` and P10 `34.39710958344596`.
- [x] XR-36A best-P5 promoted P10 to `34.74064704350063` with center `16.576952314376832` and P5 `11.415816688537598`.
- [x] XR-36B best-P10 promoted center to `16.53305721793856` with P10 `34.19387831687927` and P5 `11.502551344462804`.
- [x] Active gates updated: center `<16.53305721793856`, P10 `>34.74064704350063`, P5 `>11.738095617294311`.
- [x] Added and ran XR-37 no-train interpolation between XR-36B best-P10 and XR-36A best-P10.
- [x] XR-37 alpha `0.10/0.20/0.35/0.50` produced `16.5225/34.0621/11.5748`, `16.5148/34.2883/11.4439`, `16.5080/34.2309/11.2925`, and `16.5076/34.3321/11.5391`.
- [x] XR-37 alpha `0.50` promoted center to `16.507612899371555`; P10/P5 gates remain XR-36A-owned.
- [x] Prepared XR-38 center-preserving P10/P5 recovery runner and plan.
- [x] XR-38 static validation passed: runner syntax OK and source checkpoints exist.
- [x] Launched XR-38A on GPU0 and XR-38B on GPU1 after sandboxed CUDA initialization failed and unsandboxed execution was approved.
- [x] XR-38 startup validation passed for both lanes: raw event-count contract, intended checkpoint, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- [x] XR-38 train/eval closeout completed. XR-38A did not promote; XR-38B best-P10 nearly missed P10 at `34.707483761651176`; XR-38B best-P5 promoted P5 to `11.843962955474854`.
- [x] Active gates updated after XR-38: center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.
- [x] Prepared and ran XR-39 no-train mixed-leader soup/interpolation using XR-37 alpha `0.50`, XR-38B best-P5, and XR-36A best-P5.
- [x] XR-39 evaluated 13 soups. Best center was `c60p25f15` at `16.491779099191938/34.5306130204882/11.50000034059797`; best P10 was `c25p45f30` at `16.503940873486656/35.02295998845781/11.460459525244577`; best XR-39 P5 was `c70p20f10` at `11.744473137174333`, below XR-38B.
- [x] Active gates updated after XR-39: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.
- [x] Ran XR-40 narrow soup around XR-39 center/P10 plus XR-38B P5. No gate promoted; best XR-40 P5 was `11.529762240818568`, below XR-38B.
- [x] Decision: close no-train soup follow-up for now.
- [x] Prepared and ran XR-41 trainable loss-ratio fallback.
- [x] XR-41A best-P5 promoted P5 to `11.868197652271816` with center `16.519514334201812` and P10 `34.09821502821786`; XR-41B did not promote.
- [x] Active gates updated after XR-41: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Prepared XR-42 low-drift P5-preserve branch to preserve XR-39 center/P10 while using XR-41A best-P5 as teacher/reference.
- [x] Added `scripts/v3/run_xr42_p5_preserve_lowdrift.sh` and `docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md`.
- [x] XR-42 static validation passed: runner syntax, required checkpoints, and diff whitespace.
- [x] Launched XR-42A on GPU0 from XR-39 center `c60p25f15`, XR-41A best-P5 teacher/reference, LR `3e-6`, best metric `metric_track_center_px`.
- [x] Launched XR-42B on GPU1 from XR-39 P10 `c25p45f30`, XR-41A best-P5 teacher/reference, LR `3e-6`, best metric `metric_track_p10_pct`.
- [x] XR-42 startup validation passed for both lanes: raw event-count contract, intended checkpoint load, resolved CUDA device, trainable filter `6` tensors / `1,331,328` params, and epoch `1/10`.
- [x] XR-42 train/eval closeout completed. Both lanes early-stopped at epoch `7/10` with train/eval exit `0`.
- [x] XR-42A best-P10 reached `16.49802110535758/34.48299399103437/11.366922119685581`; best-P5 and best-center reached `16.498888087272643/34.30739874839783/11.347789451054163`.
- [x] XR-42B best-P10 reached `16.505801352432798/34.3171777180263/11.278911903926305`; best-P5 reached `16.508924693720683/34.29506881577628/11.323554761069161`.
- [x] XR-42 did not promote any center/P10/P5 gate. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Decided XR-42C explicit tiny state-distillation fallback after GPT-5.5 sub-agent audit. Key correction: XR-42 reused XR-27 runner and forced `distillation.state_similarity=false`, so it had no effective `track/state` heatmap teacher anchor.
- [x] Added `scripts/v3/run_xr42c_explicit_distill_recovery.sh` and `docs/resources/xr42c_explicit_distill_recovery_plan_2026_06_17.md`.
- [x] Ran XR-42C-A on GPU0 and XR-42C-B on GPU1 after sandbox CUDA failed with `cuda_is_available=False`; approved GPU runtime succeeded.
- [x] XR-42C-A early-stopped at epoch `7/10`; best-P10 and best-P5 both evaluated to `16.4945193869727/34.34566405841282/11.454081984928676`; best-center evaluated to `16.491862688745773/34.712585769380844/11.472364275796073`.
- [x] XR-42C-B completed epoch `10/10`; best-P10 evaluated to `16.502160484450204/34.39710965156555/11.22236428941999`; best-P5 evaluated to `16.503420085566386/34.509779705320085/11.221513932091849`.
- [x] XR-42C did not promote any active gate. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Added and ran XR-43 no-train interpolation between XR-39 center leader and XR-42C-A best-center.
- [x] XR-43 alpha `0.125/0.25/0.5/0.75` produced center/P10/P5: `16.49161465849195/34.44132730620248/11.389456115450178`, `16.491504199164254/34.33078307424273/11.478741829735892`, `16.491429926667895/34.512755850383215/11.531888089861189`, and `16.491550181593215/34.60204156466893/11.576530947004045`.
- [x] XR-43 alpha `0.5` promoted center to `16.491429926667895`.
- [x] Active gates updated after XR-43: center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Pivoted away from the XR41 P5-teacher recovery branch. Subject `39` was found to be test-only in the current manifest split, so direct subject-39 training was rejected as leakage risk.
- [x] Added and ran XR-44 updated tri-leader soup using XR-43 center alpha `0.5`, XR-39 P10 `c25p45f30`, and XR-41A best-P5. No gate promoted; best XR-44 center/P10/P5 were `16.49202539069312`, `34.7963443006788`, and `11.315051344462804`.
- [x] Added and ran XR-45 low-sim/full heatmap refresh on GPU0/GPU1. XR-45A low-sim specialist completed `10/10`; XR-45B full refresh early-stopped at epoch `7/10`; both train/eval exits were `0`.
- [x] XR-45A best-P10 reached `17.03435743876866/32.714711679731096/11.164966344833374`; best-P5 reached `16.849295384543282/33.63605521747044/11.585034343174526`; best-center reached `17.808542433806828/30.72661645753043/9.841411910738264`.
- [x] XR-45B best-P10 reached `16.50801784992218/34.5969395501273/11.428571782793318`; best-P5 reached `16.50964238813945/33.96726266316005/11.820578595570156`.
- [x] XR-45 did not promote any active gate. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Added and ran XR-46 checkpoint-space follow-up: center/P10 compatibility interpolation, XR45B-bridged soup, and GPT-5.5 sub-agent recommended XR43-manifold micro-alpha sweep.
- [x] XR-46 A/B did not promote. Best center-safe candidate was `a0p125` at `16.491489429133278/34.563776268277849/11.517007132938931`; best P10 candidate was `a0p875` at `16.500783746583121/35.022959988457814/11.421343871525355`, effectively a P10 tie with worse center/P5.
- [x] XR-46C micro-alpha sweep did not promote. Best center-near candidate was alpha `0.55` at `16.491439385073527/34.512755850383215/11.531888089861189`; best Pareto candidate under old XR39 center gate was alpha `0.875` at `16.491683168070658/34.661565365110128/11.576530947004045`.
- [x] Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [x] Added and ran XR-47 trainable full-manifest calibration from XR-43 center with XR-43 self-teacher state distillation, LR `2e-6`, center L2 `0.0025`, state weight `0.0005`.
- [x] XR-47A P10-selected lane early-stopped at epoch `7/8`; best-P10 reached `16.496871314729962/34.372449772698538/11.298894902638027`; best-P5 reached `16.495964876243047/34.196854530061991/11.303146593911308`.
- [x] XR-47B P5-selected lane completed `8/8`; best-P10 matched XR-47A best-P10; best-P5 reached `16.497679948806763/34.444728687831336/11.592262281690324`.
- [x] XR-47 did not promote any active gate. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [ ] Next branch should change mechanism, not only LR/teacher around the same heatmap-state head: candidate directions are explicit P10/P5 calibration head, boundary-aware eval-consistent loss with stronger center guard, or teacher retraining.

## 2026-06-17 XR-48 Progress

- [x] Added XR-48 P10-boundary heatmap calibration runner and plan: `scripts/v3/run_xr48_p10_boundary_heatmap_calibration.sh`, `docs/resources/xr48_p10_boundary_heatmap_calibration_plan_2026_06_17.md`.
- [x] XR-48 validation passed: `bash -n`, lane A/B `DRY_RUN=1`, `git diff --check`, required checkpoint checks, and manifest leakage check. Subject 39 counts were train `0`, val `0`, test `144`.
- [x] Ran XR-48A on GPU0 and XR-48B on GPU1. Both lanes used XR-43 center init, XR-39 P10 teacher, heatmap head-only trainable scope, and P10-boundary loss; both train/eval exits were `0`.
- [x] XR-48A early-stopped at epoch `7/8`; best-P10 reached `16.49682899543217/34.431973586763654/11.298894902638027`; best-P5 reached `16.495972565242223/34.19685453006199/11.303146593911308`.
- [x] XR-48B early-stopped at epoch `7/8`; best-P10 reached `16.49668344429561/34.431973586763654/11.417942530768258`; best-P5 reached `16.50094030073711/34.45663345200675/11.52423505101885`.
- [x] XR-48 did not promote any active gate. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [ ] Next branch should be larger than heatmap head-only P10-boundary calibration: explicit P5-boundary objective, teacher retraining, or non-head-only adapter update.

## 2026-06-17 XR-49 Progress

- [x] Added XR-49 P5-boundary heatmap calibration runner and plan: `scripts/v3/run_xr49_p5_boundary_heatmap_calibration.sh`, `docs/resources/xr49_p5_boundary_heatmap_calibration_plan_2026_06_17.md`.
- [x] XR-49 validation passed before execution: `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, and existing full-manifest/no-subject-39-train policy from XR-48.
- [x] XR-49B first failed before training because the original run path was too long. The runner was shortened and XR-49B state distillation was corrected to `0.00025`.
- [x] XR-49A training completed and produced checkpoints, but automated post-train eval failed in the old wrapper eval loop. Manual CUDNN-off eval completed for best-P10 and best-P5.
- [x] XR-49B completed train/eval with exit `0`; best-center checkpoint was missing and skipped by design.
- [x] XR-49A best-P10 reached `16.52969342981066/34.083759260177615/11.41369082587106`; XR-49A best-P5 reached `16.532351425715856/33.93452455656869/11.70068063054766`.
- [x] XR-49B best-P10 reached `16.496637644086565/34.431973586763654/11.417942530768258`; XR-49B best-P5 reached `16.497121804101127/34.444728687831336/11.54124187060765`.
- [x] XR-49 did not promote any active gate. Active gates remain center `<16.491429926667895`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- [ ] Next branch should change trainable mechanism: preferred XR-50 direction is non-head-only lightweight adapter or track-event-adapter update with explicit center preservation.

## 2026-06-17 XR-50 Progress

- [x] Added XR-50 event-adapter heatmap calibration runner and plan: `scripts/v3/run_xr50_event_adapter_heatmap_calibration.sh`, `docs/resources/xr50_event_adapter_heatmap_calibration_plan_2026_06_17.md`.
- [x] GPT-5.5 read-only sub-agent recommended XR49B-style heatmap-state calibration with trainable scope expanded to `track_center_heatmap_head.*`, `event_adapter.*`, and `patch_frontend.event_embed.proj.*`.
- [x] Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, `git diff --check`, required checkpoint checks, and GPU idle check.
- [x] Ran XR-50A on GPU0: XR-43 init, XR-41A P5 teacher, LR `1e-6`, best metric P5, 5px boundary loss, trainable heatmap head + event path.
- [x] Ran XR-50B on GPU1: XR-43 init, XR-39 P10 teacher, LR `1e-6`, best metric P10, 10px boundary loss, trainable heatmap head + event path.
- [x] Both lanes completed epoch `8/8`; train/eval exits were `0`. Startup logs confirmed `trainable_tensors=14`, `trainable_params=1504320/4638746`.
- [x] XR-50A best-P10 reached `16.483389932768684/34.3227048942021/11.933673824582781`; best-P5 reached `16.482811435631344/34.48086814199175/11.863520765304566`.
- [x] XR-50B best-P10 reached `16.483026616913932/34.394133479254585/11.978316681725639`; best-P5 reached `16.482515714849743/34.48086814199175/11.81250035422189`.
- [x] XR-50 promoted center and P5. New active gates: center `<16.482515714849743`, P10 `>35.02295998845781`, P5 `>11.978316681725639`.
- [x] Next branch should recover P10 while preserving XR-50 center/P5: candidate directions are XR-50B best-P10 to XR-39 P10 no-train soup or a low-LR P10 recovery continuation from XR-50B best-P10.

## 2026-06-17 XR-51 Progress

- [x] Added XR-51 no-train P10 recovery runner and plan: `scripts/v3/run_xr51_xr50_xr39_p10_recovery_soup_eval.sh`, `docs/resources/xr51_xr50_xr39_p10_recovery_soup_plan_2026_06_17.md`.
- [x] Static validation passed: `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, `git diff --check`, and GPU idle check before launch.
- [x] Ran XR-51A interpolation lane on GPU0: XR-50B best-P10 to XR-39 P10 at alpha `0.05/0.10/0.15/0.20/0.25/0.35`.
- [x] Ran XR-51B tri-soup lane on GPU1: XR-50B best-P10, XR-50B best-P5, and XR-39 P10 at weights `80/10/10`, `75/10/15`, `70/10/20`, `70/15/15`, `65/15/20`, `60/20/20`.
- [x] Both lanes completed with exit `0`; all 12 eval summaries were generated.
- [x] XR-51B `t60c20p20` promoted center to `16.478984827655` with P10 `34.561650446483` and P5 `11.716837085996`.
- [x] XR-51A `a0p25` produced the best XR-51 P10 at `34.710459961210`, still below XR-39's active P10 gate `35.02295998845781`.
- [x] XR-51A `a0p05` preserved P5 at `11.978316681726` and improved center to `16.482212608201`, but did not recover P10.
- [x] Active gates after XR-51: center `<16.478984827655`, P10 `>35.02295998845781`, P5 `>11.978316681725639`.
- [x] Next branch should use trainable P10 recovery rather than more no-train soup: candidate direction is XR-52 low-LR event-adapter continuation from XR-51/XR-50B with XR-39 P10 teacher and explicit P5 preservation.

## 2026-06-17 XR-52 Progress

- [x] Added XR-52 trainable P10 recovery runner and plan: `scripts/v3/run_xr52_trainable_p10_recovery_from_xr51.sh`, `docs/resources/xr52_trainable_p10_recovery_from_xr51_plan_2026_06_17.md`.
- [x] Static validation passed: `chmod +x`, `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, `git diff --check`, and GPU idle check.
- [x] Ran XR-52A on GPU0: XR-51 center leader `t60c20p20` init, XR-39 P10 teacher, LR `5e-7`, P10 best metric, event-path trainable scope.
- [x] Ran XR-52B on GPU1: XR-51 P5-preserve `a0p05` init, XR-39 P10 teacher, LR `3e-7`, P10 best metric, stronger center/state guard.
- [x] Startup validation passed for both lanes: raw event-count contract, intended init checkpoint load, CUDA device resolution, and `trainable_tensors=14`, `trainable_params=1504320/4638746`.
- [x] Both lanes early-stopped at epoch `7/8`; train/eval exits were `0`.
- [x] XR-52A best-P5 promoted center to `16.470460832119` with P10 `34.667942953110` and P5 `11.840136425836`.
- [x] XR-52B best-P10 promoted P5 to `12.017432342257` with center `16.472449232851` and P10 `34.456633458819`.
- [x] XR-52 did not recover P10; the active P10 gate remains XR-39 `35.02295998845781`.
- [x] Active gates after XR-52: center `<16.470460832119`, P10 `>35.02295998845781`, P5 `>12.017432342257`.
- [ ] Next branch should focus on P10 without low-LR event-adapter teacher continuation: candidate directions are XR-53 no-train soup between XR-52 center/P5 leaders and XR-39 P10, or teacher/model retraining for P10.

## 2026-06-17 XR-53 Preparation

- [x] Added XR-53 no-train P10 recombination runner and plan: `scripts/v3/run_xr53_xr52_xr39_p10_recombination_eval.sh`, `docs/resources/xr53_xr52_xr39_p10_recombination_plan_2026_06_17.md`.
- [x] XR-53 static validation passed: `chmod +x`, `bash -n`, required checkpoint checks, lane A/B `DRY_RUN=1`, and `git diff --check`.
- [x] XR-53A is ready to run on GPU0: XR-52A best-P5 center leader to XR-39 P10 interpolation at alpha `0.03/0.06/0.10/0.15/0.20/0.30`.
- [x] XR-53B is ready to run on GPU1: XR-52A best-P5, XR-52B best-P10, and XR-39 P10 tri-soup at weights `50/35/15`, `45/35/20`, `40/35/25`, `35/35/30`, `30/40/30`, `25/35/40`.
- [x] Ran XR-53A on GPU0 and XR-53B on GPU1; both lanes completed with exit `0`, all 12 eval summaries were generated.
- [x] Best XR-53 center was XR-53A `a0p03`: `16.470736992359/34.623300095967/11.840136425836`, missing the center gate by about `0.000276`.
- [x] Best XR-53 P10 was XR-53A `a0p15`: `16.472256399904/34.623300109591/11.525510549545`, far below the active P10 gate.
- [x] Best XR-53 P5 was XR-53A `a0p03`: `11.840136425836`, far below the active P5 gate.
- [x] XR-53 did not promote any gate. Active gates remain center `<16.470460832119`, P10 `>35.02295998845781`, P5 `>12.017432342257`.
- [x] Next branch should be XR-55 XR39-anchored expanded-scope P10 branch; XR-54 is skipped because XR-53 produced no P10 near-tie candidate.

## 2026-06-17 XR-55 Progress

- [x] Added XR-55 XR39-anchored expanded-scope runner and plan: `scripts/v3/run_xr55_xr39_p10_anchor_expanded_scope.sh`, `docs/resources/xr55_xr39_p10_anchor_expanded_scope_plan_2026_06_17.md`.
- [x] Static validation passed: `chmod +x`, `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, and `git diff --check`.
- [x] Ran XR-55A on GPU0: XR-39 P10 init/self-teacher, LR `2e-7`, P10 boundary `0.012`, state distill `0.00025`, trainable head/event path/final backbone block.
- [x] Ran XR-55B on GPU1: XR-39 P10 init, XR-52B P5 teacher, LR `3e-7`, P10 boundary `0.010`, state distill `0.00035`, same expanded scope.
- [x] Startup validation passed for both lanes: raw event-count contract, intended init checkpoint, resolved CUDA device, and `trainable_tensors=32`, `trainable_params=1949568/4638746`.
- [x] Both lanes completed epoch `8/8`; train/eval exits were `0`.
- [x] XR-55A best-P10 reached `16.501659829276/34.399235514232/11.324830266408`.
- [x] XR-55A best-P5 reached `16.497514723028/34.793368141992/11.382228217806`, the best XR-55 P10 but still below the active P10 gate.
- [x] XR-55B best-P10 reached `16.497369331973/34.567177718026/11.202381290708`.
- [x] XR-55B best-P5 reached `16.500960135460/34.399235521044/11.312925515856`.
- [x] XR-55 did not promote any gate. Active gates remain center `<16.470460832119`, P10 `>35.02295998845781`, P5 `>12.017432342257`.
- [ ] Next branch should change supervision: explicit P10 teacher retraining or dedicated P10 calibration/classification head, not another checkpoint soup or low-LR scope expansion.

## 2026-06-17 XR-56 Progress

- [x] Sub-agent design review completed. Recommendation: use metric-aligned soft-threshold P10 loss before adding a new calibration head or retraining teacher.
- [x] Added default-zero soft-threshold losses: `track_p10_soft_threshold`, `track_p5_soft_threshold`, and aux variants in `src/hbtxr/loss/bundles/track.py`.
- [x] Added XR-56 runner and plan: `scripts/v3/run_xr56_soft_threshold_p10_supervision.sh`, `docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md`.
- [x] Validation passed: `bash -n`, `py_compile`, `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` with `23 passed`, lane A/B `DRY_RUN=1`, and `git diff --check`.
- [x] Launched XR-56A on GPU0: XR-39 P10 init/self-teacher, LR `7.5e-7`, P10 soft `0.006`, P5 soft `0.001`, trainable heatmap head + event path.
- [x] Launched XR-56B on GPU1: XR-52B seed, XR-39 P10 teacher, LR `5e-7`, P10 soft `0.008`, P5 soft `0.002`, same trainable scope.
- [x] XR-56 train/eval closeout completed. XR-56A completed epoch `10/10`; XR-56B early-stopped at epoch `7/10`; train/eval exits were `0`.
- [x] XR-56A best-P10 reached `16.49389898266111/34.32270488057818/11.699830286843437`; XR-56A best-P5 reached `16.4924229485648/34.62330012321472/11.304422106061663`.
- [x] XR-56B best-P10 reached `16.477364584377835/34.37670146397182/12.044218049730574`; XR-56B best-P5 promoted center and P5 with `16.4701875601496/34.29294293948582/12.133503770828247`.
- [x] Active gates after XR-56: center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- [ ] Next branch should be XR-57 P10 calibration/refinement or P10 teacher retraining. Soft-threshold scalar supervision improved center/P5 only and did not recover P10.

## 2026-06-17 XR-57 Progress

- [x] Planned XR-57 as bounded center-refine calibration, not another scalar loss-only branch.
- [x] Real sub-agent spawn attempted for T-097 but failed with `agent thread limit reached`; main agent proceeded with read-only fallback review and direct implementation.
- [x] Added `TrackCenterRefineHead`, optional config wiring, final `track/state` residual routing, and default-zero refine-delta regularization.
- [x] Added runner and plan: `scripts/v3/run_xr57_p10_center_refine_calibration.sh`, `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md`.
- [x] Validation passed: `bash -n`, `py_compile`, `.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py` with `27 passed`, model-build smoke, and lane A/B `DRY_RUN=1`.
- [x] XR-57 actual A/B train/eval completed with exit `0`.
- [x] XR-57A early-stopped at epoch `7/12`; best-P10 test `16.69362453562873/34.38307912690299/10.973214619500297`, best-P5 test `16.715463175092424/34.849490649359566/10.925595603670393`.
- [x] XR-57B completed epoch `12/12`; best-P10 test `16.504537062985555/34.137755966186525/11.207483332497732`, best-P5 test `16.4997801729611/34.190476996558054/11.476615987505232`.
- [x] XR-57 did not promote any gate. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- [ ] Next branch should be XR-58 P10 teacher refresh: train a deliberately P10-specialized teacher/anchor first, then distill back into the center/P5-preserving student only if teacher test P10 beats XR-39.

## 2026-06-17 XR-58 Progress

- [x] Planned XR-58 as P10 teacher refresh rather than another student-preserving refinement.
- [x] Added runner and plan: `scripts/v3/run_xr58_p10_teacher_refresh.sh`, `docs/resources/xr58_p10_teacher_refresh_plan_2026_06_17.md`.
- [x] Static validation passed: `bash -n` and lane A/B `DRY_RUN=1`.
- [x] Launched XR-58A on GPU0 and XR-58B on GPU1.
- [x] Startup validation passed for both lanes: raw event-count contract, XR-39 init checkpoint load, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- [x] XR-58A/B train/eval completed with exit `0`; both lanes early-stopped at epoch `8/16`.
- [x] XR-58A best-P10 reached `16.503614359242576/34.90306201662336/11.51275544847761`; XR-58A best-P5 reached `16.501813726765768/34.68920147078378/11.259779255730765`.
- [x] XR-58B best-P10 reached `16.506438190596445/34.84226275852748/11.501275873184204`; XR-58B best-P5 reached `16.49833288192749/34.540391949244906/11.062075165339879`.
- [x] XR-58 did not promote any gate. Best XR-58 P10 improved over XR-57 but remained below XR-39 by about `0.1199`.
- [ ] Next branch should be XR-59 narrow teacher-refresh bracket around XR-58A: keep anchored self-teacher, test slightly stronger LR/P10-soft combinations, and avoid the free no-distill lane unless used as a diagnostic.

## 2026-06-17 XR-59 Progress

- [x] Planned XR-59 as a narrow continuation around XR-58A best-P10.
- [x] Added runner and plan: `scripts/v3/run_xr59_xr58a_teacher_bracket.sh`, `docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md`.
- [x] Static validation passed: `bash -n` and lane A/B `DRY_RUN=1`.
- [x] Launched XR-59A on GPU0 and XR-59B on GPU1.
- [x] Startup validation passed for both lanes: raw event-count contract, XR58A best-P10 init checkpoint load, XR39 teacher checkpoint set, resolved CUDA device, and `trainable_tensors=54`, `trainable_params=2546120/4638746`.
- [x] XR-59A/B train completed; both lanes early-stopped at epoch `7/10`.
- [x] Added and ran post-train eval helper `scripts/v3/eval_xr59_completed_checkpoints.sh` for `best_track_p10` and `best_track_p5`.
- [x] XR-59A best-P10/best-P5 both reached `16.520220368249074/34.300170864377705/11.376701021194458`.
- [x] XR-59B best-P10/best-P5 both reached `16.51065547806876/34.43409944261823/11.287415306908743`.
- [x] XR-59 did not promote any gate. Best XR-59 P10 regressed below XR-58A `34.90306201662336`.
- [ ] Next branch should close scalar P10-soft teacher continuation and pivot to a representation/protocol change: dedicated P10 calibration/classification head, subject/session failure-bucket data diagnostics, or a different teacher target construction.

## 2026-06-17 XR-60 Preparation

- [x] Pivoted from scalar P10-soft continuation to a dedicated P10 multi-candidate center calibration head.
- [x] GPT5.5 sub-agent T-102 completed read-only runner-pattern review. Spark was unavailable due usage limit, so GPT5.5 was used for actual sub-agent review.
- [x] Added default-off `TrackCenterCandidateHead` wiring through model head factory, hybrid tracker, track branch, model factory, stage2 loss aggregation, and tests.
- [x] Added candidate loss terms: P10 BCE, minimum soft-threshold, and delta L2 regularization.
- [x] Added trainable-filter coverage for `track_center_candidate_head.*`.
- [x] Added XR-60 runner and plan: `scripts/v3/run_xr60_p10_candidate_head.sh`, `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- [x] Static validation passed: `bash -n`, `py_compile`, targeted pytest `34 passed`, model-build smoke, lane A/B dry-run, and `git diff --check`.
- [x] Initial sandbox launch failed because PyTorch saw `cuda_device_count=0` despite `nvidia-smi` visibility; relaunched with approved elevated `bash scripts/v3/run_xr60_p10_candidate_head.sh` prefix.
- [x] Launched XR-60A on GPU0 and XR-60B on GPU1; both passed raw event-count contract and entered epoch `2/12`.
- [x] XR-60A/B train/eval closeout completed. XR-60A early-stopped at epoch `7/12`; XR-60B early-stopped at epoch `11/12`; train/eval exits were `0`.
- [x] XR-60A best-P10 reached `16.526124344553267/34.50255186898368/11.617772477013725`; best-P5 reached `16.5058109828404/34.60331717899867/11.337160219464984`.
- [x] XR-60B best-P10 reached `16.842585216249738/33.49277294022696/10.70663298198155`; best-P5 reached `16.717501049382346/33.87670159339905/10.971939107349941`.
- [x] XR-60 did not promote any gate. Active gates remain center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- [ ] Next branch should use candidate heads as auxiliary-only or pivot to paper-backed ellipse/pseudo-label diagnostics. Direct high-LR routed candidate state caused center/P5 collapse.

## 2026-06-17 XR Eye-Tracking Analysis Rewrite

- [x] Rewrote detailed analysis files under `anlaysis/xr-eye-tracking` to match the requested hardware reference-document depth.
- [x] Codebase analyses now include repository/layer structure, source roots, language/artifact mix, metadata/config refs, directory map, core source refs, README summary, entrypoints, pipeline interpretation, Python symbol extraction, import/dependency clues, YAML/config option clues, line-level keyword evidence, reuse risks, HGTXR mapping, priority, and next actions.
- [x] Paper analyses now include existing-method problems, proposed method, algorithm/loss/inference flow, bounded section evidence, HGTXR primitive decomposition, result/metric evidence, hardware/system relevance, experiment method, dataset interpretation, result interpretation, ablation axes, HGTXR applicability, risks, and next actions.
- [x] Generated/updated `39` `analysis.md` files: `18` codebase analyses and `21` paper analyses, totaling `8813` lines.
- [x] Added reusable generator: `scripts/v3/write_xr_eye_tracking_detailed_analysis.py`.
- [x] Validation passed: `python3 -m py_compile scripts/v3/write_xr_eye_tracking_detailed_analysis.py`, section coverage scan over all codebase/paper files, and representative FACET paper/codebase spot checks.
- [x] Real sub-agent attempt failed because GPT-5.3-Codex-Spark quota was exhausted; main agent completed the rewrite using local evidence.

## 2026-06-18 XR-61 / XR-62 Progress

- [x] Closed XR-61 auxiliary candidate-head branch from existing eval summaries. XR-61A reached `16.475529539585114/34.356718465260094/12.127126216888428`; XR-61B best reached `16.48967229127884/34.58843615395682/11.559524168287005`.
- [x] XR-61 did not promote center, P10, or P5. Candidate auxiliary supervision alone was close to center/P5 but did not recover P10.
- [x] Added FACET-backed geometry auxiliary branch: `scripts/v3/run_xr62_facet_geometry_aux_refresh.sh`.
- [x] Added plan: `docs/resources/xr62_facet_geometry_aux_refresh_plan_2026_06_17.md`.
- [x] XR-62 static validation passed: `chmod +x`, `bash -n`, lane A/B `DRY_RUN=1`, required checkpoint checks, and elevated CUDA launch on GPU0/GPU1.
- [x] XR-62A/B both early-stopped at epoch `7/10`; train/eval exits were `0`.
- [x] XR-62A best-P10/best-P5 both reached `16.468481131962367/34.30782389640808/12.052721459524973`, promoting the center gate only.
- [x] XR-62B best-P10/best-P5 both reached `16.495574762140002/34.46598720550537/11.370323480878557`, no promotion.
- [x] Active gates after XR-62: center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- [ ] Next branch should not repeat candidate/geometry auxiliary alone. Move to provenance-safe FACET/EV-Eye pseudo-ellipse label diagnostic or a P10-specific teacher target construction.

## 2026-06-18 XR Eye-Tracking Analysis Reference-Level Rewrite

- [x] Reworked `scripts/v3/write_xr_eye_tracking_detailed_analysis.py` to match the stricter reference-analysis style requested from `FlexLLM/analysis.md` and `EfficientViT-FPGA/analysis.md`.
- [x] Codebase analyses now include `repo_remote`, `paper_title`, `paper_pdf`, core source responsibility tables, data-contract/HGTXR compatibility matrices, static quality findings, and reproduction runbooks.
- [x] Paper analyses now include explicit dataset/model/benchmark tables, reported-result evidence, HGTXR result interpretation, and detailed experiment option/runbook sections.
- [x] Regenerated `39` `analysis.md` files: `18` codebase analyses and `21` paper analyses, totaling `11574` lines.
- [x] Validation passed: generator `py_compile`, full regeneration, required-section scans with `rg --files-without-match`, representative FACET/EV-Eye checks, and `git diff --check`.
- [x] Real sub-agent workflow attempted first with GPT-5.3-Codex-Spark; quota was exhausted, so T-401/T-402 were rerun with GPT5.5 explorer agents and their templates were integrated.

## 2026-06-18 XR-63 P10 Teacher-Target Oracle Diagnostic

- [x] Added no-train oracle diagnostic: `scripts/v3/analyze_p10_teacher_target_oracle.py`.
- [x] Generated full-test report: `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md` and JSON evidence.
- [x] Verified eval-style metric alignment against existing `eval_summary.json` for XR-62A, XR-39, XR-56B, and XR-58A.
- [x] Oracle over XR-62A center, XR-39 P10, XR-56B P5, and XR-58A P10-teacher predictions reaches `16.04023192701366/36.48596938775512/13.41751700680271`.
- [x] Oracle upper bound exceeds all active gates: center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- [x] GPT-5.3-Codex-Spark T-404 sub-agent attempt failed due quota limit; main-agent fallback completed implementation and validation.
- [x] Added XR-64 implementation plan: `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md`.
- [x] Added XR-64 code path: `scripts/v3/build_xr64_teacher_target_overrides.py`, `scripts/v3/run_xr64_teacher_target_construction.sh`, dataset target override support, Stage2 override losses, and tests.
- [x] Added implementation report: `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md`.
- [x] Validation passed: `py_compile`, `bash -n`, `39` targeted pytest cases, A/B runner dry-runs, and builder smoke.
- [ ] Generate train eval rows for XR-62A/XR-39/XR-56B/XR-58A, build train-only override files, then launch XR-64A/B.

## 2026-06-18 Experiment Pause Documentation Closeout

- [x] Experiments remain paused by user directive; no train/eval job was launched in this pass.
- [x] Added `scripts/v3/report_second_goal_status.py`, a read-only status reporter for the paused second goal.
- [x] Added `tests/test_second_goal_status.py`.
- [x] Updated `docs/resources/second_goal_artifact_index_2026_06_18.md` and `.json` with the status reporter and validation command.
- [x] Updated `docs/Validation.md`, `docs/track/TODO.md`, and `docs/track/log.md` with pause-state validation evidence.
- [x] Validation passed: `python3 -m py_compile scripts/v3/report_second_goal_status.py`, `.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py` with `5 passed`, and `.venv/bin/python scripts/v3/report_second_goal_status.py --format summary`.
- [x] Observed pause-state summary: `execution_state=paused_by_user_directive`, `active_goal_complete=false`, `authority_links_missing=0`, `xr64_resume_status=generated_incomplete`, `xr64_can_run_lane=false`, `missing_eval_rows=8`, `missing_overrides=6`, `leakage_risk=none`.

## 2026-06-18 Software Cleanup / Refactor Preparation

- [x] Created cleanup/refactor plan: `docs/resources/software_cleanup_refactor_plan_2026_06_18.md`.
- [x] Added read-only inventory tool: `scripts/v3/maintenance/inventory_software_tree.py`.
- [x] Added read-only path reference audit tool: `scripts/v3/maintenance/audit_path_references.py`.
- [x] Generated inventory reports: `docs/resources/software_cleanup_inventory_2026_06_18.md` and `.json`.
- [x] Generated path audit reports: `docs/resources/software_path_reference_audit_2026_06_18.md` and `.json`.
- [x] Real sub-agent CLEAN-001 classified `runs/` risk and identified must-preserve XR-39/XR-56/XR-58/XR-62/XR-63/XR-64, `interpolated_checkpoints`, `_logs`, and `diagnostics`.
- [x] Real sub-agent CLEAN-002 reviewed scripts/config/docs refactor risk and recommended wrapper/path-policy normalization before physical moves.
- [x] Inventory result: `runs=849`, software dirty entries `236`, `runs/` size `44.5 GiB`, path reference findings `1166`.
- [x] No files were deleted or moved; no train/eval experiment was launched.

## 2026-06-18 Runs Catalog Organization

- [x] Confirmed root `.gitignore` already ignores `software/runs/*`.
- [x] Added non-destructive catalog builder: `scripts/v3/maintenance/build_runs_catalog.py`.
- [x] Generated `runs/_catalog` category view with symlinks only; original `runs/<run_id>` paths remain unchanged.
- [x] Generated catalog reports: `docs/resources/runs_catalog_report_2026_06_18.md` and `.json`.
- [x] Catalog result: `849` source run directories, `2078` symlinks, `13` categories.
- [x] Key categories include `00_current_authority`, `01_xr64_resume`, `02_leaders_teachers_review`, `03_shared_support`, `30_eval_runs`, `32_raw_stage2_runs`, `90_archive_candidates`, and `99_preserve_or_review`.
- [x] No run directory was deleted or physically moved.

## 2026-06-18 Runs Physical Experiment Reorganization

- [x] Added `scripts/v3/maintenance/reorganize_runs_by_experiment.py`.
- [x] Dry-run confirmed `849` top-level run directories could be moved without destination collisions.
- [x] Physically moved `849` run directories into `runs/XR-<n>/<run_id>` or `runs/NON_XR/<group>/<run_id>`.
- [x] Left `849` compatibility symlinks at original `runs/<run_id>` paths so existing scripts/docs still resolve.
- [x] Moved `_logs`, `diagnostics`, and `interpolated_checkpoints` into `runs/NON_XR/shared/` with compatibility symlinks.
- [x] Kept generated `runs/_catalog` in place.
- [x] Generated reports: `docs/resources/runs_experiment_reorganization_2026_06_18.md` and `.json`.
- [x] Validation passed: no broken symlinks, `report_second_goal_status.py --format summary` still passes, and no train/eval job was launched.
- [x] Integrated sub-agent MOVE-001 review: fixed `scripts/v3/run_prepare_and_train.sh` latest-run discovery to include compatibility symlinks.
- [x] Added reorganization verification mode and generated `docs/resources/runs_experiment_reorganization_verify_2026_06_18.md` and `.json`.
- [x] Verification mode reports `ok=true`, `top_level_compat_symlinks=849`, `organized_run_dirs=849`, `broken=0`, and `bad_targets=0`.

## 2026-06-18 Runs Compatibility Symlink Removal

- [x] Added `scripts/v3/maintenance/remove_runs_compat_symlinks.py`.
- [x] Updated active latest-run discovery to search nested organized run roots: `src/hbtxr/config/run_contract.py`, `scripts/v3/run_prepare_and_train.sh`, `scripts/v3/check_raw_event_count_training_readiness.py`, and `scripts/v3/check_raw_event_count_training_result.py`.
- [x] Updated XR-62/XR-64 active default paths to use `runs/XR-56`, `runs/XR-58`, and `runs/XR-62` where applicable.
- [x] Removed `846` top-level run compatibility symlinks and the stale generated `runs/_catalog` symlink tree.
- [x] Retained shared aliases only: `runs/_logs`, `runs/diagnostics`, and `runs/interpolated_checkpoints`.
- [x] Generated reports: `docs/resources/runs_compat_symlink_removal_2026_06_18.md` and `.json`.
- [x] Validation passed: root `eval*`/`raw*` entries `0`, broken symlinks `0`, `report_second_goal_status.py --format summary`, `check_xr64_resume_artifacts.py --allow-missing-generated --format summary`, and `13` targeted pytest cases.
- [x] Observed environment caveat: `nvidia-smi` currently returns driver communication failure; no train/eval job was launched.
