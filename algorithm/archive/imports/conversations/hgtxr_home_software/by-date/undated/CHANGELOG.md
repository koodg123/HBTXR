# HGTXR-SW Changelog

## 2026-06-15

- Added XR-Eye-Tracking reference analysis under `anlaysis/xr-eye-tracking`, including per-codebase and per-paper folders.
- Added `anlaysis/xr-eye-tracking/experiment_integration.md` with prioritized second-goal experiments.
- Integrated XR-Eye-Tracking-derived P0/P1/P2 experiments into the paper-backed experiment plan, master plan, sub-agent plan, TODO, and progress tracking.
- Added detailed XR-Eye-Tracking analysis refresh:
  - `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md` with SRC_CASE_MODULE_GUIDE-style code/module/dataflow/call-stack analysis.
  - `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md` with detailed problem, method, algorithm, hardware architecture, experiment, dataset, result, option, and HGTXR mapping fields.

## 2026-06-16

- Added XR-27 track-center heatmap-state representation and runner; clean LR `1e-4` run promoted all active gates to center `17.2746`, P10 `31.9154`, P5 `10.2900`.
- Updated the second-goal plan to treat XR-27 as completed/promoted and queue XR-28 consolidation diagnostics instead of repeating LR-only, hinge, boundary, or broad optimizer sweeps.
- Generated XR-28 failure-bucket diagnostics comparing XR-27A against XR-06C, XR-20A, and XR-22; XR-27A improves weighted overall metrics to `17.1708/32.1921/10.3219` and mainly fixes `similarity_target <= 0.3`.
- Integrated detailed XR-Eye-Tracking analyses into the second-goal plan and clarified `XR-01` training-track vs `XR-02` no-retrain post-process ordering.
- Added `scripts/v3/eval_eyelorin_refinement.py` for EyeLoRiN-style M2F median refinement evaluation with time-gap gating, before/after center/P10/P5 metrics, and jitter proxy metrics.
- Added `tests/test_eyelorin_refinement.py`.
- Ran XR-02 smoke on AdamW fixed255k best-center checkpoint with 64 test rows; sparse trajectory gap gating prevented invalid smoothing and produced no metric change, so dense trajectory evaluation is still required.
- Started XR-01 AdamW local count bracket outside the sandbox in tmux: fixed250k on GPU0 and fixed260k on GPU1 at LR `8e-6`.
- Added `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`, linking the PAPER_REF 30-paper corpus to concrete second-goal Head/Loss/LR/Optimizer/Distillation/Teacher/Gating/Hardware experiment axes.
- Added `scripts/v3/run_xr03_ellipsestate_adamw_leader_probe.sh` for AdamW-leader warm-start ellipse-state axis/angle loss probing.
- Added `scripts/v3/compare_xr01_adamw_bracket.py` and made the XR-03 runner init checkpoint overrideable for XR-01-promoted leaders.
- Completed XR-01 AdamW fixed250k/fixed260k LR `8e-6` count bracket; all four best-center/best-P10 test evals failed the current leader promotion gate.
- Extended `scripts/v3/compare_xr01_adamw_bracket.py` to scan the next fixed255k LR `6e-6` and `1e-5` branch.
- Started XR-01 fixed255k LR bracket in tmux: `6e-6` on GPU0 and `1e-5` on GPU1.
- Added `scripts/v3/summarize_eval_failure_buckets.py` and generated `runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json`; low-similarity rows and specific `session_201` buckets are now identified as the dominant leader failure modes.
- Completed XR-01 fixed255k LR bracket. LR `6e-6` failed; LR `1e-5` best-center promoted to new software leader with test center `26.1749`, P10 `16.5021`, P5 `5.0999`. LR `1e-5` best-P10 also promoted on P10 but remains secondary because center/P5 are weaker.
- Launched XR-03 ellipse-state geometry probes from the promoted LR `1e-5` leader checkpoint: main LR `1e-5` branch on GPU0 and stability LR `6e-6` branch on GPU1.
- Completed XR-03 ellipse-state geometry probes. LR `1e-5`, axis `0.025`, angle `0.01` best-center promoted to the new overall software leader with test center `21.6822`, P10 `21.8202`, P5 `7.0446`; next queue is XR-03A/B stronger axis/angle `0.05/0.02` from this checkpoint.
- Launched XR-03A/B stronger geometry sweep from the promoted XR-03 checkpoint: GPU0 LR `1e-5` and GPU1 LR `6e-6`, both axis `0.05`, angle `0.02`.
- Completed XR-03A/B stronger geometry sweep. XR-03A LR `1e-5`, axis `0.05`, angle `0.02` best-center promoted to the current overall software leader with test center `20.4539`, P10 `25.8686`, P5 `8.3104`; XR-03B LR `6e-6` also promoted but remains secondary.
- Launched XR-03C/D bounded stronger-geometry sweep from XR-03A best-center: GPU0 LR `1e-5` and GPU1 LR `6e-6`, both axis `0.075`, angle `0.03`.
- Completed XR-03C/D bounded stronger-geometry sweep. XR-03D LR `6e-6`, axis `0.075`, angle `0.03`, best-P10 checkpoint promoted as the center-first leader with test center `20.4336`, P10 `25.7866`, P5 `7.7491`; XR-03A remains P10/P5-balanced secondary.
- Added XR-04 low-similarity-aware track-loss weighting, config, and runner; validation passed.
- Launched XR-04 low-similarity-aware probes from the XR-03D center leader checkpoint: GPU0 LR `6e-6`, GPU1 LR `1e-5`, both threshold `0.3`, scale `1.0`.
- Completed XR-04 low-similarity-aware closeout. No promotion; closest branch was LR `6e-6` best-center with center `20.4347`, P10 `25.7844`, P5 `8.4286`.
- Added XR-05A direct track-state auxiliary head, config, runner, and tests.
- Launched XR-05A probes from the XR-03D center leader checkpoint: GPU0 LR `6e-6`, GPU1 LR `3e-6`, both aux center/axis/angle `0.001/0.025/0.01`.
- Completed XR-05A closeout. GPU0 LR `6e-6` best-P10 promoted as center-first leader with center `20.3170`; GPU1 LR `3e-6` best-P10 became P10 leader with P10 `26.5761`; GPU1 best-center became P5/balanced secondary with P5 `8.6956`.
- Started XR-05B lighter direct-aux refinement on GPU1 from the XR-05A LR `3e-6` best-P10 checkpoint: aux `0.0005/0.0125/0.005`, LR `2e-6`.
- Started XR-05C midpoint-LR direct-aux refinement on GPU0 from the XR-03D best-P10 checkpoint: aux `0.001/0.025/0.01`, LR `4.5e-6`.
- Completed XR-05B/XR-05C closeout. XR-05B best-P10 promoted as new center-first leader with center `20.2927`; XR-05C did not promote.
- Started XR-05D/XR-05E refinements: XR-05D GPU1 ultra-light aux `0.00025/0.00625/0.0025`, LR `1e-6`; XR-05E GPU0 mid-light aux `0.00075/0.01875/0.0075`, LR `1.5e-6`.
- Completed XR-05D/XR-05E closeout. XR-05D did not promote. XR-05E best-center promoted the P10 gate to `26.6539`, with center `20.3673` and P5 `8.4830`.
- Added `configs/v3/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_finetune_fullwidth.yaml` and `scripts/v3/run_xr06_weakdistill_trackstateaux_probe.sh` as a validated weak-distill fallback.
- Added `scripts/v3/run_xr07_trackstateaux_centerhinge_probe.sh` and launched XR-07A/B tiny center-hinge recovery from XR-05B on GPU1/GPU0.
- Closed XR-07A/B as no-promotion on validation metrics and recorded XR-02 sparse-manifest gap statistics; XR-06 remains the next prepared fallback.
- Launched XR-06 weak-distill fallback on GPU1 from XR-05B best-P10 as both init and teacher.
- Launched XR-06B weak-distill complementary branch on GPU0 from XR-05E best-center as both init and teacher.
- Completed XR-06A closeout. Best-center promoted the center gate to `20.2838` with P10 `26.2772` and P5 `8.4724`; best-P10 did not promote.
- Completed XR-06B closeout. Best-center `20.2950/26.2177/8.5191` and best-P10 `20.4695/25.9320/8.6310`; no promotion.
- Launched XR-05F no-distill control on GPU1 from XR-05E best-center using lighter aux `0.0005/0.0125/0.005` and LR `1e-6`.
- Completed XR-05F no-distill control closeout. Best-center `20.3562/26.4328/8.2980` and best-P10 `20.3195/26.1509/8.1429`; no promotion.
- Launched XR-06C/XR-05G ultra-low-LR polish from XR-06A best-center: XR-06C weak-distill on GPU0 and XR-05G no-distill on GPU1, both aux `0.0005/0.0125/0.005`, LR `5e-7`.
- Completed XR-05G closeout with no promotion. Best-center `20.2920/26.4422/8.4566`; best-P10 `20.3760/26.2742/8.2815`.
- Completed XR-06C closeout. Best-center promoted center to `20.2831`; best-P10 promoted P10 to `26.7449`.
- Launched XR-06D/XR-06E from XR-06C leaders at LR `2.5e-7`: XR-06D no-distill center-preserve on GPU0 and XR-06E weak-distill P10-preserve on GPU1.
- Completed XR-06D/XR-06E closeout. Neither promoted: XR-06D `20.2956/26.1586/8.5047`, XR-06E best-center `20.2942/26.2806/8.5387`, XR-06E best-P10 `20.3149/26.1339/8.6811`.
- Added XR-08 checkpoint interpolation utility/runner and completed alpha `0.50` smoke. Result `20.3259/26.4983/8.3865`; no promotion.
- Launched XR-05I no-distill and XR-06F weak-distill P5-preserve branches from the XR-05A P5/balanced checkpoint at aux `0.001/0.025/0.01`, AdamW LR `5e-7`, on GPU0/GPU1.
- Completed XR-05I/XR-06F closeout. XR-05I best-center/best-P10 produced `20.3510/26.1501/8.4290` and `20.3939/26.1701/8.4503`; XR-06F best-center/best-P10 produced `20.3222/26.2007/8.6259` and `20.4218/26.3946/8.6301`. No active gate promoted; next priority shifts to targeted XR-04B failure-bucket training or dense-trajectory XR-02.
- Launched XR-04B/XR-04C targeted low-similarity branches from XR-03D best-P10. XR-04B uses threshold `0.1`, scale `2.0`, LR `6e-6` on GPU0; XR-04C uses threshold `0.1`, scale `2.0`, LR `3e-6` on GPU1. Startup validation passed for both.
- Completed XR-04B/XR-04C targeted low-similarity closeout. XR-04B best-center/best-P10 produced `20.4841/25.6003/8.2776` and `20.6738/25.4014/7.8754`; XR-04C best-center/best-P10 both produced `20.5883/25.6582/7.8499`. No active gate promoted; next priority moves to manifest/sampler-level failure-bucket targeting or dense-trajectory XR-02.
- Added failure-bucket manifest subset tooling and launched XR-04D/XR-04E subset experiments on `similarity_target <= 0.1` + `session_201`. Subset counts are train `1186`, val `182`; XR-04D uses LR `6e-6` on GPU0 and XR-04E uses LR `3e-6` on GPU1.
- Completed XR-04D/XR-04E manifest-subset closeout. XR-04D best-center/best-P10 produced `24.5081/16.7526/4.5179` and `25.7039/15.3074/4.2381`; XR-04E best-center/best-P10 produced `23.5577/18.1594/5.1063` and `23.4280/18.2555/5.0310`. No active gate promoted; hard-subset training is closed as a full-test accuracy path.
- Added XR-09 full-manifest weighted-sampler support in the dataloader, plus weighted-sampler config, runner, and tests. Initial train-manifest weight distribution for low-sim `3x` + session_201 `2x` + cap `6x` is `6x=1186`, `3x=438`, `2x=1686`, `1x=2619`.
- Launched XR-09A/XR-09B weighted-sampler branches in tmux on GPU0/GPU1; startup validation passed for raw contract, checkpoint load, CUDA device resolution, full train/val manifest counts, and epoch `1/12` entry.
- Completed XR-09A/XR-09B weighted-sampler closeout. XR-09A best-center/best-P10 produced `20.5793/25.6327/8.0102` and `20.6414/24.7428/8.0412`; XR-09B best-center/best-P10 produced `20.4154/26.1964/8.3057` and `20.5593/25.6824/7.9328`. No active gate promoted.
- Added XR-10 full-manifest loss-side sample weighting in Stage2 track losses, plus loss-weight config, runner, and tests. Initial train-manifest loss-weight distribution for low-sim `2x` + session_201 `1.5x` + cap `3x` is `3x=1186`, `2x=438`, `1.5x=1686`, `1x=2619`.
- Launched XR-10A/XR-10B loss-weight branches in tmux on GPU0/GPU1; startup validation passed for raw contract, checkpoint load, CUDA device resolution, full train/val manifest counts, and epoch `1/12` entry.
- Completed XR-10A/XR-10B loss-weight closeout. XR-10A best-center/best-P10 produced `20.3488/26.2062/8.1280` and `20.4297/26.0480/7.9996`; XR-10B best-center/best-P10 produced `20.3266/25.9702/8.1335` and `20.3897/26.1871/7.9945`. No active gate promoted.
- Added XR-11 paper-backed track-state SimDR auxiliary head/loss/config/runner. Validation passed: py_compile, bash syntax, model-build smoke with `track/state_simdr` shape `[2,2,64]`, `19 passed` targeted tests, and diff whitespace check.
- Launched XR-11 SimDR center and P10 lanes on GPU0/GPU1 from XR-06C best-center and best-P10 checkpoints; both passed raw contract and entered epoch `1/12`.
- Completed XR-11 closeout. Both train runs exited `0`, all four eval summaries parsed, and no active gate promoted. Best XR-11 result was P10-init best-P10 `20.2893/26.3253/8.3099`, missing the center gate by about `0.0062px`.
- Updated second-goal experiment integration anchors from the old AdamW `27.0897` baseline to current XR-06C gates and queued XR-12 weak-distill light-SimDR as the next bounded P0.
- Added XR-12 weak-distill light-SimDR config and runner. Validation passed for bash syntax, config/model smoke, checkpoint existence, and diff whitespace.
- Launched XR-12 on both GPUs: center lane on GPU0 from XR-06C best-center and P10 lane on GPU1 from XR-06C best-P10, both with SimDR weight `0.0001`, fixed255k, AdamW LR `5e-7`, and weak distillation teacher=init.
- Closed XR-12 train/eval. No center/P10 gate promoted; center-init best-P10 produced a P5-only scalar promotion (`8.7917`) but regressed center/P10, so leaders remain XR-06C.
- Selected XR-13 head-only weak-distill SimDR as the next P0 to reduce full-model SimDR representation drift.
- Added XR-13 head-only weak-distill SimDR config and runner. Validation passed for bash syntax, checkpoint existence, config/model/trainable-filter smoke, and startup logs.
- Launched XR-13 on both GPUs: center lane on GPU0 from XR-06C best-center and P10 lane on GPU1 from XR-06C best-P10, both with head-only trainable filter, SimDR weight `0.0001`, fixed255k, AdamW LR `5e-7`, and weak distillation teacher=init.
- Closed XR-13 train/eval. No active gate promoted; best center was center-init best-P10 at `20.2897`, still above the center gate `20.2831`, and best P10 was P10-init best-P10 at `26.5514`, below the P10 gate `26.7449`.
- Selected XR-02 dense/continuous prediction trajectory preparation as the next P0, because EyeLoRiN-style refinement is implemented but sparse manifest gaps currently invalidate smoothing.
- Added `scripts/v3/check_xr02_dense_trajectory_availability.py` and generated `runs/diagnostics/xr02_dense_trajectory_availability_20260616_084506.json`.
- XR-02 availability check found no labeled dense trajectory in the test manifest or canonical annotations (`dense_pair_count=0`, longest segment `1`), so EyeLoRiN-style M2F cannot be promoted on current canonical1 labels.
- Next priority moves to XR-04 diagnostics refresh over blink/open-eye/low-event/fixation/saccade buckets before launching another temporal/head training branch.
- Completed XR-04 diagnostics refresh on XR-06C best-center and best-P10 leaders. Artifacts: `runs/diagnostics/xr04_refresh_xr06c_bestcenter_failure_buckets_20260616.json` and `runs/diagnostics/xr04_refresh_xr06c_bestp10_failure_buckets_20260616.json`.
- XR-04 refresh confirmed the dominant weighted failure remains low `similarity_target <= 0.1`, especially `session_201` and subjects `42/45/39`; blink/closed-eye is not the immediate weighted-metric driver.
- Search/event fallback diagnostics joined `0/2238` rows because the active `trackonly` config disables search/event heads after normalization.
- Added `--limit` to `scripts/v3/infer_hbtxr.py` for bounded confidence/quality diagnostic smoke runs.
- Next priority moves to a no-train `track_pred` confidence/quality diagnostic before any new training branch.
- Added `scripts/v3/build_confidence_probe_manifest.py` and built `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/test_manifest.jsonl` with 128 low-sim and 128 high-sim test rows.
- Ran XR-06C best-center and best-P10 confidence/quality probes. Both joined `256/256` rows.
- Probe result: current `track_pred` confidence/quality is saturated at `0.99+` for all rows and is higher on the harder low-sim bucket than on easier high-sim rows.
- Decision: reject no-train confidence gate from current logits. Next experiment must add calibrated confidence supervision with a real fallback path, or change representation/branch availability.
- Added XR-14 fallback diagnostics. Previous-state/blend fallback degraded raw XR-06C best-center metrics, and `track_state_aux` was unusable as a fallback on the 256-row probe.
- Added XR-14A all-head relocalization warm-start config and runner to restore `search_state`/`event_state` alternate predictions before any confidence-calibrated fallback training.
- Closed XR-14A all-head relocalization. Both lanes early-stopped at epoch `8/12`, all four test evals completed, and no active XR-06C gate promoted. `search_state`/`event_state` full-test buckets scored P10/P5 `0.0`, so calibrated search/event fallback is closed until a later branch produces a valid alternate state.
- Added XR-15 support-adaptive fixed-count event-window plan/config/runner. This is the next P0 data-density branch after XR-14A, preserving the XR-06C track-only weak-distill/direct-aux contract while adapting event support around fixed255k.
- Launched XR-15A support-adaptive center lane on GPU0 and P10 lane on GPU1. Both lanes passed startup validation. Added `scripts/v3/eval_xr15_p5_checkpoint.sh` for post-train best-P5 evaluation.
- Closed XR-15A center/P10 evals. P10-lane best-center promoted the center gate to `20.2257` with P10 `26.5009` and P5 `8.3044`; P10 and P5 gates remain XR-06C/XR-05A. XR-15 diagnostics still concentrate on low-similarity/session_201/subjects 42 and 45. Next queue is XR-15B narrow adaptive window `224k/255k/288k`.
- Closed XR-15B best-center evals. Center-lane best-center promoted the center gate to `20.2157` with P10 `26.5753` and P5 `8.6276`; P10 and P5 gates remain XR-06C/XR-05A. Best-P10 eval artifacts stalled after `hypers/*`, so only best-center closeout is complete.
- Launched XR-15C wider adaptive window `160k/255k/384k` on GPU0/GPU1. Both lanes passed startup validation and entered epoch `1/12`.
- Closed XR-15C best-center evals. Center-lane best-center promoted the center gate to `20.1909` with P10 `26.0098` and P5 `8.4362`; P10 and P5 gates remain XR-06C/XR-05A. Next queue moves from event-support range sweeps to XR-15D bounded track-adapter/coordinate-head training.
- Added and launched XR-15D track-adapter/coordinate-head probe on GPU0/GPU1. Runner `scripts/v3/run_xr15d_trackadapters_probe.sh` passed syntax/contract/model-build checks; both lanes entered epoch `1/16`.
- Closed XR-15D with no promotion. Added and launched XR-15E weak-distill low-LR track-adapter follow-up; both lanes passed startup validation and entered epoch `1/12`.
- Closed XR-15E with no promotion. Added validated XR-16A weak-distill event-path adapter config/runner and launched center/P10 lanes on GPU0/GPU1.
- Closed XR-16A and XR-16B with no promotion. Filled the XR-15C best-P5 artifact gap; XR-15C P10-lane best-P5 promoted the P5/balanced gate to `8.7032`, so current active gates are center `<20.1909`, P10 `>26.7449`, P5 `>8.7032`.
- Added and ran XR-17A same-branch XR-15C center-to-P5 interpolation sweep. Alpha `0.75` promoted the P5/balanced gate to `8.7330`; current active gates are center `<20.1909`, P10 `>26.7449`, P5 `>8.7330`.
- Added XR-17A failure-bucket comparison diagnostics. Result: XR-17A alpha `0.75` improves weighted P10/P5 but slightly worsens weighted center, so the next P0 is a bounded XR-15C-center-preserving branch using XR-17A only as a P5-direction anchor.
- Added and ran XR-17B P5-anchor support-adaptive branch. It promoted center to `20.1823` and P5 to `8.8448`; P10 remains XR-06C at `26.7449`.
- Added and ran XR-18A no-train P10 interpolation from XR-17B center to XR-06C P10. No promotion; next P0 is trainable P10 recovery.
- Added and ran XR-19A trainable P10-recovery micro polish. It promoted center slightly to `20.1812`, but did not recover P10/P5; next P0 is stronger constrained P10 recovery.
- Ran XR-20A/XR-20B P10-recovery LR ladder. XR-20A LR `2.5e-7` best-center promoted center to `20.1755`; XR-20B LR `5e-7` did not promote. P10/P5 gates remain XR-06C/XR-17B, so the LR-only P10-recovery ladder is closed.
- Added XR-21 P10-margin configs/runners and ran secondary plus primary probes. All completed with train/eval exit `0`, but no active gate promoted; next P0 moves to a stronger representation/supervision change.
- Added XR-22 weighted checkpoint soup utility/runner and ran six tri-leader no-train evals. `c34/p33/f33` promoted P5 to `8.8690`; center/P10 leaders remain XR-20A/XR-06C.
- Added XR-23 P10-preserving optimizer probe config/runner and ran ADOPT `2.5e-7` on GPU0 plus Lion `1e-7` on GPU1. Both completed with train/eval exit `0`, but no active gate promoted; next P0 uses XR-22 as bounded P10-recovery anchor.
- Added XR-24 XR-22-anchor P10-recovery config/runner and ran AdamW LR `1.25e-7` on GPU0 plus LR `2.5e-7` on GPU1. Both completed with train/eval exit `0`, but no active gate promoted; next P0 should use ADOPT-specific defaults or a true P10-specific head/loss branch.
- Added XR-25 XR-22-anchor ADOPT-defaults config/runner and ran LR `1.25e-7` on GPU0 plus LR `2.5e-7` on GPU1 with `betas=[0.9,0.9999]`, `eps=1e-6`. Both completed with train/eval exit `0`, but no active gate promoted; next P0 is XR-26 P10-boundary loss from XR-06C best-P10 init/teacher.
- Added XR-26 P10-boundary head-only loss/config/runner and ran default boundary `0.02/0.01` on GPU0 plus light boundary `0.01/0.005` on GPU1. Both early-stopped at epoch `6/8` with train/eval exit `0`, but no active gate promoted; next P0 should pivot to coordinate representation or teacher quality.
- Closed XR-27/XR-28 track-heatmap promotion path. XR-27 introduced the heatmap-state coordinate representation and promoted all gates; XR-28 LR `1.5e-4` refinement promoted them again to center `17.0849`, P10 `32.0812`, P5 `10.5026`. Added `docs/resources/xr28_heatmap_lr_refinement_results_2026_06_16.md`; next P0 is XR-29 post-XR-28 failure-bucket diagnostics and bounded LR/epoch refinement.
- Added XR-29 post-XR-28 failure-bucket diagnostic artifact and launched LR-neighbor runs. Promoted XR-28 weighted aggregate is `16.9868/32.3454/10.5774`; residual risks are low similarity, subject `39`, subjects `42/45`, and session-heavy failures. LR `1.25e-4` runs on GPU0 and LR `1.75e-4` runs on GPU1.
- Closed XR-29 LR-neighbor runs. LR `1.25e-4` improved only P5, while LR `1.75e-4` promoted all active gates to center `17.0496`, P10 `32.5646`, P5 `11.5047`. Added `docs/resources/xr29_lr_neighbor_results_2026_06_16.md`; next P0 is XR-30 LR micro-bracket `1.625e-4`/`1.875e-4`.
- Closed XR-30 LR micro-bracket. LR `1.625e-4` did not promote; LR `1.875e-4` promoted center only to `17.0355` while P10/P5 regressed to `32.4247/11.2666`. Added `docs/resources/xr30_lr_micro_bracket_results_2026_06_16.md`; follow-up was XR-31 no-train interpolation between XR-29 and XR-30.
- Closed XR-31 no-train interpolation between XR-29 LR `1.75e-4` and XR-30 LR `1.875e-4`. Alpha `0.25` promoted P10 to `32.5795` with center `17.0466`, but P5 `11.5038` narrowly missed the strict XR-29 P5 gate. Added `scripts/v3/run_xr31_xr29_xr30_heatmap_interp_eval.sh` and `docs/resources/xr31_xr29_xr30_interpolation_results_2026_06_16.md`; next P0 is trained LR `1.8125e-4`.
- Closed XR-32/XR-33 trained LR midpoint probes. XR-32 LR `1.8125e-4` did not promote; XR-33 LR `1.84375e-4` promoted P10 to `32.6207` while center/P5 regressed to `17.0392/11.3622`. Added `docs/resources/xr32_xr33_lr_midpoint_results_2026_06_16.md`; next P0 is heatmap loss-ratio refinement or bounded XR-29/XR-33 anchor/soup regularization.
- Added and launched XR-34 heatmap loss-ratio refinement. XR-34A uses XR-29 center/P5 anchor with heatmap/offset/center `0.004/0.0015/0.0015` on GPU0; XR-34B uses XR-33 P10 anchor with `0.006/0.001/0.001` on GPU1. Added `scripts/v3/run_xr34_heatmap_lossratio_refinement.sh` and `docs/resources/xr34_heatmap_lossratio_plan_2026_06_16.md`.
- Closed XR-34. XR-34B best-P10 promoted center/P10 to `16.53321223940168 / 33.77168447630746`, but P5 stayed below XR-29 at `11.276786088943481`. Added XR-35 XR-29/XR-34B no-train interpolation plan and runner for P5 recovery.
- Ran XR-35 XR-29/XR-34B no-train interpolation on GPU1. Alpha `0.03125/0.0625/0.09375/0.125` did not promote; next P0 is trainable P5-anchor continuation.
- Added XR-36 P5-anchor continuation runner and plan. XR-36A starts from XR-34B best-P10 at LR `5e-5`; XR-36B starts from XR-35 alpha `0.09375` at LR `8.75e-5`; both use original heatmap loss ratio `0.005/0.001/0.001`.
- Closed XR-36. XR-36B best-P10 promoted center to `16.53305721793856`; XR-36A best-P5 promoted P10 to `34.74064704350063`; XR-36A best-P10 promoted P5 to `11.738095617294311`. Added XR-37 XR-36B/XR-36A interpolation plan and runner.
- Closed XR-37 no-train XR-36B/XR-36A interpolation. Alpha `0.50` promoted center to `16.507612899371555` with P10 `34.33205857958112` and P5 `11.539116007941109`; P10/P5 gates remain XR-36A-owned.
- Added XR-38 center-preserving P10/P5 recovery runner and plan. XR-38A uses XR-37 alpha `0.50` init plus XR-36A best-P10 reference at LR `2.5e-5`; XR-38B uses XR-37 alpha `0.50` init plus XR-36A best-P5 reference at LR `1.25e-5`.
- Closed XR-38. XR-38A did not promote; XR-38B best-P10 narrowly missed P10 at `34.7075`; XR-38B best-P5 promoted P5 to `11.8440`. Active gates are now center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`; next P0 is XR-39 no-train mixed-leader soup/interpolation.
- Closed XR-39 mixed-leader soup. Evaluated 13 no-train soups. `c60p25f15` promoted center to `16.4918`; `c25p45f30` promoted P10 to `35.0230`; no XR-39 soup beat XR-38B P5. Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.
- Closed XR-40 XR39-leader/P5 soup. Evaluated 7 no-train soups; no gate promoted. Best center `16.4935`, best P10 `34.7751`, and best P5 `11.5298` all missed active gates. No-train soup follow-up is closed; next P0 is trainable loss-ratio fallback.
- Closed XR-41 loss-ratio P5 fallback. XR-41A best-P5 promoted P5 to `11.8682`; center and P10 remain XR-39-owned. Active gates are now center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Added and launched XR-42 P5-preserve low-drift branch. XR-42A uses XR-39 center as init plus XR-41A best-P5 teacher/reference on GPU0; XR-42B uses XR-39 P10 as init plus the same teacher/reference on GPU1. Both use LR `3e-6`, heatmap/offset/center `0.0035/0.00125/0.0020`, and head-only trainable scope.
- Closed XR-42 with no promotion. Best XR-42 center was `16.4980`, best P10 was `34.4830`, and best P5 was `11.3669`, all below active gates. Next candidate is XR-42C only if it adds explicit tiny distillation or regularization; teacher-as-reference alone was insufficient.
- Added XR-56 direct P10 soft-threshold supervision. Added default-zero soft-threshold losses, tests, runner, and plan. Static validation passed and XR-56A/B launched on GPU0/GPU1.
- Closed XR-56. XR-56B best-P5 promoted center/P5 to `16.4701875601496/12.133503770828247`; P10 remained below XR-39 with best XR-56 P10 `34.62330012321472`. Active gates are center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Closed XR-57 bounded center-refine calibration with no promotion. Best XR-57 P10 was Lane A `best_track_p5` at `34.8495`, still below XR-39 `35.0230`; center/P5 also missed the XR-56B gates. Next P0 is XR-58 P10 teacher refresh.
- Closed XR-58 P10 teacher refresh with no promotion. XR-58A best-P10 improved the current teacher-refresh branch to `34.9031`, but still missed XR-39 `35.0230` by about `0.1199`. Next P0 is XR-59 narrow anchored-teacher bracket.
- Added XR-59 XR58A teacher-bracket runner and plan. The branch continues from XR-58A best-P10 with XR-39 teacher anchoring and narrow LR/P10-soft variations.
- Closed XR-59 with no promotion. Added `scripts/v3/eval_xr59_completed_checkpoints.sh` for post-train checkpoint eval; best XR-59 P10 was `34.43409944261823`, below XR-58A and XR-39.

## 2026-06-10

- Added accuracy recovery master plan, sub-agent plan, spec, execution, validation, progress, ADR, TODO, and conversation tracking.
- Added Stage2 raw event-count ablation configs for lower LR, full-width, no/weak distillation, and density-adaptive ROI.
- Added Stage2 ablation runner.
- Added training history summarizer.

## 2026-06-12

- Added `docs/track/HBTXR_V3_0_PAST_EXPERIMENT_RESULTS.md`, consolidating legacy `HBTXR_v3_0/docs` experiment results and mapping them to current raw event-count training decisions.

## 2026-06-17 XR-60

- Added default-off `TrackCenterCandidateHead` and candidate P10 losses.
- Added targeted tests for candidate head/loss routing and trainable include pattern.
- Added `scripts/v3/run_xr60_p10_candidate_head.sh`.
- Added `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- Closed XR-60 with no gate promotion; recorded full-test A/B metrics in `docs/Validation.md` and `docs/track/PROGRESS.md`.

## 2026-06-17 XR Eye-Tracking Analysis Rewrite

- Added `scripts/v3/write_xr_eye_tracking_detailed_analysis.py`.
- Regenerated all detailed `analysis.md` files under `anlaysis/xr-eye-tracking`.
- Expanded codebase analyses to include repo/layer structure, directory map, entrypoints, source refs, Python symbols, import/dependency clues, config option clues, line evidence, reuse risks, HGTXR conversion design, priority, and next actions.
- Expanded paper analyses to include prior-method problems, proposed method, algorithm/loss/inference flow, bounded section evidence, HGTXR primitive decomposition, result/metric evidence, hardware/system relevance, experiment method, dataset/result interpretation, ablation axes, applicability, risks, and next actions.
- Validated `39` generated analysis files with generator `py_compile`, line-count audit (`8813` total lines), section coverage scan, and FACET paper/codebase spot checks.

## 2026-06-18 XR-61 / XR-62

- Closed XR-61 auxiliary candidate-head branch with no gate promotion.
- Added `scripts/v3/run_xr62_facet_geometry_aux_refresh.sh`.
- Added `docs/resources/xr62_facet_geometry_aux_refresh_plan_2026_06_17.md`.
- Ran XR-62A/B on GPU0/GPU1. XR-62A promoted the center gate to `16.468481131962367`; P10 and P5 gates remain unchanged.

## 2026-06-18 XR Eye-Tracking Reference-Level Analysis Rewrite

- Updated `scripts/v3/write_xr_eye_tracking_detailed_analysis.py` to generate codebase reports closer to the requested FlexLLM-style reference, including repository metadata, paper linkage, core source responsibility, HGTXR contract matrices, quality findings, and reproduction runbooks.
- Updated paper reports closer to the requested EfficientViT-FPGA-style reference, including dataset/model/benchmark tables, reported-result evidence, result interpretation, and detailed experiment option runbooks.
- Regenerated all `39` XR eye-tracking analysis files under `anlaysis/xr-eye-tracking`; total line count is now `11574`.
- Validated generator syntax, regeneration, required-section coverage, representative FACET/EV-Eye outputs, and whitespace with `git diff --check`.

## 2026-06-18 XR-63

- Added `scripts/v3/analyze_p10_teacher_target_oracle.py`.
- Generated `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` and `.md`.
- XR-63 no-train oracle over XR-62A, XR-39, XR-56B, and XR-58A reaches `16.04023192701366/36.48596938775512/13.41751700680271`, exceeding all current active gates as an upper bound.
- Next P0 is leakage-safe P10 teacher-target construction; do not repeat candidate-only or geometry-only auxiliary training without the new selector/pseudo-label protocol.
- Added `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md` with the next worker plan.

## 2026-06-18 XR-64

- Added target override builder: `scripts/v3/build_xr64_teacher_target_overrides.py`.
- Added XR-64 runner: `scripts/v3/run_xr64_teacher_target_construction.sh`.
- Added additive dataset/loss support for `track_target_override_state` and test-manifest leakage guard.
- Added `tests/test_track_target_override.py` and extended track loss tests.
- Added implementation report: `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md`.
