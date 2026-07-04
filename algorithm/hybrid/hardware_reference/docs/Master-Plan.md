# HGTXR-SW Accuracy Recovery Master Plan

Date: 2026-06-10

## 2026-06-27 Current Master-State Addendum

Current execution authority is Stage1 frame-based Search improvement.

The active Stage1 baseline is:

- `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Search P10 `28.27717937613433`
- Search P5 `10.153863744915656`
- Search center `17.257583906065744`

Current strategy:

1. Keep the Stage1 baseline unchanged until a candidate completes at least `50` epochs and beats the strict P10/center gate.
2. Treat broad completed Stage1 ablations and frozen-output gates as closed unless a new mechanism is introduced.
3. Execute S1-OC integrated Search candidate A/B as the next meaningful branch.
4. Return to Stage2/XR teacher-target construction only after Stage1 Search has a stronger anchor or S1-OC produces a useful diagnostic.

Canonical current documents:

- `docs/resources/current_project_synthesis_2026_06_27.md`
- `docs/HANDOVER_2026_06_27.md`
- `HANDOVER.md`

## Goal

Raise HBTXR software accuracy toward the `PAPER_WORKS/10_submission_initial` target by using evidence from `PAPER_WORKS/01_REFERENCES/PAPER_REF`, current raw EV-Eye training results, and controlled Stage2 experiments.

## Current Baseline And Leader

- Dataset: `/home/kjm26/project/dataset/EV_Eye`
- Baseline raw event-count contract: original frames, fixed-count event window, `event_count_target=5000`
- Current best software contract: fixed-count event window, `event_count_target=255000`, support-adaptive count range `160000/255000/384000`, full-width Stage2 track path, AdamW, head-only track-center heatmap-state coordinate representation, and state-similarity distillation disabled.
- Stage1 full run: `runs/raw_mode1_stage1_event_count_20260610_192838`
- Stage2 full run: `runs/raw_mode1_stage2_event_count_20260610_193719`
- Stage1 best validation `metric_search_p10_pct`: `3.0829`
- Stage2 best validation `metric_track_p10_pct`: `7.2653`
- Stage2 final validation `metric_track_center_px`: `43.9917`
- Current center leader: XR-56B best-P5, `runs/xr56_xr52seed_xr39teacher_directp10p5soft_c255000_lr5e_7_g32_hm0_004_off0_0015_c0_0030_p10soft0_008_p5soft0_0020_s0_00035_m160k_M384k_r4000kus_p0p5_20260617_072340/train/best_track_p5.pt`, test center `16.4702`, P10 `34.2929`, P5 `12.1335`.
- Current P10 practical leader: XR-39 mixed-leader soup `c25p45f30`, `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`, test center `16.5039`, P10 `35.0230`, P5 `11.4605`.
- Current P5 practical leader: XR-56B best-P5, test center `16.4702`, P10 `34.2929`, P5 `12.1335`.
- Current active gates are mixed after XR-56: center `<16.4701875601496` from XR-56B best-P5; P10 `>35.02295998845781` from XR-39 `c25p45f30`; P5 `>12.133503770828247` from XR-56B best-P5.
- Previous center/P10/P5 leaders are retained as historical anchors only: XR-20A center `20.1755`, XR-06C P10 `26.7449`, and XR-22 P5 `8.8690`.
- Submission target claim: hybrid deployed pupil-center error `0.1812 px`, latency `0.43 ms`

## 2026-06-15 Leader And Second-Goal Update

- Current software leader was AdamW fixed255k LR `8e-6` best-center: `runs/raw_mode1_stage2_count255000_adamw_lr8e_6_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260611_112215`, test center `27.0897`, P10 `15.7109`, P5 `4.6301`.
- The user-requested Stage1 200ep + Stage2 200ep retry with Stage2 LR `1e-4` completed but did not promote; best Stage2 checkpoint appeared early and was much worse than the AdamW fixed255k leader.
- XR-Eye-Tracking reference analysis was added under `anlaysis/xr-eye-tracking` and reinforced on 2026-06-18 to `39` detailed analysis files (`11574` total lines) using the requested codebase/paper reference-document style.
- PAPER_REF-specific second-goal experiment mapping was added as `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`.
- Second-goal focus: preserve the AdamW/alpha-init leader contract, then add no-retrain post-processing, ellipse-state supervision, blink/fixation diagnostics, bounded temporal heads, and only later sparse/distilled deployment work.

## 2026-06-16 Runtime Update

- XR-01 fixed250k/fixed260k AdamW LR `8e-6` count bracket completed; all four best-center/best-P10 test evals failed the prior leader promotion gate.
- XR-01 fixed255k LR bracket completed. LR `6e-6` failed, but LR `1e-5` promoted the new software leader.
- New software leader: fixed255k AdamW LR `1e-5` best-center, `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt`, test center `26.1749`, P10 `16.5021`, P5 `5.0999`.
- LR `1e-5` best-P10 also promoted by the gate with center `26.5078`, P10 `16.8036`, P5 `4.2598`; best-center remains the primary leader because it preserves stronger center and P5.
- XR-03 ellipse-state probes launched from the new leader checkpoint: LR `1e-5` on GPU0 as the main branch, and LR `6e-6` on GPU1 as the stability branch. Both use axis weight `0.025` and angle weight `0.01`.
- XR-03 closeout: LR `1e-5` best-center promoted to the new overall software leader with test center `21.6822`, P10 `21.8202`, P5 `7.0446`. LR `1e-5` best-P10 also promoted with center `21.9530`, P10 `21.7453`, P5 `6.7351`. LR `6e-6` promoted versus the prior leader but is weaker than LR `1e-5`.
- XR-03A/B stronger geometry sweep completed from the promoted XR-03 checkpoint. XR-03A axis/angle `0.05/0.02`, LR `1e-5` promoted to the current software leader with test center `20.4539`, P10 `25.8686`, P5 `8.3104`. XR-03B axis/angle `0.05/0.02`, LR `6e-6` also promoted versus XR-03 but is weaker than XR-03A.
- Next P0 refinement: continue around the new XR-03A leader with a bounded stronger-geometry pair, preferably axis/angle `0.075/0.03` at LR `1e-5` and `6e-6`, or a conservative midpoint `0.0625/0.025` if over-regularization appears.
- XR-03C/D bounded stronger-geometry refinement completed. XR-03C LR `1e-5` best-center produced center `20.4750`, P10 `26.3202`, P5 `7.5374`. XR-03D LR `6e-6` best-P10 produced the new test-center leader with center `20.4336`, P10 `25.7866`, P5 `7.7491`; tradeoff: XR-03A remains stronger on P10/P5.
- XR-04 failure-bucket diagnostic artifact was generated at `runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json`. Low `similarity_target` rows and `session_201` buckets dominate current leader error.
- XR-04 low-similarity-aware loss completed and did not promote. Closest branch was LR `6e-6` best-center with center `20.4347`, P10 `25.7844`, P5 `8.4286`; it misses the center gate by `0.0011 px` and misses the balanced P10 gate.
- XR-05A direct track-state auxiliary head completed and promoted. GPU0 LR `6e-6` best-P10 is the center-first leader with center `20.3170`; GPU1 LR `3e-6` best-P10 is the P10 leader with P10 `26.5761`; GPU1 best-center is the P5/balanced secondary with P5 `8.6956`.
- XR-05B lighter direct-aux refinement completed and promoted on center. Best-P10 checkpoint reached center `20.2927`, P10 `26.0446`, P5 `8.0782`; this is the new center-first leader. Best-center checkpoint was balanced but did not beat current P10/P5 gates: center `20.3479`, P10 `26.2317`, P5 `8.6446`.
- XR-05C midpoint-LR direct-aux refinement completed and did not promote. Best-P10 checkpoint reached center `20.3236`, P10 `26.1671`, P5 `7.8508`.
- XR-05D ultra-light aux polish completed and did not promote. Best-center produced center `20.3298`, P10 `26.2934`, P5 `8.2980`; best-P10 produced center `20.2981`, P10 `25.8873`, P5 `7.8176`.
- XR-05E mid-light balanced polish completed. Best-center promoted the P10 gate to `26.6539` with center `20.3673` and P5 `8.4830`; best-P10 did not promote with center `20.3076`, P10 `25.8788`, P5 `8.1037`.
- XR-06 weak-distill track-state-aux polish partially promoted. XR-06A GPU1 used XR-05B best-P10 as init/teacher with aux `0.0005/0.0125/0.005`, LR `1e-6`; best-center produced center `20.2838`, P10 `26.2772`, P5 `8.4724`, becoming the new center-first leader. XR-06A best-P10 did not promote: center `20.3687`, P10 `26.0043`, P5 `8.1641`.
- XR-06B GPU0 completed from XR-05E best-center as init/teacher with aux `0.00075/0.01875/0.0075`, LR `1e-6`, and did not promote. Best-center produced center `20.2950`, P10 `26.2177`, P5 `8.5191`; best-P10 produced center `20.4695`, P10 `25.9320`, P5 `8.6310`.
- XR-05F no-distill control from XR-05E best-center with lighter aux `0.0005/0.0125/0.005`, LR `1e-6`, did not promote. Best-center produced center `20.3562`, P10 `26.4328`, P5 `8.2980`; best-P10 produced center `20.3195`, P10 `26.1509`, P5 `8.1429`.
- XR-06C/XR-05G ultra-low-LR polish completed. XR-05G did not promote. XR-06C promoted both center and P10: best-center reached center `20.2831`, P10 `26.3635`, P5 `8.4366`; best-P10 reached center `20.3829`, P10 `26.7449`, P5 `8.5268`. New active gates are center `<20.283125744547164`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- XR-06D/XR-06E follow-up from XR-06C leaders completed and did not promote. XR-06D no-distill center-preserve best-center/best-P10 both produced center `20.2956`, P10 `26.1586`, P5 `8.5047`. XR-06E weak-distill P10-preserve best-center produced `20.2942/26.2806/8.5387`; best-P10 produced `20.3149/26.1339/8.6811`.
- XR-08 checkpoint interpolation support was added. Alpha `0.50` between XR-06C best-center and best-P10 produced center `20.3259`, P10 `26.4983`, P5 `8.3865`; no promotion. Interpolation is useful as a low-cost diagnostic but not the next likely promotion path.
- XR-07A/B tiny center-hinge recovery completed training from the XR-05B center leader and did not promote: best validation center stayed near `23.63px`, below the current second-goal gate. Test eval artifacts were incomplete after long-running eval attempts, so the branch is closed on validation evidence and recorded as no-promotion.
- XR-02 EyeLoRiN-style post-process remains implementation-ready but blocked for promotion: current test manifest is sparse, with median adjacent timestamp gap about `4,000,003us` and minimum gap `360,000us`, so the `50,000us` trajectory gap gate prevents valid smoothing.
- XR-05I/XR-06F P5-preserve fallback from the XR-05A P5/balanced checkpoint completed and did not promote. XR-05I no-distill reached best-center `20.3510/26.1501/8.4290` and best-P10 `20.3939/26.1701/8.4503`. XR-06F weak-distill reached best-center `20.3222/26.2007/8.6259` and best-P10 `20.4218/26.3946/8.6301`.
- Current gates remain center `<20.283125744547164`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- Next P0 decision: local low-LR polish is saturated. Prioritize a targeted XR-04 failure-bucket branch around low `similarity_target` / `session_201`, while keeping XR-02 ready for dense trajectories. Defer heatmap/KL or temporal heads until this failure-bucket pass is exhausted.
- XR-04B/XR-04C targeted failure-bucket branches completed and did not promote. XR-04B best-center reached `20.4841/25.6003/8.2776`; XR-04B best-P10 reached `20.6738/25.4014/7.8754`; XR-04C best-center and best-P10 both reached `20.5883/25.6582/7.8499`.
- Current gates remain center `<20.283125744547164`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- Next P0 decision: low-similarity weighting alone is exhausted. Prefer a manifest/sampler-level failure-bucket experiment around low `similarity_target` / `session_201`, or generate dense trajectories for XR-02. Defer new temporal/heatmap heads until this targeted data-selection pass is checked.
- XR-04D/XR-04E manifest-subset failure-bucket branches completed and did not promote. XR-04D LR `6e-6` produced best-center `24.5081/16.7526/4.5179` and best-P10 `25.7039/15.3074/4.2381`; XR-04E LR `3e-6` produced best-center `23.5577/18.1594/5.1063` and best-P10 `23.4280/18.2555/5.0310`.
- Current decision: hard-subset fine-tuning on `similarity_target <= 0.1` + `session_201` overfits the failure bucket and damages full-test aggregate accuracy. Do not repeat hard subset or global low-similarity weighting. Next useful options are weighted sampler/full-manifest balancing, broader diagnostic subset only as analysis, or XR-02 after dense trajectories are available.
- XR-09 full-manifest weighted sampler support was added and closed. It kept all train rows while oversampling low-similarity/session_201 rows. XR-09A best-center/best-P10 reached `20.5793/25.6327/8.0102` and `20.6414/24.7428/8.0412`; XR-09B best-center/best-P10 reached `20.4154/26.1964/8.3057` and `20.5593/25.6824/7.9328`. No active gate promoted.
- Current decision: sampler-only failure-bucket balancing did not recover the XR-06C gates. Next useful options are loss-side sample weighting on the full manifest, dense-trajectory XR-02 preparation, or a new paper-backed architecture/head change rather than another sampler-only replay.
- XR-10 loss-side sample weighting was added and closed. It kept normal sampling, preserved `meta.session_key`, and applied opt-in `loss.track_sample_weight` only to Stage2 track-related losses. XR-10A best-center/best-P10 reached `20.3488/26.2062/8.1280` and `20.4297/26.0480/7.9996`; XR-10B best-center/best-P10 reached `20.3266/25.9702/8.1335` and `20.3897/26.1871/7.9945`. No active gate promoted.
- Current decision: failure-bucket rebalancing is exhausted across loss-only, hard subset, sampler-only, and loss-side full-manifest variants. Next useful branch should be paper-backed model/head change or dense-trajectory XR-02 preparation, not another low-sim/session_201 weighting replay.
- XR-11 track-state SimDR auxiliary was implemented and closed. No active gate promoted: best center was P10-init best-P10 `20.2893`, still worse than the center gate `20.283125744547164`; best P10 was center-init best-P10 `26.5455`, below the P10 gate `26.74489871433803`.
- Current decision: no-distill SimDR caused small representation drift. Next useful branch is XR-12 weak-distill light-SimDR, preserving XR-06C init/teacher with lower SimDR weight before attempting broader temporal adapters or heatmap heads.
- XR-12 weak-distill light-SimDR was executed and closed. Center-init best-center reached `20.3049/26.1514/8.2207`; center-init best-P10 reached `20.3599/26.0821/8.7917`; P10-init best-center/best-P10 both reached `20.3070/26.1514/8.1696`. Only the center-init best-P10 checkpoint exceeded the P5 gate, but it regressed center/P10 and is not selected.
- Current decision: both full-model no-distill and weak-distill SimDR drift away from XR-06C center/P10 gates. Next useful branch is XR-13 head-only weak-distill SimDR, freezing the backbone/track trunk and training only prediction heads before broader temporal adapters or heatmap heads.
- XR-13 head-only weak-distill SimDR was executed and closed. Center-init best-center reached `20.2916/26.2976/8.4694`; center-init best-P10 reached `20.2897/26.2679/8.5204`; P10-init best-center reached `20.3555/26.4622/8.4906`; P10-init best-P10 reached `20.3696/26.5514/8.5332`. No active gate promoted.
- Current decision: XR-11/XR-12/XR-13 exhausted the bounded SimDR branch. Next P0 is XR-02 dense/continuous prediction trajectory preparation so the implemented EyeLoRiN-style post-process can be evaluated on valid time-contiguous sequences.
- XR-02 dense trajectory availability check completed. Test manifest and canonical annotations both have dense pair count `0` under `50000us`, min gap `360000us`, median gap `4000003us`, and longest dense segment `1`.
- Current decision: XR-02 metric promotion is blocked on current canonical1 labels. Next P0 is XR-04 diagnostics refresh over blink/open-eye/low-event/fixation/saccade buckets before another temporal/head training branch.
- XR-04 diagnostics refresh completed for XR-06C best-center and best-P10 leaders. Low `similarity_target <= 0.1`, `session_201`, and subjects `42/45/39` remain the dominant weighted failure mode; blink/closed-eye is not the immediate weighted-metric driver.
- Current decision: low-sim/session_201 rebalancing is exhausted across loss weighting, hard subset, weighted sampler, and loss-side sample weighting. Next P0 is no-train `track_pred` confidence/quality diagnostics. Only launch confidence/relocalization gating or confidence auxiliary loss if confidence separates low-sim failures.
- XR-04 confidence/quality probe completed on a balanced 256-row low/high-sim manifest. Existing `track_pred` confidence/quality is saturated at `0.99+` and is not lower on low-sim failure rows.
- Current decision: reject no-train confidence gating from current logits. No-train previous-state/blend fallback and `track_state_aux` fallback are also negative. XR-14A all-head relocalization also failed to promote and did not produce usable `search_state`/`event_state` fallback branches. XR-15A, XR-15B, and XR-15C support-adaptive event-window training promoted the center gate, and XR-17A same-branch interpolation promoted the P5 gate. Current best center result is XR-15C center-lane best-center `20.19088832650866 / 26.009779623576573 / 8.436224787575858`. Active gates are now center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.732993507385254`.
- XR-15A diagnostics still show the same failure structure: low `similarity_target <= 0.1`, subject `42/45`, and `session_201`. Best-P5 helper attempts generated only `hypers/*` and no `eval_summary.json` under sandbox/logged timeout runs, so P5 checkpoint eval remains an artifact gap, not a blocker for the center promotion decision.
- Next P0 is a bounded XR-15C-center-preserving branch that retains the XR-17A P5 direction. XR-15D/E and XR-16A showed adapter fine-tuning drift; XR-16B showed cross-branch interpolation is weak; XR-17A showed same-branch interpolation can recover P5, and the completed failure-bucket comparison shows XR-17A should be used as a P5-direction anchor rather than a replacement center seed.
- XR-15D completed with no promotion. Best result was P10-lane best-P10 `20.410626077651976 / 26.1883510862078 / 8.385629544939313`, still below all active gates.
- XR-15E completed with no promotion. Best scalar result was center-lane best-center `20.281941563742503 / 26.2521265574864 / 8.633928898402623`, below the center, P10, and P5 gates.
- XR-16A completed with no promotion. It reproduced the XR-15E-level results while using event-path-only adapter scope.
- XR-16B one-shot checkpoint interpolation between XR-15C center and XR-06C P10 at alpha `0.125` completed with no promotion: `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.
- XR-15C missing best-P5 checkpoint eval completed and promoted P5: P10-lane best-P5 reached `20.245368467058455 / 26.487245675495693 / 8.703231593540737`.
- XR-17A same-branch interpolation sweep completed for alpha `0.25/0.50/0.625/0.75/0.875`. Alpha `0.75` promoted P5 to `8.732993507385254` with center `20.22669484274728` and P10 `26.34566399029323`; center/P10 gates remain unchanged.
- XR-17A failure-bucket comparison against the XR-15C center leader completed. Row-weighted aggregate changed from `20.0095/26.1625/8.4824` to `20.0415/26.5202/8.7890`: P10/P5 improve broadly while center regresses slightly. Gains are strongest on low-similarity, high-similarity, left-eye, and subject `42/45/39` buckets; regressions remain on subject `43/41` and right-eye P5. Next P0 should not use XR-17A as the sole center seed; use XR-15C center as primary seed/teacher and XR-17A alpha `0.75` as a bounded P5-direction teacher, checkpoint-soup anchor, or regularizer.
- XR-17B P5-anchor support-adaptive branch completed. It implemented the XR-17A diagnostic decision directly: XR-15C center checkpoint as init, XR-17A alpha `0.75` as weak teacher, AdamW LR `2.5e-7`, epochs `8`, support-adaptive `160k/255k/384k`. Best-center promoted center to `20.182338142395018`; best-P5 promoted P5 to `8.844813244683403`; P10 remains XR-06C `26.74489871433803`.
- XR-18A no-train interpolation between XR-17B best-center and XR-06C best-P10 completed for alpha `0.025/0.05/0.075/0.10`. No gate promoted. Best P10 was alpha `0.10` with `20.193220179421562 / 26.20748372077942 / 8.629677173069545`, but it lost the XR-17B center gate and remained far below the XR-06C P10 gate.
- XR-19A trainable P10-recovery micro polish completed. It used XR-17B best-P5 as init, XR-06C best-P10 as teacher, AdamW LR `1.25e-7`, and the same support-adaptive fixed255k contract. Best-P5 checkpoint promoted center slightly to `20.181213889803207`, but P10/P5 remained below active gates; best-P10 reached `20.20784169435501 / 26.232143613270352 / 8.609694181169782`.
- XR-20A/XR-20B P10-recovery LR ladder completed from the XR-17B best-P5 init and XR-06C best-P10 teacher. XR-20A LR `2.5e-7` best-center promoted center to `20.175542894431523` with P10 `25.884354482378278` and P5 `8.486394848142352`; XR-20B LR `5e-7` did not promote. Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`.
- XR-21 P10-margin mechanism-change probes completed. Added squared decoded-center hinge loss at a 10 px margin and tested XR-20A-init secondary lanes (`w=0.0015`, `w=0.003`) plus a P10-leader lane from XR-06C best-P10 (`w=0.0005`, best metric `metric_track_p10_pct`). No active gate promoted. Best secondary P10/P5 were `26.126701450347902` and `8.791666977746146`; the primary P10-leader best-P10/best-P5 both reached `20.26367484842028 / 26.20110617365156 / 8.696003689084733`. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`.
- XR-22 tri-leader checkpoint soup completed as a no-train representation/supervision preflight. Added weighted N-way checkpoint averaging and evaluated six convex mixtures of XR-20A center, XR-06C P10, and XR-17B P5 leaders. The equal-ish `c34/p33/f33` soup promoted P5 to `8.869047941480364` while preserving useful P10 `26.4604599407741`; center/P10 leaders remain XR-20A/XR-06C. Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.
- XR-23 P10-preserving optimizer probe completed from XR-06C best-P10 with ADOPT `2.5e-7` and Lion `1e-7`. Neither optimizer promoted any active gate; best XR-23 P10 was ADOPT best-P10 `26.318027945927213`.
- XR-24 XR-22-anchor P10-recovery completed from the XR-22 `c34/p33/f33` soup init and XR-06C best-P10 teacher. AdamW LR `1.25e-7` and `2.5e-7` both failed to promote; best XR-24 P10 was `26.415817070007325` and best XR-24 P5 was `8.791666984558105`.
- XR-25 XR-22-anchor ADOPT-defaults validation completed with `betas=[0.9,0.9999]`, `eps=1e-6`. It did not promote; best P10 was `26.46045993396214`, and best P5 was `8.86607174192156`, narrowly below the P5 gate.
- XR-26 P10-boundary head-only polish from XR-06C best-P10 init/teacher completed. Default and light boundary weights both failed to promote; best test P10 was `26.45790890966143`, and best test P5 was `8.541666957310268`.
- XR-27 track-heatmap coordinate representation promoted all gates to `17.274589475563594 / 31.915391901561193 / 10.289966331209456`.
- XR-28 heatmap LR refinement promoted all gates again. LR `1.5e-4` best-center/best-P10/best-P5 all resolve to epoch `10` and test `17.08485197339739 / 32.081208263124736 / 10.502551344462804`.
- XR-29 post-XR-28 failure-bucket diagnostics completed for the promoted LR `1.5e-4` checkpoint. Weighted aggregate is `16.9868 / 32.3454 / 10.5774`; residual risks are `similarity <=0.1`, high-similarity P10/P5 relative to XR-20A/XR-22, subject `39`, subjects `42/45`, and `session_201`-heavy failures.
- XR-29 LR-neighbor closeout completed. LR `1.25e-4` improved only P5, while LR `1.75e-4` promoted all active gates to `17.04961508342198 / 32.56462665285383 / 11.50467722075326`.
- XR-30 LR micro-bracket completed. LR `1.625e-4` did not promote; LR `1.875e-4` improved center to `17.035534060001375` but regressed P10/P5 to `32.42474567549569 / 11.266581957680838`.
- XR-31 no-train interpolation completed. Alpha `0.25` promoted P10 to `32.57950758934021` and improved center versus XR-29 to `17.04659355367933`, but P5 `11.503826883860997` remained slightly below the strict XR-29 P5 gate.
- XR-32/XR-33 trained LR midpoint probes completed. XR-32 LR `1.8125e-4` did not promote with `17.041867678506033 / 32.5463443006788 / 11.37500034059797`. XR-33 LR `1.84375e-4` promoted P10 to `32.62074908529009` with center `17.039179919447218` and P5 `11.362245225906372`.
- XR-34 heatmap loss-ratio refinement completed. XR-34A reached `17.026217068944657 / 32.430698088237214 / 10.911139822006225` and promoted center only. XR-34B best-P10 reached `16.53321223940168 / 33.77168447630746 / 11.276786088943481`, promoting center and P10 but not P5.
- Active gates after XR-34 were mixed: center `<16.53321223940168`, P10 `>33.77168447630746`, and P5 `>11.50467722075326`.
- XR-35 no-train interpolation between XR-29 and XR-34B completed with no promotion. Best center among the sweep was alpha `0.125` at `16.923010180677686 / 32.352041605540684 / 11.255102368763515`; best P5 was alpha `0.09375` at `16.952843945366997 / 32.42432054110936 / 11.420918709891183`.
- XR-36 trainable P5-anchor continuation completed. XR-36A best-P10 reached `16.59279990025929 / 34.39710958344596 / 11.738095617294311`, promoting P5 and giving the strongest practical P10/P5 checkpoint. XR-36A best-P5 reached `16.576952314376832 / 34.74064704350063 / 11.415816688537598`, promoting P10. XR-36B best-P10 reached `16.53305721793856 / 34.19387831687927 / 11.502551344462804`, promoting center and P10.
- XR-37 no-train interpolation between XR-36B best-P10 and XR-36A best-P10 completed. Alpha `0.50` promoted center to `16.507612899371555` with P10 `34.33205857958112` and P5 `11.539116007941109`, but did not promote P10/P5.
- XR-38 center-preserving P10/P5 recovery completed. XR-38B best-P5 promoted P5 to `11.843962955474854`; XR-38B best-P10 narrowly missed the P10 gate at `34.707483761651176`; XR-38A did not promote.
- XR-39 mixed-leader soup completed. `c60p25f15` promoted center to `16.491779099191938`; `c25p45f30` promoted P10 to `35.02295998845781`; no XR-39 soup beat the XR-38B P5 gate.
- Active gates after XR-39 were mixed: center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.843962955474854`.
- XR-40 narrow follow-up soup over XR-39 center/P10 leaders and XR-38B P5 leader completed with no promotion. Best XR-40 P5 was `11.529762240818568`, below XR-38B.
- XR-41 trainable loss-ratio fallback completed. XR-41A best-P5 promoted P5 to `11.868197652271816`; center/P10 remain XR-39-owned.
- Active gates are now mixed: center `<16.491779099191938`, P10 `>35.02295998845781`, and P5 `>11.868197652271816`.
- XR-42 low-drift P5-preserve branch completed with no promotion. Best XR-42 center/P10/P5 were `16.4980/34.4830/11.3669`, below active gates.
- Current P0 is to decide whether to run XR-42C with explicit tiny distillation/regularization, or pivot away from the XR-41 P5 teacher branch.

## Expert Council

1. Computer vision: validate target coordinate system, event-window representation, ROI geometry, and track residual definition before expensive training.
2. Training systems: prioritize Stage2-only ablations using fixed Stage1 checkpoint; avoid rerunning Stage1 unless Stage2 evidence improves.
3. Model compression/hardware: defer slimming/distillation until full-width teacher/student accuracy is credible.

## Strategy

1. Establish metric sanity: confirm whether current px metrics are in 256x256 ROI space, sensor space, or paper table space.
2. Fix optimization surface first: run Stage2-only experiments with lower LR, full width, and no or weak distillation.
3. Improve event evidence: sweep fixed-count windows and adaptive event crop, then compare event/track metrics. Current best count is fixed35k by center; fixed28k is active to check the 30k-35k bracket.
4. Add paper-backed center supervision: FACET-style ellipse/center parameter focus and decoded-center L2; fixed20k center-L2 improved P10 but not center, while fixed30k center-L2 is queued.
5. Restore paper pipeline later: mode2 dense supervision, TimeLens/V2E support, Grounded-SAM annotations, and Stage3 slimming.
6. Updated second-goal order from XR-Eye-Tracking references:
   - Training track first action: AdamW fixed255k local count/LR bracket.
   - No-retrain/post-process track: EyeLoRiN-style refinement can run immediately when dense prediction trajectories exist.
   - Geometry track: FACET/EllSeg ellipse-state auxiliary, `(sin2theta, cos2theta)` angle handling, and decoded-state losses.
   - Robustness track: Swift-Eye/EX-Gaze gated failure-bucket handling before another local crop branch.
   - Temporal track: TDTracker/BRAT/MambaPupil or CB-ConvLSTM-style temporal heads only after P0 axes plateau.

## Execution DAG

```text
Paper + submission evidence
  -> metric/target sanity check
  -> Stage2 low-cost ablation matrix
  -> best Stage2 candidate selection
  -> event-count optimum sweep
  -> second-goal split:
       training track: AdamW fixed255k LR/count bracket
       post-process track: EyeLoRiN M2F/OFE only on dense trajectories
       geometry track: FACET/EllSeg ellipse-state auxiliary
       robustness track: blink/open-eye/fixation/saccade failure buckets
       temporal track: bounded heatmap/KL or CB-ConvLSTM adapter
      current next action: post-XR-59 P10 representation/protocol pivot
  -> teacher-quality Stage1 rerun if needed
  -> sparse/quantized/distilled edge path only after full-width accuracy is strong
```

## Acceptance Criteria

- Each experiment has config, command, checkpoint path, and metric summary.
- Claims cite source files, training histories, or command outputs.
- No result is considered improved unless it beats Stage2 baseline on validation `metric_track_p10_pct` and `metric_track_center_px`.
- Full objective remains open until tested accuracy approaches submission-level result or a documented blocker proves the gap is from data/metric mismatch.

## 2026-06-21 Stage1 Frame-Search Baseline Priority Update

- User-directed current priority: improve Stage1 frame-based Search accuracy and use the promoted Stage1 checkpoint as the next baseline.
- This temporarily moves the active next action before XR-64 execution.
- Stage1 gate is tracked separately from Stage2 gates:
  - promote if validation Search P10 exceeds `23.34905708960767`;
  - prefer validation Search center below `18.319516586807538 px`.
- Runner: `scripts/external/run_stage1_frame_search_baseline_matrix.sh`.
- Plan artifact: `docs/resources/stage1_frame_search_baseline_plan_2026_06_21.md`.

## 2026-06-17 XR-56 Update

- XR-55 closed expanded last-block P10 tuning with no promotion.
- XR-56 changes the supervision objective directly: add default-zero soft-threshold center losses for P10/P5 and activate them only in the XR-56 runner.
- Runner: `scripts/external/run_xr56_soft_threshold_p10_supervision.sh`
- Plan: `docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md`
- Lane A is XR-39 P10 self-teacher on GPU0; lane B is XR-52B seed with XR-39 teacher on GPU1.
- XR-56B best-P5 promoted center/P5 to `16.4701875601496 / 12.133503770828247`; P10 remains XR-39-owned.
- Active gates after XR-56 are center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- XR-57 added a zero-initialized bounded center-refine head that routes `delta_xy` into final `track/state`.
- Runner: `scripts/external/run_xr57_p10_center_refine_calibration.sh`
- Plan: `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md`
- XR-57 train/eval completed with no promotion. Best XR-57 P10 was Lane A `best_track_p5` at `34.849490649359566`, still below the active P10 gate `35.02295998845781`; center and P5 also regressed versus XR-56B.
- Next action: XR-58 should deliberately train a P10-specialized teacher/anchor before distilling back into a center/P5-preserving student.
- XR-58 train/eval completed with no promotion. Best XR-58 P10 was Lane A best-P10 at `34.90306201662336`, improving over XR-57 but still missing XR-39 `35.02295998845781` by about `0.1199`.
- Next action: XR-59 narrow anchored-teacher bracket around XR-58A before closing this teacher-refresh family.
- XR-59 runner and plan were prepared and executed: `scripts/external/run_xr59_xr58a_teacher_bracket.sh`, `docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md`. Static validation and A/B dry-runs passed.
- XR-59 completed with no promotion. Best XR-59 P10 was Lane B `34.43409944261823`, regressing below XR-58A `34.90306201662336`.
- Next action: XR-60 dedicated multi-candidate P10 calibration head. This is the first representation/protocol pivot after closing scalar P10-soft teacher continuation.
- XR-60 artifacts prepared: `scripts/external/run_xr60_p10_candidate_head.sh`, `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- XR-60 uses XR-59B best-P10 as init, XR-39 P10 as teacher, and tests candidate-only vs candidate+heatmap trainable scopes.
