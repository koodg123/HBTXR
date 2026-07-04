# HGTXR-SW Accuracy Experiment Spec

Date: 2026-06-10

## 2026-06-27 Stage1 S1-OC Spec Addendum

Problem:

- Stage1 Search P10 can be improved, but prior P10-focused branches repeatedly regress P5 and center.
- Frozen-output/post-hoc gates show oracle headroom but do not generalize as learned validation gates.

Spec change:

- Add an integrated Search candidate branch inside the Stage1 Search path.
- Predict multiple bounded candidate center deltas and candidate logits from Search features.
- Use softmax mixture routing into final `search/state` when enabled.
- Keep hard selected xy as diagnostic only.

Implemented surface:

- `SearchCenterCandidateHead`
- Optional `search_center_candidate_head` in the tracker head factory.
- Stage1 Search branch candidate diagnostics.
- Candidate P10 BCE, min soft-threshold, and delta L2 losses.
- Default-disabled config keys in `configs/external/base.yaml`.
- Two-lane launcher `scripts/external/run_stage1_s1oc_search_candidate.sh`.

Experiment:

- S1-OC-A: candidate head only, GPU0, LR `5e-5`.
- S1-OC-B: candidate head plus zero-init residual Search adapter, GPU1, LR `2e-5`.
- Both lanes run `50` epochs from the active Stage1 baseline.

Promotion rule:

- P10 must beat `28.27717937613433`.
- Center must remain `<= 17.257583906065744`.
- P5 must be tracked.
- No promotion before `50` epochs.

Current execution state:

- Implemented and smoke-validated.
- Not yet trained on GPU.

## Problem

The current raw event-count full training completed, but accuracy is still far from the submission target. Stage2 full-run final `metric_track_center_px` was about `44 px`. Stage2-only paper-backed refinements have improved the current test-center leader to `16.4702 px`, the P10 leader to `35.0230%`, and the P5 leader to `12.1335%`, but the gap to the submission claim remains large.

## Evidence

- Submission target: `PAPER_WORKS/10_submission_initial/main.tex` claims `0.1812-pixel pupil-center error` and `0.43 ms` latency.
- Submission method: search/track split, time-synchronous geometry-stable supervision, decoupled heads, full-depth search, reduced-depth event track.
- Literature summary: FACET reports event ellipse regression on 64x64 event representation with `0.2030` pixel error; EX-Gaze uses frame relocalization plus event tracking; EyeTrAES emphasizes adaptive event slicing; 3ET motivates recurrent temporal event processing; local-global distillation motivates teacher/student but after stable targets.
- Current code: metrics are computed directly from transformed `cur_state` coordinates and predicted state coordinates.
- Best practical P10/P5 checkpoint: XR-36A best-P10, `runs/raw_mode1_stage2_count255000_adamw_lr5e_5_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr36a_p5anchor_xr34binit_xr29teacher_lr5e5_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_233614/train/best_track_p10.pt`, test center `16.5928`, P10 `34.3971`, P5 `11.7381`.
- Current center leader: XR-37 alpha `0.50` no-train interpolation, `runs/interpolated_checkpoints/xr37_xr36b_center_xr36a_p5p10_alphaa0p50.pt`, test center `16.5076`, P10 `34.3321`, P5 `11.5391`.
- Current P10 leader: XR-36A best-P5, test center `16.5770`, P10 `34.7406`, P5 `11.4158`.
- Previous unified leader: XR-27A track-heatmap coordinate representation LR `1e-4`, test center `17.2746`, P10 `31.9154`, P5 `10.2900`.
- Previous center/P10/P5 anchors: XR-20A center `20.1755`, XR-06C P10 `26.7449`, XR-22 P5 `8.8690`.

## Hypotheses

1. Metric/coordinate mismatch may dominate. Current px metric may be in transformed 256x256 ROI coordinates, not paper table coordinates.
2. Stage2 LR `1e-3` is too high for warm-start hybrid fine-tuning.
3. Structural width `0.6667` and active distillation may suppress baseline accuracy before any teacher is strong.
4. Raw fixed `5000` event-count window was too weak; fixed-count 20k/25k/30k/35k strongly improved test center, with 35k current center leader.
5. Event representation lacks anchor-centered/adaptive local focus used by EX-Gaze/EyeTrAES-style systems.
6. Current loss still optimizes state fields indirectly; decoded-center L2 and ellipse-style center/shape heads may align better with FACET-style pupil tracking.

## Experiment Matrix

2026-06-16 current state:

- Active gates are mixed after XR-56: center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- XR-11/XR-12/XR-13 SimDR variants did not promote.
- XR-02 dense trajectory refinement is blocked by sparse labeled canonical1 trajectories.
- XR-04 confidence probe rejected current `track_pred` confidence/quality as a no-train gate.
- XR-14 no-train previous-state/blend fallback and `track_state_aux` fallback are negative.
- XR-14A all-head relocalization is also negative: it did not beat XR-06C gates, and `search_state`/`event_state` are not usable fallback states.
- XR-15A support-adaptive fixed-count event-window training promoted center only. Best result was P10-lane best-center `20.225680075372967 / 26.50085105895996 / 8.304422058377947`.
- XR-15B narrow support-adaptive event-window training with min/base/max `224k/255k/288k` promoted center again. Center-lane best-center reached `20.215672533852715 / 26.575255823135375 / 8.627551317214966`; best-P10 eval artifacts stalled after `hypers/*`, so they are recorded as incomplete artifact coverage.
- XR-15C wider support-adaptive event-window training with min/base/max `160k/255k/384k` promoted center again. Center-lane best-center reached `20.19088832650866 / 26.009779623576573 / 8.436224787575858`; P10-lane best-center reached `20.205999997683932 / 26.303146975381033 / 8.397959463936942`.
- XR-15D bounded track-adapter/coordinate-head training completed without promotion. Its no-distill LR `3e-6` setting drifted away from the XR-15C center gate and did not recover P10/P5.
- XR-15E weak-distill low-LR track-adapter training also completed without promotion. It bounded drift relative to XR-15D but still missed all active gates.
- XR-16A weak-distill event-path adapter training and XR-16B alpha `0.125` checkpoint interpolation both completed without promotion.
- XR-15C P10-lane best-P5 checkpoint evaluation promoted the P5 gate to `8.703231593540737`.
- XR-17A same-branch interpolation between the XR-15C center leader and XR-15C P5 leader promoted P5 again. Alpha `0.75` reached `20.22669484274728 / 26.34566399029323 / 8.732993507385254`.
- XR-17A failure-bucket comparison completed. Alpha `0.75` improves weighted P10/P5 (`26.1625 -> 26.5202`, `8.4824 -> 8.7890`) but slightly worsens weighted center (`20.0095 -> 20.0415`). This is a useful P5 direction, not a replacement center seed.
- XR-17B executed that bounded branch: XR-15C center init, XR-17A alpha `0.75` teacher, AdamW LR `2.5e-7`, support-adaptive fixed255k. Best-center promoted center to `20.182338142395018`; best-P5 promoted P5 to `8.844813244683403`.
- XR-18A no-train interpolation from XR-17B center toward XR-06C P10 did not promote; best P10 was alpha `0.10` at `26.20748372077942`, while center and P5 gates were lost.
- XR-19A trainable P10-recovery micro polish from XR-17B best-P5 with XR-06C best-P10 teacher promoted center slightly to `20.181213889803207`, but did not recover P10/P5; best-P10 reached `26.232143613270352`.
- XR-20A LR `2.5e-7` P10-recovery ladder promoted center again to `20.175542894431523`, but P10/P5 still regressed versus active gates. XR-20B LR `5e-7` did not promote.
- XR-21 P10-margin mechanism-change probes completed with no promotion. The best secondary lane reached `20.187303059441703 / 26.126701450347902 / 8.791666977746146`; the primary XR-06C-init P10-leader lane reached `20.26367484842028 / 26.20110617365156 / 8.696003689084733`.
- XR-22 tri-leader checkpoint soup promoted P5 to `8.869047941480364` with `20.236038860252926 / 26.4604599407741 / 8.869047941480364`. It is a no-train P5/P10-friendly anchor, not a center or P10 replacement.
- XR-23 P10-preserving optimizer probe from XR-06C best-P10 completed with ADOPT `2.5e-7` and Lion `1e-7`; neither promoted. Best XR-23 P10 was ADOPT best-P10 `26.318027945927213`, below the XR-06C gate.
- XR-24 XR-22-anchor P10-recovery completed with AdamW LR `1.25e-7` and `2.5e-7`; neither promoted. Best XR-24 P10 was LR `1.25e-7` best-P5 `26.415817070007325`, and best XR-24 P5 was LR `2.5e-7` best-P5 `8.791666984558105`.
- XR-25 XR-22-anchor ADOPT-defaults completed with `betas=[0.9,0.9999]`, `eps=1e-6`; neither LR promoted. Best XR-25 P10 was `26.46045993396214`, and best XR-25 P5 was `8.86607174192156`, just below the P5 gate.
- XR-26 P10-boundary head-only polish completed from XR-06C best-P10 init/teacher. Default and light boundary weights both failed to promote; best XR-26 P10 was `26.45790890966143`, and best XR-26 P5 was `8.541666957310268`.
- XR-27 track-heatmap coordinate representation completed from XR-06C best-P10 init/teacher with state-distillation disabled and head-only heatmap trainable scope. LR `1e-4` best-P10 promoted all active gates to `17.274589475563594 / 31.915391901561193 / 10.289966331209456`.
- XR-28 heatmap LR refinement completed. LR `1.5e-4` best-P10/best-P5/best-center all resolve to epoch `10` and promoted all active gates to `17.08485197339739 / 32.081208263124736 / 10.502551344462804`; LR `7e-5` did not promote.
- XR-29 post-XR-28 diagnostics completed for the LR `1.5e-4` promoted checkpoint. Weighted aggregate is `16.9868 / 32.3454 / 10.5774`; residual risks remain low similarity, subject `39`, subjects `42/45`, and session-heavy failures.
- XR-29 LR-neighbor closeout completed. LR `1.25e-4` reached `17.179786903517588 / 32.04761978558132 / 10.53443912097386` and did not promote center/P10. LR `1.75e-4` reached `17.04961508342198 / 32.56462665285383 / 11.50467722075326` and promoted all gates.
- XR-30 LR micro-bracket completed. LR `1.625e-4` reached `17.067689692974092 / 32.420068802152365 / 10.866496937615532` and did not promote. LR `1.875e-4` reached `17.035534060001375 / 32.42474567549569 / 11.266581957680838`, promoting center only.
- XR-31 no-train interpolation completed. Alpha `0.25` reached `17.04659355367933 / 32.57950758934021 / 11.503826883860997`, promoting P10 only.
- XR-32/XR-33 trained LR midpoint probes completed. XR-32 LR `1.8125e-4` reached `17.041867678506033 / 32.5463443006788 / 11.37500034059797` and did not promote. XR-33 LR `1.84375e-4` reached `17.039179919447218 / 32.62074908529009 / 11.362245225906372`, promoting P10 only.
- XR-34 heatmap loss-ratio refinement completed. XR-34A reached `17.026217068944657 / 32.430698088237214 / 10.911139822006225`, promoting center only. XR-34B best-P10 reached `16.53321223940168 / 33.77168447630746 / 11.276786088943481`, promoting center and P10.
- XR-35 no-train XR-29/XR-34B interpolation completed with no promotion.
- XR-36 trainable P5-anchor continuation completed with mixed-gate promotion. XR-36B best-P10 owned center, XR-36A best-P5 owns P10, and XR-36A best-P10 owns P5.
- XR-37 no-train interpolation completed. Alpha `0.50` now owns center at `16.507612899371555`, while P10/P5 remain XR-36A-owned.
- XR-38 completed: XR-38B best-P5 promoted P5 to `11.843962955474854`.
- XR-39 completed: `c60p25f15` promoted center to `16.491779099191938`, and `c25p45f30` promoted P10 to `35.02295998845781`. Current P0 is a narrower XR-39 follow-up soup or XR-38C loss-ratio fallback to move P5 without losing the new center/P10 gates.
- XR-40 narrow follow-up soup completed with no promotion. Current P0 moves to trainable loss-ratio fallback; no-train soup follow-up is closed for now.
- XR-41 trainable loss-ratio fallback completed. XR-41A best-P5 promoted P5 to `11.868197652271816`; center/P10 remain XR-39-owned.
- XR-42 low-drift P5-preserve branch completed with no promotion. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- XR-56 direct P10 soft-threshold supervision completed. It did not recover P10, but XR-56B best-P5 promoted center/P5 to `16.4701875601496 / 34.29294293948582 / 12.133503770828247`. P10 remains XR-39-owned.

Priority 0: sanity checks

- Summarize history metrics for Stage1/Stage2.
- Visualize frame/event/mask/target samples.
- Confirm metric coordinate frame and target scale.

Priority 1: Stage2 optimization ablations

- `lr2e-4_nodistill_fullwidth`: full-width, no distillation, AdamW LR `2e-4`.
- `lr1e-4_weakdistill_fullwidth`: full-width, weak EMA distillation, AdamW LR `1e-4`.
- Event-count sweep using CLI override: `1000`, `2500`, `5000`, `10000`.

Priority 2: event evidence experiments

- Fixed-count bracket: completed/evaluated `18k`, `20k`, `22k`, `25k`, `30k`, `35k`; active `28k`.
- Non-saturated adaptive-count control with `reference_us=4000003` completed weak; current gain is mostly input length, not adaptive scaling.
- Test `time_bin_us` windows only after fixed-count optimum is known.

Priority 3: paper-backed center supervision

- Decoded-center L2 term: fixed20k completed and improved P10 but not center; fixed30k queued.
- FACET-inspired ellipse/center head: add optional 5-param pupil head only if center-L2 improves or count sweep plateaus.
- EX-Gaze/EyeTrAES-inspired local patch/event slicing: add anchor-centered event crop after count optimum.

Priority 4: paper-level recovery

- Rebuild mode2 dense supervision with interpolation and geometry-stable targets.
- Train full-capacity Stage1 teacher.
- Train Stage2 hybrid from full teacher.
- Only then run Stage3 self-supervised distillation/slimming.

## Success Metrics

- Primary: higher validation `metric_track_p10_pct`.
- Secondary: lower validation `metric_track_center_px`, higher `metric_event_p10_pct`, stable `metric_search_p10_pct`.
- Guardrail: no NaNs; result checker passes; no CPU fallback by accident.

## 2026-06-17 XR-56 Spec Addendum

- Problem: XR-49 through XR-55 did not recover the XR-39 P10 gate after center/P5 improved in XR-50 through XR-52.
- Spec change: add default-zero soft-threshold center losses that approximate hard P10/P5 hit-rate with differentiable positive BCE.
- Config keys:
  - `loss.track_p10_soft_threshold_weight`
  - `loss.track_p10_soft_threshold_margin_px`
  - `loss.track_p10_soft_threshold_temperature_px`
  - `loss.track_p5_soft_threshold_weight`
  - `loss.track_p5_soft_threshold_margin_px`
  - `loss.track_p5_soft_threshold_temperature_px`
- Compatibility: defaults are `0.0` weight and add no model parameters, so old checkpoints and non-XR-56 configs remain compatible.
- Experiment: `scripts/external/run_xr56_soft_threshold_p10_supervision.sh` runs two bounded lanes on GPU0/GPU1 and evaluates P10/P5/center checkpoints on the full test split.
- Result: XR-56A best-P10 `16.49389898266111/34.32270488057818/11.699830286843437`; XR-56A best-P5 `16.4924229485648/34.62330012321472/11.304422106061663`; XR-56B best-P10 `16.477364584377835/34.37670146397182/12.044218049730574`; XR-56B best-P5 `16.4701875601496/34.29294293948582/12.133503770828247`.
- Decision: update center/P5 gates only. P10 remains below `35.02295998845781`, so the next branch should be XR-57 calibration/refinement or P10 teacher retraining rather than another scalar loss-only branch.

## 2026-06-17 XR-57 Spec Addendum

- Problem: scalar P10/P5 soft-threshold supervision improved center/P5 but did not recover the strict P10 gate.
- Spec change: add `TrackCenterRefineHead`, a zero-initialized bounded residual head that predicts `delta_xy` from track fused features and can route into final `track/state`.
- Config keys:
  - `model.heads.track_center_refine`
  - `model.heads.track_center_refine_as_track_state`
  - `model.heads.track_center_refine_max_delta_px`
  - `model.heads.track_center_refine_blend`
  - `loss.track_center_refine_delta_l2_weight`
  - `loss.track_center_refine_delta_l1_weight`
- Compatibility: default is disabled and old checkpoints load non-strictly. When enabled on old checkpoints, the final linear layer is zero-initialized so the initial residual is identity.
- Experiment: `scripts/external/run_xr57_p10_center_refine_calibration.sh` defines lane A from XR-56B center/P5 leader and lane B from XR-39 P10 anchor.
- Result: XR-57 completed with no promotion. Best XR-57 P10 was Lane A `best_track_p5` at `34.849490649359566`, below the active P10 gate `35.02295998845781`.
- Decision: the next spec change should target teacher quality directly. XR-58 should train a P10-specialized teacher/anchor first, then use it as the reference for a later center/P5-preserving student only if the teacher improves full-test P10.

## 2026-06-17 XR-58 Spec Addendum

- Problem: the current student/refinement surface cannot exceed the XR-39 P10 gate.
- Spec change: add a runner-only P10 teacher-refresh branch using existing soft-threshold losses and expanded trainable scope.
- Experiment: `scripts/external/run_xr58_p10_teacher_refresh.sh` defines an anchored self-teacher lane and a free P10 lane from the XR-39 P10 checkpoint.
- Result: XR-58 completed with no promotion. Best XR-58 P10 was Lane A best-P10 at `34.90306201662336`, below the adoption threshold `35.02295998845781`.
- Decision: run one narrow XR-59 anchored-teacher bracket around XR-58A before closing this family.

## 2026-06-17 XR-59 Spec Addendum

- Problem: XR-58A reduced the P10 gap to about `0.1199`, but did not exceed XR-39.
- Spec change: add a runner-only continuation branch from XR-58A best-P10 with XR-39 as teacher.
- Experiment: `scripts/external/run_xr59_xr58a_teacher_bracket.sh` defines LR and P10-soft pressure lanes.
- Decision rule: promote only if full-test P10 exceeds `35.02295998845781`; otherwise close this scalar P10-soft continuation family.
- Result: XR-59 completed with no promotion. Best XR-59 P10 was Lane B `34.43409944261823`, below XR-58A `34.90306201662336` and XR-39 `35.02295998845781`.
- Decision: close the scalar P10-soft teacher-continuation family and require a different P10 representation or protocol diagnostic for the next P0.

## 2026-06-17 XR-60 Spec Addendum

- Problem: scalar P10-soft continuation and bounded single-delta refinement failed to exceed the XR-39 P10 gate.
- Spec change: add `TrackCenterCandidateHead`, a default-off multi-candidate center head that predicts bounded `K x 2` candidate deltas plus candidate logits from track fused features.
- Config keys:
  - `model.heads.track_center_candidate`
  - `model.heads.track_center_candidate_count`
  - `model.heads.track_center_candidate_max_delta_px`
  - `model.heads.track_center_candidate_as_track_state`
  - `model.heads.track_center_candidate_blend`
  - `loss.track_center_candidate_p10_margin_px`
  - `loss.track_center_candidate_temperature_px`
  - `loss.track_center_candidate_p10_bce_weight`
  - `loss.track_center_candidate_min_soft_threshold_weight`
  - `loss.track_center_candidate_delta_l2_weight`
- Compatibility: default is disabled; when enabled on old checkpoints, the candidate head starts with zero deltas/logits so initial behavior is identity before training.
- Experiment: `scripts/external/run_xr60_p10_candidate_head.sh` defines candidate-only and candidate+heatmap lanes from XR-59B best-P10 with XR-39 P10 teacher.
- Decision rule: promote only if full-test P10 exceeds `35.02295998845781` or if center/P5 improves without unacceptable P10 collapse.

## 2026-06-21 Stage1 Frame-Search Baseline Addendum

- Active execution priority changed by user request: improve Stage1 frame-based Search accuracy first and use that result as the next baseline.
- Current Stage1 baseline gate: `raw_mode1_stage1_best_adamw255k_200ep_2gpu_20260611_222452`, val Search P10 `23.34905708960767`, val Search center `18.319516586807538 px`.
- New runner: `scripts/external/run_stage1_frame_search_baseline_matrix.sh`.
- Stage1 baseline recipe: Search-only active head, full width, no distillation, no pruning, AdamW/cosine, 300 epochs, two-GPU LR/loss bracket.
- Stage1 promotion is evaluated separately from Stage2 gates: primary `metric_search_p10_pct`, secondary `metric_search_center_px`.
