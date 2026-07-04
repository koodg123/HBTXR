# HGTXR-SW Paper-Backed Accuracy Experiment Plan

Date: 2026-06-16

## 2026-06-18 Experiment Pause Documentation

Status:

- Experiments are paused by user directive.
- No new train/eval job should be launched until the user explicitly resumes execution.
- Current process check found no active HGTXR train/eval process.
- GPU state at the pause point: GPU0 `15 MiB` used / `15827 MiB` free / `0%`, GPU1 `15 MiB` used / `15827 MiB` free / `0%`.

Documented current state:

- Consolidated pause/current-state artifact: `docs/resources/experiment_pause_documentation_2026_06_18.md`.
- Current work summary and next experiment list: `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- XR-64 helper status: `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` now supports split/teacher selection through `XR64_SPLITS`, `XR64_TEACHERS`, and `XR64_ACTIONS`, plus foreground logging and JSON validation.
- XR-64 artifact status: no full train/val teacher `eval_rows.json` files and no train/val override JSON files exist yet under `data/_internal/manifests/manifest1/xr64_teacher_targets/`.
- The direct 8-row XR-62A smoke eval is retained as path validation only; it is not a full experiment or promotion result.

Recorded next experiments remain XR-64A/B/C followed by XR-65/XR-66/XR-67/XR-68, but they are execution-paused.

## 2026-06-18 Current Runtime Snapshot

- Active gates after XR-62: center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Current center leader: XR-62A FACET geometry auxiliary refresh, checkpoint `runs/xr62_xr56b_facetaux_p10_preserve_c255000_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_s0_00035_m160k_M384k_r4000kus_p0p5_20260618_003308/train/best_track_p10.pt`, full-test `16.468481131962367/34.30782389640808/12.052721459524973`.
- Current P10 leader remains XR-39 mixed-leader soup `c25p45f30`, P10 `35.02295998845781`.
- Current P5 leader remains XR-56B best-P5, P5 `12.133503770828247`.
- XR-61 auxiliary candidate heads did not promote: best A `16.475529539585114/34.356718465260094/12.127126216888428`, best B `16.48967229127884/34.58843615395682/11.559524168287005`.
- XR-62 confirms FACET-style geometry auxiliary can still improve center, but it does not recover P10 or P5. Candidate-only and geometry-only branches should not be repeated without a new data/teacher signal.
- XR-63 no-train teacher-target oracle diagnostic completed: `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md`. Eval-style oracle over XR-62A center, XR-39 P10, XR-56B P5, and XR-58A teacher predictions reaches `16.04023192701366/36.48596938775512/13.41751700680271`, exceeding all active gates as an upper bound.
- XR-64 plan added: `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md`.
- XR-64 code path added: `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md`. It adds train/val-only target override generation, additive dataset/loss support, test-manifest leakage guard, and an A/B runner skeleton.
- Next P0: construct a provenance-safe P10 teacher-target branch from XR-63 evidence. Do not repeat candidate-only or geometry-only training without a learnable selector/pseudo-label protocol. Required acceptance remains full-test center/P10/P5 gate improvement with train/val/test leakage checked explicitly.
- Current consolidation artifact: `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- XR-64 prep helper added: `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`. It is intended to generate train/val eval rows for XR-62A/XR-39/XR-56B/XR-58A and then build `xr64a/xr64b/xr64c` train/val override files. Initial all-in-one prep attempt stopped without output rows; direct 8-row smoke eval of XR-62A succeeded, so the next run should either re-run the helper or execute split/teacher evals one by one with explicit logs.
- Updated next experiment list:
  1. `XR-64A`: conservative teacher-target selector, XR-62A init, XR-39 teacher, LR `3e-7`, override center weight `0.0006`.
  2. `XR-64B`: threshold-priority selector, XR-56B init, XR-39 teacher, LR `5e-7`, override center weight `0.0008`.
  3. `XR-64C`: min-error diagnostic, XR-62A init, LR `3e-7`, override center weight `0.0010`; run only after A/B or as diagnostic.
  4. `XR-65`: teacher-target temporal-lite residual after XR-64; AdamW `2e-7~3e-7`, state distill `0.00020~0.00035`, stop on center drift.
  5. `XR-66`: EX-Gaze confidence-gated local update diagnostic after XR-64; no local-crop training without fallback/confidence gate.
  6. `XR-67`: EyeLoRiN/dense-trajectory reopening only when dense trajectories exist.
  7. `XR-68`: sparse/quant/distilled edge branch only after stable full-width teacher.

## Latest Runtime Snapshot

- fixed-count sweep through fixed270k is complete; fixed265k/fixed270k did not promote over fixed255k/fixed205k, confirming a count-only plateau.
- post_count22 mixed probes completed.
- Best practical P10 checkpoint: XR-39 `c25p45f30`, support-adaptive fixed255k, no-train mixed-leader heatmap-state soup, test center `16.5039`, P10 `35.0230`, P5 `11.4605`.
- Best practical P5 checkpoint: XR-41A best-P5, support-adaptive fixed255k, head-only heatmap-state coordinate representation, test center `16.5195`, P10 `34.0982`, P5 `11.8682`.
- Current center gate is XR-39 `c60p25f15`, test center `16.4918`; P10 gate is XR-39 `c25p45f30`, P10 `35.0230`; P5 gate is XR-41A best-P5, P5 `11.8682`.
- Previous center/P10/P5 anchors: XR-20A center `20.1755`, XR-06C P10 `26.7449`, XR-22 P5 `8.8690`.
- XR-21 P10-margin mechanism-change probes completed and did not promote any active gate; best secondary P5 was `8.7917`, and primary P10-leader best-P10 reached `20.2637/26.2011/8.6960`.
- Previous leaders now demoted: fixed255k best-center center `29.6082`, P10 `13.7228`, P5 `3.3431`; fixed205k best-center P10 `15.0965`, center `30.3966`, P5 `4.0693`.
- `prev_pupil_anchor` local-crop probe at fixed255k completed and failed badly: best-center test center `65.0978`, P10 `9.9877`, P5 `3.0629`; best-P10 test center `65.6465`, P10 `9.4957`, P5 `3.1365`.
- Fixed `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh` to use `eval_hbtxr.py --experiment-name`; the old `infer_hbtxr.py --output-dir` path caused the first GPU1 eval attempt to fail.
- XR-01 fixed250k/fixed260k AdamW LR `8e-6` count bracket completed on 2026-06-16 and did not promote. The follow-up fixed255k LR bracket completed: LR `6e-6` failed, while LR `1e-5` promoted the new leader.

## Current Evidence

- Submission target: `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial/main.tex` and review drafts report HBTXR as a search/track hybrid with shared pupil-state supervision, anchor refresh, event residual tracking, Stage3 distillation, and final `0.1812 px` / `0.43 ms` target.
- Paper reference inventory: `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF/eye_tracking_document_analysis_ko.md` summarizes 30 reference papers and extracted text under `tmp_pdf_text/`.
- PAPER_REF experiment mapping: `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` maps the 30-paper corpus to Head/Loss/LR/Optimizer/Self-supervised Distillation/Teacher/Gating/Hardware actions for the second goal.
- Best practical P10 software checkpoint: XR-39 `c25p45f30`, `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`, test center `16.5039`, P10 `35.0230`, P5 `11.4605`.
- Best practical P5 software checkpoint: XR-41A best-P5, `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_weakdistill_trackonly_heatmapstate_g32_hm0_004_off0_0015_c0_0015_xr41a_p5init_xr38b_bestp5_xr39p10_ref_lossratio_lr1e5_p5select_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260617_013052/train/best_track_p5.pt`, test center `16.5195`, P10 `34.0982`, P5 `11.8682`.
- Current active gates: center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Previous useful anchors: XR-27A unified `17.2746/31.9154/10.2900`, XR-20A center `20.1755`, XR-06C P10 `26.7449`, XR-22 P5 `8.8690`.
- Previous software leaders: fixed-count 250000 best-center, test center `29.6603`, P10 `13.5514`, P5 `2.7968`; fixed-count 245000 best-center, test center `29.7768`, P10 `13.6025`, P5 `3.2687`; fixed-count 240000 best-center, test center `29.8589`, P10 `13.5132`, P5 `3.3580`; fixed-count 235000 best-center, test center `29.9446`, P10 `13.6692`, P5 `3.5761`; fixed-count 230000 best-center, test center `29.9516`, P10 `14.4677`, P5 `3.6050`; fixed-count 220000 best-center, test center `30.0077`, P10 `14.3967`, P5 `4.2096`; fixed-count 210000 best-center, test center `30.1415`, P10 `14.7037`, P5 `4.2636`; fixed-count 205000 best-center, test center `30.3966`, P10 `15.0965`, P5 `4.0693`.
- Active count/loss sweep: fixed18k through fixed270k have been evaluated. fixed265k/fixed270k did not promote, and the 2026-06-16 fixed250k/fixed260k AdamW LR `8e-6` retest also failed promotion. The fixed255k LR `{6e-6,1e-5}` branch promoted AdamW LR `1e-5`; XR-03 geometry probing then promoted the new overall leader.
- Current code surfaces already support decoded-center L2 (`loss.track_center_l2_weight`), linear/squared decoded-center threshold hinge (`loss.track_center_hinge_weight`, `loss.track_center_hinge_sq_weight`), ellipse GWD (`ellipse_gwd_loss`), teacher/student distillation, self-regularization, and optimizer variants (`adamw`, `lion`, `adopt`, `soap`, `sophia_g`, `prodigy`, etc.).

## Prior-Project Experiment-Tracking Lessons

Imported reference:

- `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md`

Operational lessons from `external_hybrid_package/docs` that must shape this plan:

1. Separate tracking axes.
   - Keep validation-center leader, test-center leader, P10 leader, and P5/tail behavior separate.
   - Legacy Mode0 Stage2 showed `loss_total` can improve while `track_p10` degrades.
2. Preserve spatial and initialization contracts.
   - Legacy Stage1/Stage2 runs failed when transform/routing assumptions changed between stages.
   - Current alpha-init/AdamW leader should be treated as a contract, not just a checkpoint.
3. Prefer short gated Stage2 experiments over blind long runs.
   - Legacy Stage2 and current 200/200 retry both peaked early and then plateaued or degraded.
4. Track failure buckets, not only aggregate metrics.
   - All-48 v2 showed `eye_failed=0`, with residual errors concentrated in blink/no-detection, oversized geometry, and tiny ROI escape buckets.
5. Avoid raw tuple interpolation as a default accuracy path.
   - TSGSS found raw event tuple interpolation artifacts; derived or ROI-first event features are safer if interpolation returns.

## XR-Eye-Tracking Reference Integration

Imported analysis root:

- `anlaysis/xr-eye-tracking/index.md`
- `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/codebases/*/analysis.md`
- `anlaysis/xr-eye-tracking/papers/*/analysis.md`
- `anlaysis/xr-eye-tracking/experiment_integration.md`

New second goal: improve beyond the AdamW fixed255k leader while staying inside the HGTXR paper concept of hybrid search/track, event residual tracking, pupil-state supervision, and deployment-aware efficiency.

Reference-backed judgment:

1. `EyeLoRiN` gives the lowest-risk quick win because it can refine existing prediction trajectories without retraining.
2. `FACET`, `EllSeg`, `E-Track`, and `RITnet` converge on explicit pupil geometry; the next paper-compatible model change should strengthen ellipse-state supervision before replacing the backbone.
3. `EX-Gaze` and `Swift-Eye` show that local event tracking needs confidence/relocalization gates; the failed HGTXR `prev_pupil_anchor` crop should not be extended without those gates.
4. `3ET`, `MambaPupil`, `BRAT`, `TDTracker`, AIS2024, and AIS2025 justify temporal context only after cheaper optimizer/window and geometry-loss axes plateau.
5. `Retina`, `SEE/ESDA`, `EX-Gaze`, and `DistillGaze` support a hardware-facing branch, but only after the full-width teacher is stable.

Prioritized second-goal experiments. `XR-01` is the first training-track action. `XR-02` is a no-retrain post-process action and may run immediately when dense prediction trajectories are available.

| ID | Priority | Experiment | Evidence | Gate |
|---|---:|---|---|---|
| XR-01 | P0 | AdamW fixed255k local count/LR bracket | Current HGTXR leader + EV-Eye/FACET fixed-count behavior | Completed; LR `1e-5` best-center promoted to center `26.1749`, P10 `16.5021`, P5 `5.0999` |
| XR-02 | P0 | EyeLoRiN-style inference-time smoothing/refinement | EyeLoRiN M2F/OFE + jitter metric | Implemented, but current canonical1 labels have no dense segment under `50000us`; metric promotion blocked until dense labels or a continuous split exist |
| XR-03 | P0 | FACET/EllSeg direct ellipse-state auxiliary or stronger decoded geometry loss | FACET, EllSeg, E-Track | Completed bounded stronger decoded ellipse-state sweep; XR-03D LR `6e-6`, axis `0.075`, angle `0.03`, best-P10 checkpoint promoted to center `20.4336`; XR-03A remains P10/P5 secondary |
| XR-04 | P1 | Failure-bucket split plus confidence/low-similarity-aware training or gated local event patch | Swift-Eye, EX-Gaze, BRAT, MambaPupil | Low-sim weighting/subset/sampler/loss-side branches completed with no promotion; next XR-04 action is diagnostics refresh over blink/open-eye/low-event/fixation/saccade buckets |
| XR-05A | P1 | Direct track-state auxiliary head from `track/fused` | FACET, EllSeg, E-Track | Completed and promoted; center leader `20.3170`, P10 leader `26.5761`, P5 secondary `8.6956` |
| XR-05B | P0 | Lighter direct-aux refinement from XR-05A leader | FACET, EllSeg, E-Track | Completed; best-P10 promoted center leader `20.2927` |
| XR-05C | P0 | Same-aux midpoint LR micro-grid | FACET, EllSeg, E-Track | Completed; no promotion |
| XR-05D | P0 | Ultra-light center polish from XR-05B center leader | FACET, EllSeg, E-Track | Completed; no promotion |
| XR-05E | P0 | Mid-light balanced polish from XR-05A P5 checkpoint | FACET, EllSeg, E-Track | Completed; best-center promoted P10 leader `26.6539` |
| XR-06 | P1 | Very weak teacher-student distillation polish from XR-05B/XR-05E | Local-global distillation, DistillGaze | XR-06A best-center promoted center to `20.2838`; XR-06B completed with no promotion |
| XR-07 | P0 | Tiny center-hinge tail recovery from XR-05B | EyeLoRiN tail metric framing, FACET/EllSeg center stability | Completed; validation no-promotion, test eval incomplete |
| XR-08 | P1 | Small temporal adapter or heatmap/KL coordinate head | 3ET, TDTracker, BRAT, MambaPupil, AIS2024/2025 | SimDR variants XR-11/XR-12/XR-13 closed with no active gate promotion; defer new temporal/head training until XR-04 diagnostics refresh |
| XR-09 | P2 | Sparse/quantized/distilled edge path | Retina, SEE/ESDA, EX-Gaze, DistillGaze | Start only after stable software teacher |
| XR-10 | P2 | Synthetic/unlabeled pretraining | Data-scarcity paper, DistillGaze | Start only if subject/data scarcity overfit is confirmed |

XR-01 execution checkpoint:

- Completed first local count bracket from the AdamW fixed255k leader contract:
  - fixed250k, AdamW LR `8e-6`, best-center test center `27.1403`, P10 `15.3044`, P5 `4.3635`; no promotion.
  - fixed250k, AdamW LR `8e-6`, best-P10 test center `28.1357`, P10 `14.6688`, P5 `4.3401`; no promotion.
  - fixed260k, AdamW LR `8e-6`, best-center test center `27.2089`, P10 `15.6429`, P5 `4.3741`; no promotion.
  - fixed260k, AdamW LR `8e-6`, best-P10 test center `32.2040`, P10 `11.9898`, P5 `3.1862`; no promotion.
- Completed second branch:
  - fixed255k, AdamW LR `6e-6`, best-center/best-P10 test center `28.6105`, P10 `14.5111`, P5 `4.0374`; no promotion.
  - fixed255k, AdamW LR `1e-5`, best-center test center `26.1749`, P10 `16.5021`, P5 `5.0999`; promoted.
  - fixed255k, AdamW LR `1e-5`, best-P10 test center `26.5078`, P10 `16.8036`, P5 `4.2598`; promoted by the gate but secondary to best-center.
- `scripts/external/compare_xr01_adamw_bracket.py` scans both the completed count branch and fixed255k LR branch.

XR-03 preparation checkpoint:

- Added `scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh`.
- The runner warm-starts from the current AdamW fixed255k best-center checkpoint and applies decoded ellipse-state axis/angle losses from `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_finetune_fullwidth.yaml`.
- Next command uses the XR-01-promoted checkpoint and keeps LR `1e-5` to preserve the new optimizer contract:
  `bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh 255000 0.025 0.01 1e-5 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt`.
- Launched main branch on GPU0 as tmux `hgtxr_xr03_ellipsestate_lr1e5_gpu0_20260616`, log `runs/_logs/xr03_ellipsestate_lr1e-5_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016`.
- Launched parallel stability branch on GPU1 as tmux `hgtxr_xr03_ellipsestate_lr6e6_gpu1_20260616`, log `runs/_logs/xr03_ellipsestate_lr6e-6_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012145`.
- Closeout:
  - LR `1e-5` best-center: center `21.6822`, P10 `21.8202`, P5 `7.0446`; new overall leader.
  - LR `1e-5` best-P10: center `21.9530`, P10 `21.7453`, P5 `6.7351`; promoted but secondary.
  - LR `6e-6` best-center: center `22.8794`, P10 `20.3835`, P5 `5.9660`; promoted over the prior leader but weaker than LR `1e-5`.
  - LR `6e-6` best-P10: center `23.0640`, P10 `20.2372`, P5 `6.0910`; promoted over the prior leader but weaker than LR `1e-5`.
- XR-03A/B stronger-geometry closeout:
  - XR-03A: warm-start from the promoted XR-03 best-center checkpoint, axis `0.05`, angle `0.02`, LR `1e-5`; best-center test center `20.4539`, P10 `25.8686`, P5 `8.3104`; new overall leader.
  - XR-03B: same checkpoint and weights, LR `6e-6`; best-center test center `21.3059`, P10 `23.5102`, P5 `7.5702`; promoted but secondary.
- XR-03C/D closeout:
  - XR-03C: GPU0, axis `0.075`, angle `0.03`, LR `1e-5`, log `runs/_logs/xr03c_axis0p075_angle0p03_lr1e-5_gpu0_20260616.log`.
  - XR-03D: GPU1, axis `0.075`, angle `0.03`, LR `6e-6`, log `runs/_logs/xr03d_axis0p075_angle0p03_lr6e-6_gpu1_20260616.log`.
  - XR-03C LR `1e-5` best-center/best-P10: center `20.4750`, P10 `26.3202`, P5 `7.5374`.
  - XR-03D LR `6e-6` best-center: center `20.4940`, P10 `25.9651`, P5 `8.0446`.
  - XR-03D LR `6e-6` best-P10: center `20.4336`, P10 `25.7866`, P5 `7.7491`; new center-first leader.

XR-02 implementation checkpoint:

- Added `scripts/external/eval_eyelorin_refinement.py` for M2F-style center-only median refinement, jitter proxy reporting, and before/after center/P10/P5 metrics.
- Added time-gap gating (`--max-gap-us`, default `50000`) because current test manifest samples are sparse and should not be smoothed as a dense trajectory.
- Smoke run `eval_eyelorin_fixed255k_adamw_bestcenter_smoke64_gap50k_gpu0` on 64 test rows produced no metric change after gap gating: center `24.3144`, P10 `21.5686`, P5 `7.8431` before and after. This confirms the guard works but does not prove XR-02 benefit.
- Next XR-02 requirement: generate or evaluate dense/continuous prediction trajectories, then rerun M2F windows and only then consider OFE local event-flow refinement.

XR-04 diagnostic checkpoint:

- Added `scripts/external/summarize_eval_failure_buckets.py`.
- Full leader artifact: `runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json`.
- Main target: low-similarity rows. `similarity_target <=0.1` has weighted center `42.8642`, P10 `7.1259`, P5 `1.4252`; `user41/right/session_201` is the worst session bucket by weighted center.
- Consequence: avoid another ungated local crop. If XR-01 LR branch does not promote, prefer XR-03 geometry plus low-similarity-aware loss/gating over blink-only logic, because blink/closed-eye rows are mostly zero-weighted under the current promotion metric.
- XR-04 low-similarity training candidate:
  - Added disabled-by-default `loss.track_low_similarity_*` weighting in the track loss.
  - Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml`.
  - Added runner `scripts/external/run_xr04_lowsim_ellipsestate_probe.sh`.
  - Closeout: no promotion. LR `6e-6` best-center was closest with center `20.4347`, P10 `25.7844`, P5 `8.4286`; center misses XR-03D by `0.0011 px`, and P10 misses the XR-03A balanced gate.
  - Focused retry closeout: threshold `0.1`, scale `2.0` from XR-03D best-P10 did not promote. XR-04B LR `6e-6` best-center reached `20.4841/25.6003/8.2776`; best-P10 reached `20.6738/25.4014/7.8754`. XR-04C LR `3e-6` best-center and best-P10 both reached `20.5883/25.6582/7.8499`.
  - Manifest-subset closeout: `similarity_target <= 0.1` + `session_201` hard-subset training did not promote. XR-04D LR `6e-6` best-center/best-P10 reached `24.5081/16.7526/4.5179` and `25.7039/15.3074/4.2381`; XR-04E LR `3e-6` best-center/best-P10 reached `23.5577/18.1594/5.1063` and `23.4280/18.2555/5.0310`.
  - Decision: low-similarity loss weighting and hard-subset fine-tuning are exhausted. Next failure-bucket attempt must keep full-manifest coverage and change sampler/loss balancing, or defer to XR-02 after dense trajectories are generated.
  - XR-09 closeout: full-manifest weighted sampler did not promote. XR-09A low-sim `3x`/session `2x`/cap `6x` reached best-center `20.5793/25.6327/8.0102` and best-P10 `20.6414/24.7428/8.0412`; XR-09B low-sim `2x`/session `2x`/cap `4x` reached best-center `20.4154/26.1964/8.3057` and best-P10 `20.5593/25.6824/7.9328`.
  - Decision: sampler-only balancing is exhausted for this failure bucket. Keep the implementation for controlled ablations, but prioritize loss-side sample weighting, dense-trajectory XR-02, or a new paper-backed head/architecture change.
  - XR-10 closeout: full-manifest loss-side sample weighting did not promote. XR-10A low-sim `2x`/session `1.5x`/cap `3x` reached best-center `20.3488/26.2062/8.1280` and best-P10 `20.4297/26.0480/7.9996`; XR-10B low-sim `1.5x`/session `1.5x`/cap `2.5x` reached best-center `20.3266/25.9702/8.1335` and best-P10 `20.3897/26.1871/7.9945`.
  - Decision: failure-bucket weighting is exhausted across loss-only, hard-subset, sampler-only, and loss-side variants. Next branch should change representation/head/temporal evidence or prepare dense trajectories for XR-02.

XR-05A direct track-state auxiliary checkpoint:

- Added opt-in `track/state_aux` head from `track/fused` to absolute state6 `(x, y, a, b, u, v)`.
- The main inference path remains `track/pupil -> track/state`; auxiliary supervision is training-only unless the config enables `model.heads.track_state_aux`.
- Added config `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_finetune_fullwidth.yaml`.
- Added runner `scripts/external/run_xr05a_trackstateaux_probe.sh`.
- Validation: py_compile passed; `tests/test_track_center_l2_loss.py` passed `8` tests; raw event-count readiness returned `ready=true`; model build smoke created `TrackStateAuxHead` with `150918` parameters.
- Closeout from XR-03D best-P10 center leader:
  - GPU0 LR `6e-6` best-center: center `20.4657`, P10 `24.8372`, P5 `8.1322`.
  - GPU0 LR `6e-6` best-P10: center `20.3170`, P10 `25.8639`, P5 `8.2398`; new center-first leader.
  - GPU1 LR `3e-6` best-center: center `20.3833`, P10 `26.2105`, P5 `8.6956`; P5/balanced secondary.
  - GPU1 LR `3e-6` best-P10: center `20.3220`, P10 `26.5761`, P5 `8.0336`; P10 leader.
- Active XR-05B refinement:
  - GPU1 LR `2e-6`, aux center/axis/angle `0.0005/0.0125/0.005`, init XR-05A LR `3e-6` best-P10, log `runs/_logs/xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_lr2e-6_gpu1_20260616.log`.
- Active XR-05C refinement:
  - GPU0 LR `4.5e-6`, aux center/axis/angle `0.001/0.025/0.01`, init XR-03D best-P10, log `runs/_logs/xr05c_trackstateaux_c0p001_axis0p025_angle0p01_lr4p5e-6_gpu0_20260616.log`.
- XR-05B/C closeout:
  - XR-05B best-center: center `20.3479`, P10 `26.2317`, P5 `8.6446`.
  - XR-05B best-P10: center `20.2927`, P10 `26.0446`, P5 `8.0782`; new center-first leader.
  - XR-05C best-center: center `20.5962`, P10 `25.2691`, P5 `7.8350`.
  - XR-05C best-P10: center `20.3236`, P10 `26.1671`, P5 `7.8508`; no promotion.
- Active XR-05D/E:
  - XR-05D GPU1 LR `1e-6`, aux center/axis/angle `0.00025/0.00625/0.0025`, init XR-05B best-P10, log `runs/_logs/xr05d_trackstateaux_c0p00025_axis0p00625_angle0p0025_lr1e-6_gpu1_20260616.log`.
  - XR-05E GPU0 LR `1.5e-6`, aux center/axis/angle `0.00075/0.01875/0.0075`, init XR-05A LR `3e-6` best-center, log `runs/_logs/xr05e_trackstateaux_c0p00075_axis0p01875_angle0p0075_lr1p5e-6_gpu0_20260616.log`.
- XR-05D/E closeout:
  - XR-05D best-center: center `20.3298`, P10 `26.2934`, P5 `8.2980`; no promotion.
  - XR-05D best-P10: center `20.2981`, P10 `25.8873`, P5 `7.8176`; no promotion.
  - XR-05E best-center: center `20.3673`, P10 `26.6539`, P5 `8.4830`; promotes P10 gate.
  - XR-05E best-P10: center `20.3076`, P10 `25.8788`, P5 `8.1037`; no promotion.
- XR-06 weak-distill fallback:
  - Added weak-distill track-state-aux config and runner, defaulting to XR-05B best-P10 as both init and teacher checkpoint.
  - XR-06A GPU1 closeout: XR-05B best-P10 init/teacher, aux `0.0005/0.0125/0.005`, LR `1e-6`, log `runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log`. Best-center promoted center to `20.2838` with P10 `26.2772`, P5 `8.4724`; best-P10 did not promote with center `20.3687`, P10 `26.0043`, P5 `8.1641`.
  - XR-06B GPU0 closeout: XR-05E best-center P10-leader init/teacher, aux `0.00075/0.01875/0.0075`, LR `1e-6`, log `runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log`. Best-center reached center `20.2950`, P10 `26.2177`, P5 `8.5191`; best-P10 reached center `20.4695`, P10 `25.9320`, P5 `8.6310`. No promotion.
  - Current gates after XR-06A: center `<20.2838`, P10 `>26.6539`, or P5 `>8.6956` with acceptable tradeoff.
- XR-05F no-distill control:
  - Launched on GPU1 from XR-05E best-center with lighter aux `0.0005/0.0125/0.005`, LR `1e-6`, log `runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log`.
  - Closeout: best-center reached center `20.3562`, P10 `26.4328`, P5 `8.2980`; best-P10 reached center `20.3195`, P10 `26.1509`, P5 `8.1429`; no promotion.
  - Interpretation: XR-06A's center gain was not reproduced by no-distill low-LR lighter-aux polish from XR-05E.
- XR-06C/XR-05G ultra-low-LR polish:
  - XR-06C launched on GPU0 from XR-06A best-center as both init and teacher, weak-distill enabled, aux `0.0005/0.0125/0.005`, LR `5e-7`, log `runs/_logs/xr06c_weakdistill_xr06a_lr5e-7_gpu0_20260616.log`.
  - XR-05G launched on GPU1 from XR-06A best-center, no distillation, aux `0.0005/0.0125/0.005`, LR `5e-7`, log `runs/_logs/xr05g_nodistill_xr06a_lr5e-7_gpu1_20260616.log`.
  - XR-05G closeout: best-center `20.2920/26.4422/8.4566`, best-P10 `20.3760/26.2742/8.2815`; no promotion.
  - XR-06C closeout: best-center `20.2831/26.3635/8.4366`, promoting center by `0.00072px`; best-P10 `20.3829/26.7449/8.5268`, promoting P10.
  - New gates after XR-06C: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
- XR-06D/XR-06E follow-up from XR-06C:
  - XR-06D center-preserve micro-polish: start from XR-06C best-center, no distillation, aux `0.0005/0.0125/0.005`, LR `2.5e-7`, GPU0.
  - XR-06E P10-preserve weak-distill: start from XR-06C best-P10 as init and teacher, aux `0.0005/0.0125/0.005`, LR `2.5e-7`, GPU1.
  - Closeout: XR-06D best-center/best-P10 both reached `20.2956/26.1586/8.5047`; XR-06E best-center reached `20.2942/26.2806/8.5387`; XR-06E best-P10 reached `20.3149/26.1339/8.6811`. No gate promotion.
  - Decision: ultra-low-LR local polish from XR-06C is saturated; next P0 should target checkpoint interpolation/SWA-style averaging or P5-preserve direct-aux fallback before broadening architecture.
- XR-08 checkpoint interpolation:
  - Added `scripts/external/interpolate_hbtxr_checkpoints.py` and `scripts/external/run_xr08_checkpoint_interp_eval.sh`.
  - Alpha `0.50` between XR-06C best-center and XR-06C best-P10 created `runs/interpolated_checkpoints/xr08_xr06c_center_p10_alpha0p50r.pt`.
  - Test eval `runs/eval_fixed255k_xr08_xr06c_center_p10_interp_alpha0p50r_gpu0_w0_20260616_053156/eval/test/eval_summary.json` produced `20.3259/26.4983/8.3865`; no promotion.
  - Decision: do not spend primary GPU time on dense alpha sweep unless all training fallbacks plateau. Move next to XR-05I P5-preserve direct-aux fallback.
- XR-05I/XR-06F P5-preserve fallback:
  - XR-05I no-distill from XR-05A P5/balanced best-center, aux `0.001/0.025/0.01`, LR `5e-7`, log `runs/_logs/xr05i_p5preserve_nodistill_xr05a_lr5e-7_gpu0_20260616.log`.
  - XR-06F weak-distill from the same XR-05A P5/balanced checkpoint as init and teacher, aux `0.001/0.025/0.01`, LR `5e-7`, log `runs/_logs/xr06f_p5preserve_weakdistill_xr05a_lr5e-7_gpu1_20260616.log`.
  - XR-05I best-center: center `20.3510`, P10 `26.1501`, P5 `8.4290`; no promotion.
  - XR-05I best-P10: center `20.3939`, P10 `26.1701`, P5 `8.4503`; no promotion.
  - XR-06F best-center: center `20.3222`, P10 `26.2007`, P5 `8.6259`; no promotion.
  - XR-06F best-P10: center `20.4218`, P10 `26.3946`, P5 `8.6301`; no promotion.
  - Decision: P5-preserve fallback improved neither the XR-05A P5 gate nor the XR-06C center/P10 gates. Keep current gates: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.
  - Next priority: stop local polish repetition. Use the XR-04 failure-bucket artifact for a targeted low-similarity/high-motion branch, or resume XR-02 only when dense trajectories exist.
- XR-04B/XR-04C targeted failure-bucket relaunch:
  - Completed from XR-03D best-P10 checkpoint with axis/angle `0.05/0.02`, low-sim threshold `0.1`, and scale `2.0`.
  - XR-04B: LR `6e-6`, GPU0, log `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log`.
  - XR-04C: LR `3e-6`, GPU1, log `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log`.
  - Closeout: XR-04B best-center/best-P10 reached `20.4841/25.6003/8.2776` and `20.6738/25.4014/7.8754`; XR-04C best-center/best-P10 both reached `20.5883/25.6582/7.8499`.
  - Decision: no promotion. Continue XR-04 only through data-selection or sampler targeting; do not repeat global low-sim loss weighting.
- XR-04D/XR-04E manifest-subset failure-bucket relaunch:
  - Completed from XR-03D best-P10 checkpoint on train/val rows filtered by `similarity_target <= 0.1` and `session_201`.
  - XR-04D: LR `6e-6`, GPU0, log `runs/_logs/xr04d_failbucket_sim01_s201_lr6e-6_gpu0_20260616.log`.
  - XR-04E: LR `3e-6`, GPU1, log `runs/_logs/xr04e_failbucket_sim01_s201_lr3e-6_gpu1_20260616.log`.
  - Closeout: XR-04D best-center/best-P10 reached `24.5081/16.7526/4.5179` and `25.7039/15.3074/4.2381`; XR-04E best-center/best-P10 reached `23.5577/18.1594/5.1063` and `23.4280/18.2555/5.0310`.
  - Decision: no promotion. Hard-subset training hurts full-test aggregate accuracy; next XR-04 action must use full-manifest weighted sampling/loss balancing or remain diagnostic-only.
- XR-09 full-manifest weighted sampler:
  - Added opt-in `training.sampler.type=weighted_failure_bucket` in the dataloader.
  - Initial weight rule: low similarity rows get `2x-3x`, `session_201` rows get `2x`, combined cap `4x-6x`.
  - Evaluation gate remains unchanged: center `<20.283125744547164`, P10 `>26.74489871433803`, or P5 `>8.69557854788644`.
  - Closeout: no promotion. Best observed XR-09 result was XR-09B best-center at `20.4154/26.1964/8.3057`, still behind all active gates.
- XR-10 full-manifest loss-side weighting:
  - Added opt-in `loss.track_sample_weight`, applied to Stage2 track-related losses only.
  - Closeout: no promotion. Best observed XR-10 result was XR-10B best-center at `20.3266/25.9702/8.1335`, still behind all active gates.
- XR-07A/B closeout:
  - XR-07A GPU1 linear center hinge margin `10`, weight `0.01`, LR `1e-6`, aux `0.0005/0.0125/0.005`, init XR-05B best-P10.
  - XR-07B GPU0 squared center hinge margin `10`, weight `0.001`, LR `1e-6`, aux `0.0005/0.0125/0.005`, init XR-05B best-P10.
  - Validation bests: XR-07A val center `23.6261`, P10 `21.8980`, P5 `7.7943`; XR-07B val center `23.6286`, P10 `21.8980`, P5 `7.7943`.
  - Decision: no promotion. Test eval directories were created but `eval_summary.json` was not produced; manual full eval also ran too long and was interrupted. Validation gap is large enough to close the branch without promotion.

## Paper-To-Experiment Map

| Paper axis | Relevant reference evidence | Software experiment |
|---|---|---|
| Adaptive event slicing | EyeTrAES emphasizes fine-grained adaptive event slicing; EX-Gaze uses local event tracking around frame relocalization | Finish fixed-count optimum first; then add anchor-centered local event crop and compare against fixed-count windows |
| Ellipse-state tracking | FACET directly regresses ellipse parameters and reports strong event-based pupil tracking; submission drafts define `(x,y,a,b,theta)` state | Extend current decoded-state supervision: center L2, center-threshold hinge, `track_geo_weight`, then optional direct ellipse auxiliary head |
| Temporal event modeling | 3ET uses ConvLSTM-style temporal event processing; RGBE-Gaze/EV-Eye motivate high-frequency multimodal state updates | After count optimum, test short temporal stack or cached previous event-token fusion only if fixed-count plateau remains |
| Search/track split | EX-Gaze and submission drafts separate frame anchor refresh from event local update | Keep Stage2 track-only alpha-init surface for low-cost count/loss tests; later re-enable search-track consistency and scheduler-quality losses |
| Teacher/distillation | Local-global distillation paper and HBTXR draft Stage3 motivate feature/state/head distillation | Defer until full-width teacher is credible; then train Stage3 reduced-depth track path from fixed-count/center-L2 leader |
| Optimizer/LR | Current gain came from conservative LR and no distillation; optimizer registry exists | After count/loss plateau, run small optimizer pool on best count: AdamW baseline, Lion, ADOPT, SOAP; keep LR `5e-6` and `2e-6` probes |
| XR reference post-processing | EyeLoRiN reports model-agnostic motion-aware filtering and local refinement | Add a no-retrain post-process evaluation on AdamW fixed255k predictions before heavier architecture changes |
| XR reference ellipse modeling | FACET, EllSeg, E-Track, RITnet emphasize ellipse/mask geometry | Prioritize direct ellipse auxiliary or decoded ellipse-state loss over full backbone replacement |
| XR reference temporal robustness | Swift-Eye, BRAT, MambaPupil, TDTracker, AIS challenge code focus on blink/fixation/saccade dynamics | Add failure-bucket diagnostics, then bounded temporal head or heatmap/KL loss if justified |
| XR reference edge deployment | Retina, SEE/ESDA, EX-Gaze, DistillGaze show sparse/quantized/distilled paths | Defer hardware branch until stable full-width software leader exists |

## Detailed Reference-To-Action Backlog

| Priority | Reference signal | Concrete HGTXR action | Status |
|---|---|---|---|
| P0 | `EV-Eye` and `EX-Gaze` show high-frequency event updates are useful when event windowing is tuned | Continue fixed-count sweep until test center/P10 plateau is observed | Plateau observed at fixed265k/fixed270k; switched to mixed probes |
| P0 | `EyeTrAES` motivates adaptive event slicing but current saturated adaptive-count result behaved like fixed20k | Revisit adaptive slicing only with non-saturated timestamp reference or local-anchor windows, not global clipped scaling | Planned after count plateau |
| P1 | `EX-Gaze` uses small local event patches around the current eye/pupil state | Implement anchor-centered local event crop using previous accepted state, then train fixed-count local residual target | Implemented as `prev_pupil_anchor`; queued after count/loss plateau |
| P1 | `FACET` directly predicts ellipse state `(x,y,a,b,theta)` and reports strong event-only pupil tracking | Add direct ellipse auxiliary head or stronger decoded ellipse-state loss at best count | Implemented as XR-05A direct `track/state_aux`; promoted to center `20.3170` |
| P1 | P10/P5 leaderboard shows threshold hit rate can move independently from mean center error | Add decoded-center hinge loss that penalizes only samples outside 10 px or 5 px thresholds | Implemented and validated; queued as post-count/loss-axis candidate |
| P1 | P10/P5 misses are tail-sensitive, so large threshold misses need stronger correction than linear hinge | Add squared decoded-center hinge loss for outlier-focused P10/P5 recovery | Implemented and validated; queued as plateau loss-axis candidate |
| P1 | Local-global distillation paper supports state-shift-aware frame/event distillation | Train a stronger full-width teacher from the count leader, then distill event/track state and pupil outputs into a reduced-depth student | Deferred until count leader stabilizes |
| P2 | `3ET`/JaneEye use temporal event sequences or compact recurrent/change-based tracking | Add short temporal stack or previous-event-token fusion only if local crop and optimizer probes plateau | Not started |
| P2 | `SEE`/Retina/JaneEye favor sparse/compact event models for latency/power | Keep hardware-facing compression separate from accuracy sweep; use it after software leader is stable | Deferred |
| P0 | `EyeLoRiN` enables inference-time refinement without architecture change | Add post-process evaluation for AdamW fixed255k prediction dumps and report jitter | New XR-Eye-Tracking integration item |
| P0 | `FACET`/`EllSeg`/`E-Track` directly support geometry-aware pupil tracking | Add direct ellipse auxiliary head or stronger decoded geometry loss around AdamW leader | New XR-Eye-Tracking integration item |
| P1 | `Swift-Eye`/`BRAT`/`MambaPupil` show blink, rest/fixation, and occlusion drive instability | Add blink/open-eye/fixation/saccade metric split before another local crop | New XR-Eye-Tracking integration item |
| P1 | `TDTracker`/AIS2025 use heatmap/KL and temporal modules | Test small temporal/heatmap auxiliary only after P0 axes plateau | New XR-Eye-Tracking integration item |
| P2 | `Retina`/`SEE`/`DistillGaze` support low-power deployment | Prepare sparse/quant/distillation report after full-width teacher stabilizes | New XR-Eye-Tracking integration item |

## Priority Experiments

0. XR-Eye-Tracking second-goal P0 additions.
   - `XR-01`: AdamW fixed255k local count/LR bracket using the alpha-init, no-distill, track-only contract.
   - `XR-02`: EyeLoRiN-style post-process on the current leader prediction trajectory.
   - `XR-03`: FACET/EllSeg ellipse-state auxiliary or stronger decoded geometry loss.
   - Do these before any new local-crop branch, because `prev_pupil_anchor` failed badly without robust confidence/relocalization gates.

1. Finish event-count optimum.
   - Completed/evaluated: `18k`, `20k`, `22k`, `25k`, `28k`, `30k`, `35k`, `40k`, `45k`, `50k`, `55k`, `60k`, `65k`, `70k`, `75k`, `80k`, `85k`, `90k`, `95k`, `100k`, `105k`, `110k`, `115k`, `120k`, `125k`, `130k`, `135k`, `140k`, `145k`, `150k`, `155k`, `160k`, `165k`, `170k`, `175k`, `180k`, `185k`, `190k`, `195k`, `200k`, `205k`, `210k`, `215k`, `220k`, `225k`, `230k`, `235k`, `240k`, `245k`, `250k`, `255k`, `260k`.
   - Latest plateau pair: `265k` on GPU1 and `270k` on GPU0 completed train/eval and did not promote.
   - post_count22 switched to mixed probes at fixed255k. post_count23/post_count24/post_count25/post_count26/post_count27/post_count28/post_count29/post_count30/post_count31 remain registered for later continuation/fallback.
   - Registered forward count queue if each pair promotes: GPU1 `255k`, `265k`, `275k`, `285k`, `295k`, `305k`, `315k`, `325k`, `335k`, `345k`, `355k`; GPU0 `260k`, `270k`, `280k`, `290k`, `300k`, `310k`, `320k`, `330k`, `340k`, `350k`, `360k`.
   - Current completed branch: fixed255k best-center produced center `29.6082`, P10 `13.7228`, P5 `3.3431`; fixed265k best-center produced center `29.7582`, P10 `13.5969`, P5 `3.5974`; fixed270k best-center produced center `29.9207`, P10 `13.3652`, P5 `3.8418`.
   - Gate: promote center only if test center `< 29.6082`; promote P10 if P10 `> 15.0965` or balanced center/P10 improves.

2. Test decoded-center L2 on the best count.
   - Completed: `center-L2 fixed20k` improved P10 to `11.7921` but did not beat center leader.
   - Completed: `center-L2 fixed30k` did not promote (`35.2231` center, `10.6845` P10).
   - Completed: `center-L2 fixed35k` did not promote; best-center `35.0627` center / `11.0225` P10, best-P10 `36.2216` center / `9.5034` P10.
   - Geometry-loss probes completed at fixed35k and did not promote.
   - Next if positive: sweep `track_center_l2_weight` in `{0.001, 0.0025, 0.005}` at the best fixed count.

2b. Test decoded-center threshold hinge loss on the best count.
   - Implemented optional loss keys: `loss.track_center_hinge_weight` and `loss.track_center_hinge_margin_px`.
   - Prepared configs:
     - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge10_finetune_fullwidth.yaml`
     - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhinge5_finetune_fullwidth.yaml`
   - Prepared runner: `scripts/external/run_centerhinge_probe_queue_20260611.sh <event_count> <margin_px> <weight> <cuda:N>`.
   - Default loss weight is `0.0`, so existing runs are unaffected.
   - Validation: unit test `tests/test_track_center_l2_loss.py`, config contract checks, and `py_compile` passed.

2c. Test squared decoded-center threshold hinge loss on the best count.
   - Implemented optional loss key: `loss.track_center_hinge_sq_weight`.
   - Uses the same `loss.track_center_hinge_margin_px` threshold as the linear hinge, then squares only the over-margin residual.
   - Prepared config: `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerhingesq10_finetune_fullwidth.yaml`.
   - Prepared runner: `scripts/external/run_centerhingesq_probe_queue_20260611.sh <event_count> <margin_px> <sq_weight> <cuda:N>`.
   - Default squared-hinge weight is `0.0`, so existing runs are unaffected.
   - Initial planned probe: best fixed-count leader, margin `10.0`, squared weight `0.005`, GPU1 only after the current fixed-count branch frees or a post-count plateau occurs.
   - Validation: `tests/test_track_center_l2_loss.py` passed with `5 passed`; raw event-count readiness contract passed; config loader resolved `track_center_hinge_sq_weight=0.005`, `track_center_hinge_margin_px=10.0`, `lr=5e-6`; `py_compile`, `bash -n`, `sh -n`, and no-arg usage checks passed.

3. Tighten ellipse/state loss.
   - Use existing `ellipse_gwd_loss` and `track_geo_weight`.
   - Candidate configs prepared:
     - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo075_finetune_fullwidth.yaml`
     - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerl2_geo100_finetune_fullwidth.yaml`
     - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_finetune_fullwidth.yaml`
   - `geo075 fixed35k` completed and did not promote.
   - `geo100 fixed35k` completed and did not promote.
   - Added checkpoint-compatible decoded-state losses:
     - `loss.track_axis_log_weight`: smooth-L1 on decoded log-axis values.
     - `loss.track_angle_cos_weight`: cosine distance on decoded double-angle unit vectors.
   - Prepared runner: `scripts/external/run_ellipsestate_probe_queue_20260611.sh <event_count> <axis_log_weight> <angle_cos_weight> <cuda:N>`.
   - Validation: `tests/test_track_center_l2_loss.py`, raw event-count contract, config loader check, `py_compile`, and runner `bash -n` passed.
   - Purpose: align more closely with FACET and HBTXR canonical pupil-state target.

4. Restore anchor-centered local event crop.
   - Implemented data crop policy: `data.mode1.event_builder.crop_policy=prev_pupil_anchor`.
   - Prepared config: `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_prevpupilcrop_finetune_fullwidth.yaml`.
   - Prepared runner: `scripts/external/run_prevpupilcrop_probe_queue_20260611.sh <event_count> <cuda:N>`.
   - Input is cropped around the previous annotation pupil bbox, with current frame/event/target transformed into that local ROI.
   - Purpose: match submission search/track wording and EX-Gaze/EyeTrAES local tracking pattern.
   - Validation: `tests/test_adaptive_roi_resolver.py`, `tests/test_event_builder.py`, config loader override check, raw event-count contract, `py_compile`, and `sh -n` passed.

5. Optimizer/LR pool.
   - Run only after count/loss leader is stable.
   - Small matrix: `adamw@5e-6`, `adamw@2e-6`, `lion@2e-6`, `adopt@5e-6`, `soap@5e-6`.
   - Guardrail: stop early if val center is worse than leader by `> 0.5 px` after 4 epochs.
   - Runner prepared: `scripts/external/run_optimizer_probe_queue_20260611.sh`; `bash -n` passed on 2026-06-11.

6. Teacher/distillation.
   - Rebuild Stage1 full teacher only after Stage2 full-width accuracy improves materially.
   - Stage3: distill feature/state/head outputs from full-depth teacher into reduced-depth track path.
   - Distillation keys already exist for `search/state`, `event/state`, `track/state`, `track/fused`, and `track/pupil`.
   - Prepared weak EMA distillation runner: `scripts/external/run_weakdistill_probe_queue_20260611.sh <event_count> <cuda:N>`.
   - Queue rule: run after count plateau or when GPU1 is idle; do not overlap with active fixed-count training on the same GPU.

## Current Queue

| GPU | Queue |
|---|---|
| GPU0 | AdamW `8e-6` probe at fixed255k active after `post_count22` mixed-probe decision; later registered GPU0 branches: post_count23 fixed290k/fallback, post_count24 fixed300k/fallback, post_count25 fixed310k/fallback, post_count26 fixed320k/fallback, post_count27 fixed330k/fallback, post_count28 fixed340k/fallback, post_count29 fixed350k/fallback, post_count30 fixed360k/fallback, post_count31 fixed370k/fallback |
| GPU1 | `prev_pupil_anchor` local-crop probe at fixed255k active after `post_count22` mixed-probe decision; later registered GPU1 branches: post_count23 fixed285k/fallback, post_count24 fixed295k/fallback, post_count25 fixed305k/fallback, post_count26 fixed315k/fallback, post_count27 fixed325k/fallback, post_count28 fixed335k/fallback, post_count29 fixed345k/fallback, post_count30 fixed355k/fallback, post_count31 fixed365k or squared-hinge fallback |

Post-count conditional policy:

- Wait until fixed55k and fixed60k best-center/best-P10 test eval summaries all exist.
- If either fixed55k or fixed60k beats current fixed50k center leader (`34.3416`) or P10 leader (`12.1276`), continue count sweep:
  - GPU1: fixed65k.
  - GPU0: fixed70k.
- Otherwise switch to LR/optimizer probes on the best count:
  - GPU1: `adamw@2e-6`.
  - GPU0: `adamw@8e-6`.

Second post-count conditional policy:

- Watcher sessions `hgtxr_post_count2_gpu0_20260611` and `hgtxr_post_count2_gpu1_20260611` wait until fixed65k/fixed70k best-center/best-P10 eval summaries all exist.
- If fixed65k or fixed70k beats current fixed60k center leader (`33.9445`) or fixed55k P10 leader (`12.5672`), continue count sweep:
  - GPU1: fixed75k.
  - GPU0: fixed80k.
- Otherwise switch to LR/optimizer probes on the best count:
  - GPU1: `adamw@2e-6`.
  - GPU0: `adamw@8e-6`.

Current second-stage decision: fixed70k promoted both leaderboards, so GPU1 fixed75k started at `2026-06-11T04:36:56+09:00` and GPU0 fixed80k started at `2026-06-11T04:38:08+09:00`.

Third post-count conditional policy:

- Watcher sessions `hgtxr_post_count3_gpu0_20260611` and `hgtxr_post_count3_gpu1_20260611` wait until fixed75k/fixed80k best-center/best-P10 eval summaries all exist.
- If fixed75k or fixed80k beats current fixed70k center/P10 leader (`33.5467` / `12.8971`), continue count sweep:
  - GPU1: fixed85k.
  - GPU0: fixed90k.
- Otherwise switch to LR/optimizer probes on the best count:
  - GPU1: `adamw@2e-6`.
  - GPU0: `adamw@8e-6`.
- Validation: `bash -n scripts/external/run_post_count3_decision_queue_20260611.sh` passed; watcher logs are `runs/_logs/post_count3_decision_gpu0_20260611.log` and `runs/_logs/post_count3_decision_gpu1_20260611.log`.

Current third-stage decision: fixed75k/fixed80k promoted over fixed70k. fixed75k evals produced center `33.3526`, P10 `13.2428`, P5 `3.5757`; fixed80k evals produced center `33.0848`, P10 `12.7543`, P5 `3.4056`. The third-stage watcher selected `ACTION=count`, `BEST_COUNT=80000`, `BEST_CENTER=33.084757`, `BEST_P10=13.242773`, then launched GPU1 fixed85k at `2026-06-11T04:53:51+09:00` and GPU0 fixed90k at `2026-06-11T04:53:49+09:00`.

Fourth post-count conditional policy:

- Watcher sessions `hgtxr_post_count4_gpu0_20260611` and `hgtxr_post_count4_gpu1_20260611` wait until fixed85k/fixed90k best-center/best-P10 eval summaries all exist.
- If fixed85k or fixed90k beats current fixed80k/fixed75k leaders (`33.0848` center or `13.2428` P10), continue count sweep:
  - GPU1: fixed95k.
  - GPU0: fixed100k.
- Otherwise switch to mixed post-count probes on the best count:
  - GPU1: decoded-center hinge loss, margin `10.0`, weight `0.05`.
  - GPU0: AdamW LR probe at `8e-6`.
- Validation: `bash -n scripts/external/run_post_count4_decision_queue_20260611.sh` passed; watcher logs are `runs/_logs/post_count4_decision_gpu0_20260611.log` and `runs/_logs/post_count4_decision_gpu1_20260611.log`.

Current fourth-stage decision: fixed85k promoted center and fixed90k did not promote. fixed85k best-center/best-P10 evals produced center `32.9858`, P10 `12.9605`, P5 `3.9660`; fixed90k best-center eval produced center `33.0128`, P10 `12.6212`, P5 `3.2836`, and best-P10 eval produced center `33.4284`, P10 `11.3542`, P5 `3.4528`. The fourth-stage watcher selected `ACTION=count`, `BEST_COUNT=85000`, `BEST_CENTER=32.985801`, `BEST_P10=13.242773`, then launched GPU1 fixed95k and GPU0 fixed100k at `2026-06-11T05:10:27+09:00`.

Fifth post-count conditional policy:

- Watcher sessions `hgtxr_post_count5_gpu0_20260611` and `hgtxr_post_count5_gpu1_20260611` wait until fixed95k/fixed100k best-center/best-P10 eval summaries all exist.
- If fixed95k or fixed100k beats current fixed85k/fixed75k leaders (`32.9858` center or `13.2428` P10), continue count sweep:
  - GPU1: fixed105k.
  - GPU0: fixed110k.
- Otherwise switch to mixed post-count probes on the best count:
  - GPU1: decoded-center hinge loss, margin `10.0`, weight `0.05`.
  - GPU0: AdamW LR probe at `8e-6`.
- Validation: `bash -n scripts/external/run_post_count5_decision_queue_20260611.sh` passed; watcher logs are `runs/_logs/post_count5_decision_gpu0_20260611.log` and `runs/_logs/post_count5_decision_gpu1_20260611.log`.

Current fifth-stage decision: fixed95k did not promote and fixed100k promoted center. fixed95k best-center/best-P10 evals produced center `32.9237`, P10 `12.2815`, P5 `3.4184`; fixed100k best-center eval produced center `32.7796`, P10 `12.5961`, P5 `3.7215`, and fixed100k best-P10 eval produced center `33.1030`, P10 `12.1105`, P5 `3.6195`. The fifth-stage watcher selected `ACTION=count`, `BEST_COUNT=100000`, `BEST_CENTER=32.779604`, `BEST_P10=13.242773`, then launched GPU1 fixed105k and GPU0 fixed110k at `2026-06-11T05:26:00+09:00`.

Sixth post-count conditional policy:

- Watcher sessions `hgtxr_post_count6_gpu0_20260611` and `hgtxr_post_count6_gpu1_20260611` wait until fixed105k/fixed110k best-center/best-P10 eval summaries all exist.
- If fixed105k or fixed110k beats current fixed100k/fixed75k leaders (`32.7796` center or `13.2428` P10), continue count sweep:
  - GPU1: fixed115k.
  - GPU0: fixed120k.
- Otherwise switch to mixed post-count probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- Validation: `bash -n scripts/external/run_post_count6_decision_queue_20260611.sh` passed; watcher logs are `runs/_logs/post_count6_decision_gpu0_20260611.log` and `runs/_logs/post_count6_decision_gpu1_20260611.log`.

Current sixth-stage decision: fixed105k/fixed110k best-center evals promoted center over fixed100k. fixed105k best-center/best-P10 evals produced center `32.6356`, P10 `13.1654`, P5 `3.6662`; fixed110k best-center eval produced center `32.5969`, P10 `13.0697`, P5 `3.8193`; fixed110k best-P10 eval produced center `32.9429`, P10 `12.6399`, P5 `3.4324`. P10 leader remains fixed75k at `13.2428`. The sixth-stage watcher selected `ACTION=count`, `BEST_COUNT=110000`, `BEST_CENTER=32.596922`, `BEST_P10=13.165392`, then launched GPU1 fixed115k at `2026-06-11T05:42:02+09:00` and GPU0 fixed120k at `2026-06-11T05:44:00+09:00`.

Seventh post-count conditional policy:

- Watcher sessions `hgtxr_post_count7_gpu0_20260611` and `hgtxr_post_count7_gpu1_20260611` wait until fixed115k/fixed120k best-center and best-P10 eval summaries all exist.
- If fixed115k or fixed120k beats current fixed110k/fixed75k leaders (`32.5969` center or `13.2428` P10), continue count sweep:
  - GPU1: fixed125k.
  - GPU0: fixed130k.
- Otherwise switch to mixed post-count probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- Validation: `bash -n scripts/external/run_post_count7_decision_queue_20260611.sh`, `sh -n scripts/external/run_post_count7_decision_queue_20260611.sh`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count7_decision_gpu0_20260611.log` and `runs/_logs/post_count7_decision_gpu1_20260611.log`.

Current seventh-stage decision: fixed120k promoted both center and P10. fixed115k best-center eval produced center `32.4400`, P10 `13.3512`, P5 `3.6305`; fixed115k best-P10 eval produced center `32.9068`, P10 `12.8835`, P5 `3.3678`. fixed120k best-center eval produced center `32.3053`, P10 `13.7976`, P5 `3.4974`; fixed120k best-P10 eval produced center `32.7800`, P10 `13.0991`, P5 `3.6922`. The seventh-stage watcher selected `ACTION=count`, `BEST_COUNT=120000`, `BEST_CENTER=32.305316`, `BEST_P10=13.797619`, then launched GPU1 fixed125k at `2026-06-11T06:00:09+09:00` and GPU0 fixed130k at `2026-06-11T05:59:57+09:00`.

Eighth post-count conditional policy:

- Watcher sessions `hgtxr_post_count8_gpu0_20260611` and `hgtxr_post_count8_gpu1_20260611` wait until fixed125k/fixed130k best-center and best-P10 eval summaries all exist.
- If fixed125k or fixed130k beats current fixed120k leaders (`32.3053` center or `13.7976` P10), continue count sweep:
  - GPU1: fixed135k.
  - GPU0: fixed140k.
- Otherwise switch to mixed post-count probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- Validation: `bash -n scripts/external/run_post_count8_decision_queue_20260611.sh`, `sh -n scripts/external/run_post_count8_decision_queue_20260611.sh`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count8_decision_gpu0_20260611.log` and `runs/_logs/post_count8_decision_gpu1_20260611.log`.

Current eighth-stage decision: fixed125k/fixed130k promoted over fixed120k. fixed125k best-center eval produced center `32.1365`, P10 `14.0676`, P5 `3.8610`; fixed125k best-P10 eval produced center `32.6232`, P10 `12.9962`, P5 `3.7636`. fixed130k best-center eval produced center `32.0159`, P10 `13.7798`, P5 `3.8457`; fixed130k best-P10 eval produced center `32.4814`, P10 `12.9940`, P5 `3.7190`. The eighth-stage watcher selected `ACTION=count`, `BEST_COUNT=130000`, then launched GPU1 fixed135k and GPU0 fixed140k at about `2026-06-11T06:18-06:19+09:00`.

Ninth follow-up policy after `post_count8`:

- Watcher sessions `hgtxr_post_count9_gpu0_20260611` and `hgtxr_post_count9_gpu1_20260611` wait for the `post_count8` action.
- If `post_count8` chooses count and fixed135k/fixed140k promote over the current fixed130k/fixed125k leaders (`32.0159` center or `14.0676` P10), continue count sweep:
  - GPU1: fixed145k.
  - GPU0: fixed150k.
- If fixed135k/fixed140k do not promote, switch to the same mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count8` already chooses mixed probes and they do not promote, move to prepared loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- If a mixed probe promotes, hold for leader review rather than stacking another branch on a changed leader.
- Validation: `bash -n scripts/external/run_post_count9_decision_queue_20260611.sh` and no-arg usage check passed. After fixed125k/fixed130k became the leaders, the threshold constants were corrected to center `32.0159` and P10 `14.0676`, and watcher sessions were restarted. Watcher logs are `runs/_logs/post_count9_decision_gpu0_20260611.log` and `runs/_logs/post_count9_decision_gpu1_20260611.log`.

Current ninth-stage decision: fixed135k/fixed140k completed. fixed135k best-center eval produced center `31.9891`, P10 `13.3274`, P5 `3.5242`; fixed135k best-P10 eval produced center `32.4352`, P10 `12.8053`, P5 `3.8559`. fixed140k best-center eval produced center `31.8833`, P10 `13.1884`, P5 `3.6637`, becoming the current center leader; fixed140k best-P10 eval produced center `32.3225`, P10 `12.7156`, P5 `3.6654`. The ninth-stage watcher selected `ACTION=count`, `BEST_COUNT=140000`, then launched GPU1 fixed145k and GPU0 fixed150k at about `2026-06-11T06:37+09:00`.

Tenth follow-up policy after `post_count9`:

- Watcher sessions `hgtxr_post_count10_gpu0_20260611` and `hgtxr_post_count10_gpu1_20260611` wait for the `post_count9` decision.
- If `post_count9` chooses count and fixed145k/fixed150k promote over the best available baseline through fixed140k, continue count sweep:
  - GPU1: fixed155k.
  - GPU0: fixed160k.
- If fixed145k/fixed150k do not promote, switch to mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count9` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Unlike earlier post-count scripts, this follower computes the active baseline dynamically from available eval summaries to avoid stale threshold constants.
- Validation: `bash -n scripts/external/run_post_count10_decision_queue_20260611.sh`, executable bit, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count10_decision_gpu0_20260611.log` and `runs/_logs/post_count10_decision_gpu1_20260611.log`.

Current tenth-stage decision: fixed145k/fixed150k completed. fixed145k best-center eval produced center `31.8802`, P10 `12.5897`, P5 `3.9073`; fixed145k best-P10 eval produced center `32.3115`, P10 `12.5221`, P5 `3.5570`. fixed150k best-center eval produced center `31.7211`, P10 `12.9154`, P5 `3.7330`, becoming the current center leader; fixed150k best-P10 eval produced center `32.1254`, P10 `12.8342`, P5 `3.4179`. P10 leader remains fixed125k best-center at `14.0676`. The tenth-stage watcher selected `ACTION=count`, `BEST_COUNT=150000`, then launched GPU1 fixed155k and GPU0 fixed160k at about `2026-06-11T06:55+09:00`.

Eleventh follow-up policy after `post_count10`:

- Watcher sessions `hgtxr_post_count11_gpu0_20260611` and `hgtxr_post_count11_gpu1_20260611` wait for fixed155k/fixed160k best-center and best-P10 eval summaries.
- If fixed155k/fixed160k promote over the best available baseline through fixed150k, continue count sweep:
  - GPU1: fixed165k.
  - GPU0: fixed170k.
- If fixed155k/fixed160k do not promote, switch to mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count10` had already chosen mixed probes and they did not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count11_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count11_decision_gpu0_20260611.log` and `runs/_logs/post_count11_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count11_gpu0_20260611` and `hgtxr_post_count11_gpu1_20260611`.

Current eleventh-stage decision: fixed155k/fixed160k completed. fixed155k best-center and best-P10 evals both produced center `31.6944`, P10 `12.8878`, P5 `3.5438`. fixed160k best-center and best-P10 evals both produced center `31.5714`, P10 `13.3108`, P5 `3.5098`, becoming the current center leader. P10 leader remains fixed125k best-center at `14.0676`. The eleventh-stage watcher selected `ACTION=count`, `BEST_COUNT=160000`, then launched GPU1 fixed165k and GPU0 fixed170k at about `2026-06-11T07:14+09:00`.

Twelfth follow-up policy after `post_count11`:

- Watcher sessions `hgtxr_post_count12_gpu0_20260611` and `hgtxr_post_count12_gpu1_20260611` wait for fixed165k/fixed170k best-center and best-P10 eval summaries.
- If fixed165k/fixed170k promote over the best available baseline through fixed160k, continue count sweep:
  - GPU1: fixed175k.
  - GPU0: fixed180k.
- If fixed165k/fixed170k do not promote, switch to mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If the prior branch is already mixed and does not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count12_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count12_decision_gpu0_20260611.log` and `runs/_logs/post_count12_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count12_gpu0_20260611` and `hgtxr_post_count12_gpu1_20260611`.

Current twelfth/thirteenth-stage decision: fixed165k/fixed170k completed. fixed165k best-center eval produced center `31.5321`, P10 `13.1152`, P5 `3.7160`; fixed165k best-P10 eval produced center `31.9779`, P10 `13.0676`, P5 `3.6352`. fixed170k best-center eval produced center `31.4726`, P10 `13.5991`, P5 `3.6437`; fixed170k best-P10 eval produced center `31.9469`, P10 `13.1144`, P5 `3.3129`. fixed175k best-center test produced center `31.2559`, P10 `13.4209`, P5 `3.8291`; fixed175k best-P10 test produced center `31.7825`, P10 `13.1718`, P5 `3.6322`. fixed180k best-center test produced center `30.9685`, P10 `14.0582`, P5 `3.7117`, becoming the current center leader; fixed180k best-P10 test produced center `31.6223`, P10 `13.4073`, P5 `3.8924`. P10 leader remains fixed125k best-center at `14.0676`. post_count13 selected `ACTION=count`, then launched GPU1 fixed185k at `2026-06-11T07:55:11+09:00` and GPU0 fixed190k at `2026-06-11T07:55:02+09:00`.

Current fixed185k/fixed190k status: both are training normally. Latest observed epoch 10: fixed185k val center `35.0572`, P10 `12.9515`; fixed190k val center `35.0466`, P10 `13.5355`.

Thirteenth follow-up policy after `post_count12`:

- Watcher sessions `hgtxr_post_count13_gpu0_20260611` and `hgtxr_post_count13_gpu1_20260611` wait for the `post_count12` decision.
- If `post_count12` chooses count and fixed175k/fixed180k promote over the best available baseline through fixed170k, continue count sweep:
  - GPU1: fixed185k.
  - GPU0: fixed190k.
- If fixed175k/fixed180k do not promote, switch to mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count12` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count13_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count13_decision_gpu0_20260611.log` and `runs/_logs/post_count13_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count13_gpu0_20260611` and `hgtxr_post_count13_gpu1_20260611`.

Fourteenth follow-up policy after `post_count13`:

- Watcher sessions `hgtxr_post_count14_gpu0_20260611` and `hgtxr_post_count14_gpu1_20260611` wait for the `post_count13` decision.
- If `post_count13` chooses count and fixed185k/fixed190k promote over the best available baseline through fixed180k, continue count sweep:
  - GPU1: fixed195k.
  - GPU0: fixed200k.
- If fixed185k/fixed190k do not promote, switch to mixed probes on the best count:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count13` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count14_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count14_decision_gpu0_20260611.log` and `runs/_logs/post_count14_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count14_gpu0_20260611` and `hgtxr_post_count14_gpu1_20260611`.

Fifteenth follow-up policy after `post_count14`:

- Watcher sessions `hgtxr_post_count15_gpu0_20260611` and `hgtxr_post_count15_gpu1_20260611` wait for the `post_count14` decision.
- If `post_count14` chooses count and fixed195k/fixed200k promote over the best available baseline through fixed190k, continue count sweep:
  - GPU1: fixed205k.
  - GPU0: fixed210k.
- If fixed195k/fixed200k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count14` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count15_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count15_decision_gpu0_20260611.log` and `runs/_logs/post_count15_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count15_gpu0_20260611` and `hgtxr_post_count15_gpu1_20260611`.

Sixteenth follow-up policy after `post_count15`:

- Watcher sessions `hgtxr_post_count16_gpu0_20260611` and `hgtxr_post_count16_gpu1_20260611` wait for the `post_count15` decision.
- If `post_count15` chooses count and fixed205k/fixed210k promote over the best available baseline through fixed200k, continue count sweep:
  - GPU1: fixed215k.
  - GPU0: fixed220k.
- If fixed205k/fixed210k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count15` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count16_decision_queue_20260611.sh`, `sh -n`, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count16_decision_gpu0_20260611.log` and `runs/_logs/post_count16_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count16_gpu0_20260611` and `hgtxr_post_count16_gpu1_20260611`.

Seventeenth follow-up policy after `post_count16`:

- Watcher sessions `hgtxr_post_count17_gpu0_20260611` and `hgtxr_post_count17_gpu1_20260611` wait for the `post_count16` decision.
- If `post_count16` chooses count and fixed215k/fixed220k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed225k.
  - GPU0: fixed230k.
- If fixed215k/fixed220k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count16` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: `bash -n scripts/external/run_post_count17_decision_queue_20260611.sh`, `sh -n`, executable bit, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count17_decision_gpu0_20260611.log` and `runs/_logs/post_count17_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count17_gpu0_20260611` and `hgtxr_post_count17_gpu1_20260611`.

Eighteenth follow-up policy after `post_count17`:

- Watcher sessions `hgtxr_post_count18_gpu0_20260611` and `hgtxr_post_count18_gpu1_20260611` wait for the `post_count17` decision.
- If `post_count17` chooses count and fixed225k/fixed230k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed235k.
  - GPU0: fixed240k.
- If fixed225k/fixed230k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count17` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: copied from post_count17, retargeted to post_count18, then `bash -n`, `sh -n`, executable-bit check, and no-arg usage check passed. Watcher logs are `runs/_logs/post_count18_decision_gpu0_20260611.log` and `runs/_logs/post_count18_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count18_gpu0_20260611` and `hgtxr_post_count18_gpu1_20260611`.

Nineteenth follow-up policy after `post_count18`:

- Watcher sessions `hgtxr_post_count19_gpu0_20260611` and `hgtxr_post_count19_gpu1_20260611` wait for the `post_count18` decision.
- If `post_count18` chooses count and fixed235k/fixed240k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed245k.
  - GPU0: fixed250k.
- If fixed235k/fixed240k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count18` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: copied from post_count18, retargeted to post_count19, then `bash -n`, `sh -n`, executable-bit check, no-arg usage check, tmux registration, and watcher-log startup checks passed. Watcher logs are `runs/_logs/post_count19_decision_gpu0_20260611.log` and `runs/_logs/post_count19_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count19_gpu0_20260611` and `hgtxr_post_count19_gpu1_20260611`.

Twentieth follow-up policy after `post_count19`:

- Watcher sessions `hgtxr_post_count20_gpu0_20260611` and `hgtxr_post_count20_gpu1_20260611` wait for the `post_count19` decision.
- If `post_count19` chooses count and fixed245k/fixed250k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed255k.
  - GPU0: fixed260k.
- If fixed245k/fixed250k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count19` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: copied from post_count19, retargeted to post_count20, then `bash -n`, `sh -n`, executable-bit check, no-arg usage check, tmux registration, and watcher-log startup checks passed. Watcher logs are `runs/_logs/post_count20_decision_gpu0_20260611.log` and `runs/_logs/post_count20_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count20_gpu0_20260611` and `hgtxr_post_count20_gpu1_20260611`.

Twenty-first follow-up policy after `post_count20`:

- Watcher sessions `hgtxr_post_count21_gpu0_20260611` and `hgtxr_post_count21_gpu1_20260611` wait for the `post_count20` decision.
- If `post_count20` chooses count and fixed255k/fixed260k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed265k.
  - GPU0: fixed270k.
- If fixed255k/fixed260k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count20` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: copied from post_count20, retargeted to post_count21, then `bash -n`, `sh -n`, executable-bit check, no-arg usage check, tmux registration, and watcher-log startup checks passed. Watcher logs are `runs/_logs/post_count21_decision_gpu0_20260611.log` and `runs/_logs/post_count21_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count21_gpu0_20260611` and `hgtxr_post_count21_gpu1_20260611`.

Twenty-second follow-up policy after `post_count21`:

- Watcher sessions `hgtxr_post_count22_gpu0_20260611` and `hgtxr_post_count22_gpu1_20260611` wait for the `post_count21` decision.
- If `post_count21` chooses count and fixed265k/fixed270k promote over the dynamic fixed-count baseline, continue count sweep:
  - GPU1: fixed275k.
  - GPU0: fixed280k.
- If fixed265k/fixed270k do not promote, switch to mixed probes:
  - GPU1: `prev_pupil_anchor` local event crop.
  - GPU0: AdamW LR probe at `8e-6`.
- If `post_count21` already chooses mixed probes and they do not promote, move to loss-axis probes:
  - GPU1: FACET-style ellipse-state loss, axis `0.05`, angle `0.02`.
  - GPU0: decoded-center hinge, margin `10.0`, weight `0.05`.
- Validation: copied from post_count21, retargeted to post_count22, then `bash -n`, `sh -n`, executable-bit check, no-arg usage check, tmux registration, and watcher-log startup checks passed. Watcher logs are `runs/_logs/post_count22_decision_gpu0_20260611.log` and `runs/_logs/post_count22_decision_gpu1_20260611.log`; tmux sessions are `hgtxr_post_count22_gpu0_20260611` and `hgtxr_post_count22_gpu1_20260611`.

## Decision Rules

- Do not start distillation before a full-width leader is identified; otherwise teacher pull can preserve weak behavior.
- Treat center and P10 as separate leaderboards; a P10 improvement with worse center may still be useful for final balanced checkpoint selection.
- Prefer test evaluation with `training.num_workers=0` for consistency.
- Always run with `HBTXR_DISABLE_CUDNN=1` on this host.
- Keep full objective open until accuracy is near the submission target or evidence proves a metric/data mismatch.

## 2026-06-16 XR-11 Track-State SimDR Auxiliary

Paper-backed decision after XR-10 no-promotion: stop rebalancing the same failure buckets and change the track-branch representation. The next P0 experiment is `XR-11`, a TDTracker/AIS-style SimDR coordinate auxiliary attached to `track/fused`.

- Motivation: TDTracker/AIS heatmap/KL and SimDR-style coordinate decomposition target tail localization error without changing runtime decode.
- Implementation surface:
  - `TrackStateSimDRHead`: predicts x/y coordinate distributions from `track/fused`.
  - Output key: `track/state_simdr`.
  - Loss key: `loss_track_state_simdr`.
  - Inference path unchanged: `track/pupil -> track/state`.
- Config/runner:
  - `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_simdr_fullwidth.yaml`
  - `scripts/external/run_xr11_trackstate_simdr_probe.sh`
- Initial launch plan:
  - Center lane: XR-06C best-center checkpoint on GPU0.
  - P10 lane: XR-06C best-P10 checkpoint on GPU1.
  - Default hyperparameters: fixed255k, AdamW LR `5e-7`, SimDR bins `64`, sigma `1.5`, SimDR weight `0.0005`, existing track-state aux `0.0005/0.0125/0.005`.
- Promotion gates remain:
  - center `<20.283125744547164`
  - P10 `>26.74489871433803`
  - P5 `>8.69557854788644`

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init | best-center | 20.338836230550495 | 26.16071500778198 | 8.301445865631104 | no |
| Center-init | best-P10 | 20.366218400001525 | 26.545493936538698 | 8.573979895455496 | no |
| P10-init | best-center | 20.34303183896201 | 25.954507521220616 | 8.235544497626169 | no |
| P10-init | best-P10 | 20.289304559571402 | 26.325255816323416 | 8.309949268613543 | no |

Decision: XR-11 no-distill SimDR is closed without promotion. Best result was P10-init best-P10 center `20.2893`, missing the center gate by about `0.0062px` and missing the P10/P5 gates. Since XR-06C leader is weak-distill and XR-11 removed distillation while adding SimDR, the next bounded P0 should test weak-distill with a much lighter SimDR auxiliary, or restrict trainable scope to the new SimDR/head layers to avoid representation drift.

## 2026-06-16 XR-12 Candidate: Weak-Distill Light SimDR

`XR-12` should preserve the XR-06C weak-distill contract and only add a lower-weight coordinate-distribution auxiliary:

- Init/teacher: same XR-06C checkpoint per lane.
- Fixed count: `255000`.
- Optimizer: AdamW.
- LR: `5e-7` first; do not lower again unless validation shows center-preserving but underfit behavior.
- Direct track-state aux: keep `0.0005/0.0125/0.005`.
- SimDR bins/sigma: keep `64/1.5`.
- SimDR weight: start `0.0001`; optional GPU1 variant `0.0002` only if using two-lane sweep.
- Gate: promote only if center `<20.283125744547164`, P10 `>26.74489871433803`, or P5 `>8.69557854788644`.

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init weak-distill | best-center | 20.30488075869424 | 26.15136126109532 | 8.22066354070391 | no |
| Center-init weak-distill | best-P10 | 20.359868260792325 | 26.082058572769164 | 8.791666991370064 | P5-only |
| P10-init weak-distill | best-center | 20.306958419936045 | 26.15136125428336 | 8.169643129621234 | no |
| P10-init weak-distill | best-P10 | 20.306958419936045 | 26.15136125428336 | 8.169643129621234 | no |

Decision: XR-12 is closed without center or P10 promotion. The center-init best-P10 checkpoint improves the standalone P5 gate (`8.7917` > `8.6956`) but degrades both center and P10, so it is not a replacement for the XR-06C center/P10 leaders. Since both no-distill and weak-distill SimDR variants drift away from the XR-06C center/P10 gates, the next P0 should restrict trainable scope rather than adding another full-model representation loss.

## 2026-06-16 XR-13 Candidate: Head-Only Weak-Distill SimDR

`XR-13` should reuse the XR-12 objective but freeze the backbone/track trunk and train only the lightweight prediction heads. This keeps the paper-backed TDTracker/AIS coordinate-distribution auxiliary while reducing representation drift.

- Init/teacher: same XR-06C checkpoint per lane.
- Fixed count: `255000`.
- Optimizer: AdamW.
- LR: `5e-7` first; do not raise until the head-only filter is validated.
- Direct track-state aux: keep `0.0005/0.0125/0.005`.
- SimDR bins/sigma/weight: keep `64/1.5/0.0001`.
- Trainable filter: include `track_head.*`, `track_state_aux_head.*`, and `track_state_simdr_head.*`; optionally add `prev_state_encoder.*` only if the head-only run underfits without moving validation center.
- Gate: same active gates, center `<20.283125744547164`, P10 `>26.74489871433803`, or P5 `>8.69557854788644`, with center/P10 leaders preserved as primary.

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init head-only weak-distill | best-center | 20.29158056122916 | 26.29761974470956 | 8.469388048989432 | no |
| Center-init head-only weak-distill | best-P10 | 20.28970977578844 | 26.26785784448896 | 8.520408460072108 | no |
| P10-init head-only weak-distill | best-center | 20.35549293586186 | 26.46216060093471 | 8.49064655985151 | no |
| P10-init head-only weak-distill | best-P10 | 20.36963668550764 | 26.551446315220424 | 8.533163567951748 | no |

Decision: XR-13 is closed without active gate promotion. The SimDR branch has now been tested as no-distill full model, weak-distill full model, and weak-distill head-only. Do not spend the next GPU slot on another SimDR weight/LR-only replay. Next P0 is XR-02 dense/continuous trajectory availability check.

## 2026-06-16 XR-02 Dense-Trajectory Availability Closeout

Diagnostic artifact:

- `runs/diagnostics/xr02_dense_trajectory_availability_20260616_084506.json`

Result:

- Test manifest: `72` groups, `2238` rows, min gap `360000us`, median gap `4000003us`, dense pair count `0`, longest dense segment `1`.
- Canonical annotations: same `72` groups and `2238` rows, dense pair count `0`, longest dense segment `1`.

Decision: XR-02 cannot produce a gate-valid M2F metric claim on current canonical1 labels. Keep the implementation, but shift immediate execution to XR-04 diagnostics refresh before another temporal/head training run.

## 2026-06-16 XR-04 Diagnostics Refresh Closeout

Artifacts:

- `runs/diagnostics/xr04_refresh_xr06c_bestcenter_failure_buckets_20260616.json`
- `runs/diagnostics/xr04_refresh_xr06c_bestp10_failure_buckets_20260616.json`

Result:

- XR-06C best-center low-sim bucket `similarity_target <= 0.1`: count `493`, weighted center `27.1247`, P10 `19.2399`, P5 `4.7506`.
- XR-06C best-P10 low-sim bucket `similarity_target <= 0.1`: count `493`, weighted center `26.9471`, P10 `19.0024`, P5 `4.9881`.
- Worst recurring buckets remain `session_201` and subjects `42/45/39`.
- Blink/closed-eye rows do not explain the active weighted metric because closed-eye and invalid-track rows are zero-weighted.
- Search/event fallback rows are unavailable for the active leader: the normalized config sets `model.heads.active=track`, disabling those branches.

Decision: XR-04 reweighting-style training is exhausted. The next paper-backed action is EX-Gaze-style confidence/relocalization evidence collection: use `track_pred` confidence/quality to check whether low-sim failures are detectable before training. If positive, implement a gated fallback or auxiliary confidence loss. If negative, move to a bounded new representation path rather than another low-sim weighting replay.

## 2026-06-16 XR-04 Confidence/Quality Probe Closeout

Artifacts:

- `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/test_manifest.jsonl`
- `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestcenter_buckets_20260616.json`
- `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestp10_buckets_20260616.json`

Result:

- Center leader low-sim rows: center `24.5408`, P10 `21.5686`, P5 `2.9412`, confidence `0.9999997`, quality `0.9999998`.
- Center leader high-sim rows are easier but not more confident: `0.6..0.9` center `15.5990`, confidence `0.9999987`; `>=0.9` center `11.1933`, confidence `0.9999980`.
- P10 leader shows the same pattern: low-sim center `24.5044`, confidence `0.9999997`; high-sim `0.6..0.9` center `15.8637`, confidence `0.9999981`.

Decision: existing confidence/quality logits are overconfident and cannot support EX-Gaze-style no-train gating. Next branch should not use current confidence as an acceptance signal. The next paper-backed options are:

1. Train calibrated confidence/quality supervision with explicit error/support targets and a concrete fallback path.
2. Re-enable/train search or event fallback state so relocalization has an alternate prediction to choose.
3. If no fallback is feasible, move to a bounded representation/head change aimed at low-sim/session failure rows.

## 2026-06-16 XR-14 No-Train Fallback Closeout

Artifacts:

- `scripts/external/eval_similarity_fallback.py`
- `runs/diagnostics/xr14_similarity_prevstate_fallback_xr06c_bestcenter_20260616.json`
- `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_infer_20260616`
- `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_trackstate_buckets_20260616.json`
- `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_auxstate_buckets_20260616.json`

Result:

- Previous-state fallback is harmful. Raw official-like batchmean center is `20.283125752718494`; best fallback candidate `blend(threshold=0.05, alpha=0.75)` is `22.57394233260353`.
- In the fallback rows for `threshold=0.05`, raw center is `27.6395` but blended fallback center is `39.0527`, a `+11.4131px` degradation.
- Direct `track_state_aux` is not a fallback predictor. On the balanced 256-row probe it scores center `158.7825`, P10 `0.0`, P5 `0.0`.

Decision:

No-train confidence/previous-state/aux-state fallback is closed. The paper-backed path now requires restoring an alternate relocalization branch before confidence calibration can be meaningful.

XR-14A executable branch:

- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml`
- Runner: `scripts/external/run_xr14_allhead_relocalize_probe.sh`
- Intent: re-enable search/event/track heads, keep mask disabled, preserve the XR-06C track leader with track-only weak distillation targets, then compare `track_state`, `search_state`, and `event_state` to evaluate whether search/event can serve as a real fallback on low-sim/session failure buckets.
- Result: both lanes early-stopped at epoch `8/12` and failed the active XR-06C gates. Center lane scored `20.342467624800545 / 25.644133370263237 / 8.11947306905474`; P10 lane scored `20.348247524670192 / 25.703657184328353 / 7.97278938974653`.
- Branch-state result: `track_state` remains near leader quality but misses gates. `search_state` and `event_state` joined all `2238` test rows but scored P10/P5 `0.0` with center around `178px`.
- Decision: close calibrated search/event fallback for the current branch. Next paper-backed 2차 목표 work should change the representation or data density for low-similarity/session failures instead of adding fallback calibration on unusable alternate states.

## 2026-06-16 XR-15 Support-Adaptive Event-Window Plan

Rationale:

- EyeTrAES and EV-Eye motivate event slicing/data-density as an accuracy axis.
- XR-04/XR-09/XR-10 exhausted low-sim weighting/sampling, but did not change event evidence quality.
- XR-14A rejected search/event fallback, not the primary track event input.

Prepared artifacts:

- `docs/XR15_SUPPORT_ADAPTIVE_PLAN.md`
- `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`
- `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`

XR-15A default:

- Init/teacher: XR-06C lane-specific checkpoint.
- Training: weak distill, AdamW `5e-7`, track-only, full-width, direct state aux `0.0005/0.0125/0.005`.
- Event window: fixed-count policy with runtime adaptive count enabled, base `255000`, min `192000`, max `320000`, reference `4000003us`, scale power `0.5`.

Decision:

Run XR-15A before any new head-heavy branch. Reject hard-subset plus SimDR as immediate P0 because hard-subset training and SimDR-only variants already failed promotion.

XR-15A closeout:

- Center lane best-center: `20.230558776855467 / 26.303146975381033 / 8.573554713385446`.
- Center lane best-P10: `20.259641446386066 / 25.812925890513828 / 8.451105751310076`.
- P10 lane best-center: `20.225680075372967 / 26.50085105895996 / 8.304422058377947`.
- P10 lane best-P10: `20.259886418070113 / 26.12457557405744 / 8.357993486949375`.
- Decision: XR-15A promotes the center gate only. New active gates are center `<20.225680075372967`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- Diagnostics: low `similarity_target <= 0.1`, subject `42/45`, and `session_201` remain the dominant weighted error buckets.
- P5 artifact note: best-P5 helper attempts created `hypers/*` but no `eval_summary.json`; treat P5 checkpoint eval as incomplete artifact coverage, not as a blocker for the center promotion.

Next queue:

1. `XR-15B`: narrow support-adaptive window `224k/255k/288k`, same XR-15A init/teacher, LR, weak distill, and direct-aux contract. Completed with center-only promotion.
2. `XR-15C`: wider support-adaptive window `160k/255k/384k`; completed with center-only promotion.
3. `XR-15D`: bounded track-adapter/coordinate-head branch from the XR-15C center leader and XR-06C P10 leader. Completed with no promotion.
4. `XR-15E`: weak-distill low-LR track-adapter branch. Completed with no promotion.
5. `XR-16A`: weak-distill event-path adapter branch. Completed with no promotion.
6. `XR-16B`: one-shot checkpoint interpolation diagnostic. Completed with no promotion.
7. `XR-17A`: same-branch XR-15C center-to-P5 checkpoint interpolation diagnostic. Completed with P5 promotion at alpha `0.75`.

XR-15B closeout:

- Center lane best-center: `20.215672533852715 / 26.575255823135375 / 8.627551317214966`; new center leader.
- P10 lane best-center: `20.21979672227587 / 26.324405458995273 / 8.28741525241307`; center improvement but not leader.
- Best-P10 eval artifacts stalled after `hypers/*`; recorded as incomplete artifact coverage.
- Decision: continue event-support branch with XR-15C wider `160k/255k/384k` before switching back to temporal/coordinate-head experiments.

XR-15C launch:

- Center lane GPU0 log `runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819`.
- P10 lane GPU1 log `runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816`.
- Startup validation passed for both lanes.

XR-15C closeout:

- Center lane best-center: `20.19088832650866 / 26.009779623576573 / 8.436224787575858`; new center leader.
- P10 lane best-center: `20.205999997683932 / 26.303146975381033 / 8.397959463936942`; center improvement but not leader.
- Decision: XR-15C confirms event-support adaptation remains useful for center error, but P10/P5 regress. Active gates are center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- Next paper-backed action: XR-15D bounded track-adapter/coordinate-head training.

XR-15D launch:

- Runner: `scripts/external/run_xr15d_trackadapters_probe.sh`.
- Config: `configs/external/mode1_stage2_raw_event_count_lr3e-6_nodistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
- Fixed contract: `data.mode1.event_builder.policy=fixed_count`, `event_count_target=255000`, `adaptive_count.enabled=false`.
- Trainable scope: `track_head.*`, `frame_adapter.*`, `event_adapter.*`, `patch_frontend.frame_embed.proj.*`, `patch_frontend.event_embed.proj.*`.
- Center lane: tmux `hgtxr_xr15d_trackadapters_center_gpu0_20260616`, log `runs/_logs/xr15d_trackadapters_center_lr3e-6_gpu0_20260616.log`, init `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt`.
- P10 lane: tmux `hgtxr_xr15d_trackadapters_p10_gpu1_20260616`, log `runs/_logs/xr15d_trackadapters_p10_lr3e-6_gpu1_20260616.log`, init `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`.
- Startup evidence: both lanes passed raw contract, loaded intended checkpoint, reported trainable filter `22` tensors / `448520` parameters, resolved CUDA device `cuda:0/1`, and entered epoch `1/16`.

XR-15D closeout:

- Center lane best-center: `20.333278461865017 / 25.986395263671874 / 8.095238372257777`.
- Center lane best-P10: `20.36335334096636 / 25.821854482378278 / 8.246173749651227`.
- P10 lane best-center: `20.362164442879813 / 26.141582359586444 / 8.562500286102296`.
- P10 lane best-P10: `20.410626077651976 / 26.1883510862078 / 8.385629544939313`.
- Decision: no promotion. The no-distill adapter branch regresses center from XR-15C and does not beat P10/P5 gates.

XR-15E launch:

- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackadapters_centerloss_finetune_fullwidth.yaml`.
- Runner: `scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh`.
- Change isolated from XR-15D: teacher=init weak distillation and LR `5e-7`; trainable scope and fixed255k contract unchanged.
- Center lane: tmux `hgtxr_xr15e_weakdistill_trackadapters_center_gpu0_20260616`, log `runs/_logs/xr15e_weakdistill_trackadapters_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_center_xr15e_fullwidth_20260616_123419`.
- P10 lane: tmux `hgtxr_xr15e_weakdistill_trackadapters_p10_gpu1_20260616`, log `runs/_logs/xr15e_weakdistill_trackadapters_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackadapters_centerloss_p10_xr15e_fullwidth_20260616_123431`.
- Startup validation passed for both lanes.

XR-15E closeout:

- Center lane best-center: `20.281941563742503 / 26.2521265574864 / 8.633928898402623`.
- Center lane best-P10: `20.297553059032985 / 26.096939522879463 / 8.618197590964181`.
- P10 lane best-center: `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
- P10 lane best-P10: `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
- Decision: no promotion. Both-adapter training remains below XR-15C center, XR-06C P10, and XR-05A P5 gates.

XR-16A launch:

- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`.
- Runner: `scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh`.
- Change isolated from XR-15E: trainable scope is reduced to `track_head.*`, `event_adapter.*`, and `patch_frontend.event_embed.proj.*`; frame adapter and frame embedding projection remain frozen.
- Static validation passed: runner `bash -n`, raw event-count contract, diff whitespace check, and trainable-filter smoke reporting `14` tensors / `324680` trainable params.
- Center lane: tmux `hgtxr_xr16a_eventadapter_center_gpu0_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_center_xr16a_fullwidth_20260616_125515`.
- P10 lane: tmux `hgtxr_xr16a_eventadapter_p10_gpu1_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_p10_xr16a_fullwidth_20260616_125515`.
- Startup validation passed for both lanes.

XR-16A closeout:

- Center lane best-center: `20.281941199302672 / 26.2521265574864 / 8.633928898402623`.
- Center lane best-P10: `20.29755315440042 / 26.096939522879463 / 8.618197590964181`.
- P10 lane best-center: `20.3268525327955 / 26.333759232929776 / 8.409864234924317`.
- P10 lane best-P10: `20.3268525327955 / 26.333759232929776 / 8.409864234924317`.
- Decision: no promotion. Event-path-only adapter isolation reproduced XR-15E-level behavior and did not beat center `<20.19088832650866`, P10 `>26.74489871433803`, or P5 `>8.69557854788644`.

XR-16B checkpoint interpolation diagnostic:

- Added `scripts/external/run_xr16b_checkpoint_interp_eval.sh`.
- Alpha `0.125` between XR-15C center best-center and XR-06C best-P10 produced `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.
- Decision: no promotion. Cross-branch interpolation did not recover the P10/P5 tradeoff.

XR-15C best-P5 artifact closeout:

- Center-lane best-P5 eval: `20.243119949953897 / 26.12159937449864 / 8.601190778187343`.
- P10-lane best-P5 eval: `20.245368467058455 / 26.487245675495693 / 8.703231593540737`.
- Decision: P10-lane best-P5 promotes the P5/balanced secondary from XR-05A `8.69557854788644` to XR-15C `8.703231593540737`.
- Active gates were center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.703231593540737` before XR-17A.

XR-17A same-branch interpolation diagnostic:

- Added `scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh`.
- Endpoint A: XR-15C center-lane best-center checkpoint.
- Endpoint B: XR-15C P10-lane best-P5 checkpoint.
- Alpha sweep results:
  - `0.25`: `20.199484479427337 / 26.20960956301008 / 8.625425474984306`.
  - `0.50`: `20.211353632381986 / 26.35204153742109 / 8.679847247259957`.
  - `0.625`: `20.218607200895036 / 26.449830661501203 / 8.703231607164655`.
  - `0.75`: `20.22669484274728 / 26.34566399029323 / 8.732993507385254`.
  - `0.875`: `20.235614109039307 / 26.527636807305473 / 8.673469693320138`.
- Decision: alpha `0.75` promotes P5 to `8.732993507385254`. Center and P10 gates remain unchanged.
- Active gates are now center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.732993507385254`.

XR-17A failure-bucket comparison:

- Artifacts:
  - `runs/diagnostics/xr17a_compare_xr15c_center_failure_buckets_20260616.json`.
  - `runs/diagnostics/xr17a_compare_xr17a_alpha0p75_failure_buckets_20260616.json`.
- Row-weighted aggregate comparison:
  - XR-15C center leader: center `20.0095`, P10 `26.1625`, P5 `8.4824`.
  - XR-17A alpha `0.75`: center `20.0415`, P10 `26.5202`, P5 `8.7890`.
- Bucket interpretation:
  - P10/P5 gains are broad, especially `similarity_target <= 0.1`, `similarity_target > 0.9`, left-eye rows, and subjects `42/45/39`.
  - Center error worsens slightly across most similarity buckets.
  - Subject `43/41` and right-eye P5 remain regression buckets.
- Decision: XR-17A is a useful P5-direction anchor, not a clean replacement for the XR-15C center seed. Next P0 should preserve XR-15C center as primary seed/teacher and constrain any XR-17A influence through dual-teacher regularization, checkpoint-soup anchoring, or a bucket-aware P5 auxiliary.

XR-17B P5-anchor support-adaptive branch:

- Artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh`.
  - Run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310`.
- Setup: XR-15C center best-center init, XR-17A alpha `0.75` teacher, AdamW LR `2.5e-7`, epochs `8`, support-adaptive `160k/255k/384k`, `reference_us=4000003`, `scale_power=0.5`.
- Test results:
  - Best-center: `20.182338142395018 / 26.113946315220424 / 8.474490090778895`; center promotion.
  - Best-P10: `20.22993471963065 / 26.166242245265416 / 8.603316634041922`; no promotion.
  - Best-P5: `20.234592584201266 / 26.09183748109 / 8.844813244683403`; P5 promotion.
- Bucket diagnostics:
  - Best-center weighted aggregate: `20.0015 / 26.2647 / 8.5335`.
  - Best-P5 weighted aggregate: `20.0492 / 26.2136 / 8.8401`.
- Decision: XR-17B validates the bounded P5-anchor hypothesis. Active gates are now center `<20.182338142395018`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`. Next branch should recover P10 without sacrificing the new center/P5 gates.

XR-18A XR-17B-center to XR-06C-P10 interpolation:

- Runner: `scripts/external/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh`.
- Alpha sweep:
  - `0.025`: `20.184962025710515 / 26.113946315220424 / 8.474490090778895`.
  - `0.05`: `20.18765983411244 / 26.054422501155308 / 8.525510501861572`.
  - `0.075`: `20.190412517956325 / 26.156463316508702 / 8.525510501861572`.
  - `0.10`: `20.193220179421562 / 26.20748372077942 / 8.629677173069545`.
- Decision: no promotion. Small interpolation toward XR-06C P10 improves P10/P5 only weakly and quickly loses the XR-17B center gate. Next P0 should be trainable and explicitly constrained, not another no-train alpha sweep.

XR-19A trainable P10-recovery micro polish:

- Artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr19a_p10recovery_micro_polish.sh`.
  - Run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_micro_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_152610`.
- Setup: XR-17B best-P5 init, XR-06C best-P10 teacher, AdamW LR `1.25e-7`, epochs `8`, support-adaptive `160k/255k/384k`, `reference_us=4000003`, `scale_power=0.5`.
- Test results:
  - Best-center: `20.182335831437793 / 25.973640203475952 / 8.588435670307705`.
  - Best-P10: `20.20784169435501 / 26.232143613270352 / 8.609694181169782`.
  - Best-P5: `20.181213889803207 / 25.928997346333094 / 8.508503695896694`; center promotion.
- Bucket diagnostic for best-P10: weighted aggregate `20.0236 / 26.3669 / 8.6357`; low-similarity rows remain weak at `26.5449` center and `17.5772` P10 for `similarity_target < 0.1`.
- Decision: XR-19A confirms P10-teacher micro-polish is center-safe but too weak for P10/P5 recovery. Active gates are now center `<20.181213889803207`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`. Next P0 should be stronger constrained P10 recovery, not another `1.25e-7` micro polish.

XR-20A/XR-20B P10-recovery LR ladder:

- Setup: XR-17B best-P5 init, XR-06C best-P10 teacher, AdamW, epochs `8`, support-adaptive `160k/255k/384k`, `reference_us=4000003`, `scale_power=0.5`.
- XR-20A LR `2.5e-7` run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr2p5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155340`.
- XR-20B LR `5e-7` run root: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155656`.
- Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-20A LR `2.5e-7` | best-center | 20.175542894431523 | 25.884354482378278 | 8.486394848142352 | center promotion |
| XR-20A LR `2.5e-7` | best-P10 | 20.202199470996856 | 26.031038168498448 | 8.74914996964591 | no P10/P5 promotion |
| XR-20A LR `2.5e-7` | best-P5 | 20.211376798152923 | 26.00552796636309 | 8.71088467325483 | no promotion |
| XR-20B LR `5e-7` | best-center | 20.257083107743945 | 25.838861281531198 | 8.380102341515677 | no promotion |
| XR-20B LR `5e-7` | best-P10 | 20.183283712182725 | 25.8312082358769 | 8.423894848142352 | no promotion |
| XR-20B LR `5e-7` | best-P5 | 20.218086302280426 | 25.805698026929583 | 8.588435690743582 | no promotion |

- Decision: XR-20A promotes center only. The higher LR `5e-7` branch worsens P10 and center, so the P5-init/XR-06C-P10-teacher LR-only ladder is closed. Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`.
- Bucket diagnostic for XR-20A best-center: `runs/diagnostics/xr20a_lr2p5e7_bestcenter_failure_buckets_20260616.json`, weighted aggregate `19.9949 / 26.0092 / 8.5335`; low-similarity `similarity_target < 0.1` remains weak at weighted `26.5436` center, `17.8147` P10, `5.2257` P5. Worst session buckets remain `user42/*/session_201`, `user39/left/session_201`, and `user45/right/session_201`.
- Next P0: do not spend the next GPU slot on another LR-only replay of this branch. Use a paper-backed mechanism change: P10-specialized loss/head, calibrated representation change, or a bounded branch that explicitly accepts P10-leader-only promotion.

XR-21 P10-margin mechanism-change closeout:

- Setup: decoded-center squared hinge loss at a 10 px margin. Secondary lanes used XR-20A best-center init and XR-06C best-P10 teacher. Primary lane used XR-06C best-P10 as both init and teacher, with best/scheduler metric set to `metric_track_p10_pct`.
- Added configs:
  - `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_fullwidth.yaml`
  - `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_fullwidth.yaml`
- Added runners:
  - `scripts/external/run_xr21_p10margin_supportadaptive_probe.sh`
  - `scripts/external/run_xr21_p10leader_margin_probe.sh`
- Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-21 secondary `w=0.0015` | best-center | 20.18683280604226 | 25.886480338232857 | 8.49914995602199 | no promotion |
| XR-21 secondary `w=0.0015` | best-P10 | 20.19624582529068 | 25.90688850539071 | 8.558673770087106 | no promotion |
| XR-21 secondary `w=0.0015` | best-P5 | 20.19960424559457 | 25.96938850539071 | 8.77976222038269 | no promotion |
| XR-21 secondary `w=0.003` | best-center | 20.197126933506556 | 25.900510951450894 | 8.618197584152222 | no promotion |
| XR-21 secondary `w=0.003` | best-P10 | 20.187303059441703 | 26.126701450347902 | 8.791666977746146 | no promotion |
| XR-21 secondary `w=0.003` | best-P5 | 20.200117662974765 | 25.96938850539071 | 8.728741809300013 | no promotion |
| XR-21 primary `w=0.0005` | best-P10 | 20.26367484842028 | 26.20110617365156 | 8.696003689084733 | no promotion |
| XR-21 primary `w=0.0005` | best-P5 | 20.26367484842028 | 26.20110617365156 | 8.696003689084733 | no promotion |

- Decision: XR-21 does not promote center, P10, or P5. Tiny P10-margin replay is closed. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`.
- Next P0: change representation/supervision more materially instead of repeating LR-only or hinge-weight-only probes.

XR-22 tri-leader checkpoint soup closeout:

- Setup: no-train weighted checkpoint soup across current center, P10, and P5 leaders.
  - Center anchor: XR-20A best-center.
  - P10 anchor: XR-06C best-P10.
  - P5 anchor: XR-17B best-P5.
- Added scripts:
  - `scripts/external/average_hbtxr_checkpoints.py`
  - `scripts/external/run_xr22_trileader_soup_eval.sh`
- Test results:

| Soup | Center px | P10 pct | P5 pct | Gate decision |
|---|---:|---:|---:|---|
| `c50_p25_f25` | 20.21947033064706 | 26.349915708814347 | 8.785289430618286 | no promotion |
| `c34_p33_f33` | 20.236038860252926 | 26.4604599407741 | 8.869047941480364 | P5 promotion |
| `c25_p50_f25` | 20.257381524358475 | 26.36267081669399 | 8.601190784999302 | no promotion |
| `c25_p25_f50` | 20.235262938908168 | 26.269133404323032 | 8.829932287761144 | no promotion |
| `c20_p60_f20` | 20.270640075206757 | 26.405187838418144 | 8.735119356427873 | no promotion |
| `c20_p40_f40` | 20.251396659442356 | 26.349915736062187 | 8.603316634041922 | no promotion |

- Decision: XR-22 promotes P5/balanced gate to `8.869047941480364`. Center and P10 gates remain XR-20A and XR-06C. Active gates are now center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.

XR-23 P10-preserving optimizer probe closeout:

- Setup: Stage2-only P10-leader optimizer probe seeded from XR-06C best-P10 as both init and weak-distill teacher. Hinge replay was disabled to isolate optimizer behavior. Best/scheduler metric stayed `metric_track_p10_pct`.
- Added artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr23_p10_optimizer_probe.sh`.
- Commands:
  - ADOPT/GPU0: `bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 adopt 2.5e-7 cuda:0`.
  - Lion/GPU1: `bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 lion 1e-7 cuda:1`.
- Runtime:
  - ADOPT completed epoch `8/8` with train exit `0`.
  - Lion early-stopped at epoch `6/8` with train exit `0`.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-23A ADOPT `2.5e-7` | best-P10 | 20.261837770257678 | 26.318027945927213 | 8.681122745786395 | no promotion |
| XR-23A ADOPT `2.5e-7` | best-P5 | 20.219128920350755 | 25.956633363451278 | 8.469388042177473 | no promotion |
| XR-23B Lion `1e-7` | best-P10 | 20.22320341382708 | 26.311650391987392 | 8.513180569240026 | no promotion |
| XR-23B Lion `1e-7` | best-P5 | 20.271730688640048 | 26.129677595411028 | 8.532313203811645 | no promotion |

- Decision: XR-23 does not promote center, P10, or P5. Optimizer substitution alone is insufficient to preserve or exceed XR-06C P10. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.
- Next P0: prefer a bounded P10-recovery branch using the XR-22 `c34/p33/f33` soup as a P5/P10-friendly anchor. If optimizer work is retried, use a narrower ADOPT-defaults validation (`betas=[0.9,0.9999]`, `eps=1e-6`) rather than a broad optimizer sweep.

XR-24 XR-22-anchor P10-recovery closeout:

- Setup: Stage2-only AdamW P10-recovery seeded from the XR-22 `c34/p33/f33` soup checkpoint, with XR-06C best-P10 as weak-distill teacher. Hinge replay was disabled. Best/scheduler metric stayed `metric_track_p10_pct`.
- Added artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr24_xr22_anchor_p10recovery.sh`.
- Commands:
  - GPU0 conservative LR: `bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
  - GPU1 primary LR: `bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Runtime:
  - LR `1.25e-7` completed epoch `8/8` with train exit `0`.
  - LR `2.5e-7` completed epoch `8/8` with train exit `0`.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-24A AdamW `1.25e-7` | best-P10 | 20.217043702942984 | 26.254252440588814 | 8.764881270272392 | no promotion |
| XR-24A AdamW `1.25e-7` | best-P5 | 20.223851100036075 | 26.415817070007325 | 8.722364262172155 | no promotion |
| XR-24B AdamW `2.5e-7` | best-P10 | 20.23367166178567 | 26.121599388122558 | 8.588435677119664 | no promotion |
| XR-24B AdamW `2.5e-7` | best-P5 | 20.238911376680647 | 26.07695653779166 | 8.791666984558105 | no promotion |

- Decision: XR-24 does not promote center, P10, or P5. XR-22 soup is useful as a no-train P5/P10-friendly anchor, but trainable AdamW polish from that anchor drifts below both the XR-06C P10 gate and the XR-22 P5 gate.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.
- Next P0: stop LR-only AdamW polishing from XR-22. Use either a narrow ADOPT-defaults validation on the XR-22 anchor (`betas=[0.9,0.9999]`, `eps=1e-6`) or a true P10-specific head/loss branch with the XR-06C P10 checkpoint preserved as teacher/anchor.

XR-25 XR-22-anchor ADOPT-defaults closeout:

- Setup: same XR-22 `c34/p33/f33` init and XR-06C best-P10 teacher as XR-24, but using ADOPT with its upstream-style defaults `betas=[0.9,0.9999]`, `eps=1e-6`.
- Added artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh`.
- Commands:
  - GPU0 conservative LR: `bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0`.
  - GPU1 primary LR: `bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1`.
- Runtime:
  - LR `1.25e-7` early-stopped at epoch `7/8` with train exit `0`.
  - LR `2.5e-7` completed epoch `8/8` with train exit `0`.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-25A ADOPT-defaults `1.25e-7` | best-P10 | 20.21177260194506 | 26.46045993396214 | 8.728741809300013 | no promotion |
| XR-25A ADOPT-defaults `1.25e-7` | best-P5 | 20.22005627495902 | 26.181123181751797 | 8.707483305249895 | no promotion |
| XR-25B ADOPT-defaults `2.5e-7` | best-P10 | 20.23533037390028 | 26.108844266619002 | 8.64583364214216 | no promotion |
| XR-25B ADOPT-defaults `2.5e-7` | best-P5 | 20.23760941709791 | 26.187500749315536 | 8.86607174192156 | no promotion |

- Decision: XR-25 does not promote center, P10, or P5. The best P5 result misses the XR-22 gate by about `0.0030`, and P10 remains well below XR-06C.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.
- Next P0: close optimizer-default and XR-22 trainable-polish branch. Move to XR-26 P10-boundary loss using XR-06C best-P10 as init and teacher, with head-only trainable scope and the existing `track_state_aux` side head.

XR-26 P10-boundary aux head-only closeout:

- Setup: Stage2-only head-only training from XR-06C best-P10 as both init and weak-distill teacher. Added a differentiable P10-boundary surrogate around the 10 px metric threshold on both `track/state` and `track/state_aux`.
- Added artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10boundary_headonly_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr26_p10boundary_aux_probe.sh`.
  - Loss/tests: `src/hbtxr/loss/bundles/track.py`, `tests/test_track_center_l2_loss.py`.
- Commands:
  - Default lane/GPU0: `bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:0`.
  - Light lane/GPU1: `bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt xr06cp10_init_teacher_light 0.01 0.005`.
- Runtime:
  - Both lanes early-stopped at epoch `6/8` with train exit `0`.
  - Best validation P10 stayed at `21.538634983998424` from epoch `2`, so the new boundary loss did not improve the P10-selection surface.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-26A boundary `0.02/0.01` | best-P10 | 20.32213627440589 | 26.45790890966143 | 8.535289403370449 | no promotion |
| XR-26A boundary `0.02/0.01` | best-P5 | 20.318065077917918 | 26.42814700944083 | 8.541666957310268 | no promotion |
| XR-26B boundary `0.01/0.005` | best-P10 | 20.32214218207768 | 26.45790890966143 | 8.535289403370449 | no promotion |
| XR-26B boundary `0.01/0.005` | best-P5 | 20.318073788711004 | 26.42814700944083 | 8.541666957310268 | no promotion |

- Decision: XR-26 does not promote center, P10, or P5. Boundary-weight strength had negligible effect; head-only P10-boundary polishing is closed.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, and P5 `>8.869047941480364`.
- Next P0: stop scalar P10-surrogate polish. Move to a representation/teacher branch: P10-direct coordinate-distribution or heatmap-style head with stronger P10 teacher anchoring, or a full teacher-quality rerun if the current XR-06C teacher is the limiting surface.

XR-27 track-heatmap coordinate representation closeout:

- Setup: Stage2-only head-only training from XR-06C best-P10 as init/teacher. Added a track-center heatmap head and used heatmap peak plus sub-cell offset as the coordinate representation for `track/state[:, :2]`.
- Evidence source: TDTracker/AIS heatmap/KL and SimDR-style coordinate decomposition motivate changing the coordinate representation after scalar P10-boundary and optimizer branches saturated.
- Distillation correction: state-similarity distillation was disabled because the teacher has random weights for the newly added heatmap head. Clean XR-27 runs use feature/prediction distillation only.
- Added artifacts:
  - Config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`.
  - Runner: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`.
  - Result note: `docs/resources/xr27_trackheatmap_results_2026_06_16.md`.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-27A LR `1e-4` | best-P10 | 17.274589475563594 | 31.915391901561193 | 10.289966331209456 | promotes all gates |
| XR-27A LR `1e-4` | best-P5 | 17.6397743497576 | 29.80569808142526 | 10.210459525244577 | promotes all gates, weaker than best-P10 |
| XR-27B LR `3e-5` | best-P10 | 18.388037020819528 | 27.238521099090576 | 8.488520683561052 | promotes center/P10 only |
| XR-27B LR `3e-5` | best-P5 | 19.03528357914516 | 24.192602627617973 | 6.854166896002633 | promotes center only |

- Decision: XR-27A best-P10 is the new active leader. Active gates are center `<17.274589475563594`, P10 `>31.915391901561193`, and P5 `>10.289966331209456`.
- Next P0: XR-28 consolidation. Run failure-bucket comparison against XR-06C/XR-20A/XR-22, add or evaluate best-center checkpoint selection, then run a narrow LR/heatmap-weight refinement around LR `1e-4`.

XR-28 track-heatmap consolidation diagnostics:

- Artifact: `docs/resources/xr28_trackheatmap_consolidation_diagnostics_2026_06_16.md`.
- Generated bucket summaries:
  - `runs/diagnostics/xr28_xr27a_bestp10_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr06c_bestp10_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr20a_bestcenter_failure_buckets_20260616.json`
  - `runs/diagnostics/xr28_xr22_c34p33f33_failure_buckets_20260616.json`
- Finding: XR-27A improves weighted overall metrics to `17.1708 / 32.1921 / 10.3219`; the biggest meaningful gains are in `similarity_target <= 0.3`.
- Remaining risk: high-similarity `>0.9` P10 is still better in XR-20A/XR-22, and subject `39` loses P10/P5 even though center improves.
- Runner update: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh` now supports `XR27_BEST_METRIC_NAME` and `XR27_SCHEDULER_METRIC_NAME`; existing XR-27A best-center and best-P10 both occur at epoch `10`.
- Next P0: run LR `7e-5` and `1.5e-4` refinements with `XR27_BEST_METRIC_NAME=metric_track_center_px`, same heatmap weights, and `distillation.state_similarity=false`.

XR-28 heatmap LR refinement closeout:

- Artifact: `docs/resources/xr28_heatmap_lr_refinement_results_2026_06_16.md`.
- Commands used the XR-27 heatmap runner with `XR27_BEST_METRIC_NAME=metric_track_center_px`, `XR27_SCHEDULER_METRIC_NAME=metric_track_center_px`, unchanged heatmap weights `0.005/0.001`, center L2 `0.001`, and `distillation.state_similarity=false`.
- Runtime:
  - LR `7e-5` on GPU0 completed epoch `10/10` with train exit `0`.
  - LR `1.5e-4` on GPU1 completed epoch `10/10` with train exit `0`.
  - Six eval summaries exist: best-P10, best-P5, and best-center for both LR lanes.

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-28 LR `7e-5` | best-P10 | 17.935810264519283 | 28.53273880141122 | 10.105442489896502 | no promotion |
| XR-28 LR `7e-5` | best-P5 | 19.03988778250558 | 23.74872510773795 | 7.972789362498692 | no promotion |
| XR-28 LR `7e-5` | best-center | 17.55661221402032 | 31.095664044788904 | 9.788265630177088 | no promotion |
| XR-28 LR `1.5e-4` | best-P10 | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | promotes all gates |
| XR-28 LR `1.5e-4` | best-P5 | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | promotes all gates |
| XR-28 LR `1.5e-4` | best-center | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | promotes all gates |

- Decision: XR-28 LR `1.5e-4` is the new active software leader. Active gates are center `<17.08485197339739`, P10 `>32.081208263124736`, and P5 `>10.502551344462804`.
- Next P0: XR-29 should first run post-XR-28 failure buckets, then either reproduce/extend LR `1.5e-4` or test the bounded LR neighbors `1.25e-4` and `1.75e-4`. Heatmap-weight or teacher-quality changes should wait until the bucket analysis says which failure mode remains.

XR-29 post-XR-28 diagnostic and LR-neighbor launch:

- Artifact: `docs/resources/xr29_post_xr28_failure_bucket_diagnostics_2026_06_16.md`.
- Generated diagnostic: `runs/diagnostics/xr29_xr28_lr1p5e4_bestcenter_failure_buckets_20260616.json`, joined `2238`, missing `0`.
- Weighted aggregate improved over XR-27A to `16.9868 / 32.3454 / 10.5774`.
- Remaining failure modes:
  - `similarity <=0.1`: `19.8681 / 26.60 / 7.36`.
  - `similarity >0.9`: P10/P5 still worse than XR-20A/XR-22 anchors.
  - subject `39`: `19.8477 / 18.90 / 3.15`.
  - worst sessions remain `user45/right/session_201`, `user42/*/session_201`, and subject-39 sessions.
- Launched bounded LR-neighbor runs:
  - GPU0 LR `1.25e-4`, log `runs/_logs/xr29_heatmap_centerselect_lr1p25e-4_gpu0_20260616.log`.
  - GPU1 LR `1.75e-4`, log `runs/_logs/xr29_heatmap_centerselect_lr1p75e-4_gpu1_20260616.log`.
- LR-neighbor closeout:

| XR-29 lane | Checkpoint lanes | Center | P10 | P5 | Decision |
|---|---|---:|---:|---:|---|
| LR `1.25e-4` | best-P10/best-P5/best-center | 17.179786903517588 | 32.04761978558132 | 10.53443912097386 | P5 only; not selected |
| LR `1.75e-4` | best-P10/best-P5/best-center | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 | promotes all gates |

- Decision: XR-29 LR `1.75e-4` is the new active software leader. Active gates are center `<17.04961508342198`, P10 `>32.56462665285383`, and P5 `>11.50467722075326`.
- Next P0: run LR `1.625e-4` and `1.875e-4` as a bounded micro-bracket around the promoted LR `1.75e-4`. Keep heatmap weights, teacher/init checkpoint, support-adaptive count contract, and center-selected checkpoints unchanged.

XR-30 LR micro-bracket closeout:

- Artifact: `docs/resources/xr30_lr_micro_bracket_results_2026_06_16.md`.
- Ran LR `1.625e-4` on GPU0 and LR `1.875e-4` on GPU1 with unchanged heatmap weights, support-adaptive fixed255k contract, XR-06C init/teacher, and center-selected checkpointing.
- Both train lanes completed epoch `10/10` with train exit `0`; all six best-P10/best-P5/best-center full-test eval summaries were generated.

| XR-30 lane | Checkpoint lanes | Center | P10 | P5 | Decision |
|---|---|---:|---:|---:|---|
| LR `1.625e-4` | best-P10/best-P5/best-center | 17.067689692974092 | 32.420068802152365 | 10.866496937615532 | no promotion |
| LR `1.875e-4` | best-P10/best-P5/best-center | 17.035534060001375 | 32.42474567549569 | 11.266581957680838 | center only |

- Decision: XR-30 LR `1.875e-4` is the new center leader, but XR-29 LR `1.75e-4` remains the best unified checkpoint and still owns the P5 gate.

XR-31 no-train checkpoint interpolation:

- Interpolated XR-29 LR `1.75e-4` best-center toward XR-30 LR `1.875e-4` best-center with alpha `0.125/0.1875/0.25/0.50/0.75`.
- Runner: `scripts/external/run_xr31_xr29_xr30_heatmap_interp_eval.sh`.

| Alpha | Center | P10 | P5 | Decision |
|---:|---:|---:|---:|---|
| `0.125` | 17.04816469124385 | 32.57313004221235 | 11.488945926938738 | P10-only improvement |
| `0.1875` | 17.047394837651932 | 32.528487185069494 | 11.503826883860997 | no promotion |
| `0.25` | 17.04659355367933 | 32.57950758934021 | 11.503826883860997 | P10 promoted |
| `0.50` | 17.043099636690958 | 32.50170144353594 | 11.37500034059797 | no promotion |
| `0.75` | 17.039319237640925 | 32.5080789906638 | 11.362245225906372 | no promotion |

- Decision: XR-31 alpha `0.25` became a temporary P10 leader, but it did not replace the unified checkpoint because P5 was slightly below XR-29.

XR-32/XR-33 trained LR midpoint probes:

| Run | LR | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---:|---|
| XR-32 | `1.8125e-4` | 17.041867678506033 | 32.5463443006788 | 11.37500034059797 | no promotion |
| XR-33 | `1.84375e-4` | 17.039179919447218 | 32.62074908529009 | 11.362245225906372 | P10 promoted |

- Decision: XR-33 LR `1.84375e-4` is the new P10 leader, but it does not replace the unified checkpoint because center and P5 remain worse than the active gates. Active gates are center `<17.035534060001375`, P10 `>32.62074908529009`, P5 `>11.50467722075326`.
- Next P0: stop close LR-only replay unless history shows non-convergence. Prefer heatmap loss-ratio refinement or bounded XR-29/XR-33 anchor/soup regularization.

XR-34 heatmap loss-ratio refinement:

- Runner: `scripts/external/run_xr34_heatmap_lossratio_refinement.sh`.
- XR-34A GPU0: XR-29 init/teacher, LR `1.75e-4`, heatmap/offset/center `0.004/0.0015/0.0015`. Goal: preserve center/P5 while testing stronger sub-cell and decoded-center emphasis.
- XR-34B GPU1: XR-33 init/teacher, LR `1.84375e-4`, heatmap/offset/center `0.006/0.001/0.001`. Goal: sharpen the current P10 leader direction.
- Startup validation: both lanes passed raw event-count contract, loaded intended checkpoint, resolved to `cuda:0/1`, kept trainable filter at `6` tensors / `1,331,328` params, and entered epoch `1/10`.
- Results: XR-34A reached `17.026217068944657 / 32.430698088237214 / 10.911139822006225`. XR-34B best-P10 reached `16.53321223940168 / 33.77168447630746 / 11.276786088943481`.
- Decision: XR-34B is the new center/P10 leader. It does not replace XR-29 as the strict P5/unified anchor because P5 remains below `11.50467722075326`.
- XR-35 no-train interpolation results: alpha `0.03125/0.0625/0.09375/0.125` produced `17.0163/32.4753/11.3346`, `16.9840/32.4690/11.3912`, `16.9528/32.4243/11.4209`, and `16.9230/32.3520/11.2551`. No active gate promoted.
- XR-36 trainable P5-anchor continuation completed. XR-36A best-P10 reached `16.59279990025929 / 34.39710958344596 / 11.738095617294311`, restoring and promoting P5. XR-36A best-P5 reached `16.576952314376832 / 34.74064704350063 / 11.415816688537598`, promoting P10. XR-36B best-P10 reached `16.53305721793856 / 34.19387831687927 / 11.502551344462804`, narrowly promoting center and also promoting P10.
- XR-37 no-train interpolation completed. Alpha `0.10/0.20/0.35/0.50` produced `16.5225/34.0621/11.5748`, `16.5148/34.2883/11.4439`, `16.5080/34.2309/11.2925`, and `16.5076/34.3321/11.5391`. Alpha `0.50` promotes center only.
- Next: XR-38 center-preserving P10/P5 recovery from the XR-37 alpha `0.50` anchor with explicit XR-36A P10/P5 references.

XR-38 center-preserving P10/P5 recovery:

- Runner: `scripts/external/run_xr38_center_preserve_p10p5_recovery.sh`.
- Plan: `docs/resources/xr38_center_preserve_p10p5_recovery_plan_2026_06_17.md`.
- Lane A: XR-37 alpha `0.50` init, XR-36A best-P10 reference, LR `2.5e-5`, center-selected. Goal: preserve center while softly recovering P5/P10.
- Lane B: XR-37 alpha `0.50` init, XR-36A best-P5 reference, LR `1.25e-5`, P10-selected. Goal: recover strict P10 with bounded LR drift.
- Closeout: XR-38A did not promote. XR-38B best-P10 nearly recovered P10 at `16.510086681161606 / 34.707483761651176 / 11.223214626312256`, but missed the strict P10 gate. XR-38B best-P5 promoted P5 to `11.843962955474854` with center `16.513694180761064` and P10 `33.93452457700457`.
- Decision: update the strict P5 gate only. Active gates after XR-38 are center `<16.507612899371555`, P10 `>34.74064704350063`, P5 `>11.843962955474854`.
- Next: XR-39 no-train mixed-leader soup/interpolation using XR-37 alpha `0.50`, XR-38B best-P5, and XR-36A best-P5; fallback XR-38C adjusts heatmap/offset/center ratio to `0.004/0.0015/0.0015`.

XR-39 mixed-leader soup:

- Runner: `scripts/external/run_xr39_mixed_leader_soup_eval.sh`.
- Plan/results: `docs/resources/xr39_mixed_leader_soup_plan_2026_06_17.md`.
- Evaluated 13 soups over XR-37 center, XR-36A P10, and XR-38B P5 anchors.
- Best center: `c60p25f15`, `16.491779099191938 / 34.5306130204882 / 11.50000034059797`.
- Best P10: `c25p45f30`, `16.503940873486656 / 35.02295998845781 / 11.460459525244577`.
- Best XR-39 P5: `c70p20f10`, `16.492875189440593 / 34.34566400391715 / 11.744473137174333`, below the XR-38B P5 gate.
- Decision: XR-39 promotes center and P10; P5 remains XR-38B-owned.
- Active gates after XR-39 are center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.843962955474854`.

XR-40 XR39-leader/P5 follow-up soup:

- Runner: `scripts/external/run_xr40_xr39leader_p5_soup_eval.sh`.
- Plan/results: `docs/resources/xr40_xr39leader_p5_soup_plan_2026_06_17.md`.
- Evaluated 7 soups over XR-39 center, XR-39 P10, and XR-38B P5 anchors.
- Best center: `c45p35f20`, `16.493494159834725 / 34.77508579662868 / 11.255527530397687`, below the center gate.
- Best P10: `c40p40f20`, `16.494016419138227 / 34.7750858102526 / 11.249149976457868`, below the P10 gate.
- Best P5: `c35p25f40`, `16.49522715806961 / 34.64115719795227 / 11.529762240818568`, below the P5 gate.
- Decision: no promotion. No-train soup follow-up is closed; next P0 is trainable loss-ratio fallback for P5 recovery.

XR-41 loss-ratio P5 fallback:

- Runner: `scripts/external/run_xr41_lossratio_p5_fallback.sh`.
- Plan/results: `docs/resources/xr41_lossratio_p5_fallback_plan_2026_06_17.md`.
- XR-41A used XR-38B best-P5 init, XR-39 P10 reference, LR `1e-5`, best metric `metric_track_p5_pct`, loss ratio `0.004/0.0015/0.0015`.
- XR-41B used XR-39 P10 init, XR-38B P5 reference, LR `8e-6`, best metric `metric_track_p10_pct`, same loss ratio.
- XR-41A best-P5 promoted P5 to `11.868197652271816` with center `16.519514334201812` and P10 `34.09821502821786`.
- Decision: update P5 gate only. Active gates after XR-41 are center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.

XR-42 low-drift P5-preserve branch:

- Runner: `scripts/external/run_xr42_p5_preserve_lowdrift.sh`.
- Plan/results: `docs/resources/xr42_p5_preserve_lowdrift_plan_2026_06_17.md`.
- XR-42A used XR-39 center `c60p25f15` init, XR-41A best-P5 reference, LR `3e-6`, best metric `metric_track_center_px`, loss ratio `0.0035/0.00125/0.0020`.
- XR-42B used XR-39 P10 `c25p45f30` init, XR-41A best-P5 reference, LR `3e-6`, best metric `metric_track_p10_pct`, same loss ratio.
- XR-42A best-P10 reached `16.49802110535758 / 34.48299399103437 / 11.366922119685581`.
- XR-42B best-P10 reached `16.505801352432798 / 34.3171777180263 / 11.278911903926305`.
- Decision: no gate promoted. Active gates remain center `<16.491779099191938`, P10 `>35.02295998845781`, P5 `>11.868197652271816`.
- Next option: XR-42C only if explicit tiny distillation/regularization is added; XR-42A/B show teacher-as-reference alone is insufficient.

XR-56 direct P10 soft-threshold supervision:

- Runner: `scripts/external/run_xr56_soft_threshold_p10_supervision.sh`.
- Plan: `docs/resources/xr56_soft_threshold_p10_supervision_plan_2026_06_17.md`.
- Motivation: XR-49/XR-52/XR-55 showed that band-limited boundary loss, checkpoint-space recombination, and low-LR scope expansion are insufficient for the strict P10 gate.
- Mechanism: optimize `softplus((center_error - margin_px) / temperature_px)` with P10 margin `10 px`; add a smaller P5 guard with margin `5 px`.
- Lane A: XR-39 P10 self-teacher, LR `7.5e-7`, P10 soft `0.006`, P5 soft `0.001`.
- Lane B: XR-52B seed with XR-39 P10 teacher, LR `5e-7`, P10 soft `0.008`, P5 soft `0.002`.
- Closeout: XR-56A reached best-P10 `16.49389898266111 / 34.32270488057818 / 11.699830286843437` and best-P5 `16.4924229485648 / 34.62330012321472 / 11.304422106061663`. XR-56B reached best-P10 `16.477364584377835 / 34.37670146397182 / 12.044218049730574` and best-P5 `16.4701875601496 / 34.29294293948582 / 12.133503770828247`.
- Decision: XR-56 promoted center and P5 through XR-56B best-P5, but did not recover P10. Active gates are now center `<16.4701875601496`, P10 `>35.02295998845781`, P5 `>12.133503770828247`. Next P0 is XR-57 calibration/refinement or P10 teacher retraining, not another checkpoint soup or scalar loss-only branch.

XR-57 bounded P10 center-refine calibration:

- Runner: `scripts/external/run_xr57_p10_center_refine_calibration.sh`.
- Plan: `docs/resources/xr57_p10_center_refine_calibration_plan_2026_06_17.md`.
- Mechanism: add zero-initialized bounded residual `delta_xy` from track fused features and route it into final `track/state`.
- Lane A: XR-56B best-P5 init, XR-39 P10 teacher, LR `1e-4`, max delta `4 px`, refine head + heatmap head trainable.
- Lane B: XR-39 P10 init/self-teacher, LR `7.5e-5`, max delta `3 px`, refine head-only trainable.
- Closeout: no promotion. XR-57A best-P10 reached `16.69362453562873/34.38307912690299/10.973214619500297`; XR-57A best-P5 reached `16.715463175092424/34.849490649359566/10.925595603670393`; XR-57B best-P10 reached `16.504537062985555/34.137755966186525/11.207483332497732`; XR-57B best-P5 reached `16.4997801729611/34.190476996558054/11.476615987505232`.
- Decision: residual center refinement is not sufficient to recover the strict P10 gate. Next P0 is XR-58 P10 teacher refresh: train a deliberately P10-specialized teacher/anchor, then use it for distillation only if its full-test P10 beats XR-39.

XR-58 P10 teacher refresh:

- Runner: `scripts/external/run_xr58_p10_teacher_refresh.sh`.
- Plan: `docs/resources/xr58_p10_teacher_refresh_plan_2026_06_17.md`.
- Mechanism: train a P10-specialized teacher from XR-39 P10 with expanded trainable scope, accepting temporary center/P5 drift.
- Lane A: anchored self-teacher, LR `1e-6`, P10 soft `0.014`, state distill `0.00010`.
- Lane B: free P10 teacher, LR `5e-7`, P10 soft `0.018`, distillation off.
- Closeout: no promotion. XR-58A best-P10 reached `16.503614359242576/34.90306201662336/11.51275544847761`; XR-58A best-P5 reached `16.501813726765768/34.68920147078378/11.259779255730765`; XR-58B best-P10 reached `16.506438190596445/34.84226275852748/11.501275873184204`; XR-58B best-P5 reached `16.49833288192749/34.540391949244906/11.062075165339879`.
- Decision: XR-58A narrowed the P10 gap but still missed XR-39 by about `0.1199`. Next P0 is XR-59 narrow anchored-teacher bracket around XR-58A, not another free no-distill branch.

XR-59 XR58A teacher bracket:

- Runner: `scripts/external/run_xr59_xr58a_teacher_bracket.sh`.
- Plan: `docs/resources/xr59_xr58a_teacher_bracket_plan_2026_06_17.md`.
- Mechanism: continue from XR-58A best-P10 with XR-39 as P10 teacher, keeping anchored state distillation.
- Lane A: LR `7.5e-7`, P10 soft `0.016`, temp `0.9`, center L2 `0.0004`.
- Lane B: LR `5e-7`, P10 soft `0.022`, temp `0.75`, center L2 `0.0003`.
- Closeout: no promotion. XR-59A best-P10/best-P5 both reached `16.520220368249074 / 34.300170864377705 / 11.376701021194458`; XR-59B best-P10/best-P5 both reached `16.51065547806876 / 34.43409944261823 / 11.287415306908743`.
- Decision: scalar P10-soft continuation regressed from XR-58A and should be closed. Next P0 should change representation or protocol, for example a dedicated P10 calibration/classification head, subject/session failure-bucket data diagnostics, or different teacher target construction.

XR-60 P10 candidate-head calibration:

- Runner: `scripts/external/run_xr60_p10_candidate_head.sh`.
- Plan: `docs/resources/xr60_p10_candidate_head_plan_2026_06_17.md`.
- Mechanism: add a zero-initialized default-off multi-candidate head that predicts bounded candidate center deltas plus candidate logits, then routes the top-logit candidate into final `track/state` for metric-visible P10 calibration.
- Lane A: XR-59B best-P10 init, XR-39 P10 teacher, candidate-only trainable scope, `K=4`, max delta `8 px`, blend `0.50`, LR `1e-4`.
- Lane B: same init/teacher, candidate+heatmap trainable scope, `K=6`, max delta `10 px`, blend `0.75`, LR `7.5e-5`.
- Candidate losses: P10 BCE, min soft-threshold, and delta L2; scalar P10 soft is off, scalar P5 soft remains as a small guard.
- Decision target: recover P10 beyond `35.02295998845781` without giving back the XR-56B center/P5 gates.
