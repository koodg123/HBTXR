# HGTXR Software Current Project Synthesis - 2026-06-27

## Purpose

This document consolidates the current HGTXR software work state, experiment plans, experiment results, conversation decisions, and continuation path as of 2026-06-27 KST.

It is intended to be the high-level companion to:

- `HANDOVER.md`
- `docs/HANDOVER_2026_06_27.md`
- `docs/resources/current_stage1_work_progress_2026_06_25.md`
- `docs/resources/stage1_frame_search_escalation_plan_2026_06_22.md`
- `docs/track/PROGRESS.md`
- `docs/track/log.md`

## Current Authority

The current active work priority is Stage1 frame-based Search accuracy improvement.

Active Stage1 frame-search baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Validation Search P10: `28.27717937613433`
- Validation Search P5: `10.153863744915656`
- Validation Search center: `17.257583906065744`
- Minimum promotion epoch gate: `50` epochs.

Strict Stage1 promotion policy:

- P10 must exceed `28.27717937613433`.
- Center must be `<= 17.257583906065744`.
- P5 must be tracked and should recover toward `10.153863744915656`.
- No candidate can replace the baseline before it completes at least `50` epochs.

## Conversation And Decision Summary

The working directory was corrected to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`.

The early environment issue was a driver/NVML mismatch:

- Kernel module was initially on NVIDIA `595.45.04`.
- User-space `libnvidia-ml.so.1` and `libcuda.so.1` pointed at `610.43.02`.
- The later readiness output showed the host repaired to NVIDIA `595.71.05`, CUDA `13.2`, two RTX 5080 GPUs, and PyTorch CUDA availability.

Raw event-count training readiness was later confirmed:

- Raw event-count contract passed.
- Manifest split counts were train `5929`, val `844`, test `2238`.
- CUDA device count was `2`.
- Torch version was `2.12.0+cu130`.
- Stage2 checkpoint existed at `runs/raw_mode1_stage2_event_count_config_smoke_20260610_092747/train/best_track_p10.pt`.

The user then moved from raw training execution into result analysis, experiment planning, reference-code/paper analysis, run directory cleanup, git hygiene, and finally a Stage1-first strategy because Stage1 Search accuracy was not high enough to serve as a strong baseline.

Major durable directives:

- Treat `software` as the hard working directory for this track.
- Use real sub-agent workflow when work is non-trivial and sub-agent execution is available.
- Do not claim sub-agent use unless a real sub-agent call was made.
- Keep `runs/` local/generated and categorize its contents without treating run artifacts as source.
- Stop experiments when requested and document current state before launching more.
- For Stage1, do not judge a branch before at least `50` epochs.
- For accuracy work, avoid repeating experiment families that already failed unless a new mechanism is introduced.

## Reference Analysis Work

XR-Eye-Tracking and PAPER_REF analysis was created and integrated into HGTXR planning.

Important artifacts:

- `anlaysis/xr-eye-tracking/codebases/*/analysis.md`
- `anlaysis/xr-eye-tracking/papers/*/analysis.md`
- `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/experiment_integration.md`
- `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`
- `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md`
- `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.md`

Reference-derived experiment themes:

- FACET/EllSeg-style geometry and ellipse-state supervision.
- EyeLoRiN-style no-retrain or lightweight refinement, blocked by sparse trajectory availability.
- EX-Gaze-style confidence/relocalization, but current confidence signals were not discriminative enough.
- DistillGaze/local-global distillation, useful only after stable teacher targets exist.
- Temporal robustness ideas from 3ET, MambaPupil, TDTracker, AISSM, deferred until current frame/search accuracy and target quality are credible.

## Stage2 / XR Second-Goal Summary

The Stage2/XR track made large improvements from the initial raw event-count baseline.

Initial raw Stage2 baseline:

- Best validation P10 around `7.2653`.
- Final validation center around `43.9917 px`.

The later Stage2/XR work introduced event-count sweeps, center-aware checkpointing, heatmap-state representation, soft-threshold losses, candidate/refine heads, checkpoint soups, teacher-target diagnostics, and targeted failure-bucket experiments.

Key Stage2/XR gates before the Stage1-first pivot:

- Center leader near `16.46 px`.
- P10 leader near `35.02`.
- P5 leader near `12.13`.

Important XR-64 to XR-66 findings:

- XR-63 no-train oracle showed sample-wise teacher routing headroom: center `16.04023192701366`, P10 `36.48596938775512`, P5 `13.41751700680271`.
- XR-64 implemented leakage-safe teacher-target construction from train/val rows and override files.
- XR-64A/B/C improved center only slightly and did not solve P10/P5.
- XR-65 P10/P5 recovery did not promote.
- XR-66 low-similarity targeted recovery improved the low-sim bucket but did not promote full-test gates.

Current interpretation:

- Stage2/XR has useful mechanisms and diagnostics, but downstream tracking is limited by the strength and calibration of the Stage1 frame Search anchor.
- The active priority is therefore Stage1 frame-based Search baseline improvement before returning to Stage2/XR teacher-target work.

## Stage1 Experiment Results

The active Stage1 baseline was promoted from the center-polish self-distill lane:

- It beat the earlier Stage1 baseline on P10, P5, and center.
- It remains the strict baseline for all current Stage1 work.

Completed Stage1 families with no replacement:

- Promoted-polish retry lanes.
- S1-K/S1-L geometry and low-LR self-distill lanes.
- S1-MA/S1-MB mask-assisted lanes.
- S1-MC/S1-MD center-preserving follow-up lanes.
- S1-ME/S1-MF baseline rescue lanes.
- S1-MG/S1-MH Search P10/P5 soft-threshold lanes.
- S1-MI/S1-MJ mask-guard lanes.
- S1-MK/S1-ML P10-seed recovery lanes.
- Residual Search head variants.
- Search bbox/OBB auxiliary variants.
- Depth-8 and mask-cascade probes.
- HeadFactory ROI/eye variants.
- DeiT-Tiny direct preload.
- Optimizer, loss, scheduler, augmentation, and combined guard suites.

Representative no-promotion diagnostic:

- S1-MJ reached P10 `28.482705062290407`.
- It failed promotion because P5 was `9.205975082685363` and center was `17.31833925787008`.
- Decision: S1-MJ is a P10-rich seed, not a baseline.

Core plateau diagnosis:

- Stage1 P10 can be pushed above the current baseline.
- Those P10 gains repeatedly degrade P5 and center.
- The current problem is not lack of P10 signal; it is lack of a mechanism that improves P10 while preserving fine localization.

## Closed Post-Hoc Gate Family

The post-hoc/frozen-output gate family is closed as a primary experiment path.

Important diagnostic:

- Oracle output selection reached P10 `30.908581067930978`, P5 `12.384321968510466`, center `16.39156784201568`.
- This proves checkpoint outputs are complementary at sample level.

But learned frozen-output gates did not convert the oracle headroom into validation improvement:

- Output-only hard/soft gates did not promote.
- Confidence/context softmin gates did not promote.
- Confidence-only gates did not promote.
- Regularized top-k gates did not promote.

Decision:

- Do not launch more frozen-output/post-hoc gate ablations unless a genuinely new mechanism is introduced.
- The next mechanism must be integrated into the trainable model path or target construction path.

## S1-OC Integrated Search Candidate Branch

S1-OC is the current next planned Stage1 mechanism.

Reason:

- It moves sample-wise candidate selection/calibration inside the Stage1 model.
- It avoids the closed frozen-output gate failure mode.
- It uses differentiable softmax mixture routing, while hard selection is kept diagnostic only.

Implemented code surface:

- `src/hbtxr/models/heads.py`
  - Added `SearchCenterCandidateHead`.
  - `PupilSearchHead` supports `legacy`, `residual_mlp`, and `deep_residual_mlp`.
- `src/hbtxr/models/tracker/head_factory.py`
  - Added optional `search_center_candidate_head` wiring.
- `src/hbtxr/models/hybrid_tracker.py`
  - Added Search candidate and residual config passthrough.
- `src/hbtxr/models/tracker/search_branch.py`
  - Emits Search candidate diagnostics and routes soft candidate xy into `search/state` when enabled.
- `src/hbtxr/loss/bundles/search_event.py`
  - Added `pupil_center_candidate_losses`.
- `src/hbtxr/loss/stage1.py`
  - Added candidate loss integration and Search soft-threshold logging.
- `configs/external/base.yaml`
  - Added default-disabled Search residual and Search candidate config keys.
- `scripts/external/run_stage1_s1oc_search_candidate.sh`
  - Added two-lane S1-OC launcher.

S1-OC lanes:

| Lane | GPU | Trainable Scope | LR | Purpose |
|---|---:|---|---:|---|
| S1-OC-A | GPU0 | `search_center_candidate_head.*` | `5e-5` | Candidate head only |
| S1-OC-B | GPU1 | `search_center_candidate_head.*`, `search_head.residual.*` | `2e-5` | Candidate head plus zero-init residual adapter |

Both lanes:

- Epochs: `50`.
- Baseline init: active Stage1 baseline `best_search_p10.pt`.
- Teacher ensemble: baseline P10/P5/center checkpoints.
- Early stopping: disabled.
- Gate metric: `metric_search_p10_pct`.

S1-OC current execution status:

- Implemented: yes.
- Static validation: yes.
- Dry-run command materialization: yes.
- Dummy CPU forward/loss smoke: yes.
- Real 50-epoch GPU training: no.
- Current file-system check found no `*s1oc*` run directory under `runs/NON_XR/raw` and no `*s1oc*` log under `runs/NON_XR/shared/_logs`.

## Validation Already Performed

S1-OC validation completed:

- `python3 -m py_compile` for modified model/loss/training files.
- `bash -n scripts/external/run_stage1_s1oc_search_candidate.sh`.
- `bash -n scripts/external/run_stage1_frame_search_ensemble_teacher.sh`.
- `DRY_RUN=1 RUN_TAG=stage1_s1oc_search_candidate_dryrun bash scripts/external/run_stage1_s1oc_search_candidate.sh`.
- Dummy CPU forward/loss smoke with tiny model:
  - Candidate outputs existed.
  - Candidate loss appeared in Stage1 loss output.
  - `loss_total` was finite.
- `git diff --check` passed on relevant paths during the prior documentation pass.

This documentation pass additionally verified:

- No S1-OC run directories currently exist.
- No S1-OC log files currently exist.
- The S1-OC launcher exists and points to the active baseline.
- Real sub-agents were spawned and closed for read-only synthesis/evaluator tasks.

## Immediate Continuation Runbook

Before launching:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
find runs/NON_XR/raw -maxdepth 1 -type d -name '*s1oc*'
find runs/NON_XR/shared/_logs -maxdepth 1 -type f -name '*s1oc*'
```

Launch S1-OC A/B:

```bash
DRY_RUN=0 DETACH=1 RUN_TAG=stage1_s1oc_search_candidate_$(date +%Y%m%d_%H%M%S) \
  bash scripts/external/run_stage1_s1oc_search_candidate.sh
```

Monitor expected logs:

```bash
tail -f runs/NON_XR/shared/_logs/stage1_frame_search_ensemble_teacher_mo_${RUN_TAG}.log
tail -f runs/NON_XR/shared/_logs/stage1_frame_search_ensemble_teacher_mp_${RUN_TAG}.log
```

Promotion evaluation:

- Wait until both lanes complete at least `50/50` epochs.
- Compare against the active baseline:
  - P10 `> 28.27717937613433`
  - center `<= 17.257583906065744`
  - P5 tracked and preferably close to or above `10.153863744915656`

## If S1-OC Fails

Do not repeat the same post-hoc gate family.

Recommended next directions:

1. Integrated selector with stronger feature supervision, if S1-OC diagnostics show useful candidate diversity.
2. Teacher-target construction for Stage1 Search, analogous to the safer XR-64 target-override logic.
3. Pretrained adaptation phase for DeiT-Tiny or larger teacher models, instead of direct preload.
4. Target-quality and label construction audit if P10 improvements consistently harm P5/center.

## Sub-Agent Use In This Documentation Pass

Task Card T-001:

```yaml
task_id: T-001
sub_agent: gpt5.3-codex-spark
role: analyst
objective: Summarize current HGTXR software progress, experiment plans, and experiment results from existing docs only.
file_ownership: []
assigned_skill: [autosci-exp-status, agent-hierarchy-runtime-manager]
outputs: [progress summary, experiment result summary, next experiment plan summary, handover risks]
validation: [source file evidence]
```

Task Card T-002:

```yaml
task_id: T-002
sub_agent: codex-gpt5.5
role: evaluator
objective: Define what a complete HANDOVER document must contain for the current Stage1 work.
file_ownership: []
assigned_skill: [agent-hierarchy-runtime-manager]
outputs: [handover outline, continuation commands, residual risks, quality gate checklist]
validation: [do not invent unverified experiment execution, separate implemented/validated/executed states]
```

Both agents were read-only. The main agent integrated their outputs and verified the current file-system state directly.

## Risks And Open Items

- S1-OC is not yet proven on real data.
- S1-OC changes final `search/state` when candidate routing is enabled, so metrics measure the integrated branch by design.
- Host GPU access and approval are required to launch real training.
- The worktree contains many prior modified/untracked files; future commits must stage carefully.
- Stage1 Search gates and Stage2/XR gates must remain separate.
- Direct comparison to the paper submission claim remains blocked by protocol/coordinate-frame differences.

## Best-Output Criteria Check

- Completeness: current progress, experiment plans, results, conversation decisions, S1-OC state, and handover path are covered.
- Evidence: claims are tied to existing project docs, run paths, script paths, and direct file-system checks.
- Executability: continuation commands are included.
- Consistency: Stage1 and Stage2/XR gates are separated.
- Safety: no credentials or hidden reasoning are stored.
- Maintainability: docs point to canonical current-state and handover artifacts.
