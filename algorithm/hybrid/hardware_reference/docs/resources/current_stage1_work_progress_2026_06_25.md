# Current Stage1 Work Progress - 2026-06-25

## Summary

This document records the current HGTXR software Stage1 frame-based Search work state as of 2026-06-25 KST.

Current decision:

- Keep the active Stage1 frame-search baseline unchanged.
- Stop repeating frozen-output/post-hoc gate ablations.
- Move the next meaningful Stage1 experiment to `S1-OC`, an integrated end-to-end Search candidate branch inside the Stage1 model.
- `S1-OC` code and launch script are implemented and smoke-validated, but the 50-epoch GPU run has not started because the escalated launch approval was aborted.

## Active Baseline

Active Stage1 frame-search baseline:

- Run: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949`
- Checkpoint: `runs/NON_XR/raw/stage1_frame_search_centerpolish_selfdistill_lr7p5e6_xy2_ema999_stage1_centerpolish_after_s1ef_20260622_0128_20260622_021949/train/best_search_p10.pt`
- Validation Search P10: `28.27717937613433`
- Validation Search P5: `10.153863744915656`
- Validation Search center: `17.257583906065744`
- Gate: promoted follow-up baseline, minimum epoch requirement satisfied.

Promotion rule remains strict:

- P10 must exceed `28.27717937613433`.
- Center must be `<= 17.257583906065744`.
- P5 must be tracked and should recover toward `10.153863744915656`.
- Any replacement candidate must complete at least `50` epochs before promotion.

## Completed Stage1 Experiment Findings

The main Stage1 conclusion is stable across the completed experiments:

- P10 can be raised slightly, but the improvement usually comes with worse P5 and worse center.
- P5/center-preserving branches tend to tie or underperform P10.
- The strict baseline has not been replaced.

Important positive diagnostic:

- S1-MJ reached P10 `28.482705062290407`.
- S1-MJ failed promotion because P5 `9.205975082685363` and center `17.31833925787008` were worse than the baseline.
- Interpretation: S1-MJ is a P10-rich seed, not a baseline.

Important no-promotion groups:

- Promoted-polish retry lanes: no promotion.
- S1-K/S1-L geometry and low-LR self-distill lanes: no promotion.
- S1-MA/S1-MB mask-assisted lanes: no promotion.
- S1-MC/S1-MD center-preserving follow-up lanes: no promotion.
- S1-ME/S1-MF baseline rescue lanes: no promotion.
- S1-MG/S1-MH Search P10/P5 soft-threshold lanes: no promotion.
- S1-MI/S1-MJ mask-guard lanes: no promotion, but S1-MJ is a useful P10 diagnostic.
- S1-MK/S1-ML P10-seed recovery lanes: no promotion.
- Residual Search head / search bbox aux / search OBB aux: no promotion.
- Depth-8 and mask-cascade probes: no promotion.
- HeadFactory ROI/eye variants: no promotion.
- DeiT-Tiny direct preload: no promotion; current target needs a dedicated adaptation phase.
- Optimizer, loss, scheduler, augmentation, and combined guard suites: no promotion.

Representative completed results:

| Group | Best/Notable Result | Decision |
|---|---:|---|
| S1-MJ P10 seed | P10 `28.482705062290407`, center `17.31833925787008` | P10 diagnostic only |
| S1-NF Adopt/step | P10 about `28.35018`, center worse | no promotion |
| S1-NA/NB HeadFactory ROI | P10 about `28.32772`, center/P5 worse | no promotion |
| S1-MY depth-8 | P10 about `28.31087`, center/P5 worse | no promotion |
| S1-OA combined guard | P10 tie, P5/center worse | no promotion |

## Post-Hoc Gate Family Closeout

The output-level gate experiments had diagnostic value but are now closed as a low-return path.

Static/output ensemble diagnostics:

- Static weighted ensembles did not exceed the validation P10 gate.
- Oracle output selection reached:
  - P10 `30.908581067930978`
  - P5 `12.384321968510466`
  - center `16.39156784201568`
- Interpretation: checkpoint outputs are complementary at sample level.

Learned gate experiments:

- `scripts/external/train_stage1_search_output_gate.py` was implemented for frozen-output gate training.
- It audits train/val/test manifest overlap and reports both batch-mean and global-weighted metrics.
- Train/val/test overlap audit found `0` overlaps in the full gate run.

Closed gate variants:

- Output-only gate: no promotion.
- Confidence/context softmin gate: no promotion.
- Confidence-only softmin and oracle-CE gates: no promotion.
- Regularized top-k gate with entropy/balance terms: no promotion.

Representative validation results:

| Gate Variant | Validation P10 | Validation P5 | Validation Center | Decision |
|---|---:|---:|---:|---|
| output-only hard | `27.83692722371966` | `10.222371967654992` | `17.400613902092832` | no promotion |
| output-only soft | `28.039083557951468` | `9.67430368373765` | `17.31651116715608` | no promotion |
| output_conf_context softmin soft | `28.18171608265946` | `8.876909254267746` | `17.321360889128062` | no promotion |
| output_conf softmin soft | `28.06379155435757` | `8.876909254267746` | `17.320374500826468` | no promotion |
| output_conf_context topk2 top-k | `27.85040431266845` | `9.168912848158133` | `17.362701884669868` | no promotion |

Decision:

- The oracle upper bound is real.
- Frozen-output MLP gates did not generalize the oracle headroom on validation.
- Do not launch more post-hoc gate ablations unless a new mechanism is introduced.

## S1-OC Integrated Candidate Branch

`S1-OC` is the next meaningful Stage1 mechanism because it moves selection/calibration inside the trainable Stage1 model instead of training a frozen-output post-hoc gate.

Implemented code surface:

- `src/hbtxr/models/heads.py`
  - Added `SearchCenterCandidateHead`, reusing the zero-initialized candidate-delta pattern from the track candidate head.
  - Existing `PupilSearchHead` already supports `legacy`, `residual_mlp`, and `deep_residual_mlp`.
- `src/hbtxr/models/tracker/head_factory.py`
  - Added optional `search_center_candidate_head` module wiring.
- `src/hbtxr/models/hybrid_tracker.py`
  - Added config passthrough for Search candidate head count, max delta, routing, and blend.
- `src/hbtxr/models/tracker/search_branch.py`
  - Emits `search/center_candidate_delta`, `search/center_candidate_logits`, `search/center_candidate_xy`, `search/center_candidate_weights`, `search/center_candidate_soft_xy`, and hard diagnostic selected xy.
  - Uses softmax mixture for trainable routing into `search/state`; hard argmax is diagnostic only.
- `src/hbtxr/loss/bundles/search_event.py`
  - Added `pupil_center_candidate_losses`.
- `src/hbtxr/loss/stage1.py`
  - Added Stage1 Search candidate loss integration.
  - Added Search P10/P5 soft-threshold loss support.
- `configs/external/base.yaml`
  - Added default-disabled config keys for Search residual and Search candidate branch.
- `scripts/external/run_stage1_s1oc_search_candidate.sh`
  - Added two-lane S1-OC launcher.

S1-OC lanes:

| Lane | GPU | Trainable Scope | LR | Purpose |
|---|---:|---|---:|---|
| S1-OC-A | GPU0 | `search_center_candidate_head.*` | `5e-5` | Test candidate head only |
| S1-OC-B | GPU1 | `search_center_candidate_head.*`, `search_head.residual.*` | `2e-5` | Test candidate head plus zero-init residual adapter |

Both lanes:

- Epochs: `50`.
- Baseline init: active Stage1 baseline `best_search_p10.pt`.
- Teacher ensemble: baseline P10/P5/center checkpoints.
- Early stopping: disabled.
- Gate metric: `metric_search_p10_pct`.
- Promotion requires P10 improvement and center preservation.

Current S1-OC execution status:

- Dry-run command materialization passed.
- Dummy forward/loss smoke passed.
- No S1-OC run directory exists under `runs/NON_XR/raw`.
- No S1-OC log exists under `runs/NON_XR/shared/_logs`.
- The 50-epoch GPU launch was attempted via escalated execution but the approval was aborted, so no S1-OC training has started.

## Validation Performed

Completed validation:

- Python compile:
  - `src/hbtxr/models/heads.py`
  - `src/hbtxr/models/tracker/head_factory.py`
  - `src/hbtxr/models/tracker/search_branch.py`
  - `src/hbtxr/models/hybrid_tracker.py`
  - `src/hbtxr/training/model_factory.py`
  - `src/hbtxr/loss/stage1.py`
  - `src/hbtxr/loss/bundles/search_event.py`
  - `src/hbtxr/loss/bundles/__init__.py`
- Shell syntax:
  - `scripts/external/run_stage1_s1oc_search_candidate.sh`
  - `scripts/external/run_stage1_frame_search_ensemble_teacher.sh`
- Dry-run:
  - `DRY_RUN=1 RUN_TAG=stage1_s1oc_search_candidate_dryrun bash scripts/external/run_stage1_s1oc_search_candidate.sh`
- Dummy CPU forward/loss smoke:
  - Built a tiny Stage1 model with `search_center_candidate=true`.
  - Verified candidate outputs exist.
  - Verified `loss_search_center_candidate_p10_bce` is included.
  - Verified finite `loss_total`.

Remaining validation:

- Add focused pytest coverage for Search candidate branch.
- Run a short real-data smoke if GPU launch is approved.
- Run full S1-OC A/B for at least `50` epochs.
- Evaluate promotion against the strict baseline gate.

## Current Risks

- S1-OC is implemented but not yet proven on real data.
- Because S1-OC routes candidate xy into final `search/state`, metrics and Search losses now evaluate the integrated candidate branch when the feature is enabled. This is intended, but it means the 50-epoch gate is mandatory before any conclusion.
- The current worktree contains many unrelated modified/untracked files from earlier software and hardware work. Future commits should stage carefully.
- Direct DeiT preload was poor in previous runs; any future large-model/pretrained route likely needs a separate adaptation phase, not direct substitution.

## Next Actions

1. Add focused tests for S1-OC candidate forward/loss behavior.
2. Launch S1-OC A/B with host GPU access:

```bash
DRY_RUN=0 DETACH=1 RUN_TAG=stage1_s1oc_search_candidate_$(date +%Y%m%d_%H%M%S) \
  bash scripts/external/run_stage1_s1oc_search_candidate.sh
```

3. Monitor until both lanes reach `50/50`.
4. Promote only if a candidate beats P10 `28.27717937613433` and keeps center `<= 17.257583906065744`.
5. If S1-OC fails, shift Stage1 work toward target construction or teacher-target methods rather than more frozen-output gate variants.
