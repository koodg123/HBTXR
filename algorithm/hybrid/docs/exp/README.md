# Hybrid Experiment Import Candidates

This note curates frame-event hybrid experiment results that are worth carrying
into the hybrid algorithm track.

## Source Scope

- `external_hybrid_package/docs/exps/*`
- `external_hybrid_package/exps/configs/*`
- `hgtxr_software/docs/resources/xr*.md`
- `hgtxr_software/docs/track/RAW_EVENT_COUNT_TRAINING.md`

## P0 Candidates

### Stage2 objective stability and failure modes

Source:

- `external_hybrid_package/docs/exps/20260401_042845_13_mode0_stage2_experiment_results_analysis.md`

Useful result:

- DataParallel crash was fixed.
- Zeroing `search`, `event`, and `track_geo` weights to `0` was necessary for
  numerically sane Stage2 in the tested setup.
- Geo-on multigpu run was stable at infrastructure level but geometry loss
  dominated:
  validation `loss_total = 2795.529`,
  search P10 `85.6469`,
  event P10 `3.0267`,
  track P10 `13.35018`.
- Geo-off sanity runs trained stably, but track quality did not improve; best
  track P10 appeared at epoch `1` in the longer runs.
- Main conclusion: stable optimization did not translate into better tracking,
  and `loss_total` was a misleading success signal.

Recommended action:

- Keep Stage2 as a gated hybrid objective, not a direct long-run recipe.
- Add promotion gates on track center/P10/P5, not only loss.
- Require checkpoint-level comparison against epoch-1 and teacher-initialized
  baselines.

Risk:

- Stage2 can look stable while degrading the actual tracking objective.

### Stage1 mask-guided teacher for hybrid initialization

Source:

- `external_hybrid_package/docs/exps/20260402_010646_15_mode0_stage1_mask_guidance_ablation_analysis.md`

Useful result:

- Hard mask-centroid and soft mask-cascade variants are strong center solvers.
- Soft center consistency is the safest regularized improvement.

Recommended action:

- Use hard mask-centroid as a teacher/upper-bound path for Stage1 search.
- Prefer soft center consistency for a deployable teacher candidate.
- Do not pass mask-guided results into Stage2 unless transform policy, mask
  quality, and checkpoint selection are fixed in the manifest.

### HGTXR second-goal experiment sequence

Sources:

- `hgtxr_software/anlaysis/xr-eye-tracking/experiment_integration.md`
- `hgtxr_software/anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`
- `hgtxr_software/docs/resources/second_goal_*`
- `hgtxr_software/docs/resources/xr64_*`
- `hgtxr_software/docs/resources/xr65_p10p5_recovery_results_2026_06_22.md`
- `hgtxr_software/docs/resources/xr66_lowsim_targeted_recovery_results_2026_06_22.md`

Useful result:

- Current gates in the HGTXR software line were:
  center `<16.468481131962367`,
  P10 `>35.02295998845781`,
  P5 `>12.133503770828247`.
- XR-64 teacher-target construction became the P0 mechanism after earlier
  optimizer/count/geometry experiments plateaued.
- XR-65 preserved center but failed P10/P5 promotion:
  best P10 `34.31505176680429`, best P5 `12.08248336655753`.
- XR-66 low-similarity hard-subset training improved targeted bucket metrics
  but did not transfer to full-test promotion.

Recommended action:

- Import the experiment governance:
  explicit gates, train/val-only target overrides, full-test evaluation without
  target override, and no-promotion closure rules.
- Use XR-64 style leakage-safe teacher-target construction as the next serious
  hybrid experiment family, not blind Stage2 reruns.
- Treat XR-65/XR-66 as negative/diagnostic evidence.

Risk:

- HGTXR software results and external hybrid package results may use different
  manifests, coordinate frames, and metric protocols. Bridge them before direct
  ranking.

## P1 Candidates

### Raw event-count hybrid contract

Source:

- `hgtxr_software/docs/track/RAW_EVENT_COUNT_TRAINING.md`

Recommended action:

- Keep as a reproducibility contract for Stage1/Stage2 raw-event hybrid runs.
- Use as a smoke/full-run validation template when moving from external package
  code into the reorganized `algorithm/hybrid` code.

### Bbox cascade as hybrid auxiliary

Source:

- `external_hybrid_package/docs/exps/20260408_105839_17_mode0_stage1_expE_bbox_head_experiments.md`

Recommended action:

- Keep dense-eye guided bbox cascade as an auxiliary localization branch only.
- Do not replace mask/state outputs with bbox outputs until pupil bbox IoU is
  materially improved.

## Do Not Import As Active Hybrid Recipes

- Any Stage2 run whose best tracking metric occurs only at epoch `1` unless it
  is explicitly used as an initialization baseline.
- Geometry-on Stage2 recipes that let geometry loss dominate total loss.
- Hard low-similarity subset training as the primary recovery method.
- HGTXR paper-target numbers as direct promotion gates without metric bridge.
