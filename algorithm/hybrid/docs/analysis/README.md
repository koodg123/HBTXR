# Hybrid Analysis Import Candidates

This note captures analysis material that should guide the hybrid frame-event
algorithm track.

## Source Scope

- `external_hybrid_package/docs/prj/*`
- `external_hybrid_package/docs/exps/*`
- `hgtxr_software/anlaysis/xr-eye-tracking/*`
- `hgtxr_software/anlaysis/paper-ref/*`
- `hgtxr_software/docs/resources/*`

## P0 Analysis To Keep

### Hybrid search/track contract

Source:

- `hgtxr_software/anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`

Useful contract:

- Hybrid event-frame search/track split.
- Shared geometric pupil state `(x, y, a, b, theta)`.
- Frame-guided search anchor refresh.
- Event-guided local residual track update.
- Time-synchronous geometry-stable supervision.
- Modality-aware backbone with decoupled search and track heads.
- Stage1 frame teacher, Stage2 hybrid learning, Stage3 distillation.

Recommended action:

- Use this as the hybrid architecture contract.
- Treat current software experiments as accuracy-recovery work toward this
  contract, not as final proof of paper-level performance.

### Experiment decision logic

Sources:

- `hgtxr_software/anlaysis/xr-eye-tracking/experiment_integration.md`
- `hgtxr_software/docs/resources/second_goal_*`

Useful logic:

- Keep center, P10, P5, and jitter-like stability as separate leaderboards.
- Use explicit acceptance gates before changing crop, sampling, representation,
  or post-processing policy.
- Defer sparse/quantized hardware paths until a full-width teacher is stable.
- Do not compare paper pixel errors directly unless coordinate frame and ROI
  scale match.

Recommended action:

- Build the hybrid experiment registry around gates and negative evidence, not
  only run names.
- Add a metric bridge before mixing external package results with HGTXR software
  results.

### Stage2 failure interpretation

Source:

- `external_hybrid_package/docs/exps/20260401_042845_13_mode0_stage2_experiment_results_analysis.md`

Useful conclusion:

- Stage2 training can be numerically stable while failing the track objective.
- Track metrics and checkpoint selection must be first-class outputs.

Recommended action:

- Require all hybrid Stage2 analyses to report:
  best search metrics, event metrics, track metrics, best epoch per metric,
  final metrics, and whether best track came from epoch `1`.

## P1 Analysis To Keep

### Reference-code family map

Source:

- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`

Useful mapping:

- Geometry-first online trackers provide scheduler/fallback ideas.
- Dense event-frame neural trackers provide event encoder and temporal model
  ideas.
- Segmentation teachers provide label and auxiliary supervision ideas.
- Hardware-facing sparse systems provide later deployment constraints.

Recommended action:

- Reimplement ideas in the current PyTorch/HBTXR package rather than directly
  porting old TensorFlow or paper-reference code.

### Paper-to-experiment mapping

Source:

- `hgtxr_software/anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- `hgtxr_software/anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`

Useful mapping:

- FACET/EllSeg/E-Track style geometry supports ellipse-state auxiliary losses.
- 3ET/TDTracker/Mamba-like temporal models support temporal-lite adapters only
  after dense trajectory or stronger teacher-target evidence exists.
- EX-Gaze/Swift-Eye/E-BTS support local update and blink/failure buckets, but
  prior blind local-crop/fallback attempts were harmful.
- Distillation and sparse/quantized deployment should be Stage3 or later.

Recommended action:

- Keep temporal/local/distillation branches conditional on teacher quality and
  failure-bucket diagnostics.

## P2 Analysis To Keep

- Runs catalog and cleanup resources:
  useful for provenance and artifact hygiene.
- Paper-target bridge resources:
  useful when preparing publication-style comparisons, not as immediate
  training commands.
- Hardware/sparse references:
  useful after software accuracy stabilizes and quantization work resumes.

## Import Boundary

Do import:

- Metric gates, result synthesis, and no-promotion decisions.
- Dataset/manifest contracts.
- Stable transform/event-window semantics.
- Negative ablations that prevent repeated failed work.

Do not import blindly:

- Historical immediate-priority queues that were superseded.
- Paper-reported absolute numbers as direct local gates.
- Source-specific paths, environment repair commands, or run names as package
  APIs.
