# HGTXR-SW Current Experiment Result Synthesis

Date: 2026-06-20

Experiments are paused by user directive. This document summarizes validated results only; it is not a new train/eval result.

## Current Best Gates

| Metric | Best validated value | Direction | Owner | Interpretation |
|---|---:|---|---|---|
| `metric_track_center_px` | `16.468481131962367` | lower is better | XR-62A FACET geometry auxiliary refresh | best center gate; useful as geometry/center guard |
| `metric_track_p10_pct` | `35.02295998845781` | higher is better | XR-39 mixed-leader soup `c25p45f30` | best P10 gate; still the hardest active threshold gate |
| `metric_track_p5_pct` | `12.133503770828247` | higher is better | XR-56B best-P5 | best P5 gate; strong but fragile under P10-focused branches |

These are not a single-model claim. The best center, P10, and P5 values come from different experiment owners.

## XR-63 Oracle Diagnostic

XR-63 reports:

| Metric | Oracle value |
|---|---:|
| `metric_track_center_px` | `16.04023192701366` |
| `metric_track_p10_pct` | `36.48596938775512` |
| `metric_track_p5_pct` | `13.41751700680271` |

This is useful because it shows complementary sample-wise predictions across the current leader set.
It is not promotable as a trained result because it is a no-train, test-aligned teacher-target oracle.

Operational implication:

- XR-64 is the leakage-safe route to test whether this oracle signal transfers through train/val-only target construction.

## Submission Target Relation

The submission draft target is:

| Target | Value |
|---|---:|
| hybrid pixel error | `0.1812 px` |
| hybrid P10 | `99.97%` |
| hybrid P5 | `99.72%` |
| hybrid P1 | `99.61%` |
| hybrid latency | `0.43 ms` |

Direct comparison is currently invalid.

Reasons:

- Current metrics are in the `post-transform input coordinate frame`.
- Current P10/P5 thresholds are not proven to be raw sensor-pixel thresholds.
- Current active software gates are track-centered, not proven full hybrid scheduler results.
- `metric_track_p1_pct` is now implemented, but paper-frame/full-test P1 evidence is not part of the current active gate evidence.
- Hardware latency must remain separate from software accuracy promotion.

Until the metric/protocol bridge is unblocked, software promotion uses the current center/P10/P5 gates only.

## XR-64 Resume Gate

Current XR-64 state:

| Item | State |
|---|---:|
| status | `generated_incomplete` |
| ready to train | `false` |
| can run lane | `false` |
| missing train/val teacher eval rows | `8` |
| missing train/val override JSON files | `6` |
| leakage risk | `none` |

Next sequence after the user explicitly resumes experiments:

1. Run `XR-64-prep`.
2. Generate all train/val teacher eval rows.
3. Generate all train/val override JSON files.
4. Run the strict XR-64 checker.
5. Launch XR-64A on GPU0 and XR-64B on GPU1 only after strict readiness passes.
6. Use XR-64C only if A/B evidence shows useful but underfit teacher-target pressure.

## Closed Or Low-Return Standalone Axes

These are retained as history, not current P0 work:

- geometry-only replay after XR-62.
- candidate-head-only continuation after XR-60/XR-61.
- scalar P10-soft continuation after XR-56/XR-58/XR-59.
- low-similarity weighting/subset/sampler after XR-04/XR-09/XR-10.
- checkpoint soup/interpolation as primary work.
- prev-pupil/local crop without confidence fallback.

## Conclusion

The current experiment set has identified credible accuracy headroom, but the second goal remains incomplete.
No single trained model owns all gates, XR-63 is diagnostic only, XR-64 generated artifacts are missing, and paper-level metric/protocol comparison is still blocked.

Machine-readable source: `docs/resources/second_goal_current_result_synthesis_2026_06_20.json`.
