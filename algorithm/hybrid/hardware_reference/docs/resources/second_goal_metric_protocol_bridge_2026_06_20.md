# HGTXR-SW Metric And Protocol Bridge

Date: 2026-06-20

Experiments are paused by user directive. This document is a verification bridge, not a training result.

## Purpose

The active second goal asks to maximize accuracy toward the `10_submission_initial` paper target. The submission draft reports:

- hybrid pupil-center error: `0.1812 px`
- hybrid P10: `99.97%`
- hybrid P5: `99.72%`
- hybrid P1: `99.61%`
- hybrid latency: `0.43 ms`

The current HGTXR-SW evaluator reports:

- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`
- `metric_track_p1_pct`

This bridge defines what must be true before those two metric spaces can be compared.

## Current Metric Implementation

Current track metrics are computed in `src/hbtxr/loss/metrics.py`:

```text
track_state = resolve_track_state(batch, outputs)
track_error = norm(track_state[:, :2] - batch["cur_state"][:, :2])
metric_track_center_px = weighted mean(track_error, quality * track_geom)
metric_track_p10_pct = hit_rate(track_error <= 10 px)
metric_track_p5_pct = hit_rate(track_error <= 5 px)
```

Therefore, the metric target is `batch["cur_state"][:, :2]`, not an independently confirmed raw EV-Eye sensor-coordinate target.

## Coordinate Sanity Evidence

Command run:

```bash
PYTHONPATH=src .venv/bin/python scripts/external/check_metric_coordinate_sanity.py --manifest data/_internal/manifests/manifest1/val_manifest.jsonl --limit 20 --hit-threshold-px 10.0
```

Observed evidence:

| Field | Value |
|---|---:|
| sample count | `20` |
| metric frame | `post-transform input coordinate frame` |
| directly sensor px | `false` |
| resize policy | `facet_square_direct` |
| target size | `256x256` |
| mean scale x | `1.0756302521008403` |
| mean scale y | `1.7181208053691275` |
| P10 sensor-equivalent x | `9.296875 px` |
| P10 sensor-equivalent y | `5.8203125 px` |

Conclusion:

Current `metric_*_center_px` and P5/P10 are measured after ROI/sensor spatial transform, not in raw EV-Eye sensor pixels. Paper-level pixel claims must be compared only after matching the same coordinate frame or inverse-transforming predictions.

## Bridge Status

Current status: `blocked_until_metric_frame_and_protocol_match`.

Direct submission comparison allowed: `false`.

Paper-target promotion claim allowed: `false`.

## Protocol Contract

The submission draft says it follows an EX-Gaze-style split protocol, but current software completion must still use the concrete HGTXR manifest evidence:

| Item | Current value |
|---|---:|
| manifest root | `data/_internal/manifests/manifest1` |
| train rows | `5929` |
| val rows | `844` |
| test rows | `2238` |
| test-derived training targets allowed | `false` |
| XR-64 train/val eval rows required | `8` |
| XR-64 train/val override JSON required | `6` |
| XR-64 ready to train | `false` |

This means the bridge cannot be unblocked by paper claims alone. It needs concrete current-manifest and evaluator evidence.

## Hybrid Protocol Match

The submission table reports a scheduler-driven hybrid mode. The current active software metrics are still primarily:

- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`

Current gaps:

- `metric_track_p1_pct` is now implemented in the evaluator, but paper-frame/full-test P1 evidence is not yet available.
- full hybrid scheduler evaluation rows are not currently available.
- search/track selected-mode provenance is not currently part of the active accuracy gate.

Hybrid match status: `not_proven`.

Reasons:

- `metric_track_center_px` is not yet proven to use the same pixel frame as the paper's `0.1812 px`.
- P10/P5 thresholds are in transformed input coordinates and can map anisotropically to sensor-space pixels.
- Current active software queue does not yet prove the full hybrid scheduler policy used by the submission table.
- Submission P1 is not yet available as paper-frame full-test evidence.
- Hardware latency must remain separate from software accuracy promotion.

## Required Evidence To Unblock

Before claiming paper-level accuracy closure, produce:

1. Prediction rows with raw sensor-space center and transformed center for the same samples.
2. An evaluator path that either inverse-transforms track predictions to sensor pixels or replays the paper evaluator in the same coordinate frame.
3. EV-Eye split manifest counts and provenance matched to the EX-Gaze-following split claim.
4. Hybrid scheduler evaluation rows containing search/track mode, chosen state, P10/P5/P1, and center error for the same target definition.
5. A documented rule for whether software promotion uses transformed-input gates or paper-frame sensor-pixel gates.

## Current Operational Rule

Until this bridge is unblocked:

- use current center/P10/P5 gates for software promotion.
- keep XR-64-prep, XR-64A, and XR-64B as P0.
- do not claim progress toward `0.1812 px` from `metric_track_center_px` alone.
- require this bridge and `scripts/external/check_metric_protocol_bridge.py` before any final accuracy-closure claim.
