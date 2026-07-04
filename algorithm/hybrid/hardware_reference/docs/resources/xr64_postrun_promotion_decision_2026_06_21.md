# XR-64 Post-Run Promotion Decision

Date: 2026-06-21

This is the current promotion-decision placeholder for future XR-64A/B post-run results. It does not run train/eval jobs.

## Current Decision

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| decision status | `missing_postrun_evidence` |
| software promotion allowed | `false` |
| single-model SOTA claim allowed | `false` |
| paper-level completion allowed | `false` |
| candidate eval summaries | `0` |
| P1 evidence candidates | `0` |
| axis certification required | `true` |
| axis-certified candidates | `0` |
| axis claims allowed | `false` |
| bounded tradeoff candidates | `0` |
| unbounded tradeoff candidates | `0` |
| control variables checked | `false` |
| rejection reason | `missing_postrun_evidence` |

## Gates

| Metric | Rule |
|---|---|
| center | `metric_track_center_px < 16.468481131962367` |
| P10 | `metric_track_p10_pct > 35.02295998845781` |
| P5 | `metric_track_p5_pct > 12.133503770828247` |

Future paper-target bridge evidence should also include `metric_track_p1_pct` in candidate eval summaries. The helper tracks this as evidence, but it does not use P1 as a current software-promotion gate.

## Bounded Tradeoff Scorecard

Future candidates are now classified before promotion:

| Field | Rule |
|---|---|
| center severe regression | delta below `-0.3 px` |
| P10 severe regression | delta below `-1.0 percentage point` |
| P5 severe regression | delta below `-1.0 percentage point` |
| `clean_multi_gate_improvement` | at least two gates improve and no severe regression |
| `single_gate_bounded_tradeoff` | one gate improves and non-winning gates stay within floors |
| `diagnostic_unbounded_tradeoff` | a gate improves but at least one non-winning gate has severe regression |
| `no_gate_improvement` | no active software gate improves |

Only candidates with bounded tradeoff can become the winning software-promotion candidate. Unbounded tradeoff remains diagnostic and must route to follow-up analysis rather than promotion.

Future candidate eval summaries must also include ablation proof fields:

- `ablation_axis`: must cover `head`, `loss`, `lr`, and `teacher_model_training`.
- `ablation_changed_keys`: object keyed by `head`, `loss`, `lr`, and `teacher_model_training`, each with a non-empty key list.
- `ablation_benchmark_baseline_id`: `XR-64-prep`.
- `axis_certified`: `true`.

Future candidate eval summaries must also prove that final test evaluation was run with target overrides cleared:

- `data_track_target_override_path`: JSON `null` only. This is the flat eval-summary field for config path `data.track_target_override_path`.
- `data_allow_test_target_override`: JSON boolean `false` only. Numeric `0` or string `"false"` is invalid.
- `loss_track_target_override_center_l2_weight`: JSON numeric `0.0` only. String `"0.0"` or boolean `false` is invalid.
- `loss_track_state_aux_target_override_center_l2_weight`: JSON numeric `0.0` only. String `"0.0"` or boolean `false` is invalid.

The decision helper intentionally requires the flat eval-summary fields above. Dotted config-path keys alone, for example `data.track_target_override_path`, are documentation/config references and do not satisfy the future promotion schema.

The artifact checker `scripts/external/check_xr64_postrun_promotion_decision.py` validates this current no-evidence placeholder against the decision helper's no-candidate output. It prevents drift in the active gates, six-candidate XR-64A/B matrix, override-clear schema, bounded-tradeoff floors, and blocked promotion flags.

## Current Blockers

- Experiments are paused by user directive.
- XR-64A/B training runs do not exist.
- XR-64A/B post-run test eval summaries do not exist.
- No trained XR-64 checkpoint has improved current software gates.
- Paper-level completion remains blocked by metric/protocol bridge.

## Future Decision Command

When XR-64A/B test eval summaries exist, run the read-only decision helper with one candidate per checkpoint-kind eval summary.
The example below shows the full expected center/P10/P5 checkpoint-kind coverage for both lanes; fewer candidates may be useful for debugging, but they are not enough to satisfy the post-run evidence contract.

```bash
.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py \
  --candidate XR-64A:best_metric_track_center_px:path/to/xr64a/best_metric_track_center_px/eval_summary.json \
  --candidate XR-64A:best_track_p10:path/to/eval_summary.json \
  --candidate XR-64A:best_track_p5:path/to/xr64a/best_track_p5/eval_summary.json \
  --candidate XR-64B:best_metric_track_center_px:path/to/xr64b/best_metric_track_center_px/eval_summary.json \
  --candidate XR-64B:best_track_p10:path/to/xr64b/best_track_p10/eval_summary.json \
  --candidate XR-64B:best_track_p5:path/to/eval_summary.json \
  --format summary
```

Machine-readable source: `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json`.

Artifact validation:

```bash
.venv/bin/python scripts/external/check_xr64_postrun_promotion_decision.py --format summary
.venv/bin/python -m pytest -q tests/test_xr64_postrun_promotion_decision.py
```
