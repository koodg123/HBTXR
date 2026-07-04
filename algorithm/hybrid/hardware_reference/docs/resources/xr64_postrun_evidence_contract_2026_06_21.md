# XR-64 Post-Run Evidence Contract

Date: 2026-06-21

This contract defines what evidence must exist after XR-64A/B run before any promotion or completion claim. It does not run train/eval jobs.

## Current State

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| evidence state | `missing_postrun_evidence` |
| promotion claim allowed | `false` |
| second-goal completion allowed | `false` |
| XR-64A training evidence | `false` |
| XR-64B training evidence | `false` |
| XR-64A eval summaries | `0` |
| XR-64B eval summaries | `0` |

## Required Lanes

| Lane | Role | Device | Required |
|---|---|---|---:|
| `XR-64A` | center-safe conservative teacher-target branch | `cuda:0` | `true` |
| `XR-64B` | threshold-priority P10/P5 recovery branch | `cuda:1` | `true` |

Optional:

- `XR-64C`: min-error diagnostic branch only after XR-64A/B evidence shows useful but underfit teacher-target pressure.

## Required Per-Lane Evidence

Each required lane must provide:

- run root.
- `hypers/resolved_config.json`.
- `train/ablation_lr_manifest.json`.
- `train/optimizer_config_snapshot.json`.
- `train/teacher_provenance.json`.
- `train/ablation_attribution_report.json`.
- `train/best_track_p10.pt`.
- `train/best_track_p5.pt`.
- `train/best_metric_track_center_px.pt`.
- test `eval_summary.json` for each required checkpoint kind.
- test `eval_rows.json` for each required checkpoint kind.
- runner log with `XR64_TRAIN_EXIT` and `XR64_EVAL_EXIT` markers.

Required checkpoint kinds:

- `best_track_p10`
- `best_track_p5`
- `best_metric_track_center_px`

## Ablation Evidence Contract

XR-64A/B post-run evidence must be able to support the requested second-goal experiment axes, not only a raw promotion decision.

Required axes:

- `head`
- `loss`
- `lr`
- `teacher_model_training`

Control variables:

- same manifest1 train/val/test split protocol.
- same current software gates.
- test target override path cleared.
- XR-64-prep generated artifacts fixed before XR-64A/B comparison.

Lane-to-axis mapping:

| Lane | Head | Loss | LR | Teacher |
|---|---|---|---|---|
| `XR-64A` | heatmap/state heads plus teacher-target override supervision | conservative override center/P10/P5/state-distill losses | low-LR conservative continuation | XR-62A center init with XR-39 teacher-target transfer |
| `XR-64B` | heatmap/state heads plus threshold-priority teacher-target override supervision | threshold-priority override center/P10/P5/state-distill losses | low-LR threshold-priority continuation | XR-56B P5 init with XR-39 teacher-target transfer |

Each lane must declare:

- `baseline_reference=XR-64-prep`.
- `ablation_design.axis_tested`: `head`, `loss`, `lr`, and `teacher_model_training`.
- `ablation_design.varied`: lane-specific loss/init/LR policy changes.
- `ablation_design.held_fixed`: manifest1 split protocol, XR-64-prep generated artifacts, and test evaluator/gates.
- `axis_control_key_values`: concrete config key paths for head, loss, LR, and teacher model training.
- `teacher_provenance_required`: teacher checkpoint ID, checksum, source manifest, and override policy.

Future post-run decisions must include `axis_claims`, `control_variables_checked`, `gate_tradeoff_table`, and `rejection_reason`.

Required metrics:

- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`
- `metric_track_p1_pct`

`metric_track_p1_pct` is required as future paper-target bridge evidence. It is not a current software-promotion gate until coordinate-frame, split, and hybrid-scheduler protocol matching are proven.

## Test Eval Contract

Final test evaluation must clear target override paths:

| Override | Required value |
|---|---|
| `data.track_target_override_path` | `null` |
| `data.allow_test_target_override` | `false` |
| `loss.track_target_override_center_l2_weight` | `0.0` |
| `loss.track_state_aux_target_override_center_l2_weight` | `0.0` |

## Promotion Rules

Current software gates:

| Metric | Promotion rule |
|---|---|
| center | `metric_track_center_px < 16.468481131962367` |
| P10 | `metric_track_p10_pct > 35.02295998845781` |
| P5 | `metric_track_p5_pct > 12.133503770828247` |

Promotion requires a trained XR-64 checkpoint, test eval summary, test eval rows, override-cleared test evaluation, run/config provenance, and documented tradeoff against all current gates.

Reject promotion if:

- only command/prelaunch artifacts exist.
- only train/val override artifacts exist.
- XR-63 oracle is used as promotion evidence.
- test override path is non-null.
- center/P10/P5 tradeoff is undocumented.
- metric/protocol bridge is still blocked but paper-level completion is claimed.

## Validation

```bash
python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py
.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary
.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_xr64_postrun_evidence_contract.py
```

Machine-readable source: `docs/resources/xr64_postrun_evidence_contract_2026_06_21.json`.
