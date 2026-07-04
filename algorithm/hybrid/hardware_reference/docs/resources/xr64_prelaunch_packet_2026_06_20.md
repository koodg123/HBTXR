# XR-64 Prelaunch Packet

Date: 2026-06-20

This packet is a future-only resume artifact. It does not run train/eval jobs.

## Current State

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| allowed to execute | `false` |
| completion allowed | `false` |
| completion blockers | `28` |
| XR-64 status | `generated_incomplete` |
| XR-64 ready to train | `false` |
| missing eval rows | `8` |
| missing override JSON files | `6` |
| eval static inputs ready | `true` (`6 / 6`) |
| build inputs ready | `false` (`0 / 8`) |
| post-run review commands | `2` |
| resume matrix axes | `6` |
| resume matrix gates | `5` |
| next P0 lanes | `XR-64-prep`, `XR-64A`, `XR-64B` |
| XR-64 prep ready-after-resume units | `8` |
| XR-64 prep blocked units | `2` |
| blocker map blockers | `28` |
| paper-target preflight evidence count | `7` |
| paper-target preflight output schemas | `10` |
| leakage risk | `none` |
| metric/protocol bridge | `blocked_until_metric_frame_and_protocol_match` |
| paper-target evidence manifest | `missing_required_evidence` |

## Strict Preconditions

1. The user explicitly resumes experiments.
2. XR-64-prep generates all `8` train/val teacher `eval_rows.json` files.
3. XR-64-prep generates all `6` train/val override JSON files.
4. `scripts/external/check_xr64_resume_artifacts.py --format summary` reports `ready_to_train=true`, `can_run_lane=true`, `missing_eval_rows=0`, `missing_overrides=0`, and `leakage_risk=none`.
5. No test split target override is generated or used for training.
6. Final test evaluation for promoted checkpoints clears target override paths.
7. XR-64A/B launch is allowed only after strict readiness passes.
8. Post-run candidate review is allowed only after XR-64A/B test eval summaries include ablation axis certification.
9. The second-goal completion gate must pass before any final completion claim.

## Phase Packet

| Phase | Stage | Paused? | Evidence |
|---|---|---:|---|
| `PHASE-0-READONLY-STATUS` | precheck | allowed | status reporter, completion gate, command manifest checker |
| `PHASE-1-XR64-PREP-EVAL-ROWS` | artifact generation after resume | not allowed | `8` train/val teacher eval rows |
| `PHASE-2-XR64-PREP-OVERRIDES` | artifact generation after resume | not allowed | `6` train/val override JSON files |
| `PHASE-3-STRICT-READY` | strict gate after resume | not allowed | XR-64 checker ready state |
| `PHASE-4-XR64A-XR64B-LAUNCH` | training after strict ready | not allowed | XR-64A and XR-64B run/checkpoint/test summaries |
| `PHASE-5-POSTRUN-CANDIDATE-REVIEW` | post-run decision review after launch | not allowed | certified candidates and promotion helper command |

## Command Review

These commands print future commands for review. They do not execute experiments:

```bash
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section summary --format summary
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section prep --format commands
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section launch --format script
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section postrun --format commands
```

## Validation

```bash
python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py
.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary
.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary
.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary
.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section postrun --format commands
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python scripts/external/check_second_goal_resume_readiness_matrix.py --format summary
.venv/bin/python scripts/external/check_xr64_prep_progress_ledger.py --format summary
.venv/bin/python scripts/external/check_second_goal_blocker_artifact_map.py --format summary
.venv/bin/python scripts/external/check_paper_target_metric_bridge_preflight.py --format summary
.venv/bin/python -m pytest -q tests/test_xr64_prelaunch_packet.py
```

## Remaining Blockers

- Experiments are paused by user directive.
- XR-64 eval rows are missing.
- XR-64 override JSON files are missing.
- XR-64A/B/C training evidence is missing.
- Metric/protocol bridge blocks direct submission comparison.
- Paper-target comparison evidence manifest is missing required future artifacts.
- No single trained model owns all final evidence yet.

Machine-readable source: `docs/resources/xr64_prelaunch_packet_2026_06_20.json`.
