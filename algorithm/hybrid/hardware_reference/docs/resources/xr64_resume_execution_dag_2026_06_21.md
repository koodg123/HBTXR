# XR-64 Resume Execution DAG

Date: 2026-06-21

Experiments remain paused by user directive. This artifact is a non-executing DAG contract for the future XR-64 resume path.

## Purpose

The command manifest lists the commands and the design contract fixes lane hyperparameters. This DAG fixes the dependency order across the whole resume path:

1. precheck
2. prep eval rows
3. prep override builds
4. strict readiness
5. XR-64A/B launch
6. post-run review

## Current State

| Field | Value |
|---|---:|
| ready to train | `false` |
| can run lane | `false` |
| missing eval rows | `8` |
| missing overrides | `6` |
| leakage risk | `none` |
| current next node | `XR64-EVAL-TRAIN-XR62A` |
| completion allowed | `false` |

## DAG Counts

| Node type | Count |
|---|---:|
| precheck | `2` |
| prep total | `10` |
| eval | `8` |
| build | `2` |
| strict readiness | `1` |
| launch | `2` |
| postrun | `2` |
| total | `17` |

## Critical Gates

- `XR64-BUILD-TRAIN` depends on all four train teacher eval nodes.
- `XR64-BUILD-VAL` depends on all four val teacher eval nodes.
- `XR64-STRICT-READY` depends on both build nodes.
- `XR64A-LAUNCH` and `XR64B-LAUNCH` depend on `XR64-STRICT-READY`.
- Post-run candidate collection depends on both XR-64A and XR-64B launch evidence.

## Rejection Rules

- Do not run non-precheck nodes while `execution_state=paused_by_user_directive`.
- Do not include `test` in any prep eval/build split.
- Do not launch XR-64A/B before strict readiness reports `ready_to_train`.
- Do not run post-run review before XR-64A/B produce override-cleared test eval summaries.
- Do not mark the second goal complete from DAG readiness alone.

Machine-readable source: `docs/resources/xr64_resume_execution_dag_2026_06_21.json`.
