# FACET Reproduction Plan Progress Audit

Date: 2026-06-26

Source plan:

```text
references/report/FACET/FACET_reproduction_plan_2026-06-25.md
```

Evidence files:

```text
references/report/FACET/FACET_reproduction_status_2026-06-26.md
references/report/FACET/FACET_reproduction_status_2026-06-26.json
references/report/FACET/FACET_full_training_monitor_2026-06-26.md
references/report/FACET/FACET_validation_smoke_suite_2026-06-26.md
```

## Summary

The FACET reproduction plan is still active and incomplete. Phase 1 through
Phase 3 data/preparation gates are complete. Phase 4 and Phase 4B are running
or waiting on long training completion and final evaluation artifacts.

The current automation state is suitable for waiting: training watchdogs,
checkpoint watchers, final evaluation runners, artifact validators, summary
synchronization, and regression smoke tests are in place. Routine monitoring is
set to a 1-hour cadence.

## Plan Progress Checklist

| Plan item | Current state | Evidence |
|---|---|---|
| Phase 1 subset EPNet baseline data | Passed | `DeanDataset/manifest.json`, `num_samples: 8911` |
| Phase 1 subset EPNet smoke checkpoint | Passed | `EPNet_local_train_smoke`, checkpoint count 2 |
| Phase 2 U-Net labelled PNG dataset | Passed | `DavisWithMaskDataset_labelled_subset/manifest.json`, `num_samples: 9011` |
| Phase 2 U-Net smoke checkpoint | Passed | `RGBUNet_local_train_smoke`, checkpoint count 2 |
| Phase 2 full U-Net checkpoint | Passed | `RGBUNet_local_subset`, checkpoint count 4 |
| Phase 3 full `DeanDataset_full_unet` | Passed | manifest `valid_ellipse_count: 1457820` |
| Phase 4 EPNet full training checkpoint | Partially passed | `EPNet_full_unet`, checkpoint count 4 |
| Phase 4 EPNet full training completion | Missing | no `max_epochs=70` completion marker yet |
| Phase 4 EPNet `fpn_dw` ablation | Waiting | waiter/evaluation watcher present, no checkpoint yet |
| Phase 4 final EPNet evaluation artifacts | Missing | final result/comparison JSON and Markdown not produced yet |
| Phase 4B HBTXR full training checkpoint | Missing | `HBTXR_full_unet` checkpoint count 0 |
| Phase 4B HBTXR full training completion | Missing | no `max_epochs=70` completion marker yet |
| Phase 4B HBTXR effective-batch-32 follow-up | Waiting | waiter/evaluation watcher present, no checkpoint yet |
| Phase 4B HBTXR-vs-EPNet comparison artifacts | Missing | final pairwise JSON and Markdown not produced yet |

## Current Status Snapshot

Latest status checker state:

```text
overall_status: incomplete
passed: 10
missing: 8
```

Important interpretation:

- Intermediate checkpoints do not satisfy the reproduction goal.
- The final gate requires full training completion markers and full-validation
  evaluation artifacts.
- The fpn_dw and effective-batch-32 follow-up runs are intentionally waiting
  for their baseline completion/GPU-free conditions.

## Automation Readiness

Prepared scripts:

```text
run_full_checkpoint_evaluation_2026-06-26.sh
run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
watch_full_training_jobs_2026-06-26.sh
watch_full_checkpoints_and_evaluate_2026-06-26.sh
watch_followup_training_jobs_2026-06-26.sh
watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
run_validation_smoke_suite_2026-06-26.sh
```

Validation result from the current worktree:

```text
[1/4] Python syntax checks
[2/4] Shell syntax checks
[3/4] Artifact validation smoke
artifact validation smoke passed
[4/4] Pairwise and completion marker smokes
pairwise input validation smoke passed
training completion marker smoke passed
FACET validation smoke suite passed
```

## Monitoring Cadence

The active monitoring cadence is hourly. The latest observed loop cadence was:

```text
facet_full_training_watchdog: 12:25 -> 13:25 -> 14:25
facet_full_eval_watcher: 12:25 -> 13:25 -> 14:25
facet_followup_training_watchdog: 12:40 -> 13:40 -> 14:40
facet_epnet_fpn_dw_eval_watcher: 12:36 -> 13:36 -> 14:36
facet_hbtxr_effbs32_eval_watcher: 12:38 -> 13:38 -> 14:38
facet_epnet_fpn_dw_gpu0_waiter: 12:35 -> 13:35 -> 14:35
facet_hbtxr_effbs32_gpu1_waiter: 12:38 -> 13:38 -> 14:38
```

Policy:

```text
Do not run routine training-result monitoring more frequently than once per hour.
Use one-off checks only for explicit user requests, error recovery, or static
script/config validation.
```

## Next Work

1. Wait for full EPNet and HBTXR training completion markers.
2. Let `watch_full_checkpoints_and_evaluate_2026-06-26.sh` run the full
   validation only after both model checkpoints and completion markers are
   ready.
3. After baseline evaluation, let the fpn_dw and effective-batch-32 follow-up
   waiters proceed when their baseline/GPU-free gates pass.
4. Re-run `run_validation_smoke_suite_2026-06-26.sh` after any edit to
   evaluation, watcher, validator, comparison, or summary scripts.
5. Mark the reproduction goal complete only after every status checker item is
   `passed` and final full-validation artifacts exist.
