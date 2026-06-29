# FACET Full Training Watchdog - 2026-06-26

## Summary

A long-running watchdog was added for the full EPNet/FACET and HBTXR training
jobs. The existing evaluation watcher only waits for checkpoints and final
completion markers. This new watchdog keeps the training sessions alive by
restarting a missing training tmux session unless that training log already
contains a completion marker.

The watchdog does not interrupt currently running jobs.

## Script

```text
references/report/FACET/watch_full_training_jobs_2026-06-26.sh
```

Log:

```text
references/report/FACET/FACET_full_training_watchdog_2026-06-26.log
```

tmux session:

```text
facet_full_training_watchdog
```

## Behavior

Every loop, the watchdog:

- Checks `facet_epnet_full_gpu0`.
- Checks `facet_hbtxr_full_gpu1`.
- Checks `facet_full_eval_watcher`.
- Restarts a missing EPNet/HBTXR session with its existing launcher if the
  corresponding training log does not show completion.
- Restarts the evaluation watcher if it is missing and final artifacts do not
  exist.
- Requests a guarded hourly refresh through
  `run_hourly_status_refresh_guard_2026-06-26.sh`, which refreshes at most once
  per hour:
  - `FACET_reproduction_status_2026-06-26.{json,md}`
  - `FACET_full_training_progress_snapshot_2026-06-26.{json,md}`
  - `FACET_reproduction_completion_audit_2026-06-26.{json,md}`

Defaults:

```text
FACET_TRAINING_WATCHDOG_INTERVAL_SECONDS=3600
FACET_TRAINING_WATCHDOG_MAX_LOOPS=0
FACET_TRAINING_WATCHDOG_START_EVAL_WATCHER=1
```

Update on 2026-06-26 15:31 KST:

```text
The watchdog no longer refreshes status/progress directly. It calls the hourly
guard so overlapping watchdog loops cannot update training-result artifacts
more frequently than the configured 3600-second interval.
```

## Validation

Syntax:

```text
bash -n references/report/FACET/watch_full_training_jobs_2026-06-26.sh
```

The script was dry-run inside tmux with one loop:

```text
facet_training_watchdog_dryrun
```

The direct sandbox `bash script.sh` dry run cannot manage tmux reliably in this
environment, so the meaningful validation is the tmux-internal dry run. That
run correctly reported:

```text
EPNet session alive: facet_epnet_full_gpu0
HBTXR session alive: facet_hbtxr_full_gpu1
evaluation watcher alive: facet_full_eval_watcher
max loops reached; exiting watchdog
```

The live watchdog first loop reported:

```text
watchdog loop=1
EPNet session alive: facet_epnet_full_gpu0
HBTXR session alive: facet_hbtxr_full_gpu1
evaluation watcher alive: facet_full_eval_watcher
```

After the hourly-guard update, the watchdog session was restarted without
interrupting the training sessions. The first loop of the restarted watchdog
reported:

```text
watchdog loop=1
EPNet session alive: facet_epnet_full_gpu0
HBTXR session alive: facet_hbtxr_full_gpu1
evaluation watcher alive: facet_full_eval_watcher
skip refresh: latest artifact age=425s, next_due_in=3175s, min_interval=3600s
```

## Current Live State

Active tmux sessions:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
facet_full_training_watchdog
```

GPU compute apps:

```text
GPU0: EPNet/FACET, PID 1428589, 4214 MiB
GPU1: HBTXR, PID 1483023, 11334 MiB
```

Current progress snapshot from the first live watchdog loop:

```text
EPNet_full_unet: epoch 0, 12622 / 36415, 34.66%, 11.48 it/s
HBTXR_full_unet: epoch 0, 2577 / 291315, 0.88%, 5.46 it/s
```

No full checkpoint exists yet.

## Remaining Gates

The overall FACET reproduction goal remains incomplete:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.
