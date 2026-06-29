# FACET Follow-up Training Watchdog

Date: 2026-06-26

## Summary

The baseline watchdog only supervises the active EPNet/HBTXR full baseline jobs and the baseline final evaluation watcher. A separate follow-up watchdog was added for the planned successor experiments:

```text
EPNet fpn_dw ablation
HBTXR effective-batch-32 comparison run
```

No active training process was interrupted.

## Added Script

```text
references/report/FACET/watch_followup_training_jobs_2026-06-26.sh
```

Default interval:

```text
FACET_FOLLOWUP_WATCHDOG_INTERVAL_SECONDS=3600
```

## Supervised Sessions

EPNet fpn_dw:

```text
waiter:       facet_epnet_fpn_dw_gpu0_waiter
eval watcher: facet_epnet_fpn_dw_eval_watcher
launcher:     references/report/FACET/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh
eval watcher script: references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
```

HBTXR effective-batch-32:

```text
waiter:       facet_hbtxr_effbs32_gpu1_waiter
eval watcher: facet_hbtxr_effbs32_eval_watcher
launcher:     references/report/FACET/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh
eval watcher script: references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

## Restart Policy

The watchdog restarts a missing waiter only when:

```text
the corresponding final artifacts do not exist
and the corresponding training completion marker has not been found
```

It restarts a missing evaluation watcher only when:

```text
the corresponding final artifacts do not exist
```

## Status Refresh Policy

The watchdog requests status refresh through:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

This means the follow-up watchdog does not directly rewrite status artifacts.
If another watchdog has refreshed status/progress recently, the guard skips the
refresh and preserves the user-requested 1-hour monitoring cadence.

Update on 2026-06-26 15:31 KST:

```text
The follow-up watchdog was changed to call the hourly guard instead of running
check_reproduction_status.py directly.
```

## Registered Session

```text
facet_followup_training_watchdog
```

Current tmux evidence:

```text
facet_followup_training_watchdog: alive
facet_epnet_fpn_dw_gpu0_waiter: alive
facet_epnet_fpn_dw_eval_watcher: alive
facet_hbtxr_effbs32_gpu1_waiter: alive
facet_hbtxr_effbs32_eval_watcher: alive
```

## Verification

```text
bash -n watch_followup_training_jobs_2026-06-26.sh: passed
script executable: passed
tmux session facet_followup_training_watchdog: alive
```

After the hourly-guard update, the follow-up watchdog session was restarted
without interrupting training sessions or waiter/evaluation watcher sessions.
The first loop of the restarted watchdog reported:

```text
followup watchdog loop=1
EPNet fpn_dw waiter alive: facet_epnet_fpn_dw_gpu0_waiter
EPNet fpn_dw eval watcher alive: facet_epnet_fpn_dw_eval_watcher
HBTXR effbs32 waiter alive: facet_hbtxr_effbs32_gpu1_waiter
HBTXR effbs32 eval watcher alive: facet_hbtxr_effbs32_eval_watcher
skip refresh: latest artifact age=425s, next_due_in=3175s, min_interval=3600s
```
