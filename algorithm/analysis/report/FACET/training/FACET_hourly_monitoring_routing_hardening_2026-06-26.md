# FACET Hourly Monitoring Routing Hardening

Date: 2026-06-26

## Summary

Routine FACET training-result monitoring was hardened so all recurring
watchdog/checkpoint watcher loops route status/progress/audit refreshes through
the hourly guard:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

This preserves the user policy that routine training-result monitoring should
not run more frequently than once per hour.

No training process was stopped or restarted. Only watchdog/evaluation watcher
sessions were restarted after their scripts were updated.

## Guarded Routine Scripts

The following recurring scripts now call the hourly guard instead of directly
running status/progress refresh scripts:

```text
watch_full_training_jobs_2026-06-26.sh
watch_followup_training_jobs_2026-06-26.sh
watch_full_checkpoints_and_evaluate_2026-06-26.sh
watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

The hourly guard is now the only routine path that calls:

```text
check_reproduction_status.py
write_full_training_progress_snapshot.py
audit_reproduction_completion_2026-06-26.py
```

## Final Evaluation Exception

Final evaluation runners still synchronize status immediately after they create
real final artifacts:

```text
run_full_checkpoint_evaluation_2026-06-26.sh
run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

After `sync_reproduction_status_summary.py`, they also regenerate:

```text
FACET_reproduction_completion_audit_2026-06-26.json
FACET_reproduction_completion_audit_2026-06-26.md
```

This exception is intentional because final artifact creation should update the
completion decision immediately instead of waiting for the next hourly loop.

## Regression Test

Added:

```text
references/report/FACET/test_hourly_refresh_guard_routing_2026-06-26.sh
```

The smoke test fails if a routine watcher:

- lacks the hourly guard binding
- does not call the hourly guard
- directly references `check_reproduction_status.py`
- directly references `write_full_training_progress_snapshot.py`

It also verifies that the hourly guard retains responsibility for status,
progress, and completion audit refresh.

Update on 2026-06-26 15:47 KST:

```text
The routing smoke now discovers all watch_*_2026-06-26.sh scripts dynamically
instead of relying on a fixed watcher list. New routine watcher scripts will
therefore be checked automatically.
```

## Validation

Commands run:

```text
bash -n references/report/FACET/test_hourly_refresh_guard_routing_2026-06-26.sh
references/report/FACET/test_hourly_refresh_guard_routing_2026-06-26.sh
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

Observed output:

```text
hourly refresh guard routing smoke passed
FACET validation smoke suite passed
```

## Current Completion Impact

This does not complete the FACET reproduction. It reduces monitoring drift and
keeps the completion audit fresh when final evaluation artifacts are produced.
The current completion audit remains:

```text
status_overall: incomplete
can_mark_goal_complete: false
completion_decision: incomplete
```

## Latest Guard Check

Checked at 2026-06-26 15:51 KST:

```text
skip refresh: latest artifact age=1565s, next_due_in=2035s, min_interval=3600s
```

No forced refresh was run. The active watchdog/evaluation watcher sessions are
alive and route their routine refresh requests through the hourly guard, so the
next status/progress/audit refresh should occur only after the guard interval is
satisfied.

## One-Shot Refresh Reservation

Update on 2026-06-26 16:03 KST:

```text
tmux session: facet_next_hourly_refresh_once
scheduled: next hourly refresh once scheduled wait_seconds=1567 min_interval=3600
```

The one-shot session waits until the hourly guard interval is due, then calls
`run_hourly_status_refresh_guard_2026-06-26.sh` once without `--force`. Current
status/progress/audit artifacts had not yet been refreshed at the time of this
check.

Follow-up check at 2026-06-26 16:04 KST:

```text
facet_next_hourly_refresh_once: alive
status artifact mtime:   2026-06-26 15:25:29 KST
progress artifact mtime: 2026-06-26 15:25:31 KST
audit artifact mtime:    2026-06-26 15:27:34 KST
```

The one-shot refresh had not fired yet at this checkpoint.

## Duplicate-Safe One-Shot Launcher

Added:

```text
references/report/FACET/start_next_hourly_refresh_once_2026-06-26.sh
```

This launcher checks the exact tmux session name
`facet_next_hourly_refresh_once` before starting a new one-shot refresh
reservation. If the session already exists, it records the condition in
`FACET_next_hourly_refresh_once_2026-06-26.log` and exits successfully without
creating a duplicate monitor. If tmux session inspection itself fails, it
fails closed and does not attempt to start a new session.

Verification at 2026-06-26 22:49 KST:

```text
start_next_hourly_refresh_once_2026-06-26.sh
-> starting one-shot hourly refresh session: facet_next_hourly_refresh_once

second call
-> one-shot hourly refresh session already alive: facet_next_hourly_refresh_once
```

`tmux ls` showed exactly one `facet_next_hourly_refresh_once` session after the
second call. The new one-shot reservation computed:

```text
next hourly refresh once scheduled wait_seconds=2664 min_interval=3600
```

This keeps the next status/progress/audit refresh aligned to the one-hour
minimum interval instead of refreshing immediately.
