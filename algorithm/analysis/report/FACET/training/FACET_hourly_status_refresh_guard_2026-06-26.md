# FACET Hourly Status Refresh Guard

Date: 2026-06-26

## Summary

User policy: routine FACET training-result monitoring should run every 1 hour,
not more frequently.

To prevent accidental manual over-polling, a guarded refresh script was added.
It checks the latest mtime among the status/progress artifacts and exits without
reading the training logs when the latest artifact is newer than the configured
minimum interval.

## Added Script

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

Default interval:

```text
FACET_STATUS_REFRESH_MIN_INTERVAL_SECONDS=3600
```

When the 1-hour interval has elapsed, the script refreshes:

```text
references/report/FACET/FACET_reproduction_status_2026-06-26.json
references/report/FACET/FACET_reproduction_status_2026-06-26.md
references/report/FACET/FACET_full_training_progress_snapshot_2026-06-26.json
references/report/FACET/FACET_full_training_progress_snapshot_2026-06-26.md
references/report/FACET/FACET_reproduction_completion_audit_2026-06-26.json
references/report/FACET/FACET_reproduction_completion_audit_2026-06-26.md
```

The script logs decisions to:

```text
references/report/FACET/FACET_hourly_status_refresh_guard_2026-06-26.log
```

## Usage

Normal guarded refresh:

```bash
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

Force refresh only for explicit user requests or recovery work:

```bash
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh --force
```

## Validation

Syntax check:

```text
bash -n run_hourly_status_refresh_guard_2026-06-26.sh: passed
```

Guard behavior checked at 2026-06-26 14:59 KST:

```text
[2026-06-26T14:59:01+0900] skip refresh: latest artifact age=1080s, next_due_in=2520s, min_interval=3600s
```

This confirms the script did not refresh status/progress/audit artifacts when
the last status or progress artifact was younger than 1 hour.

Update on 2026-06-26 15:28 KST:

```text
The guarded refresh script now also regenerates the completion audit after a
real status/progress refresh. The skip path still exits before reading training
logs or regenerating artifacts.
```

Update on 2026-06-26 15:33 KST:

```text
The full training watchdog, follow-up watchdog, and checkpoint evaluation
watchers now call this guard instead of directly rewriting status/progress
artifacts during routine loops. Final evaluation runners may still synchronize
status immediately after creating real final artifacts.
```

The script was also added to the shell syntax checks in:

```text
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```
