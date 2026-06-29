# FACET Validation Smoke Suite

Date: 2026-06-26

## Summary

A single validation smoke suite was added to rerun the static checks and
regression smokes for the FACET reproduction automation added around final
artifact validation.

No training process was stopped or restarted by this update.

## Added Script

```text
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

The suite runs:

1. Python syntax checks for the reproduction status, summary, pairwise, sync,
   artifact validation, and completion audit scripts.
2. Shell syntax checks for the final evaluation runners, watcher scripts, and
   smoke tests, including the hourly status refresh guard.
3. Artifact validation smoke, including rejection of smoke JSON, null metric
   values, missing checkpoint paths, incomplete comparison rows, and incomplete
   summary labels. It also checks final artifact context mismatches and
   incomplete Markdown report content, final artifact date suffix parsing, and
   invalid summary entry paths.
4. Pairwise input validation smoke.
5. Training completion marker smoke.
6. Evaluation runner completion gate smoke, including post-evaluation status
   synchronization and completion audit regeneration checks.
7. Hourly refresh guard routing smoke, which ensures routine watcher loops
   call `run_hourly_status_refresh_guard_2026-06-26.sh` instead of directly
   refreshing status/progress artifacts.
8. Hourly guard skip order smoke, which ensures the skip branch exits before
   status/progress/audit refresh commands can run.
9. Monitoring interval defaults smoke, which ensures routine watcher/waiter
   interval defaults remain at 3600 seconds.
10. Next hourly refresh once smoke, which ensures the one-shot refresh helper
   waits for the remaining interval and calls the hourly guard without `--force`.
11. Completion audit fail gate smoke, which ensures `--fail-on-incomplete`
   exits nonzero and still writes auditable JSON/Markdown outputs.
12. Completion audit pass gate smoke, which ensures a synthetic all-passed
   status can produce `can_mark_goal_complete: true`.

The individual smoke tests write only to `/tmp`, so the suite does not modify
long-running training outputs or final reproduction artifacts.

## Validation

Commands run:

```text
chmod +x references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
bash -n references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

Observed output:

```text
[1/4] Python syntax checks
[2/4] Shell syntax checks
[3/4] Artifact validation smoke
artifact validation smoke passed
[4/4] Pairwise, completion marker, and routing smokes
pairwise input validation smoke passed
training completion marker smoke passed
evaluation runner completion gate smoke passed
hourly refresh guard routing smoke passed
hourly guard skip order smoke passed
monitoring interval defaults smoke passed
next hourly refresh once smoke passed
completion audit fail gate smoke passed
completion audit pass gate smoke passed
FACET validation smoke suite passed
```

## Usage

Run this after changing any of:

```text
check_reproduction_status.py
build_reproduction_summary.py
validate_reproduction_artifact.py
compare_model_evaluation_results.py
sync_reproduction_status_summary.py
audit_reproduction_completion_2026-06-26.py
run_*checkpoint_evaluation_2026-06-26.sh
watch_*2026-06-26.sh
run_hourly_status_refresh_guard_2026-06-26.sh
test_hourly_refresh_guard_routing_2026-06-26.sh
test_hourly_guard_skip_order_2026-06-26.sh
test_monitoring_interval_defaults_2026-06-26.sh
test_next_hourly_refresh_once_2026-06-26.sh
test_completion_audit_fail_gate_2026-06-26.sh
test_completion_audit_pass_gate_2026-06-26.sh
```

The suite is a fast automation regression check. It is not a replacement for
the final full-validation evaluation after training completes.
