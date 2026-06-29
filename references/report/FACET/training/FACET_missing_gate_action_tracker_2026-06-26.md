# FACET Missing Gate Action Tracker

Date: 2026-06-26

## Summary

This tracker maps the current `missing 8` completion-audit gates to the
training session, watcher, runner, and artifact that should resolve each gate.
It is an execution tracker, not a final result.

Source state:

```text
plan: references/report/FACET/FACET_reproduction_plan_2026-06-25.md
status: references/report/FACET/FACET_reproduction_status_2026-06-26.json
progress: references/report/FACET/FACET_full_training_progress_snapshot_2026-06-26.json
audit: references/report/FACET/FACET_reproduction_completion_audit_2026-06-26.json
latest artifact refresh: 2026-06-26 22:33 KST
guard check: 2026-06-26 22:51 KST, skipped, next_due_in=2506s
```

Completion remains blocked by evidence, not by planning:

```text
overall_status: incomplete
passed: 10
missing: 8
can_mark_goal_complete: false
```

## Active Automation

The reproduction automation currently has these roles:

| Role | Session or Script | Purpose |
|---|---|---|
| Full EPNet training | `facet_epnet_full_gpu0` | Train `DavisEyeEllipse_EPNet_full_unet.yaml` on GPU0. |
| Full HBTXR training | `facet_hbtxr_full_gpu1` | Train `DavisEyeEllipse_HBTXR_full_unet.yaml` on GPU1. |
| Full training watchdog | `facet_full_training_watchdog` | Restart missing full-training sessions only if completion markers/final artifacts are absent. |
| Full evaluation watcher | `facet_full_eval_watcher` | Wait for full EPNet/HBTXR completion, then run full evaluation/comparison. |
| EPNet fpn_dw waiter | `facet_epnet_fpn_dw_gpu0_waiter` | Start fpn_dw ablation after baseline GPU0 conditions are satisfied. |
| EPNet fpn_dw eval watcher | `facet_epnet_fpn_dw_eval_watcher` | Evaluate fpn_dw ablation after completion. |
| HBTXR effbs32 waiter | `facet_hbtxr_effbs32_gpu1_waiter` | Start effective-batch-32 HBTXR run after baseline GPU1 conditions are satisfied. |
| HBTXR effbs32 eval watcher | `facet_hbtxr_effbs32_eval_watcher` | Evaluate HBTXR effbs32 after completion. |
| Follow-up watchdog | `facet_followup_training_watchdog` | Keep follow-up waiters/eval watchers alive. |
| Hourly refresh reservation | `facet_next_hourly_refresh_once` | Call the hourly status/progress/audit guard at the next due time. |

## Missing Gate Mapping

| # | Missing gate | Owner | Evidence required to pass | Current state |
|---:|---|---|---|---|
| 1 | Phase 4 full EPNet training completion | `facet_epnet_full_gpu0`, `facet_full_training_watchdog` | `EPNet_full_unet_gpu0_train_2026-06-26.log` contains strict `max_epochs=70` completion marker. | EPNet checkpoint gate is passed, but completion marker is absent. |
| 2 | Phase 4B full HBTXR checkpoint | `facet_hbtxr_full_gpu1`, `facet_full_training_watchdog` | `.ckpt` under `runs/logs/HBTXR_full_unet/version_*/checkpoints/`. | No full HBTXR checkpoint in latest status. |
| 3 | Phase 4B full HBTXR training completion | `facet_hbtxr_full_gpu1`, `facet_full_training_watchdog` | `HBTXR_full_unet_gpu1_train_2026-06-26.log` contains strict `max_epochs=70` completion marker. | HBTXR full training is active but not complete. |
| 4 | Phase 4 EPNet fpn_dw ablation checkpoint | `facet_epnet_fpn_dw_gpu0_waiter`, `facet_followup_training_watchdog` | `.ckpt` under `runs/logs/EPNet_fpn_dw_full_unet/version_*/checkpoints/`. | Waiter/eval watcher are alive; ablation output is not yet present. |
| 5 | Phase 4 EPNet fpn_dw ablation completion | `facet_epnet_fpn_dw_gpu0_waiter`, `facet_epnet_fpn_dw_eval_watcher` | `EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log` contains strict `max_epochs=70` completion marker. | Completion marker absent. |
| 6 | Phase 4B HBTXR effective-batch-32 checkpoint | `facet_hbtxr_effbs32_gpu1_waiter`, `facet_followup_training_watchdog` | `.ckpt` under `runs/logs/HBTXR_full_unet_effbs32/version_*/checkpoints/`. | Waiter/eval watcher are alive; effbs32 output is not yet present. |
| 7 | Phase 4B HBTXR effective-batch-32 completion | `facet_hbtxr_effbs32_gpu1_waiter`, `facet_hbtxr_effbs32_eval_watcher` | `HBTXR_full_unet_effbs32_gpu1_train_2026-06-26.log` contains strict `max_epochs=70` completion marker. | Completion marker absent. |
| 8 | Phase 4 final evaluation artifacts | full/follow-up evaluation runners | Valid final JSON/Markdown artifacts for EPNet, HBTXR, EPNet fpn_dw, HBTXR effbs32, pairwise comparisons, table comparison, and summary. | Evaluation is correctly gated until training completion markers exist. |

## Current Progress Snapshot

Latest structured progress snapshot, generated at 2026-06-26 22:33 KST:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 10 | 9908 / 36415 | 27.21% | 11.48 it/s | 38:29 | 8 |
| HBTXR_full_unet | 0 | 232098 / 291315 | 79.67% | 5.49 it/s | 2:59:55 | 0 |

This progress is not a completion proof. The completion audit intentionally
requires strict full-training completion markers and validated final artifacts.

## Next Actions

1. Keep the 1-hour monitoring cadence. Do not force status/progress refreshes
   unless debugging a failure.
2. Let `facet_next_hourly_refresh_once` trigger the next guarded refresh.
3. After HBTXR produces a full checkpoint and both full baseline logs show
   completion, let `facet_full_eval_watcher` run full EPNet/HBTXR evaluation.
4. Let follow-up waiters start EPNet fpn_dw and HBTXR effbs32 only after their
   baseline dependency conditions are satisfied.
5. Re-run the completion audit only after the hourly guard refresh or after a
   watcher records a material state transition.

## Completion Rule Reminder

The active goal must remain incomplete until:

```text
FACET_reproduction_completion_audit_2026-06-26.json
```

reports:

```text
can_mark_goal_complete: true
completion_decision: complete
```

## Continuation Check: 2026-06-26 22:53 KST

The hourly guard was called without `--force` and correctly skipped because the
latest status/progress/audit artifacts were still newer than one hour:

```text
skip refresh: latest artifact age=1170s, next_due_in=2430s, min_interval=3600s
```

Relevant tmux sessions were still present:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_training_watchdog
facet_full_eval_watcher
facet_epnet_fpn_dw_gpu0_waiter
facet_epnet_fpn_dw_eval_watcher
facet_hbtxr_effbs32_gpu1_waiter
facet_hbtxr_effbs32_eval_watcher
facet_followup_training_watchdog
facet_next_hourly_refresh_once
```

Active training processes were still present:

```text
tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
```

No completion decision changed at this checkpoint. The authoritative completion
audit remains:

```text
can_mark_goal_complete: false
completion_decision: incomplete
```

## No-Log Summary Helper

Added a helper for future continuation checks:

```text
references/report/FACET/summarize_missing_gates_2026-06-26.py
references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

It reads only:

```text
FACET_reproduction_completion_audit_2026-06-26.json
FACET_full_training_progress_snapshot_2026-06-26.json
status/progress/audit artifact mtimes
```

It does not scan training logs. This keeps routine status explanations aligned
with the one-hour monitoring policy while still showing:

```text
missing gates
completion_decision
can_mark_goal_complete
refresh_next_due_in_seconds
structured progress snapshot
```

The Markdown artifact can be regenerated with:

```bash
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/report/FACET/summarize_missing_gates_2026-06-26.py \
  --output-md references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

Runtime status can be included without scanning training logs:

```bash
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/report/FACET/summarize_missing_gates_2026-06-26.py \
  --include-runtime \
  --output-md references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

When run from a restricted Codex subprocess, host tmux/process state may be
unobservable. In that case the runtime section must report `unavailable` rather
than `missing`, so it does not falsely imply that training sessions stopped.

Validation added:

```text
references/report/FACET/test_missing_gate_summary_2026-06-26.sh
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

Verification:

```text
missing gate summary smoke passed
FACET validation smoke suite passed
```

## Hourly Guard Owns Summary Refresh

Updated:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
references/report/FACET/run_next_hourly_refresh_once_2026-06-26.sh
references/report/FACET/test_hourly_refresh_guard_routing_2026-06-26.sh
references/report/FACET/test_hourly_guard_skip_order_2026-06-26.sh
references/report/FACET/test_next_hourly_refresh_once_2026-06-26.sh
```

`run_hourly_status_refresh_guard_2026-06-26.sh` now regenerates:

```text
references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

only after it actually refreshes status/progress/audit artifacts. The skip
branch still exits before status/progress/audit/summary calls. This lets
long-running watchdogs keep using only the hourly guard while ensuring the
summary artifact is updated after real hourly refreshes.

The one-shot script now relies on the hourly guard for summary generation and
then records the completion decision. It no longer invokes the guarded summary
wrapper separately.

Verification:

```text
hourly refresh guard routing smoke passed
hourly guard skip order smoke passed
next hourly refresh once smoke passed
FACET validation smoke suite passed
```

The one-shot reservation was restarted at 2026-06-26 23:14 KST so it uses the
current script:

```text
facet_next_hourly_refresh_once: alive
scheduled: next hourly refresh once scheduled wait_seconds=1121 min_interval=3600
```

## Continuation Check: 2026-06-26 23:16 KST

No forced refresh was run. The no-log summary reported:

```text
latest_artifact_age_seconds: 2544
refresh_next_due_in_seconds: 1056
refresh_state: fresh
completion_decision: incomplete
can_mark_goal_complete: False
counts: {'passed': 10, 'missing': 8}
```

The one-shot and long-running FACET sessions remained present:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_training_watchdog
facet_full_eval_watcher
facet_epnet_fpn_dw_gpu0_waiter
facet_epnet_fpn_dw_eval_watcher
facet_hbtxr_effbs32_gpu1_waiter
facet_hbtxr_effbs32_eval_watcher
facet_followup_training_watchdog
facet_next_hourly_refresh_once
```

No completion gate changed at this checkpoint.

## Follow-Up Waiter Completion Gate Regression

Added a static regression test to keep successor experiments from starting
before their baseline runs have strict completion markers:

```text
references/report/FACET/test_followup_waiter_completion_gate_2026-06-26.sh
```

The test verifies:

```text
FACET_FPN_DW_WAIT_BASELINE_COMPLETE defaults to 1
FACET_EFFBS32_WAIT_BASELINE_COMPLETE defaults to 1
EPNet fpn_dw waits on EPNet_full_unet_gpu0_train_2026-06-26.log
HBTXR effbs32 waits on HBTXR_full_unet_gpu1_train_2026-06-26.log
both waiters loop until training_complete("${BASELINE_LOG}") is true
```

Verification:

```text
follow-up waiter completion gate smoke passed
FACET validation smoke suite passed
```

## Continuation Check: 2026-06-26 23:02 KST

The hourly guard was called without `--force` and correctly skipped because the
latest status/progress/audit artifacts were still newer than one hour:

```text
skip refresh: latest artifact age=1736s, next_due_in=1864s, min_interval=3600s
```

The no-log summary artifact was regenerated:

```text
references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

Current summary state:

```text
completion_decision: incomplete
can_mark_goal_complete: False
counts: {'passed': 10, 'missing': 8}
```

Host-level checks showed the automation path was still alive:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
tmux: facet_full_training_watchdog
tmux: facet_full_eval_watcher
tmux: facet_epnet_fpn_dw_gpu0_waiter
tmux: facet_epnet_fpn_dw_eval_watcher
tmux: facet_hbtxr_effbs32_gpu1_waiter
tmux: facet_hbtxr_effbs32_eval_watcher
tmux: facet_followup_training_watchdog
tmux: facet_next_hourly_refresh_once
```

Active training processes:

```text
tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
```

No completion gate changed at this checkpoint. The goal remains active.

## Guarded Summary Wrapper

Added:

```text
references/report/FACET/run_guarded_missing_gate_summary_2026-06-26.sh
references/report/FACET/test_guarded_missing_gate_summary_2026-06-26.sh
```

The wrapper performs the common continuation check safely:

```text
1. run run_hourly_status_refresh_guard_2026-06-26.sh without --force
2. regenerate FACET_missing_gate_summary_2026-06-26.md
3. append output to FACET_guarded_missing_gate_summary_2026-06-26.log
```

Execution at 2026-06-26 23:04 KST:

```text
skip refresh: latest artifact age=1828s, next_due_in=1772s, min_interval=3600s
completion_decision: incomplete
can_mark_goal_complete: False
counts: {'passed': 10, 'missing': 8}
```

Verification:

```text
guarded missing gate summary smoke passed
FACET validation smoke suite passed
```

## Goal Completion Check Wrapper

Added:

```text
references/report/FACET/check_goal_completion_after_guard_2026-06-26.sh
references/report/FACET/test_goal_completion_check_2026-06-26.sh
```

The wrapper runs the guarded missing-gate summary first, then reads:

```text
references/report/FACET/FACET_reproduction_completion_audit_2026-06-26.json
```

It returns success only when both conditions are true:

```text
can_mark_goal_complete == true
completion_decision == complete
```

Current execution at 2026-06-26 23:05 KST:

```text
skip refresh: latest artifact age=1913s, next_due_in=1687s, min_interval=3600s
FACET reproduction goal is incomplete according to completion audit
exit code: 2
```

This fail-closed behavior is expected while the audit remains `missing 8`.

Verification:

```text
goal completion check smoke passed
FACET validation smoke suite passed
```

## Next-Hourly Summary Update

Updated:

```text
references/report/FACET/run_next_hourly_refresh_once_2026-06-26.sh
references/report/FACET/test_next_hourly_refresh_once_2026-06-26.sh
```

The next-hourly one-shot now performs:

```text
1. wait until the status/progress refresh interval is due
2. call run_hourly_status_refresh_guard_2026-06-26.sh without --force
3. call run_guarded_missing_gate_summary_2026-06-26.sh with FACET_SKIP_HOURLY_GUARD=1
```

This means the next scheduled hourly refresh also refreshes:

```text
references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

without requiring a separate manual command.

Follow-up hardening:

```text
run_guarded_missing_gate_summary_2026-06-26.sh now supports
FACET_SKIP_HOURLY_GUARD=1.
```

The one-shot script uses this option after its own guard call, so the next
hourly reservation avoids a duplicate guard invocation while still regenerating
the missing-gate summary artifact.

## Next-Hourly Completion Decision Logging

Updated:

```text
references/report/FACET/check_goal_completion_after_guard_2026-06-26.sh
references/report/FACET/run_next_hourly_refresh_once_2026-06-26.sh
references/report/FACET/test_goal_completion_check_2026-06-26.sh
references/report/FACET/test_next_hourly_refresh_once_2026-06-26.sh
```

`check_goal_completion_after_guard_2026-06-26.sh` now supports:

```text
FACET_SKIP_GUARDED_SUMMARY=1
```

The next-hourly one-shot uses:

```text
FACET_GOAL_CHECK_ALLOW_INCOMPLETE=1
FACET_SKIP_GUARDED_SUMMARY=1
```

after the guard and summary have already run. This records the current
completion decision in `FACET_goal_completion_check_2026-06-26.log` without
causing the one-shot session to fail while the audit is legitimately incomplete.

## One-Shot Refresh Session Refresh

Action at 2026-06-26 23:11 KST:

```text
tmux kill-session -t facet_next_hourly_refresh_once
references/report/FACET/start_next_hourly_refresh_once_2026-06-26.sh
```

Only the one-shot monitoring reservation was restarted. Training sessions were
not stopped or restarted.

Reason:

```text
The existing facet_next_hourly_refresh_once session had been created before the
one-shot script was updated to run guarded summary and completion-decision
logging. Restarting the reservation makes the next hourly refresh use the
current script.
```

Verification:

```text
facet_next_hourly_refresh_once: alive
created: 2026-06-26 23:11 KST
scheduled: next hourly refresh once scheduled wait_seconds=1349 min_interval=3600
active full training processes:
  tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
  tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
```

## Missing Summary Freshness Fields

Updated:

```text
references/report/FACET/summarize_missing_gates_2026-06-26.py
references/report/FACET/test_missing_gate_summary_2026-06-26.sh
references/report/FACET/FACET_missing_gate_summary_2026-06-26.md
```

The no-log summary now includes:

```text
latest_artifact_age_seconds
refresh_state
refresh_next_due_in_seconds
```

Current generated summary at 2026-06-26 23:12 KST:

```text
latest_artifact_age_seconds: 2321
refresh_next_due_in_seconds: 1279
refresh_state: fresh
completion_decision: incomplete
can_mark_goal_complete: False
```

Verification:

```text
missing gate summary smoke passed
FACET validation smoke suite passed
```

## Continuation Check: 2026-06-26 23:17 KST

Action:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

Result:

```text
skip refresh: latest artifact age=2638s, next_due_in=962s, min_interval=3600s
```

Interpretation:

```text
The 1-hour monitoring interval has not elapsed yet. No forced status refresh,
training-log scan, training restart, or session restart was performed.
```

Current known completion state remains:

```text
completion_decision: incomplete
counts: {'passed': 10, 'missing': 8}
can_mark_goal_complete: False
```

The next scheduled refresh remains delegated to:

```text
tmux session: facet_next_hourly_refresh_once
```

## Continuation Check: 2026-06-26 23:18 KST

Action:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

Result:

```text
skip refresh: latest artifact age=2711s, next_due_in=889s, min_interval=3600s
```

Interpretation:

```text
The 1-hour refresh gate is still not due. Existing training and watcher tmux
sessions remain present, and no manual refresh, forced log scan, or process
restart was performed.
```

Current known completion state remains:

```text
completion_decision: incomplete
counts: {'passed': 10, 'missing': 8}
can_mark_goal_complete: False
```

## Hourly Refresh: 2026-06-26 23:33 KST

Refresh source:

```text
tmux session facet_next_hourly_refresh_once invoked
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

Updated artifacts:

```text
FACET_reproduction_status_2026-06-26.json: 2026-06-26 23:33:48 +0900
FACET_full_training_progress_snapshot_2026-06-26.json: 2026-06-26 23:33:53 +0900
FACET_reproduction_completion_audit_2026-06-26.json: 2026-06-26 23:33:53 +0900
FACET_missing_gate_summary_2026-06-26.md: 2026-06-26 23:33:53 +0900
```

Completion state:

```text
completion_decision: incomplete
counts: {'passed': 10, 'missing': 8}
can_mark_goal_complete: False
```

Current progress snapshot:

```text
EPNet_full_unet: epoch 11, step 11672 / 36415, progress 32.05%, rate 11.41 it/s, remaining 36:08, checkpoints 8
HBTXR_full_unet: epoch 0, step 251911 / 291315, progress 86.47%, rate 5.48 it/s, remaining 1:59:44, checkpoints 0
```

Gate interpretation:

```text
No completion gate changed. The baseline full trainings are still running, so
the follow-up fpn_dw/effective-batch-32 gates and final evaluation artifacts are
not yet ready.
```

Next refresh reservation:

```text
facet_next_hourly_refresh_once: restarted at 2026-06-26 23:34:13 +0900
next hourly refresh once scheduled wait_seconds=3580 min_interval=3600
```
