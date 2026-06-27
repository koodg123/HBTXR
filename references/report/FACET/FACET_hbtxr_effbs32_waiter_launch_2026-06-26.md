# FACET HBTXR Effective-Batch-32 Waiter Launch

Date: 2026-06-26

## Summary

The HBTXR effective-batch-32 follow-up run is now registered as wait-safe tmux sessions. This run is for stricter EPNet-vs-HBTXR comparison because the active HBTXR baseline uses physical/effective batch size 4 while EPNet uses batch size 32.

No active baseline training process was interrupted.

## Started Sessions

```text
facet_hbtxr_effbs32_gpu1_waiter
facet_hbtxr_effbs32_eval_watcher
```

## Waiter Behavior

`facet_hbtxr_effbs32_gpu1_waiter` runs:

```text
references/report/FACET/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh
```

It currently:

```text
passed DeanDataset_full_unet manifest/progress consistency gate
is waiting for HBTXR_full_unet baseline max_epochs=70 completion marker
will wait for GPU1 to become free before starting HBTXR effective-batch-32 training
uses 3600-second default wait interval
```

## Evaluation Runner

```text
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Expected individual outputs:

```text
references/report/FACET/FACET_hbtxr_effbs32_reproduction_results_2026-06-26.json
references/report/FACET/FACET_hbtxr_effbs32_reproduction_results_2026-06-26.md
```

If the EPNet baseline JSON already exists, it also writes:

```text
references/report/FACET/FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.json
references/report/FACET/FACET_epnet_vs_hbtxr_effbs32_comparison_2026-06-26.md
```

## Evaluation Watcher

`facet_hbtxr_effbs32_eval_watcher` runs:

```text
references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

It checks every 3600 seconds by default and waits for both:

```text
HBTXR_full_unet_effbs32 checkpoint
HBTXR_full_unet_effbs32 max_epochs=70 completion marker
```

The watcher treats a missing checkpoint root as normal pre-run state:

```text
checkpoint count: 0
latest checkpoint: missing
continue waiting
```

Routine status refreshes in this watcher are routed through:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

This prevents the effbs32 watcher from rewriting status artifacts more often
than the 1-hour monitoring cadence.

After this update, `facet_hbtxr_effbs32_eval_watcher` was restarted without
touching the waiter or training sessions. Its first loop reported:

```text
loop=1 effbs32_ckpt_count=0 effbs32_done=0 require_completed=1
skip refresh: latest artifact age=567s, next_due_in=3033s, min_interval=3600s
```

## Verification

```text
bash -n run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh: passed
bash -n watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh: passed
tmux session facet_hbtxr_effbs32_gpu1_waiter: alive
tmux session facet_hbtxr_effbs32_eval_watcher: alive
```
