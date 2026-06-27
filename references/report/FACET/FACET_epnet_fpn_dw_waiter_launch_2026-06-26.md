# FACET EPNet fpn_dw Waiter Launch

Date: 2026-06-26

## Summary

The planned EPNet `fpn_dw` ablation is now registered as wait-safe tmux sessions. No active baseline training process was interrupted.

## Started Sessions

```text
facet_epnet_fpn_dw_gpu0_waiter
facet_epnet_fpn_dw_eval_watcher
```

## Waiter Behavior

`facet_epnet_fpn_dw_gpu0_waiter` runs:

```text
references/report/FACET/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh
```

It currently:

```text
passed DeanDataset_full_unet manifest/progress consistency gate
is waiting for EPNet_full_unet baseline max_epochs=70 completion marker
will wait for GPU0 to become free before starting fpn_dw training
uses 3600-second default wait interval
```

## Evaluation Watcher Behavior

`facet_epnet_fpn_dw_eval_watcher` runs:

```text
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
```

It waits for:

```text
EPNet_fpn_dw_full_unet checkpoint
EPNet_fpn_dw_full_unet max_epochs=70 completion marker
```

Then it runs:

```text
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
```

Routine status refreshes in this watcher are routed through:

```text
references/report/FACET/run_hourly_status_refresh_guard_2026-06-26.sh
```

This prevents the fpn_dw watcher from rewriting status artifacts more often
than the 1-hour monitoring cadence.

After this update, `facet_epnet_fpn_dw_eval_watcher` was restarted without
touching the waiter or training sessions. Its first loop reported:

```text
loop=1 fpn_dw_ckpt_count=0 fpn_dw_done=0 require_completed=1
skip refresh: latest artifact age=567s, next_due_in=3033s, min_interval=3600s
```

Expected outputs:

```text
references/report/FACET/FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json
references/report/FACET/FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md
```

## Fix Applied

The evaluation watcher originally exited when the `EPNet_fpn_dw_full_unet` run root did not exist yet. This is normal before the fpn_dw ablation starts, so the watcher was patched to treat missing checkpoint root as:

```text
checkpoint count: 0
latest checkpoint: missing
continue waiting
```

## Verification

```text
bash -n watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh: passed
tmux session facet_epnet_fpn_dw_gpu0_waiter: alive
tmux session facet_epnet_fpn_dw_eval_watcher: alive
```
