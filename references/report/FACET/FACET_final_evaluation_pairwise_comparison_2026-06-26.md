# FACET Final Evaluation Pairwise Comparison Hardening

Date: 2026-06-26

## Summary

The final checkpoint evaluation flow was hardened so that HBTXR individual evaluation and EPNet-vs-HBTXR pairwise comparison are separate artifacts.

Previously, the HBTXR evaluation markdown path was named:

```text
references/report/FACET/FACET_epnet_vs_hbtxr_comparison_2026-06-26.md
```

That name implied a pairwise comparison, but the generated content would have been a single-model HBTXR-vs-paper evaluation table.

## Changes

Added script:

```text
references/codebase/software/FACET/EvEye/utils/scripts/compare_model_evaluation_results.py
```

Updated final evaluation runner:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
```

New final outputs:

```text
FACET_reproduction_results_2026-06-26.json
FACET_table2_comparison_2026-06-26.md
FACET_hbtxr_reproduction_results_2026-06-26.json
FACET_hbtxr_reproduction_results_2026-06-26.md
FACET_epnet_vs_hbtxr_comparison_2026-06-26.json
FACET_epnet_vs_hbtxr_comparison_2026-06-26.md
```

The pairwise comparison table includes:

```text
P10
P5
P3
P1
mean pixel error
IoU
AP
params M
trainable params M
FLOPs G
latency ms
```

Each row records EPNet value, HBTXR value, `HBTXR - EPNet`, preferred direction, and winner.

## Status Gate Update

`check_reproduction_status.py` now requires both the individual HBTXR markdown and pairwise comparison JSON/Markdown artifacts for the final evaluation gate.

`watch_full_training_jobs_2026-06-26.sh` also uses the stricter final artifact set before deciding that the evaluation watcher no longer needs to be restarted.

`watch_full_checkpoints_and_evaluate_2026-06-26.sh` now routes routine
status/progress refreshes through `run_hourly_status_refresh_guard_2026-06-26.sh`
so the checkpoint watcher cannot refresh training-result artifacts more often
than the user-requested hourly cadence.

After this update, `facet_full_eval_watcher` was restarted without touching the
training sessions. Its first loop reported:

```text
loop=1 ep_ckpt_count=4 hb_ckpt_count=0 ep_done=0 hb_done=0 require_completed=1
skip refresh: latest artifact age=567s, next_due_in=3033s, min_interval=3600s
```

## Runtime Note

No active training process was changed. This is a final-evaluation artifact correction only.
