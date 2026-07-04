# FACET Evaluation Watcher Idempotency

Date: 2026-06-26

## Summary

The Phase 4 and Phase 4B evaluation automation was hardened so that watcher
completion checks match the final reproduction artifact gates and evaluation
runners do not repeat expensive full validation when their JSON/Markdown outputs
already exist.

No training process was stopped or restarted by this update.

## Changes

Baseline checkpoint watcher:

```text
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

- Treats missing checkpoint roots as zero checkpoints instead of failing.
- Keeps the existing 1-hour polling interval.

Follow-up checkpoint watchers:

```text
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

- `fpn_dw` is complete only when its individual evaluation artifacts and the
  top-level reproduction summary artifacts exist.
- `effbs32` is complete only when its individual evaluation artifacts, the
  EPNet-vs-HBTXR-effbs32 pairwise artifacts, and the top-level reproduction
  summary artifacts exist.

Evaluation runners:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

- Skip model re-evaluation when the corresponding JSON/Markdown artifacts are
  already present.
- Still regenerate missing pairwise comparisons and the top-level reproduction
  summary as needed.
- This prevents repeated full validation runs if only a summary or pairwise
  artifact was missing.

## Validation

Static validation passed:

```text
bash -n watch_full_checkpoints_and_evaluate_2026-06-26.sh
bash -n watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
bash -n watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
bash -n run_full_checkpoint_evaluation_2026-06-26.sh
bash -n run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
bash -n run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
bash -n watch_full_training_jobs_2026-06-26.sh
bash -n watch_followup_training_jobs_2026-06-26.sh
```

Search validation confirmed that:

- watcher final-artifact checks include the reproduction summary artifacts.
- HBTXR effbs32 final-artifact checks include EPNet-vs-HBTXR-effbs32 pairwise
  artifacts.
- evaluation runners print skip messages when existing artifacts make
  re-evaluation unnecessary.

## Remaining State

The FACET reproduction goal is still incomplete. The long-running full and
follow-up training/evaluation gates must still produce the final artifacts
listed by:

```text
references/report/FACET/FACET_reproduction_status_2026-06-26.md
```
