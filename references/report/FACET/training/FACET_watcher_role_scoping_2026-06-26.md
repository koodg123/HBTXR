# FACET Watcher Role Scoping

Date: 2026-06-26

## Summary

Watcher completion checks were adjusted so each watcher stops after its own
evaluation responsibility is satisfied, instead of requiring the final top-level
summary to be globally `complete`.

No training process was stopped or restarted by this update.

## Reason

The full baseline evaluator can finish before the planned follow-up runs:

```text
EPNet fpn_dw ablation
HBTXR effective-batch-32 comparison
```

At that point `FACET_reproduction_summary_2026-06-26.json` is expected to exist
but remain `partial`, because follow-up artifacts are still missing. If the
baseline watchdog requires the summary JSON itself to validate as `complete`,
it can restart the baseline evaluation watcher once per hour until the follow-up
runs finish.

The same issue can affect the fpn_dw watcher while effbs32 is still pending.

## Updated Behavior

Baseline full-training watchdog:

```text
references/report/FACET/watch_full_training_jobs_2026-06-26.sh
```

Now stops restarting the baseline evaluation watcher when the baseline EPNet,
baseline HBTXR, baseline pairwise comparison, and top-level summary files exist.
The summary file does not need to be globally complete for the baseline watcher
to stop.

Follow-up watchdog:

```text
references/report/FACET/watch_followup_training_jobs_2026-06-26.sh
```

Now checks fpn_dw and effbs32 responsibilities separately:

- fpn_dw watcher completion requires valid fpn_dw evaluation artifacts and the
  top-level summary files.
- effbs32 watcher completion requires valid effbs32 evaluation artifacts,
  EPNet-vs-HBTXR-effbs32 pairwise artifacts, and the top-level summary files.

Individual checkpoint watchers:

```text
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

They now use the same role-scoped artifact checks.

## Completion Authority

This does not weaken the reproduction completion criteria. The authoritative
global completion gate remains:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

That checker still requires the final summary JSON to be content-valid and
`summary_state: complete` before the reproduction goal can pass.

## Validation

Static validation passed:

```text
bash -n watch_full_training_jobs_2026-06-26.sh
bash -n watch_followup_training_jobs_2026-06-26.sh
bash -n watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
bash -n watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

Search validation confirmed that watcher completion checks no longer call
`artifact_valid summary`, but still require the top-level summary file to exist.
