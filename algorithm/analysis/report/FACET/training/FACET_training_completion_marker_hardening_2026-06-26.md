# FACET Training Completion Marker Hardening

Date: 2026-06-26

## Summary

The reproduction status checker was hardened so a training log containing only a
configuration string such as `max_epochs=70` is not treated as completed
training.

No training process was stopped or restarted by this update.

## Updated Script

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Previous Python status behavior accepted:

```text
max_epochs=70
```

as a completion marker. This was too broad because logs can contain the
configured epoch count before training has finished.

New accepted markers:

```text
`max_epochs=70` reached
max_epochs=70 reached
Trainer.fit stopped
```

This matches the stricter shell watcher behavior, which already required a
`reached` or `stopped` marker.

## Updated Shell Watchers And Waiters

The shell-side completion regex was also narrowed to the same marker set:

```text
`max_epochs=70` reached|max_epochs=70 reached|Trainer\.fit.*stopped
```

Updated scripts:

```text
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/watch_full_training_jobs_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh
references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/watch_followup_training_jobs_2026-06-26.sh
```

The previous shell pattern `max_epochs=70.*reached` was too broad because it
could match text like `max_epochs=70 not reached`.

## Added Smoke Test

```text
references/report/FACET/test_training_completion_marker_2026-06-26.sh
```

The test writes only `/tmp` synthetic logs and verifies:

- `trainer config: max_epochs=70` is rejected.
- `trainer config: max_epochs=70 not reached` is rejected.
- `` `max_epochs=70` reached. `` is accepted.
- the shell regex and Python status checker agree on these cases.

## Validation

Commands run:

```text
python -m py_compile references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
chmod +x references/report/FACET/test_training_completion_marker_2026-06-26.sh
bash -n references/report/FACET/test_training_completion_marker_2026-06-26.sh
references/report/FACET/test_training_completion_marker_2026-06-26.sh
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

Observed output:

```text
training completion marker smoke passed
FACET validation smoke suite passed
```

## Remaining State

This hardening only affects completion detection. The reproduction goal remains
incomplete until the full and follow-up training logs contain real completion
markers and the final full-validation artifacts pass content validation.
