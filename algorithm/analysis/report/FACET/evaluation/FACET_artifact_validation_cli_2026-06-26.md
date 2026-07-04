# FACET Artifact Validation CLI

Date: 2026-06-26

## Summary

The evaluation runner and watcher skip conditions now use the same content
validation logic as the final reproduction status checker. This prevents stale,
partial, or smoke-only JSON files from causing evaluation to be skipped or a
watcher to exit early.

No training process was stopped or restarted by this update.

## Added Script

```text
references/codebase/software/FACET/EvEye/utils/scripts/validate_reproduction_artifact.py
```

Supported artifact types:

```text
eval
comparison
summary
```

Example:

```text
python validate_reproduction_artifact.py --type eval --path FACET_reproduction_results_2026-06-26.json
```

Exit behavior:

```text
0: artifact content is valid
1: artifact is missing, stale, partial, smoke-only, or otherwise invalid
```

The script injects the FACET root into `sys.path`, so it can be called from the
project root, the FACET root, or the report scripts.

## Runner Integration

Updated scripts:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Previous behavior:

```text
if JSON file exists and Markdown file exists: skip re-evaluation
```

New behavior:

```text
if JSON content is valid and Markdown file exists: skip re-evaluation
otherwise: regenerate the evaluation or comparison artifact
```

## Watcher Integration

Updated scripts:

```text
references/report/FACET/watch_full_training_jobs_2026-06-26.sh
references/report/FACET/watch_followup_training_jobs_2026-06-26.sh
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
references/report/FACET/watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

Watcher final-artifact checks now validate JSON content before deciding that an
evaluation watcher no longer needs to run.

## Validation

Static validation passed:

```text
python -m py_compile validate_reproduction_artifact.py check_reproduction_status.py
bash -n run_full_checkpoint_evaluation_2026-06-26.sh run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
bash -n watch_full_training_jobs_2026-06-26.sh watch_followup_training_jobs_2026-06-26.sh watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh watch_hbtxr_effbs32_checkpoints_and_evaluate_2026-06-26.sh
```

Smoke rejection check:

```text
validate_reproduction_artifact.py --type eval --path FACET_phase4_epnet_eval_smoke_2026-06-25.json
```

Observed result:

```text
ok: false
exit code: 1
issues:
- max_batches is 2, expected 0 for full validation
- dataset_root is DeanDataset, expected DeanDataset_full_unet
```

This is the expected result because the smoke artifact is not a full
reproduction evaluation artifact.
