# FACET Summary Content Validation

Date: 2026-06-26

## Summary

`build_reproduction_summary.py` was hardened so that the top-level reproduction
summary does not mark a smoke-only or stale evaluation JSON as `available`.

Before this update, a command that passed only the EPNet smoke evaluation JSON
could produce:

```text
summary_state: complete
```

even though the artifact was not from the full `DeanDataset_full_unet`
validation split. The summary builder now reuses the same content validation
logic as `check_reproduction_status.py`.

No training process was stopped or restarted by this update.

## Updated Script

```text
references/codebase/software/FACET/EvEye/utils/scripts/build_reproduction_summary.py
```

Validation now applies to:

- evaluation result JSON artifacts
- pairwise comparison JSON artifacts

Artifact state values:

```text
available: content is valid for full reproduction
missing: artifact file is absent or unreadable
invalid: artifact exists but is smoke-only, stale, partial, or not full-dataset
```

The top-level summary state is now:

```text
complete: all referenced results and comparisons are available
partial: at least one referenced result or comparison is missing or invalid
```

## Validation

Static validation passed:

```text
python -m py_compile build_reproduction_summary.py check_reproduction_status.py sync_reproduction_status_summary.py
```

Smoke rejection check:

```text
build_reproduction_summary.py --result EPNet_smoke:FACET_phase4_epnet_eval_smoke_2026-06-25.json
```

Observed result:

```text
summary_state: partial
EPNet_smoke state: invalid
validation issues:
- max_batches is 2, expected 0 for full validation
- dataset_root is DeanDataset, expected DeanDataset_full_unet
```

This keeps the top-level final Markdown report aligned with the stricter
completion gate in `check_reproduction_status.py`.
