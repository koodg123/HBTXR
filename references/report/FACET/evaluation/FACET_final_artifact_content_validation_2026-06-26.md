# FACET Final Artifact Content Validation

Date: 2026-06-26

## Summary

The final reproduction status gate was strengthened so that it no longer treats
artifact filenames alone as proof of completion. Final evaluation JSON files,
pairwise comparison JSON files, and the top-level reproduction summary JSON are
now inspected for content-level validity.

No training process was stopped or restarted by this update.

## Updated Checker

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

The final evaluation gate now validates:

- evaluation JSON files were produced from the full validation split
  (`max_batches` must be `0` or absent).
- evaluation JSON files use:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

- evaluation JSON files contain positive `evaluated_batches`.
- required metrics exist:

```text
val_p10_acc
val_p5_acc
val_p3_acc
val_p1_acc
val_mean_distance
val_IoU
val_AP
```

- comparison JSON files contain non-empty rows and valid left/right full-dataset
  evaluation summaries.
- `FACET_reproduction_summary_*.json` has:

```text
summary_state: complete
missing_artifacts: []
all results state: available
all comparisons state: available
```

## Why This Matters

Earlier smoke artifacts can share a similar shape with final artifacts but are
not valid reproduction outputs. For example:

```text
references/report/FACET/FACET_phase4_epnet_eval_smoke_2026-06-25.json
```

is intentionally rejected because:

```text
max_batches is 2, expected 0 for full validation
dataset_root is DeanDataset, expected DeanDataset_full_unet
```

This prevents a smoke test or partial summary from satisfying the Phase 4 final
reproduction gate.

## Validation

Static validation passed:

```text
python -m py_compile references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Content validation smoke:

```text
validate_eval_result_json(FACET_phase4_epnet_eval_smoke_2026-06-25.json)
```

Observed result:

```text
ok=False
issues=[
  "max_batches is 2, expected 0 for full validation",
  "dataset_root is '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset', expected '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet'"
]
```

The current reproduction status was regenerated after the checker update:

```text
references/report/FACET/FACET_reproduction_status_2026-06-26.json
references/report/FACET/FACET_reproduction_status_2026-06-26.md
```

Current state remains:

```text
overall_status: incomplete
passed: 10
missing: 8
```

The goal is still incomplete until full training, follow-up ablations, and final
full-validation evaluation artifacts pass these content checks.
