# FACET Pairwise Input Validation

Date: 2026-06-26

## Summary

Pairwise comparison generation was hardened so it cannot be built from smoke,
partial, stale, or non-full-dataset evaluation JSON inputs.

No training process was stopped or restarted by this update.

## Updated Scripts

```text
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
references/codebase/software/FACET/EvEye/utils/scripts/compare_model_evaluation_results.py
```

## Changes

`run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh` previously generated
the EPNet-vs-HBTXR-effbs32 comparison when the EPNet baseline JSON file existed:

```text
FACET_reproduction_results_2026-06-26.json
```

It now requires that file to pass:

```text
validate_reproduction_artifact.py --type eval
```

before generating the pairwise comparison.

`compare_model_evaluation_results.py` now validates both input evaluation JSON
files by default. It exits non-zero if either side is not a full
`DeanDataset_full_unet` validation artifact.

An escape hatch exists only for explicit debugging:

```text
--allow-invalid-inputs
```

Production runner scripts do not use this option.

## Validation

Static validation passed:

```text
python -m py_compile compare_model_evaluation_results.py check_reproduction_status.py
bash -n run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh run_full_checkpoint_evaluation_2026-06-26.sh
```

Smoke rejection check:

```text
compare_model_evaluation_results.py \
  --left-json FACET_phase4_epnet_eval_smoke_2026-06-25.json \
  --right-json FACET_phase4_epnet_eval_smoke_2026-06-25.json
```

Observed result:

```text
exit code: 1
reason: max_batches is 2 and dataset_root is DeanDataset instead of DeanDataset_full_unet
```

This prevents smoke outputs from becoming pairwise comparison artifacts in the
final reproduction report path.
