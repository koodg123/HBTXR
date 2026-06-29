# FACET Comparison Row Validation

Date: 2026-06-26

## Summary

Final pairwise comparison artifacts must include every metric required by the
reproduction plan. The comparison validator was strengthened so a JSON file with
only one or a few rows cannot pass as a final EPNet-vs-HBTXR comparison.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

`validate_comparison_json()` now requires exactly the planned metric rows:

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

For each row, the validator checks:

- expected `preferred_direction`
- finite numeric `left`
- finite numeric `right`
- finite numeric `right_minus_left`
- non-empty `winner`

It also continues to check both model summaries for positive
`evaluated_batches` and the expected full `DeanDataset_full_unet` root.

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Added a `/tmp` comparison JSON fixture with valid side summaries but only one
metric row. The validator must reject it as an incomplete comparison artifact.

## Validation

Commands run:

```text
python -m py_compile check_reproduction_status.py validate_reproduction_artifact.py build_reproduction_summary.py compare_model_evaluation_results.py
bash -n test_artifact_validation_smoke_2026-06-26.sh run_validation_smoke_suite_2026-06-26.sh
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
references/report/FACET/run_validation_smoke_suite_2026-06-26.sh
```

Observed suite output:

```text
[1/4] Python syntax checks
[2/4] Shell syntax checks
[3/4] Artifact validation smoke
artifact validation smoke passed
[4/4] Pairwise and completion marker smokes
pairwise input validation smoke passed
training completion marker smoke passed
evaluation runner completion gate smoke passed
FACET validation smoke suite passed
```

## Reproduction Impact

This strengthens the final Phase 4/4B comparison gate. A final comparison now
has to contain the complete plan-level metric set before the reproduction status
checker can mark final artifacts as passed.
