# FACET Summary Label Validation

Date: 2026-06-26

## Summary

The final reproduction summary must include every result and comparison required
by the reproduction plan. The summary validator was strengthened so a
`summary_state: complete` JSON cannot pass if it omits any planned model result
or pairwise comparison label.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

`validate_summary_json()` now requires these result labels:

```text
EPNet_full_unet
HBTXR_full_unet
EPNet_fpn_dw_full_unet
HBTXR_full_unet_effbs32
```

It also requires these comparison labels:

```text
EPNet_vs_HBTXR
EPNet_vs_HBTXR_effbs32
```

The validator still requires:

- `summary_state: complete`
- empty `missing_artifacts`
- all result entries marked `available`
- all comparison entries marked `available`

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Added a `/tmp` summary JSON fixture that claims `summary_state: complete` and
marks its provided entries as available, but includes only one model result
label. The validator must reject it because planned labels are missing.

## Validation

Commands run:

```text
python -m py_compile check_reproduction_status.py validate_reproduction_artifact.py build_reproduction_summary.py
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

This strengthens the final summary gate. The status checker can only accept a
summary as complete after the baseline, HBTXR, fpn_dw, and effective-batch-32
result labels plus the required pairwise comparison labels are present and
available.
