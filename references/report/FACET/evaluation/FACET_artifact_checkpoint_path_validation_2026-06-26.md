# FACET Artifact Checkpoint Path Validation

Date: 2026-06-26

## Summary

Final evaluation and comparison artifacts must point to checkpoint files that
actually exist. The artifact validator was strengthened so JSON files with
numeric metrics but stale or missing checkpoint paths cannot pass as final
reproduction results.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Added checkpoint existence checks for:

```text
evaluation.checkpoint
comparison.left.checkpoint
comparison.right.checkpoint
```

The validator now rejects:

- missing checkpoint path
- path string that does not resolve to an existing file

This validation is used through:

```text
references/codebase/software/FACET/EvEye/utils/scripts/validate_reproduction_artifact.py
references/codebase/software/FACET/EvEye/utils/scripts/build_reproduction_summary.py
references/codebase/software/FACET/EvEye/utils/scripts/compare_model_evaluation_results.py
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Added a `/tmp` evaluation JSON fixture with valid numeric metrics and the full
`DeanDataset_full_unet` root, but a non-existent checkpoint path. The validator
must reject it.

The partial comparison fixture now uses an existing `/tmp` placeholder
checkpoint so the test continues to isolate the incomplete-row failure mode.

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

This strengthens the final artifact gate by tying reported metrics back to real
checkpoint files. Completion still requires full training completion markers,
full-validation evaluation artifacts, required pairwise comparisons, and all
status checker gates to pass.
