# FACET Final Artifact Numeric Validation

Date: 2026-06-26

## Summary

The final reproduction artifacts must contain real numeric metric values, not
only the expected JSON keys. The evaluation artifact validator was strengthened
so a final result cannot pass if required metrics or runtime fields are missing,
`null`, non-numeric, or non-finite.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Validation now requires finite numeric values for:

```text
metrics.val_p10_acc
metrics.val_p5_acc
metrics.val_p3_acc
metrics.val_p1_acc
metrics.val_mean_distance
metrics.val_IoU
metrics.val_AP
params_m
trainable_params_m
flops_g
latency_ms
```

This applies through:

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

Added a `/tmp` evaluation JSON fixture with all required keys but
`metrics.val_p3_acc: null`. The validator must reject it. This prevents a stale
or incomplete final evaluation JSON from satisfying the Phase 4/4B completion
gate by shape alone.

## Validation

Commands run:

```text
python -m py_compile check_reproduction_status.py validate_reproduction_artifact.py build_reproduction_summary.py compare_model_evaluation_results.py
bash -n test_artifact_validation_smoke_2026-06-26.sh run_validation_smoke_suite_2026-06-26.sh
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
FACET validation smoke suite passed
```

## Reproduction Impact

This moves the plan closer to completion by making the final evaluation gate
stricter. Completion still requires long-running training completion markers,
full-validation evaluation artifacts, and all status checker items to pass.
