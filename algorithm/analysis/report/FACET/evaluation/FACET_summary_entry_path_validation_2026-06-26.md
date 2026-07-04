# FACET Summary Entry Path Validation

Date: 2026-06-26

## Summary

Final summary JSON entries must now point to the expected artifact filenames for
their labels. This prevents a summary from passing with only correct
`label/state` pairs while the entry paths are missing, stale, or assigned to the
wrong role.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Added path pattern checks for summary result labels:

| Summary result label | Expected path pattern |
|---|---|
| `EPNet_full_unet` | `FACET_reproduction_results_*.json` |
| `HBTXR_full_unet` | `FACET_hbtxr_reproduction_results_*.json` |
| `EPNet_fpn_dw_full_unet` | `FACET_epnet_fpn_dw_reproduction_results_*.json` |
| `HBTXR_full_unet_effbs32` | `FACET_hbtxr_effbs32_reproduction_results_*.json` |

Added path pattern checks for summary comparison labels:

| Summary comparison label | Expected path pattern |
|---|---|
| `EPNet_vs_HBTXR` | `FACET_epnet_vs_hbtxr_comparison_*.json` |
| `EPNet_vs_HBTXR_effbs32` | `FACET_epnet_vs_hbtxr_effbs32_comparison_*.json` |

The validator also checks that each path exists as a file.

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Added a `/tmp` summary JSON fixture with all required labels marked available,
but with the `EPNet_full_unet` entry pointing to the wrong filename pattern. The
validator must reject it.

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

This strengthens the final summary gate. The summary must now contain every
planned label and each entry must point to the correct role-specific artifact
file before the reproduction status can pass.
