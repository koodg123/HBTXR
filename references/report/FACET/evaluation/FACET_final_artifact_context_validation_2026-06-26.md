# FACET Final Artifact Context Validation

Date: 2026-06-26

## Summary

Final artifact filenames now have to match their JSON content. This prevents an
artifact with valid numeric metrics from being accepted under the wrong
reproduction role, for example an HBTXR result file containing EPNet config
metadata.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

`check_final_evaluation()` now validates expected context by artifact filename
pattern.

Expected evaluation JSON context:

| Artifact pattern | Expected config | Expected model type |
|---|---|---|
| `FACET_reproduction_results_*.json` | `DavisEyeEllipse_EPNet_full_unet.yaml` | `EPNet` |
| `FACET_hbtxr_reproduction_results_*.json` | `DavisEyeEllipse_HBTXR_full_unet.yaml` | `HBTXR` |
| `FACET_epnet_fpn_dw_reproduction_results_*.json` | `DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml` | `EPNet` |
| `FACET_hbtxr_effbs32_reproduction_results_*.json` | `DavisEyeEllipse_HBTXR_full_unet_effbs32.yaml` | `HBTXR` |

Expected comparison JSON context:

| Artifact pattern | Expected left label | Expected right label |
|---|---|---|
| `FACET_epnet_vs_hbtxr_comparison_*.json` | `EPNet_full_unet` | `HBTXR_full_unet` |
| `FACET_epnet_vs_hbtxr_effbs32_comparison_*.json` | `EPNet_full_unet` | `HBTXR_full_unet_effbs32` |

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

The smoke now directly verifies that:

- evaluation context mismatch creates a validation issue
- comparison label mismatch creates a validation issue

This is kept as a fast `/tmp`/in-memory regression test and does not touch
training sessions or checkpoint outputs.

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

This strengthens the final Phase 4/4B artifact gate. A final artifact must now
be valid by content, point to an existing checkpoint, use the full dataset root,
and match the expected role implied by its filename.
