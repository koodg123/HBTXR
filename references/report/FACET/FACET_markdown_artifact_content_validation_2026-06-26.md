# FACET Markdown Artifact Content Validation

Date: 2026-06-26

## Summary

Final Markdown artifacts are no longer accepted by file existence alone. The
status checker now validates that each required Markdown artifact contains the
expected report/table sections for its role.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Added `validate_markdown_artifact()` and pattern-specific required terms for:

- reproduction summary Markdown
- Table II style evaluation Markdown
- HBTXR evaluation Markdown
- EPNet-vs-HBTXR comparison Markdown
- fpn_dw ablation evaluation Markdown
- HBTXR effective-batch-32 evaluation Markdown
- EPNet-vs-HBTXR effective-batch-32 comparison Markdown

Examples of required terms:

```text
# FACET Reproduction Results
## Evaluation Artifacts
## Model Metrics
## Pairwise Comparisons
| Metric | Current | Paper Table II reference | Delta |
Evaluation Comparison
Left checkpoint:
Right checkpoint:
Right - left
Winner
```

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Added a `/tmp` Markdown fixture with unrelated content. The smoke calls
`validate_markdown_artifact()` and verifies that required report terms are
missing, so the artifact is rejected.

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

This strengthens the final report gate. A final reproduction can only pass when
both JSON artifacts and their Markdown report companions are structurally
consistent with the planned output roles.
