# FACET Final Artifact Date Consistency

Date: 2026-06-26

## Summary

Final reproduction artifacts now have to share the same date suffix. This avoids
mixing the latest JSON from one run with Markdown or comparison artifacts from
another run date when wildcard patterns are used.

No training process was stopped or restarted by this update.

## Updated Validator

File:

```text
references/codebase/software/FACET/EvEye/utils/scripts/check_reproduction_status.py
```

Added:

```text
artifact_date(path)
artifact date suffix consistency check in check_final_evaluation()
```

The expected suffix format is:

```text
_YYYY-MM-DD.json
_YYYY-MM-DD.md
```

If any selected final artifact has no date suffix, or if selected artifacts have
more than one unique date suffix, the final evaluation artifact gate becomes
invalid/partial instead of passed.

## Smoke Test Update

File:

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

The smoke now checks:

- a valid `YYYY-MM-DD` suffix is parsed
- a non-date suffix is rejected
- mixed date suffixes are detectable

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

This strengthens the final Phase 4/4B artifact gate by requiring the selected
JSON and Markdown artifacts to form one coherent dated result bundle.
