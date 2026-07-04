# FACET Artifact Validation Smoke Test

Date: 2026-06-26

## Summary

A repeatable smoke test was added for the final artifact validation path. The
test confirms that the existing EPNet smoke evaluation JSON cannot be accepted
as a full reproduction artifact and cannot make the top-level reproduction
summary `complete`.

No training process was stopped or restarted by this update.

## Added Script

```text
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

The script uses only `/tmp` outputs:

```text
/tmp/facet_artifact_validation_eval_smoke_stdout.json
/tmp/facet_artifact_validation_eval_smoke_stderr.txt
/tmp/facet_artifact_validation_summary_stdout.json
/tmp/facet_artifact_validation_summary_smoke.json
/tmp/facet_artifact_validation_summary_smoke.md
```

## Checks

The smoke test verifies:

- `validate_reproduction_artifact.py --type eval` rejects
  `FACET_phase4_epnet_eval_smoke_2026-06-25.json`.
- `build_reproduction_summary.py` marks that same artifact as:

```text
state: invalid
```

- the generated summary stays:

```text
summary_state: partial
```

- validation issues include:

```text
max_batches is 2
expected DeanDataset_full_unet
```

## Validation

Commands run:

```text
chmod +x references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
bash -n references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
references/report/FACET/test_artifact_validation_smoke_2026-06-26.sh
```

Observed output:

```text
artifact validation smoke passed
```

This test should be rerun after future changes to `check_reproduction_status.py`,
`validate_reproduction_artifact.py`, or `build_reproduction_summary.py`.
