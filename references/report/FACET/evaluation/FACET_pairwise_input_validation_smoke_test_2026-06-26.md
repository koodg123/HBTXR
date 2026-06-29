# FACET Pairwise Input Validation Smoke Test

Date: 2026-06-26

## Summary

A repeatable smoke test was added for pairwise comparison input validation. The
test confirms that smoke evaluation JSON files are rejected by default and can
only be used when the explicit debugging escape hatch is passed.

No training process was stopped or restarted by this update.

## Added Script

```text
references/report/FACET/test_pairwise_input_validation_smoke_2026-06-26.sh
```

The script uses only `/tmp` outputs:

```text
/tmp/facet_pairwise_reject_stdout.json
/tmp/facet_pairwise_reject_stderr.txt
/tmp/facet_pairwise_invalid_input_debug.json
/tmp/facet_pairwise_invalid_input_debug.md
/tmp/facet_pairwise_allow_invalid_stdout.json
```

## Checks

The smoke test verifies:

- `compare_model_evaluation_results.py` rejects
  `FACET_phase4_epnet_eval_smoke_2026-06-25.json` as both left and right input
  in default mode.
- `compare_model_evaluation_results.py --allow-invalid-inputs` can still create
  a debug pairwise output under `/tmp`.
- The debug output has expected labels and non-empty comparison rows.

## Validation

Commands run:

```text
chmod +x references/report/FACET/test_pairwise_input_validation_smoke_2026-06-26.sh
bash -n references/report/FACET/test_pairwise_input_validation_smoke_2026-06-26.sh
python -m py_compile references/codebase/software/FACET/EvEye/utils/scripts/compare_model_evaluation_results.py
references/report/FACET/test_pairwise_input_validation_smoke_2026-06-26.sh
```

Observed output:

```text
pairwise input validation smoke passed
```

This should be rerun after future changes to
`compare_model_evaluation_results.py` or `check_reproduction_status.py`.
