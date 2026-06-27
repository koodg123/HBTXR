# FACET Reproduction Summary Artifact

Date: 2026-06-26

## Summary

The reproduction plan requires a final Markdown artifact named:

```text
references/report/FACET/FACET_reproduction_results_<date>.md
```

Before this update, the full checkpoint evaluator produced the EPNet evaluation
JSON as `FACET_reproduction_results_2026-06-26.json`, but the Markdown artifact
for the EPNet paper comparison was named `FACET_table2_comparison_2026-06-26.md`.
That left the plan-level summary Markdown path unfilled.

This update adds a dedicated summary builder that combines individual model
evaluation JSON files and pairwise comparison JSON files into:

```text
references/report/FACET/FACET_reproduction_summary_2026-06-26.json
references/report/FACET/FACET_reproduction_results_2026-06-26.md
```

## Added Script

```text
references/codebase/software/FACET/EvEye/utils/scripts/build_reproduction_summary.py
```

Inputs:

```text
--result LABEL:PATH
--comparison LABEL:PATH
--status-json PATH
--output-json PATH
--output-md PATH
```

Behavior:

- Reads available evaluation JSON artifacts.
- Marks missing optional follow-up artifacts as `missing`.
- Produces a partial summary while long-running training or follow-up ablations
  are still incomplete.
- Produces a complete summary only when all referenced final evaluation and
  comparison JSON artifacts exist.

## Runner Integration

The summary builder is now called by:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

This means the plan-level Markdown summary is regenerated after:

- EPNet full + HBTXR full baseline evaluation
- EPNet `fpn_dw` ablation evaluation
- HBTXR effective-batch-32 evaluation

## Status Gate Update

`check_reproduction_status.py` now requires:

```text
FACET_reproduction_results_*.md
FACET_reproduction_summary_*.json
```

This prevents the reproduction goal from being marked complete when only the
EPNet JSON and Table II Markdown exist but the plan-level final summary is still
missing.

## Validation

Static checks:

```text
python -m py_compile build_reproduction_summary.py check_reproduction_status.py
bash -n run_full_checkpoint_evaluation_2026-06-26.sh run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh run_hbtxr_effbs32_checkpoint_evaluation_2026-06-26.sh
```

Smoke summary generation:

```text
build_reproduction_summary.py was run with the existing EPNet smoke evaluation
JSON and intentionally missing optional paths.
```

Observed result:

```text
summary_state: partial
available: EPNet_smoke
missing: Missing_model, Missing_comparison
```

No training session was stopped or restarted by this update.
