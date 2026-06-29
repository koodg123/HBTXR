# FACET Missing Gate Summary

generated_local: `2026-06-27 12:37:55 +0900`
latest_artifact: `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_completion_audit_2026-06-26.json`
latest_artifact_mtime: `2026-06-27 12:37:55 +0900`
refresh_min_interval_seconds: `3600`
latest_artifact_age_seconds: `0`
refresh_next_due_in_seconds: `3600`
refresh_state: `fresh`

## Completion

overall: `incomplete`
counts: `{'passed': 10, 'missing': 8}`
can_mark_goal_complete: `False`
completion_decision: `incomplete`

## Missing Gates

| # | Gate | Missing evidence |
|---:|---|---|
| 1 | Phase 4 full EPNet training completion | EPNet max_epochs=70 completion log |
| 2 | Phase 4B full HBTXR checkpoint | full HBTXR training output |
| 3 | Phase 4B full HBTXR training completion | HBTXR max_epochs=70 completion log |
| 4 | Phase 4 EPNet fpn_dw ablation checkpoint | EPNet fpn_dw ablation training output |
| 5 | Phase 4 EPNet fpn_dw ablation completion | EPNet fpn_dw max_epochs=70 completion log |
| 6 | Phase 4B HBTXR effective-batch-32 checkpoint | HBTXR effective-batch-32 training output |
| 7 | Phase 4B HBTXR effective-batch-32 completion | HBTXR effective-batch-32 max_epochs=70 completion log |
| 8 | Phase 4 final evaluation artifacts | FACET_reproduction_results_*.json, FACET_reproduction_results_*.md, FACET_reproduction_summary_*.json, FACET_table2_comparison_*.md, FACET_hbtxr_reproduction_results_*.json, FACET_hbtxr_reproduction_results_*.md, FACET_epnet_vs_hbtxr_comparison_*.json, FACET_epnet_vs_hbtxr_comparison_*.md, FACET_epnet_fpn_dw_reproduction_results_*.json, FACET_epnet_fpn_dw_table2_comparison_*.md, FACET_hbtxr_effbs32_reproduction_results_*.json, FACET_hbtxr_effbs32_reproduction_results_*.md, FACET_epnet_vs_hbtxr_effbs32_comparison_*.json, FACET_epnet_vs_hbtxr_effbs32_comparison_*.md |

## Progress Snapshot

| Model | Epoch | Step | Progress | Rate | Remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 24 | 6110 / 36415 | 16.78% | 11.54 it/s | 43:46 | 16 |
| HBTXR_full_unet | 0 | 198631 / 291315 | 68.18% | 5.49 it/s | 4:41:29 | 0 |

This summary only reads status/progress/audit JSON artifacts. It does not scan training logs.
