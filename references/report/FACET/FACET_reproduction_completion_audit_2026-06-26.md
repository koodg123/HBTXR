# FACET Reproduction Completion Audit

Generated at: `2026-06-27T01:37:12.262755+00:00`
Plan: `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_plan_2026-06-25.md`
Status JSON: `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_status_2026-06-26.json`
Status overall: `incomplete`
Status counts: `{'passed': 10, 'missing': 8}`
Can mark goal complete: `False`
Completion decision: `incomplete`

## Completion Rule

Goal completion requires the plan file to exist, every expected status item to be present and passed, grouped Phase 1-4B requirements to be passed, and final full-validation artifacts to be accepted by the status checker.

## Requirement Groups

| Group | State |
|---|---|
| Phase 1 subset EPNet baseline | passed |
| Phase 2 U-Net relabeling pipeline | passed |
| Phase 3 full DeanDataset expansion | passed |
| Phase 4 full EPNet reproduction | incomplete |
| Phase 4B HBTXR parallel comparison | incomplete |

## Non-Passed Status Items

| Status item | State | Missing |
|---|---|---|
| Phase 4 full EPNet training completion | missing | EPNet max_epochs=70 completion log |
| Phase 4B full HBTXR checkpoint | missing | full HBTXR training output |
| Phase 4B full HBTXR training completion | missing | HBTXR max_epochs=70 completion log |
| Phase 4 EPNet fpn_dw ablation checkpoint | missing | EPNet fpn_dw ablation training output |
| Phase 4 EPNet fpn_dw ablation completion | missing | EPNet fpn_dw max_epochs=70 completion log |
| Phase 4B HBTXR effective-batch-32 checkpoint | missing | HBTXR effective-batch-32 training output |
| Phase 4B HBTXR effective-batch-32 completion | missing | HBTXR effective-batch-32 max_epochs=70 completion log |
| Phase 4 final evaluation artifacts | missing | FACET_reproduction_results_*.json, FACET_reproduction_results_*.md, FACET_reproduction_summary_*.json, FACET_table2_comparison_*.md, FACET_hbtxr_reproduction_results_*.json, FACET_hbtxr_reproduction_results_*.md, FACET_epnet_vs_hbtxr_comparison_*.json, FACET_epnet_vs_hbtxr_comparison_*.md, FACET_epnet_fpn_dw_reproduction_results_*.json, FACET_epnet_fpn_dw_table2_comparison_*.md, FACET_hbtxr_effbs32_reproduction_results_*.json, FACET_hbtxr_effbs32_reproduction_results_*.md, FACET_epnet_vs_hbtxr_effbs32_comparison_*.json, FACET_epnet_vs_hbtxr_effbs32_comparison_*.md |

## Group Details

### Phase 1 subset EPNet baseline

State: `passed`

| Status item | State | Missing |
|---|---|---|
| Phase 1 subset DeanDataset | passed |  |
| Phase 1 EPNet smoke checkpoint | passed |  |

### Phase 2 U-Net relabeling pipeline

State: `passed`

| Status item | State | Missing |
|---|---|---|
| Phase 2 U-Net labelled PNG dataset | passed |  |
| Phase 2 U-Net smoke checkpoint | passed |  |
| Phase 2 full U-Net checkpoint | passed |  |

### Phase 3 full DeanDataset expansion

State: `passed`

| Status item | State | Missing |
|---|---|---|
| Phase 3 full DeanDataset_full_unet | passed |  |
| U-Net labelled subset visual samples | passed |  |

### Phase 4 full EPNet reproduction

State: `incomplete`

| Status item | State | Missing |
|---|---|---|
| Phase 4 full EPNet checkpoint | passed |  |
| Phase 4 full EPNet training completion | missing | EPNet max_epochs=70 completion log |
| Phase 4 EPNet fpn_dw ablation checkpoint | missing | EPNet fpn_dw ablation training output |
| Phase 4 EPNet fpn_dw ablation completion | missing | EPNet fpn_dw max_epochs=70 completion log |
| Phase 4 final evaluation artifacts | missing | FACET_reproduction_results_*.json, FACET_reproduction_results_*.md, FACET_reproduction_summary_*.json, FACET_table2_comparison_*.md, FACET_hbtxr_reproduction_results_*.json, FACET_hbtxr_reproduction_results_*.md, FACET_epnet_vs_hbtxr_comparison_*.json, FACET_epnet_vs_hbtxr_comparison_*.md, FACET_epnet_fpn_dw_reproduction_results_*.json, FACET_epnet_fpn_dw_table2_comparison_*.md, FACET_hbtxr_effbs32_reproduction_results_*.json, FACET_hbtxr_effbs32_reproduction_results_*.md, FACET_epnet_vs_hbtxr_effbs32_comparison_*.json, FACET_epnet_vs_hbtxr_effbs32_comparison_*.md |

### Phase 4B HBTXR parallel comparison

State: `incomplete`

| Status item | State | Missing |
|---|---|---|
| Phase 4B full HBTXR checkpoint | missing | full HBTXR training output |
| Phase 4B full HBTXR training completion | missing | HBTXR max_epochs=70 completion log |
| Phase 4B HBTXR effective-batch-32 checkpoint | missing | HBTXR effective-batch-32 training output |
| Phase 4B HBTXR effective-batch-32 completion | missing | HBTXR effective-batch-32 max_epochs=70 completion log |
| Phase 4 final evaluation artifacts | missing | FACET_reproduction_results_*.json, FACET_reproduction_results_*.md, FACET_reproduction_summary_*.json, FACET_table2_comparison_*.md, FACET_hbtxr_reproduction_results_*.json, FACET_hbtxr_reproduction_results_*.md, FACET_epnet_vs_hbtxr_comparison_*.json, FACET_epnet_vs_hbtxr_comparison_*.md, FACET_epnet_fpn_dw_reproduction_results_*.json, FACET_epnet_fpn_dw_table2_comparison_*.md, FACET_hbtxr_effbs32_reproduction_results_*.json, FACET_hbtxr_effbs32_reproduction_results_*.md, FACET_epnet_vs_hbtxr_effbs32_comparison_*.json, FACET_epnet_vs_hbtxr_effbs32_comparison_*.md |
