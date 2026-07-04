# FACET Reports

Updated: 2026-06-29

This directory is the canonical report area for FACET/HBTXR analysis, reproduction planning, dataset records, training notes, evaluation outputs, and operational scripts.

Base paths:

- Report directory: `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET`
- Codebase: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET`
- Paper: `/home/kjm26/project/PRJXR/HBTXR/references/papers/software/FACET_Fast_and_Accurate_Event-Based_Eye_Tracking_Using_Ellipse_Modeling_for_Extended_Reality.pdf`
- EV-Eye raw data: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data`

## Standing Rule

FACET-related analysis documents, reproduction records, dataset generation records, validation reports, and follow-up documents should be saved under this directory and placed into the appropriate category folder.

## Category Layout

| Directory | Purpose | Tracked files |
|---|---|---:|
| `analysis/` | Code/paper analysis, implementation notes, environment diagnostics, and contract clarifications. | 9 |
| `planning/` | Reproduction plans, runbooks, experiment plans, and plan-progress audits. | 7 |
| `datasets/` | Dataset construction notes, split/count reports, subject-motion tables, and sample visualizations. | 46 |
| `training/` | Training launch reports, monitoring notes, recovery reports, checkpoints/status gates, and probe summaries. | 30 |
| `evaluation/` | Evaluation reports, validation notes, comparison artifacts, CSV summaries, and figures. | 40 |
| `operations/` | Runnable helper scripts, watchdogs, one-shot checkers, and machine-readable status/audit JSON used by operations. | 44 |

Root is reserved for this `README.md`, the generated `CATALOG.md`, and local ignored runtime byproducts such as `*.log` files. Large per-sample evaluation CSV files are kept under `evaluation/HBTXR_val_motion_eval/` but remain ignored by git.

## Full Catalog

See `CATALOG.md` for the current file-by-file index and classification criteria.
