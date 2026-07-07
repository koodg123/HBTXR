# HBTXR-Pool P0/P1/P2 Import Manifest

Date: 2026-07-07

## Scope

This manifest records the non-codebase archive import requested for HBTXR-Pool:

- P0: high-value hybrid experiment and operational documents.
- P1: legacy experiment scripts, configs, and reference tests.
- P2: HGTXR hardware/quantization evidence and project context documents.
- Conversation/progress logs: project-classified and date-reorganized records.

No source directories were removed or renamed.

## P0 Hybrid Documents

| Source | Target | Purpose |
|---|---|---|
| `HBTXR_v3_0/docs/exps/` | `HBTXR/algorithm/hybrid/docs/exp/v3_legacy/` | Experiment results, ablation analysis, progress logs, and qualitative resources. |
| `HBTXR_v3_0/docs/prj/` | `HBTXR/algorithm/hybrid/docs/analysis/v3_operational/` | Runtime, config, data contract, call-stack, and preprocessing operation guides. |
| `HBTXR_v3_0/docs/plan/` | `HBTXR/algorithm/hybrid/docs/analysis/v3_plans/` | Historical planning and stabilization checklists. |
| `HBTXR_v3_0/docs/others/` | `HBTXR/algorithm/hybrid/docs/analysis/v3_research_archive/` | Paper drafts, reference analyses, optimizer notes, update reports, and archived research context. |

## P1 Scripts, Configs, Tests

| Source | Target | Purpose |
|---|---|---|
| `HBTXR_v3_0/exps/scripts/` | `HBTXR/algorithm/hybrid/scripts/experiments/v3_legacy/` | Experimental one-off scripts for GroundedSAM, target-FPS, TimeLens-XL, v2e, visualization, and dataset construction. |
| `HBTXR_v3_0/scripts/` | `HBTXR/algorithm/hybrid/scripts/runtime_legacy/v3/` | Legacy top-level runtime scripts for prepare/train/eval/export/infer workflows. |
| `HBTXR_v3_0/exps/configs/` | `HBTXR/algorithm/hybrid/configs/experiments/v3_legacy/` | Historical experiment presets and GroundedSAM experiment configs. |
| `HBTXR_v3_0/tests/` | `HBTXR/algorithm/hybrid/tests/v3_legacy_reference/` | Reference tests and snapshots for future regression-porting. |

These files are imported as reference/legacy material. They are not assumed to
run against the current `algorithm/hybrid/src` API without adaptation.

## P2 Hardware And Quantization Evidence

| Source | Target | Purpose |
|---|---|---|
| `HGTXR/HGTXR-etri-server/docs/resources/final_*` and selected evidence resources | `HBTXR/hardware/docs/resources/hgtxr_final_evidence/` | Final evidence, resource matrix, operator audits, Q4W8A validation artifacts, and source/requirements traces. |
| `HGTXR/HGTXR-etri-server/hardware/analysis/integrated-2026-06-26/` | `HBTXR/hardware/docs/analysis/integrated-2026-06-26/` | Integrated hardware analysis with provenance and coverage manifests. |
| `HGTXR/HGTXR-etri-server/docs/Quantization.md` and related context docs | `HBTXR/quantization/docs/hgtxr_context/` | HGTXR quantization and paper-reproduction context. |
| `HGTXR/HGTXR-etri-server/docs/*.md` | `HBTXR/algorithm/analysis/imported-docs/hgtxr_project_docs/` | Top-level HGTXR project planning, execution, validation, and paper-mapping documents. |

## Conversation And Progress Logs

Logs were copied into:

```text
HBTXR/algorithm/analysis/conversations/<project>/by-date/<YYYY-MM-DD|undated>/
```

Project buckets:

- `hbtxr_v3_0`
- `hbtxr_etri_desktop`
- `hbtxr_etri_server`
- `hbtxr_home`
- `hgtxr_etri_server_project`
- `hgtxr_etri_server_hardware`
- `hgtxr_etri_server_software`
- `hgtxr_etri_server_quantization`
- `hgtxr_home_project`
- `hgtxr_home_hardware`
- `hgtxr_home_software`

Date extraction rule:

- Filenames containing `YYYYMMDD` are placed under `YYYY-MM-DD`.
- Filenames containing `YYYY-MM-DD` are placed under that date.
- Files without an explicit date are placed under `undated`.

## Import Counts

Observed after import:

- P0 hybrid experiment documents: `80` files under `algorithm/hybrid/docs/exp/v3_legacy`.
- P0 hybrid operational/planning/research documents: `147` files under `algorithm/hybrid/docs/analysis/v3_*`.
- P1 scripts/configs/tests: `175` files.
- P2 hardware/quantization/project evidence documents and artifacts: `71` files.
- Conversation/progress logs: `127` files.

## Follow-Up

The next useful step is to create a short curated reading order from the imported
documents, then port only the scripts/tests that still match the current
`algorithm/hybrid/src` API.
