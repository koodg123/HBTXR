# Imported Materials

This directory preserves source-lineage material moved out of active package
surfaces during the 2026-07-07 structure cleanup.

`IMPORT_MANIFEST_P0_P1_P2.md` records the detailed source-to-target mapping for
the larger import batch.

| Directory | Upstream Source | Status |
|---|---|---|
| `legacy_hybrid/docs/` | `HBTXR_v3_0/docs/*` | Reference-only experiment, analysis, plan, and research documents. |
| `legacy_hybrid/scripts/` | `HBTXR_v3_0/exps/scripts/` and `HBTXR_v3_0/scripts/` | Reference scripts; not assumed to run against current `algorithm/hybrid/src` without adaptation. |
| `legacy_hybrid/configs/experiments/` | `HBTXR_v3_0/exps/configs/` | Historical experiment presets. |
| `legacy_hybrid/configs/package_legacy/` | former consolidated HBTXR algorithm config surface | Pre-existing package-level compatibility config. |
| `legacy_hybrid/tests/` | `HBTXR_v3_0/tests/` | Reference regression tests and snapshots for future porting. |
| `hgtxr/docs/` | `HGTXR/HGTXR-etri-server/docs/*.md` | Project planning, execution, validation, and paper-mapping context. |
| `conversations/` | HBTXR/HGTXR project tracking files | Conversation and progress history grouped by project and date. |

The upstream source names are kept here for provenance. Active package directory
names are intentionally general.
