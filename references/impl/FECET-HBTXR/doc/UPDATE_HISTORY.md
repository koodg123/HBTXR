# Update History

## 2026-03-25

- Created the simplified `FECET-HBTXR` baseline by flattening the FACET-style data flow and `HBTXR_v3_0` runtime/training stack into a compact package structure.
- Added FACET-compatible event representation, ellipse supervision, simplified manifests, 2-stage training, and Search/Track FSM runtime tracing.
- Added `tools/prepare_facet_gsam_dataset.py` to build a FACET-style dataset from raw EV-Eye `Data_davis` with optional Grounded-SAM annotation.
- Enforced a non-destructive data preparation policy so the raw EV-Eye root and the Grounded-SAM root are treated as read-only inputs.
- Added Linux `sh` wrappers under `scripts/` for dataset preparation, training, evaluation, inference, and visualization.
- Added `tools/overlay_preview.py` and `scripts/overlay_preview.sh` for per-session random annotation overlay preview generation.
- Added `tools/prepare_facet_reference_dataset.py` to bridge canonical/manifests data into the sample-wise cache format expected by the original FACET loader.
- Added `tools/facet_train.py`, `tools/facet_eval.py`, and `tools/facet_infer.py` as FACET comparison wrappers with configurable dataset root, checkpoint, and device selection.
- Added `scripts/facet_*.sh` and `scripts/fecet_compare_*.sh` for FACET-vs-FECET-HBTXR quantitative comparison runs.
- Added `doc/FACET_QUANT_COMPARISON.md` to document the comparison workflow.
- Added repository hygiene files: `.gitignore`, expanded `requirements.txt`, and a rewritten `README.md`.
- Added this `doc/` directory for change history and conversation-history management.
- Added `doc/PROGRESS_CHECKLIST.md` to track plan-vs-progress status and to separate implemented scope from not-yet-executed real-data validation.
- Refreshed the history documents to include the latest overlay-preview work and the current status review request.
