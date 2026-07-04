# Changelog

## 2026-07-03

- Added HBTXR-style result packages under `analysis/results`.
- Added Retina subject37-48 pixel-error distribution package.
- Added ERVT subject37-48 pixel-error distribution package.
- Added JETCAS-style Excel workbooks for Retina, ERVT, and combined Retina/ERVT.
- Added scripts for inference, HBTXR label-map rebuild, and Excel generation.
- Documented the joined motion label-map decision and validation evidence.

## 2026-07-04

- Added model IO, complexity, head-output, and prediction-semantics report for HBTXR comparison targets and additional report models.
- Added measured or profiled rows for `EIDet / ElNet`, `E-Track`, `EV-Eye`, `EX-Gaze`, and `Swift-Eye`.
- Added local venv ignore rule for `tmp/venvs/`.
- Documented profiling caveats for missing `DCNv2`, TensorFlow custom blocks, EX-Gaze MMRotate sourcing, and Swift-Eye dummy ROI proposals.
- Added Swift-Eye direct HBTXR cache training preparation script and launch wrappers.
- Changed Swift-Eye direct preparation to 64x64 2-channel backbone input, with reduced detector/temporal anchors and temporal crop geometry.
