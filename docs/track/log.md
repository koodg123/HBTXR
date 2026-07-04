# Work Log

## 2026-07-03

- Generated and validated Retina/ERVT subject37-48 result packages.
- Added scripts to generate predictions, rebuild tables from HBTXR joined motion labels, and create JETCAS-style Excel files.
- Rebuilt Retina/ERVT Excel files after identifying that raw metadata motion labels caused blank Saccade columns.
- Verified that Saccade counts are nonzero for every subject 37-48.

## 2026-07-04

- Continued on branch `etri-desktop`.
- Documented model input/output, Params, MACs, FLOPs, head semantics, and final prediction semantics for the HBTXR comparison set and additional report targets.
- Added FACET `EIDet / ElNet` to the measured-target table format and profiled it with a DCNv2-compatible shim because local `DCNv2` is absent.
- Profiled `EV-Eye` in the main `.venv`.
- Created additional local venvs under `tmp/venvs/` for incompatible profiling stacks.
- Installed and used TensorFlow 2.6.0 plus `jakeret/unet` to profile `E-Track`.
- Installed and used PyTorch 1.13.0, MMCV-full 1.7.2, MMDetection 2.28.2, and local Swift-Eye MMRotate code to profile `Swift-Eye`.
- Installed and used PyTorch 1.13.1, MMCV 2.0.1, MMEngine 0.8.5, MMDetection 3.1.0, and MMRotate 1.0.0rc1 to profile `EX-Gaze`.
- Added `tmp/venvs/` to `.gitignore`.
- Restored profiling-generated tracked `__pycache__` updates.
