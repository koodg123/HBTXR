# Validation

## Planned Checks

- Confirm requested root directories exist.
- Confirm migrated package tree exists at `algorithm/src/EvEye`.
- Confirm branch-specific additions exist:
  - Retina/ERVT scripts under `algorithm/analysis/scripts`.
  - Annotation analysis under `algorithm/analysis/annotation-analysis`.
  - Frame dataset under `algorithm/src/EvEye/dataset/DavisEyeEllipse`.
- Run syntax/import smoke checks where local dependencies allow.
- Print final `tree` output for the modified `HBTXR` directory.

## 2026-07-05 Results

- Confirmed root entries: `.gitignore`, `README.md`, `algorithm`, `hardware`,
  `quantization`, `references`, `third`.
- Confirmed migrated package files:
  - `algorithm/src/EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py`
  - `algorithm/src/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseFrameDataset.py`
  - `algorithm/analysis/scripts/create_retina_ervt_jetcas_tables.py`
  - `algorithm/analysis/annotation-analysis/README.md`
- Ran `python3 -m py_compile` on selected migrated files:
  - `algorithm/src/EvEye/dataset/dataset_factory.py`
  - `algorithm/src/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseFrameDataset.py`
  - `algorithm/scripts/facet/train.py`
  - `algorithm/src/EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py`
  - `algorithm/src/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py`
- Removed validation-generated `__pycache__` directories.
- Confirmed no nested `.git` directories under `HBTXR`.
- Confirmed no files over 100 MB remain under `references/legacy-codebase`.
- `git status -- HBTXR` could not be run because
  `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool` was not recognized as a Git
  repository in this session.
