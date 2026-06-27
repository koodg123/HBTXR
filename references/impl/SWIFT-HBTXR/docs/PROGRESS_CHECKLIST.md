# Progress Checklist

## Scope

This checklist compares the current `SWIFT-HBTXR` repository against the agreed integration plan:

- remove `mmrotate` from the project implementation
- reuse the `HBTXR_v3_0` style model, trainer, and Search/Track FSM
- preserve the `Swift-Eye` anti-blink UNet behavior
- integrate `timelens` through a strong external wrapper
- simplify the repository into a flat, readable structure
- keep a single understandable execution path from data prep to inference

## Status Legend

- `[x]`: implemented and verified in repository-level tests
- `[-]`: implemented in code, but not yet validated on the real dataset / real checkpoint / full training run
- `[ ]`: not done yet

## Plan vs Progress

### 1. Repository Simplification

- [x] Flat repository structure created around `swift_hbtxr/`, `tools/`, `scripts/`, `configs/`, `tests/`, and `docs/`
- [x] Main package renamed and flattened from the copied `FECET-HBTXR` scaffold
- [x] Execution surface reduced to seven intended entry points
- [x] Root README rewritten to match the simplified flow

### 2. HBTXR Core Reuse

- [x] `HBTXR_v3_0` style tracker core preserved in [model.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/model.py)
- [x] Search/Track host-side FSM preserved in [scheduler.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/scheduler.py)
- [x] 2-stage trainer path preserved in [trainer.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/trainer.py)
- [x] Runtime trace path preserved and extended in [runtime.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/runtime.py)
- [x] Stage1 and stage2 smoke training succeeded on a real-data subset
- [-] Full training on the target real dataset has not been executed in this session

### 3. Swift-Eye Preservation

- [x] Anti-blink UNet preserved in standalone form in [antiblink.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/antiblink.py)
- [x] `open_extent` calculation added in [geometry.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/geometry.py)
- [x] `closed_eye_flag` and `hold_last` runtime handling added
- [x] Partial Swift-Eye checkpoint import tool added
- [x] Real Swift-Eye anti-blink checkpoint import validated with key remapping
- [x] Runtime anti-blink gating now guards against unarmed tracking state and implausible reference ellipses
- [ ] Full Swift-Eye feature parity with the original `mmrotate` pipeline

### 4. TimeLens Integration

- [x] TimeLens wrapper implemented in [interpolation.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/interpolation.py)
- [x] TimeLens input-preparation bridge implemented in [prepare_timelens_inputs.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_timelens_inputs.py)
- [x] CLI entry point added in [interpolate_timelens.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/interpolate_timelens.py)
- [x] Shell wrapper added in [prepare_timelens_inputs.sh](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/scripts/prepare_timelens_inputs.sh)
- [x] Shell wrapper added in [interpolate.sh](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/scripts/interpolate.sh)
- [x] Expected upstream checkpoint location and filename behavior identified from the local `timelens` reference
- [x] Real local session preparation succeeded for a `timelens`-ready folder
- [x] Real `timelens` checkpoint downloaded and executed on a local real session
- [x] End-to-end interpolation validation completed for one real session

### 5. Data Contract And Geometry

- [x] Manifest rows now carry `frame_path`, `interpolated_frame_path`, `event_window`, `ellipse_xywht`, `state6`, `open_extent`, and `antiblink_source`
- [x] Manifest rows now distinguish exact `timelens` relinks through `interpolated_frame_matched`
- [x] External geometry contract fixed to `ellipse_xywht`
- [x] Internal runtime/training contract fixed to `state6`
- [x] Dataset loader updated to prefer interpolated frames when present
- [x] Real manifest generation succeeded on the discovered canonical dataset
- [x] Windows-side raw frame fallback added for canonical reparse-point frame paths
- [x] Manifest rebuild logic now indexes real `timelens` outputs by `timestamp.txt`
- [x] Real manifest generation with actual interpolated outputs succeeded

### 6. Dependency Reduction

- [x] Project source under `swift_hbtxr/` and `tools/` is free of `mmrotate`, `mmdet`, and `mmcv`
- [x] Repository test checks enforce that absence
- [ ] Removal of all legacy wording such as `FACET` from every comment or helper name

### 7. Documentation And Validation

- [x] `docs/README.md` added
- [x] `docs/UPDATE_HISTORY.md` added
- [x] `docs/CONVERSATION_HISTORY.md` added
- [x] `docs/PROGRESS_CHECKLIST.md` added
- [x] Repository-level automated tests pass: `20 passed, 1 skipped`
- [x] Real canonical tree discovered and connected to the project configuration
- [x] Real-data sample loading verified after raw-frame fallback integration
- [x] Real-data smoke `eval` and `infer` succeeded on a subset
- [x] Real `timelens` input preparation succeeded on one local session
- [x] Synthetic validation exists for exact timestamp-based interpolated relinking
- [x] Real interpolated manifest rebuild validated with `4` exact timestamp matches
- [x] Real anti-blink forward-pass validation succeeded after checkpoint import
- [x] Full end-to-end smoke execution completed across all intended CLI entry points on a real-data subset
- [x] Runtime anti-blink gating was revalidated against the smoke checkpoint after policy fixes
- [-] Real-data validation is still pending beyond subset smoke runs and single-session interpolation

### 8. Repository Publication

- [x] Local repository state audited before publication
- [x] Generated outputs and temporary directories excluded through [`.gitignore`](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/.gitignore)
- [x] Git repository initialized on `main`
- [x] `origin` configured for [SWIFT-HBTXR GitHub](https://github.com/CloudeResume/SWIFT-HBTXR)
- [x] Initial publication commit created: `81a296e`
- [x] `origin/main` push succeeded
- [x] Post-push working tree confirmed clean

### 9. Documentation Maintenance

- [x] Implementation history updated in [UPDATE_HISTORY.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/UPDATE_HISTORY.md)
- [x] Conversation decisions updated in [CONVERSATION_HISTORY.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/CONVERSATION_HISTORY.md)
- [x] Plan-vs-progress review updated in [PROGRESS_CHECKLIST.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/PROGRESS_CHECKLIST.md)
- [x] `docs/` remains limited to lightweight history and status tracking rather than generated data products

### 10. Full Smoke And Runtime Stabilization

- [x] Full pipeline smoke workspace created and analyzed under [runs/smoke_full_20260327](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/smoke_full_20260327)
- [x] Full smoke subset built from a real session with interpolation-aware manifest rows
- [x] Anti-blink collapse in smoke-time runtime was reproduced and root-caused
- [x] Runtime arming guard implemented for anti-blink hold-last
- [x] Runtime plausibility guard implemented for anti-blink reference ellipses
- [x] Final guarded runtime behavior matches the no-anti-blink FSM/output distribution on the smoke checkpoint

## Current Assessment

- Strongly completed at repository level:
  - project scaffold
  - core HBTXR integration
  - anti-blink preservation
  - TimeLens wrapper path
  - documentation and tests
- Strongly completed on local data assets:
  - canonical EV-Eye tree connection
  - manifest generation on the real canonical tree
  - one-sample real-data loader verification
- subset smoke runs for train/eval/infer
- real `timelens` interpolation on one local session
- real interpolated manifest rebuild
- real Swift-Eye anti-blink import validation
- full end-to-end smoke execution across all public CLI stages
- runtime anti-blink stabilization for weak smoke-time ellipse predictions
- GitHub publication on `origin/main`
- lightweight project-history documentation
- Partially completed:
  - real-dataset training and inference
- Not yet completed:
  - benchmark-grade performance validation against original Swift-Eye or `HBTXR_v3_0`

## Recommended Next Checklist

- [x] Place the actual `timelens` checkpoint at the expected local path or set it in [configs/base.yaml](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/configs/base.yaml)
- [x] Place the actual Swift-Eye anti-blink checkpoint and set it in [configs/base.yaml](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/configs/base.yaml)
- [x] Run `prepare_dataset.sh` on the intended canonical data tree
- [x] Run `prepare_timelens_inputs.sh` on at least one real session
- [x] Run `interpolate.sh` on at least one real session
- [x] Rebuild manifests with `--interpolated-root`
- [x] Fix manifest rebuilding so it matches real `timelens` outputs by timestamp
- [x] Run `infer.sh` on a real-data subset sequence
- [x] Run stage1 and stage2 smoke training on the real dataset subset
- [x] Initialize Git and publish the repository to GitHub
- [x] Refresh project docs after publication
- [x] Inspect and stabilize `hold_last` behavior against a real anti-blink checkpoint on the smoke runtime path
- [ ] Run full stage1 and stage2 training on the real dataset
- [x] Record real-data results back into this document
