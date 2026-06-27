# Conversation History

## 2026-03-27

### Full Smoke-Test Request

- The user requested a full smoke test across the entire pipeline and asked for the results to be analyzed.

### Full Smoke-Test Outcome

- A dedicated end-to-end smoke workspace was created under [runs/smoke_full_20260327](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/smoke_full_20260327).
- The following stages were executed end-to-end on a small real-data subset:
  - dataset preparation
  - `timelens` input preparation
  - interpolation
  - interpolation-aware manifest rebuilding
  - Swift-Eye anti-blink import
  - stage1 training
  - stage2 training
  - evaluation
  - inference
  - demo trace generation
- The smoke result was treated as functionally successful, but the analysis found a runtime caveat:
  - anti-blink gating forced `hold_last` on nearly all frames for the smoke-trained checkpoint

### Runtime-Fix Follow-Up Request

- The user requested to continue after the smoke analysis rather than stopping at the report.

### Runtime-Fix Decision

- The anti-blink collapse was treated as a runtime policy problem, not a checkpoint-import failure.
- Two corrective decisions were made:
  - only arm anti-blink gating after a stable `track` state has been observed
  - ignore anti-blink gating when the reference ellipse is implausibly small or otherwise invalid

### Runtime-Fix Outcome

- [runtime.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/runtime.py) was patched to add:
  - delayed anti-blink arming
  - ellipse plausibility guarding
- [test_runtime_scheduler.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tests/test_runtime_scheduler.py) was expanded with deterministic runtime tests for the new policy.
- Runtime comparison after the final guard showed:
  - anti-blink enabled with guard: `search 4 / track 4`
  - anti-blink disabled: `search 4 / track 4`
- This was taken as sufficient evidence that the runtime output path no longer freezes under the smoke-trained checkpoint.

### Latest Documentation And Release Request

- The user then requested:
  - update documents and history
  - commit the latest changes
  - push them to GitHub

### Git Publication Request

- The user requested that the project be committed and pushed to:
  - `https://github.com/CloudeResume/SWIFT-HBTXR`

### Git Publication Decision

- The repository was first audited before publication because it was not yet a Git repository and still contained many generated local outputs.
- The publication strategy was fixed as:
  - initialize a fresh Git repository in [SWIFT-HBTXR](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR)
  - keep only source, configs, tests, scripts, and docs under version control
  - exclude generated runs, cached data products, interpolated outputs, and temporary directories through [`.gitignore`](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/.gitignore)

### Git Publication Outcome

- The repository was initialized on `main`.
- `origin` was added for:
  - `https://github.com/CloudeResume/SWIFT-HBTXR.git`
- The initial repository commit was created:
  - `81a296e Initial SWIFT-HBTXR integration`
- The commit was pushed successfully to `origin/main`.
- The working tree was then confirmed clean.

### Current Documentation Request

- After publication, the user requested another documentation refresh covering:
  - updated implementation summary
  - updated conversation summary
  - updated history
  - plan-vs-progress review
  - checklist-style reporting

## 2026-03-26

### Initial Request

- The user requested a new project under `code/SWIFT-HBTXR`.
- The requested integration constraints were:
  - use `Swift-Eye` as the base reference project
  - reuse the proposed model structure and scheduler from `HBTXR_v3_0`
  - keep the `timelens` path external and referenced in place
  - analyze the Swift-Eye code and paper in detail
  - simplify the project structure
  - remove the `mmrotate` dependency
  - refer to the previous `FACET + HBTXR_v3_0` integration in `FECET-HBTXR`

### Planning Decisions

- The repository direction was fixed to:
  - full `mmrotate` removal
  - `HBTXR_v3_0` tracker/FSM reuse
  - `Swift-Eye` anti-blink UNet preservation
  - strong `timelens` integration through a wrapper
  - flat project structure similar to `FECET-HBTXR`
- The model compatibility decision was:
  - do not attempt to transplant the full Swift-Eye detector/tracker backbone into the new HBTXR core
  - limit checkpoint reuse to the anti-blink UNet and threshold-like metadata

### Implementation Decisions

- The main package was flattened under `swift_hbtxr/`.
- The geometry contract was fixed as:
  - external I/O: `ellipse_xywht`
  - internal runtime/training state: `state6(x, y, a, b, u, v)`
- The runtime policy was fixed as:
  - keep the HBTXR `search/track` FSM
  - add anti-blink evaluation
  - add `hold_last` output behavior during blink-like intervals
- The execution surface was reduced to seven entry points:
  - `prepare_dataset`
  - `interpolate`
  - `train_stage1`
  - `train_stage2`
  - `eval`
  - `infer`
  - `demo_sequence`

### Validation Outcome

- The implemented repository passed repository-level tests:
  - `16 passed, 1 skipped`
- Validation limits remaining after the session:
  - no real `timelens` checkpoint run yet
  - no full real-dataset training run yet
  - no real Swift-Eye checkpoint import against an actual released checkpoint in this session

### Current Request

- The user requested a `docs` folder and asked for:
  - update summary
  - conversation summary
  - history update
  - plan-vs-progress review
  - checklist-style output
- This directory and its files were added in response to that request.

### Follow-up Request

- The user then requested to continue with the plan.
- In response, the implementation moved from repository-only integration to real local asset connection.

### Real Asset Findings

- The following real local assets were confirmed:
  - raw EV-Eye root exists at `E:\WSL\Shared\dataset\Eye\EV_Eye\raw_data\Data_davis`
  - canonical EV-Eye tree exists at `E:\WSL\Shared\dataset\Eye\EV_Eye\canonical`
  - canonical session index exists at `canonical/indexes/sessions.jsonl`
- The following expected assets were not found locally in this session:
  - `timelens` refined model checkpoint
  - released Swift-Eye model checkpoint

### Follow-up Implementation Decision

- `prepare_dataset` was extended to consume the existing canonical session index directly.
- A Windows-safe raw-frame fallback path was added because canonical frame symlinks were not readable through the Windows Python runtime used here.
- Real manifests were then generated inside `SWIFT-HBTXR/data/_internal/manifests`.

### Follow-up Validation Outcome

- Real manifest generation on the discovered canonical tree succeeded.
- Real-data sample loading succeeded after the raw-frame fallback patch.
- A small real-data smoke path succeeded for:
  - stage1 training
  - stage2 training
  - evaluation
  - inference
- `timelens` and real Swift-Eye checkpoints are still missing locally, so interpolation and anti-blink checkpoint import remain blocked on external model files.

### Latest Follow-up Request

- The user requested to continue the planned work again instead of stopping at documentation and smoke validation.

### Latest Decision

- The next blocker was narrowed to the missing `timelens` input-preparation bridge.
- The repository was updated so a canonical session can now be materialized into the exact upstream `timelens` layout:
  - `images/*.png`
  - `images/timestamp.txt`
  - `events/*.npz`
- The preparation path was designed to:
  - read the canonical session index
  - choose canonical frames when readable
  - otherwise fall back to raw frames automatically on Windows
  - reuse the canonical `events.npz`

### Latest Validation Outcome

- The new preparation path passed repository tests.
- A real local session was prepared successfully:
  - `user01/left/session_102`
  - `8` frames materialized
  - frame source resolved to raw fallback
- Full repository validation after the new path was added:
  - `17 passed, 1 skipped`
- Full interpolation is still blocked on the missing upstream `timelens` checkpoint.

### Latest Follow-up Decision

- The next implementation gap was identified in manifest rebuilding:
  - the earlier manifest logic assumed `interpolated_root` preserved original frame filenames
  - real upstream `timelens` outputs are indexed by sequential PNGs plus `timestamp.txt`
- The manifest builder was updated to relink interpolated frames by exact timestamp instead.
- A new per-row flag was added so downstream code can tell whether a true interpolated-frame match exists:
  - `interpolated_frame_matched`

### Latest Follow-up Validation

- Synthetic tests confirmed that timestamp-matched interpolated outputs are written back into the manifest correctly.
- A real local probe against the current `timelens_ready/images` tree produced:
  - indexed sessions: `1`
  - indexed frames: `8`
  - exact matches: `0`
- This was treated as correct because that tree contains prepared `timelens` inputs, not actual interpolated outputs.
- Full repository validation after this update:
  - `18 passed, 1 skipped`

### Latest User Request

- The user explicitly requested three real-data actions:
  - obtain the real `attention.bin` and run interpolation
  - rebuild manifests against the real interpolated outputs
  - validate Swift-Eye anti-blink checkpoint import with the actual published checkpoint

### Latest Execution Outcome

- The published `TimeLens` checkpoint and published `Swift-Eye` checkpoint were downloaded from the links referenced in the local READMEs.
- A local `timelens` runner was added because the upstream sample script was not CLI-compatible and hardcoded CUDA/sample paths.
- Real interpolation was executed successfully on `user01/left/session_102`.
- `prepare_dataset --interpolated-root` was re-run against the real interpolation output tree and produced `4` exact timestamp relinks for annotated frames.
- Swift-Eye anti-blink import was first observed to import only transposed-conv/output keys.
- That mismatch was traced to UNet key naming differences rather than shape incompatibility.
- After adding the key remapping layer, the real Swift-Eye import succeeded with:
  - `118` imported UNet keys
  - `0` missing target UNet keys
- A real forward-pass check on the imported anti-blink checkpoint then succeeded.
