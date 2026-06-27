# Update History

## 2026-03-27

### End-To-End Smoke Validation

- Executed a full end-to-end smoke validation across the intended public execution surface on a small real-data subset:
  - `prepare_dataset`
  - `prepare_timelens_inputs`
  - `interpolate`
  - `prepare_dataset --interpolated-root`
  - `import_swift_eye_checkpoint`
  - `train_stage1`
  - `train_stage2`
  - `eval`
  - `infer`
  - `demo_sequence`
- Wrote the dedicated smoke workspace under:
  - [runs/smoke_full_20260327](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/smoke_full_20260327)
- Built a real-data smoke subset from `user01/left/session_102`:
  - train: `8`
  - val: `4`
  - test: `8`
- Verified the full data-preparation and interpolation bridge again inside that smoke workspace:
  - base manifest generation from the real canonical tree
  - `timelens` preparation from raw-frame fallback
  - real `attention.bin` interpolation
  - interpolation-aware manifest rebuild with `4` exact timestamp relinks
- Recorded the smoke result summary in:
  - [analysis_summary.json](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/smoke_full_20260327/analysis_summary.json)

### Runtime Stabilization For Anti-Blink Gating

- The first full smoke run revealed a runtime failure mode:
  - anti-blink inference returned `open_extent = 0.0` on all frames for the smoke-trained checkpoint
  - runtime output collapsed to `hold` for `7/8` post-bootstrap frames
- Root-cause analysis showed that:
  - the imported anti-blink checkpoint itself was not broken
  - a direct raw-frame check with the ground-truth ellipse still produced `open_extent ≈ 0.9586`
  - the collapse was caused by low-quality smoke-time ellipse predictions being fed directly into anti-blink gating
- Updated [runtime.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/runtime.py) in two stages:
  - anti-blink gating is only armed after a valid `track` state has been established
  - anti-blink gating is ignored when the reference ellipse is implausibly small or invalid in image space
- Expanded [test_runtime_scheduler.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tests/test_runtime_scheduler.py) with deterministic tests for:
  - arming anti-blink only after tracking is established
  - ignoring anti-blink when the reference ellipse is implausible
- Re-ran runtime comparisons after the guard changes:
  - original anti-blink infer: `search 1 + hold 7`
  - armed-only gating: `search 1 + track 1 + hold 6`
  - armed + plausibility guard: `search 4 + track 4`
  - no anti-blink infer: `search 4 + track 4`
- Wrote the gating-fix comparison summary to:
  - [analysis_gating_fix_summary.json](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/smoke_full_20260327/analysis_gating_fix_summary.json)

### Repository Publication

- Reviewed the repository state before publication and found that [SWIFT-HBTXR](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR) had not yet been initialized as a Git repository.
- Hardened [`.gitignore`](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/.gitignore) so publication excludes generated local artifacts:
  - `runs/`
  - `data/_internal/cache/`
  - `data/_internal/indexes/`
  - `data/_internal/interpolated_real/`
  - `data/_internal/interpolated_real_annotated/`
  - `data/_internal/manifests_interpolated_real/`
  - `data/_internal/manifests_interp_probe/`
  - `data/_internal/timelens_ready/`
  - `data/_internal/timelens_ready_annotated/`
  - `data/splits_interpolated_real/`
  - local `pytest` and temporary probe directories
- Initialized the local Git repository on branch `main`.
- Added `origin` remote:
  - `https://github.com/CloudeResume/SWIFT-HBTXR.git`
- Created the initial publication commit:
  - commit: `81a296e`
  - message: `Initial SWIFT-HBTXR integration`
- Pushed `main` successfully to the GitHub repository and set upstream tracking.
- Verified repository post-push state:
  - current branch: `main`
  - working tree: clean

### Documentation Refresh

- Updated [CONVERSATION_HISTORY.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/CONVERSATION_HISTORY.md) to include the repository publication request and the decisions made during Git publishing.
- Updated [PROGRESS_CHECKLIST.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/PROGRESS_CHECKLIST.md) to reflect the newly completed Git publication and documentation-maintenance scope.
- Updated [README.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/README.md) and this `docs/` tree earlier in the session so the repository now documents:
  - real checkpoint acquisition
  - real interpolation validation
  - real interpolated manifest rebuild
  - real anti-blink checkpoint import
  - GitHub publication state

### Validation Note

- After the runtime stabilization work, full repository validation was re-run:
  - `20 passed, 1 skipped`
- The remaining skipped test is still the Windows environment shell-check case without POSIX `sh` in `PATH`.

## 2026-03-26

### Repository Initialization

- Created a new flattened `SWIFT-HBTXR` project at `code/SWIFT-HBTXR` using `FECET-HBTXR` as the structural starting point.
- Renamed the main package to `swift_hbtxr` and reduced the repository to the intended execution surface:
  - `prepare_dataset`
  - `interpolate`
  - `train_stage1`
  - `train_stage2`
  - `eval`
  - `infer`
  - `demo_sequence`

### Core Integration

- Reused the simplified `HBTXR_v3_0` style model, trainer, scheduler, and runtime path as the core tracker implementation.
- Added [geometry.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/geometry.py) to centralize:
  - `ellipse_xywht <-> state6`
  - transform helpers
  - ellipse mask generation
  - `open_extent` computation
- Updated [dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/dataset.py) so manifests and batches now carry:
  - `interpolated_frame_path`
  - `ellipse_xywht`
  - `state6`
  - `open_extent`
  - `antiblink_source`

### Swift-Eye Preservation Scope

- Added [antiblink.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/antiblink.py) with a standalone anti-blink UNet branch derived from the Swift-Eye behavior.
- Added blink-aware runtime behavior in [runtime.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/runtime.py):
  - anti-blink inference
  - `open_extent`
  - `closed_eye_flag`
  - `hold_last`
  - `runtime/output_mode`
- Added [compat.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/compat.py) for partial Swift-Eye checkpoint import:
  - imports only `unet.*`
  - skips incompatible Swin/FPN/rotated head weights
  - writes an explicit import report

### TimeLens Integration

- Added [interpolation.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/interpolation.py) as a strong wrapper around the upstream `timelens/tests/run_attention.py` entrypoint.
- Added CLI and wrapper scripts for interpolation:
  - [interpolate_timelens.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/interpolate_timelens.py)
  - [interpolate.sh](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/scripts/interpolate.sh)

### Tooling And Documentation

- Added CLI tools:
  - [import_swift_eye_checkpoint.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/import_swift_eye_checkpoint.py)
  - [demo_sequence.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/demo_sequence.py)
- Rewrote [README.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/README.md) to document the simplified 4-step workflow:
  - dataset preparation
  - interpolation
  - training
  - inference
- Added the `docs/` directory for implementation history, conversation log, and progress tracking.

### Validation

- Added or updated tests for:
  - dataset schema and ABI
  - runtime scheduler transitions
  - `hold_last` behavior
  - TimeLens command construction
  - anti-blink checkpoint import
  - repository scaffold constraints
- Repository-level test result in this session:
  - `16 passed, 1 skipped`
- The skipped test was the POSIX `sh` syntax check on a Windows environment without `sh` in `PATH`.

### Real Canonical Tree Integration

- Updated [prepare_dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_dataset.py) so it can consume an existing canonical `indexes/sessions.jsonl` tree even when `session_package.json` is absent.
- Changed the default `prepare_dataset` index output location to the project-local path:
  - `data/_internal/indexes`
- Updated [configs/base.yaml](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/configs/base.yaml) so `canonical_root` points to the discovered real canonical dataset:
  - `E:/WSL/Shared/dataset/Eye/EV_Eye/canonical`

### Windows Real-Data Fallback

- Found that canonical frame paths are stored as reparse points that are not reliably readable from the Windows Python environment used in this session.
- Added `raw_frame_path` emission in [prepare_dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_dataset.py) using the canonical index `raw_session_dir`.
- Updated [dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/dataset.py) to:
  - prefer interpolated frames when available
  - otherwise use canonical frames
  - otherwise fall back to `raw_frame_path`

### Real Data Execution Result

- Successfully generated real manifests from the discovered canonical tree:
  - sessions: `388`
  - train: `5929`
  - val: `844`
  - test: `2238`
- Verified that a real training sample can be loaded end-to-end after the raw frame fallback was added:
  - frame tensor shape: `(1, 256, 256)`
  - event tensor shape: `(2, 256, 256)`
- Executed a real-data smoke pipeline on a small manifest subset:
  - stage1 train
  - stage2 train
  - eval
  - infer
- Smoke artifacts were written under:
  - [runs/real_smoke](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/real_smoke)
- Re-ran repository tests after the real-data integration changes:
  - `16 passed, 1 skipped`

### TimeLens Input Preparation Bridge

- Added [prepare_timelens_inputs.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_timelens_inputs.py) and [prepare_timelens_inputs.sh](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/scripts/prepare_timelens_inputs.sh) to convert canonical EV-Eye sessions into the exact folder layout expected by upstream `timelens`.
- Extended [interpolation.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/interpolation.py) with:
  - session-index loading
  - canonical/raw frame source selection
  - Windows-safe raw-frame fallback for `timelens` input preparation
  - `images/*.png + timestamp.txt + events/0000001.npz` materialization
- Updated [configs/base.yaml](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/configs/base.yaml) with default `timelens` preparation roots and modes.
- Updated [README.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/README.md) so interpolation now documents the required preparation step explicitly instead of pointing `timelens` directly at the canonical session tree.

### TimeLens Preparation Validation

- Added repository tests for the new `timelens` preparation path.
- Verified preparation on a real local session:
  - session: `user01/left/session_102`
  - prepared frames: `8`
  - selected frame source: raw fallback
  - prepared summary: [prepared_summary_real.json](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/interpolate/prepared_summary_real.json)
- Targeted validation after this change:
  - `7 passed, 1 skipped`
- Full repository validation after this change:
  - `17 passed, 1 skipped`

### Interpolated Manifest Relinking

- Updated [prepare_dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_dataset.py) so `--interpolated-root` no longer assumes original frame filenames survive inside `timelens` outputs.
- Added timestamp-based indexing of interpolated session folders through each session `timestamp.txt`.
- Manifest rows now carry:
  - `interpolated_frame_path`
  - `interpolated_frame_matched`
- Exact timestamp matches now resolve to the real interpolated PNG path.
- When an interpolated session exists but the requested annotation timestamp is not present, manifest generation now falls back safely to the source frame path instead of synthesizing a non-existent interpolated path.

### Interpolated Relinking Validation

- Added synthetic tests for timestamp-matched interpolated manifest relinking.
- Probed the current local `timelens_ready/images` tree as an `--interpolated-root` candidate:
  - indexed sessions: `1`
  - indexed frames: `8`
  - exact annotation matches: `0`
- This result is expected because `timelens_ready/images` is only the prepared input tree, not the real interpolated output tree.
- Full repository validation after this change:
  - `18 passed, 1 skipped`

### Real Checkpoint Acquisition

- Downloaded the published `TimeLens` checkpoint to [attention.bin](/E:/WSL/Shared/ETRI_SYNC/HBTXR/references/timelens/refined_model/attention.bin).
- Downloaded the published `Swift-Eye` checkpoint to [swift_eye_weights.pth](/E:/WSL/Shared/ETRI_SYNC/HBTXR/references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/swift_eye_weights.pth).
- Updated [configs/base.yaml](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/configs/base.yaml) so the default interpolation and anti-blink checkpoint fields now point to the real local assets.

### Real TimeLens Execution

- Added [run_timelens_attention.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/run_timelens_attention.py) as a local CLI runner that:
  - wraps the upstream `timelens` modules
  - removes the hardcoded sample-path dependency in the original `tests/run_attention.py`
  - supports CPU execution in the current local PyTorch environment
  - works around the missing `tqdm` package with a local shim
- Updated [interpolation.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/interpolation.py) so the wrapper now launches the local runner instead of the upstream sample script directly.
- Fixed the local runner so `timestamp.txt` includes the final boundary-frame timestamp as well.
- Executed a real interpolation run for `user01/left/session_102` with `frames_to_insert=1` and wrote the output under:
  - [interpolated_real_annotated](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/interpolated_real_annotated)
- Wrote the real interpolation summary to:
  - [summary_real_annotated.json](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/interpolate/summary_real_annotated.json)

### Real Interpolated Manifest Rebuild

- Re-ran [prepare_dataset.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/tools/prepare_dataset.py) against the real interpolation output tree.
- Wrote the rebuilt manifests to:
  - [manifests_interpolated_real](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/manifests_interpolated_real)
- Real relinking result:
  - indexed interpolation sessions: `1`
  - indexed interpolation frames: `7`
  - exact annotation matches: `4`

### Real Swift-Eye Import Validation

- Improved [compat.py](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/swift_hbtxr/compat.py) with real Swift-Eye UNet key remapping so the original `double_conv` naming scheme maps into the flattened local anti-blink UNet.
- Re-ran the actual Swift-Eye import and wrote:
  - [antiblink_detector_real.pt](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/import_swift_eye/antiblink_detector_real.pt)
  - [import_report_real.json](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/import_swift_eye/import_report_real.json)
- Real import result:
  - imported UNet keys: `118`
  - skipped non-UNet or incompatible keys: `257`
  - missing target UNet keys after remap: `0`
- Ran one forward-pass validation on a real frame after import:
  - `open_extent = 0.9724`
  - `closed_eye_flag = 0`
  - `should_hold = False`
