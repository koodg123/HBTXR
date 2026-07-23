# SWIFT-HBTXR

`SWIFT-HBTXR` is a flattened integration project that keeps the `HBTXR_v3_0` tracker and scheduler core, preserves the `Swift-Eye` anti-blink UNet and `timelens` interpolation path, and removes the `mmrotate` stack entirely.

## What This Project Keeps

- `HBTXR_v3_0` style ABI: `frame`, `event`, `prev_state`
- Search/Track runtime FSM from the HBTXR line
- `Swift-Eye` inspired anti-blink UNet with `open_extent` and `hold_last`
- `timelens` wrapper for interpolation without copying the upstream source
- Flat package layout under `swift_hbtxr/`

## Project Layout

```text
SWIFT-HBTXR/
  configs/
    base.yaml
    stage1_search.yaml
    stage2_hybrid.yaml
  scripts/
    prepare_dataset.sh
    prepare_timelens_inputs.sh
    interpolate.sh
    train_stage1.sh
    train_stage2.sh
    eval.sh
    infer.sh
    demo_sequence.sh
  tools/
    prepare_dataset.py
    prepare_timelens_inputs.py
    interpolate_timelens.py
    train.py
    eval.py
    infer.py
    demo_sequence.py
    import_swift_eye_checkpoint.py
  swift_hbtxr/
    dataset.py
    event_repr.py
    geometry.py
    interpolation.py
    antiblink.py
    model.py
    scheduler.py
    losses.py
    metrics.py
    trainer.py
    runtime.py
    compat.py
    io.py
```

## Environment

Core environment:

```sh
cd /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- The core project targets the newer HBTXR-style PyTorch environment.
- `timelens` can live in a separate Python environment. The wrapper accepts `--timelens-python` so the interpolation step can call that environment explicitly.
- `mmrotate`, `mmdet`, and `mmcv` are not used in this repository.

## 1. Prepare Dataset

Build manifests from a canonical session tree:

```sh
sh scripts/prepare_dataset.sh \
  --canonical-root /mnt/e/WSL/Shared/dataset/Eye/EV_Eye/canonical \
  --split-scheme exgaze_with_val
```

Current default in `configs/base.yaml` points to the discovered local canonical tree:

- `E:/WSL/Shared/dataset/Eye/EV_Eye/canonical`

When canonical frame files are stored as unreadable reparse points on Windows, the dataset loader falls back to the original raw frame path automatically when it is present in the manifest.

Optional interpolation-aware manifest:

```sh
sh scripts/prepare_dataset.sh \
  --canonical-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/canonical \
  --interpolated-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/interpolated
```

When `--interpolated-root` points to a real `timelens` output tree, `prepare_dataset` now reads each session `timestamp.txt` and links annotations to exact interpolated frames by timestamp instead of assuming the original frame filename still exists there.

Manifest rows include:

- `frame_path`
- `interpolated_frame_path`
- `interpolated_frame_matched`
- `event_window`
- `ellipse_xywht`
- `state6`
- `open_extent`
- `antiblink_source`

## 2. Interpolate With TimeLens

`timelens` expects a session folder with:

- `images/*.png`
- `images/timestamp.txt`
- `events/*.npz`

Prepare that structure from the canonical EV-Eye tree first:

```sh
sh scripts/prepare_timelens_inputs.sh \
  --canonical-root /mnt/e/WSL/Shared/dataset/Eye/EV_Eye/canonical \
  --session-key user01/left/session_102 \
  --max-frames 64 \
  --prepared-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/timelens_ready \
  --overwrite
```

This step automatically falls back to `raw_data/.../frames` when canonical frame paths are unreadable from the Windows runtime.

Then call the SWIFT-HBTXR local runner, which wraps the upstream `timelens` modules with a CLI-compatible entrypoint:

```sh
sh scripts/interpolate.sh \
  --image-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/timelens_ready/images \
  --event-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/timelens_ready/events \
  --output-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/interpolated \
  --timelens-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/references/timelens \
  --timelens-checkpoint /abs/path/to/attention.bin \
  --timelens-python /abs/path/to/python
```

If you already have a manifest subset, you can also derive the session list from it:

```sh
python tools/prepare_timelens_inputs.py \
  --manifest data/_internal/manifests/smoke_val_manifest.jsonl \
  --max-frames 64
```

Current validated local assets:

- [attention.bin](/E:/WSL/Shared/ETRI_SYNC/HBTXR/references/timelens/refined_model/attention.bin)
- [swift_eye_weights.pth](/E:/WSL/Shared/ETRI_SYNC/HBTXR/references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/swift_eye_weights.pth)
- [antiblink_detector_real.pt](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/runs/import_swift_eye/antiblink_detector_real.pt)
- [manifests_interpolated_real](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/data/_internal/manifests_interpolated_real)

## 3. Import Swift-Eye Anti-Blink Weights

Only the UNet branch is imported. Non-compatible backbone and rotated-head keys are skipped and reported.

```sh
python tools/import_swift_eye_checkpoint.py \
  --checkpoint /abs/path/to/swift_eye_checkpoint.pth \
  --output-checkpoint runs/import_swift_eye/antiblink_detector.pt \
  --output-report runs/import_swift_eye/import_report.json
```

## 4. Train

Stage 1:

```sh
sh scripts/train_stage1.sh \
  --device cuda
```

Stage 2:

```sh
sh scripts/train_stage2.sh \
  --device cuda \
  --stage1-checkpoint /abs/path/to/best_search_p10.pt
```

## 5. Evaluate And Infer

Evaluate:

```sh
sh scripts/eval.sh \
  --checkpoint /abs/path/to/best_track_p10.pt \
  --output runs/eval/stage2_metrics.json
```

Infer with optional anti-blink import:

```sh
sh scripts/infer.sh \
  --checkpoint /abs/path/to/best_track_p10.pt \
  --antiblink-checkpoint runs/import_swift_eye/antiblink_detector.pt \
  --output-jsonl runs/inference/runtime_trace.jsonl \
  --output-summary runs/inference/runtime_summary.json
```

The runtime trace reports:

- `runtime_state`
- `runtime_reason`
- `runtime_fsm_reason`
- `output_mode`
- `ellipse_xywht`
- `open_extent`
- `should_hold`

## 6. Demo Sequence

```sh
sh scripts/demo_sequence.sh \
  --checkpoint /abs/path/to/best_track_p10.pt \
  --session-key user01/left/session_101 \
  --max-samples 64
```

## Verification

```sh
python -m pytest
```

## Documentation

- [docs/README.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/README.md)
- [docs/UPDATE_HISTORY.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/UPDATE_HISTORY.md)
- [docs/CONVERSATION_HISTORY.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/CONVERSATION_HISTORY.md)
- [docs/PROGRESS_CHECKLIST.md](/E:/WSL/Shared/ETRI_SYNC/HBTXR/code/SWIFT-HBTXR/docs/PROGRESS_CHECKLIST.md)
