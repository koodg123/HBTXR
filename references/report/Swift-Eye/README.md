# Swift-Eye Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/Swift-Eye`

Swift-Eye is built on MMRotate and includes custom `train_swift_eye` scripts for backbone/neck training and temporal fusion.

Evidence:

- `references/codebase/software/Swift-Eye/README.md:1-7` identifies Swift-Eye and MMRotate.
- `references/codebase/software/Swift-Eye/README.md:27-43` documents data and pretrained weights.
- `references/codebase/software/Swift-Eye/README.md:45-47` documents execution via `test_interpolated.py`.
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/train_backbone_and_neck.py:28-39` registers `GazeDataset` and default config path.
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py:2-29` defines train data settings.

## Original Protocol

Backbone/neck config:

- Dataset type: `GazeDataset`.
- Annotation folder: `annotations`.
- Image folder: `images`.
- Resize scale: `(346, 346)`.
- Samples per GPU: 8.
- Workers per GPU: 8.
- Model: RoITransformer with SwinTransformer backbone and FPN neck.
- Rotated bbox task with one class: `pupil`.

Evidence:

- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py:2-29`
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py:30-79`
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py:83-113`
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py:114-218`

Temporal fusion code expects paired template/search image annotations from a pickle dataframe.

Evidence:

- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py:57-88`
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py:90-134`
- `references/codebase/software/Swift-Eye/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py:149-193`

## HBTXR Contract Compatibility

Compatibility is low-medium.

Why:

- HBTXR data is cached event segments, not MMRotate image folders with DOTA-style rotated bbox annotations.
- Original image scale is 346x346.
- Temporal fusion expects paired template/search records.
- Swift-Eye predicts rotated boxes/masks, not the FACET heatmap/ellipse target format.

Required changes:

1. Export HBTXR samples to image files at 64x64.
2. Convert each FACET ellipse to rotated bbox polygon annotation.
3. Create `training_root`, `validation_root`, and `test_root` with `images` and `annotations`.
4. Change all `img_scale=(346,346)` to `(64,64)`.
5. Recompute normalization statistics for exported event-frame images.
6. If temporal fusion is used, build a dataframe of template/search pairs without crossing session boundaries.
7. Ensure output coordinates are rescaled to the HBTXR 64x64 metric scale.

## Expected Output

Possible after adapter:

- Rotated pupil box.
- Pixel error from box center.
- Approximate ellipse/box IoU.

## Readiness

Status: not ready for immediate training.

Blocking work: rotated bbox image/annotation export and 64x64 MMRotate config.

