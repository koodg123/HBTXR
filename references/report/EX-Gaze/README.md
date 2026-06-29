# EX-Gaze Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/EX-Gaze`

EX-Gaze is an MMEngine/MMRotate-style project for event/image pupil detection and tracking.

Evidence:

- `references/codebase/software/EX-Gaze/README.md:14-18` lists `mmcv`, `mmdet`, `mmengine`, and `mmrotate`.
- `references/codebase/software/EX-Gaze/README.md:59-68` documents train and end-to-end tracking commands.
- `references/codebase/software/EX-Gaze/dataset/eye_pupil_dataset.py:9-23` defines `EyePupilDataset` metadata including classes and original frame size.
- `references/codebase/software/EX-Gaze/dataset/eye_pupil_dataset.py:24-33` documents image filename, eye region, and pupil annotation fields.
- `references/codebase/software/EX-Gaze/configs/_base_/data_split.py:1-7` shows the current local split file is reduced to user 48, with the intended split commented out.

## Original Protocol

EX-Gaze expects annotation files and event representation HDF5 files.

Examples:

- Annotation file name: `blink_seg_exp5_cont_frame_event_pre_accum_thr50_tracking_dataset.json`.
- Event representation: `event_accum_thr50_pol_event_count.hdf5`.
- Pupil detection config uses mask size `(260, 346)`.
- Train entrypoint: `python train/default_train.py`.

Evidence:

- `references/codebase/software/EX-Gaze/configs/dataset_config/pre_accum_pol_even_count_inter2000/patch_n8_s16/multi_max10_accum50_blink_exp5_ev_overlap_pupil_disp_with_rand_pre.py:7-16`
- `references/codebase/software/EX-Gaze/configs/train_config/full_eye_pupil_detector/mbv3spreX_head_retina_img_pupil_det_eye_region_crop.py:12-23`
- `references/codebase/software/EX-Gaze/configs/train_config/full_eye_pupil_detector/mbv3spreX_head_retina_img_pupil_det_eye_region_crop.py:54-60`

## HBTXR Contract Compatibility

Compatibility is medium-low.

Why it is not direct:

- HBTXR cache is not an MMEngine JSON annotation dataset.
- EX-Gaze currently carries original EV-Eye frame-size assumptions.
- The local split file is not HBTXR subject-independent; it is currently set to user 48 only.
- EX-Gaze has both frame/image pupil detector and event displacement/tracking configs, so the exact sub-model choice must be fixed before training.

Required changes:

1. Export HBTXR train/val/test samples into EX-Gaze annotation JSON.
2. Export 64x64 event representations to HDF5 or PNG, depending on the selected EX-Gaze config.
3. Replace `train_user_list`, `val_user_list`, and `test_user_list` with train 1-32, val 33-36, test 37-48.
4. Change all frame/mask sizes from `(260,346)` or `(346,346)` assumptions to 64x64.
5. Convert FACET ellipse to EX-Gaze pupil annotation `[xc, yc, w, h, theta]`.
6. Add a metric adapter to recover HBTXR-compatible pixel error and optional ellipse IoU.

## Expected Output

Possible after adapter:

- Pupil ellipse prediction.
- Pixel error.
- IoU if the predicted ellipse can be decoded consistently.

## Readiness

Status: not ready for immediate training.

Blocking work: annotation/HDF5 export and config normalization to 64x64.

