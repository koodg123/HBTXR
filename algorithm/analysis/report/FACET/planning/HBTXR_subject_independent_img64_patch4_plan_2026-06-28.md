# HBTXR Subject-Independent Img64 Patch4 Experiment Plan

Date: 2026-06-28

## Goal

Run an additional HBTXR experiment with a subject-independent EV-Eye split and produce the same style of artifacts as `HBTXR_val_motion_eval`:

1. Subject-wise pixel error / IoU distribution.
2. Subject-wise motion distribution.
3. Subject-wise mean / median / P95 / P99 pixel error.
4. Motion-wise mean / median / P95 / P99 pixel error.
5. Annotation precision and pseudo-label noise discussion.

Repeated-run confidence intervals are excluded from this run unless extra multi-seed training is started later.

## Split Decision

The user request says:

```text
1-36 Train / 33-36 Val / 37-48 Test
```

This literal split overlaps subjects 33-36 between train and val, so it is not fully subject-independent. For a leak-free subject-independent experiment, this plan uses the standard corrected split:

```text
Train: subjects 1-32
Val:   subjects 33-36
Test:  subjects 37-48
```

If the literal overlapping split is required later, the split script will support an explicit overlap mode, but the primary experiment will use the leak-free split above.

## Data Strategy

Do not rerun the U-Net full expansion. Reuse the existing pseudo-label/event cache:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

Create a new cached dataset root:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent
```

The new root will contain:

```text
train/cached_data
train/cached_ellipse
val/cached_data
val/cached_ellipse
test/cached_data
test/cached_ellipse
manifest.json
progress_state.json
```

The split is session-contiguous because the original cache was written in session order. A re-splitting script will reconstruct each original session's sample range from `progress_state.json`, then copy event segments and ellipses to the target split according to the session subject.

## Model Configuration

Create a new config:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
```

It will match `DavisEyeEllipse_HBTXR_full_unet_img64_patch4.yaml` except:

- `root_path`: `DeanDataset_full_unet_subject_independent`
- logger name: `HBTXR_subject_independent_img64_patch4`
- default root/log dirs separated from previous runs
- train split: `train`
- validation split: `val`

The model remains:

```text
img_size=64
patch_size=4
output heatmap=16x16
```

## Training Plan

Run HBTXR on one free GPU, preferably GPU1 unless occupied:

```bash
CUDA_VISIBLE_DEVICES=<gpu> FACET_DISABLE_CUDNN=1 \
  /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  references/codebase/software/FACET/tools/train.py \
  --config DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
```

Expected artifacts:

```text
references/codebase/software/FACET/runs/logs/HBTXR_subject_independent_img64_patch4/
```

## Evaluation Plan

Generalize the existing motion-eval script to support arbitrary split names and checkpoint paths.

Evaluate at least:

```text
val split:  subjects 33-36
test split: subjects 37-48
```

Output roots:

```text
references/report/FACET/HBTXR_subject_independent_img64_patch4_val_motion_eval/
references/report/FACET/HBTXR_subject_independent_img64_patch4_test_motion_eval/
```

Final report:

```text
references/report/FACET/HBTXR_subject_independent_img64_patch4_results_2026-06-28.md
```

## Motion Definition

Use the same velocity-based 3-state rule as the prior HBTXR val report:

```text
Saccade  = pseudo-label pupil-center speed > 493 px/s
Fixation = speed <= 493 px/s and session code in {101, 201}
Smooth   = speed <= 493 px/s and session code in {102, 202}
```

Session mapping follows `subject-motion-analysis`:

```text
101: session_1_0_1, saccade/fixation regime, no manual GT
102: session_1_0_2, smooth pursuit
201: session_2_0_1, saccade/fixation regime
202: session_2_0_2, smooth pursuit
```

## Validation Gates

1. Split root exists and loads with `DavisEyeEllipseDataset` for `train`, `val`, and `test`.
2. Subject membership is leak-free:
   - train only 1-32
   - val only 33-36
   - test only 37-48
3. Config parses and a single batch forward pass works.
4. Training creates a best checkpoint monitored by `val_mean_distance`.
5. Val/test motion reports include:
   - predictions with metadata CSV
   - subject error/IoU table
   - subject motion counts table
   - subject pixel error table
   - motion pixel error table
   - label precision floor table
   - pseudo-label noise table
   - figures
   - Markdown summary

## Risks

- Full cache re-splitting copies large memmap data and may take time/disk space.
- `train 1-36 / val 33-36` is ambiguous; this plan intentionally uses leak-free `train 1-32`.
- Saccade samples may be sparse under the velocity threshold and should be interpreted descriptively.
- Pseudo-label noise is measurable only where manual GT exists; session `101` has no manual GT.
