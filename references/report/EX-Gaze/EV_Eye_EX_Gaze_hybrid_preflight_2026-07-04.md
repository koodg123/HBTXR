# EV-Eye and EX-Gaze Hybrid Preflight

Date: 2026-07-04

## Scope

- EV-Eye target: GPU0, frame `1 x 128 x 128`, event `2 x 64 x 64`.
- EX-Gaze target: GPU1, frame `1 x 128 x 128`, event `2 x 64 x 64`.
- Split policy: train subjects 1-32, val subjects 33-36, test subjects 37-48.
- Reference evaluation package: `/home/kjm26/project/PRJXR/HBTXR/analysis/RESULTS/HBTXR_subject37_48_error_distribution`.

## Source Dataset

- Source root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Raw Davis root for frame recovery: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis`

| Split | Manifest Samples | Progress Valid | Sessions | Subjects | Skipped | Skip No Ellipse | Skip No Events |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 968873 | 968873 | 256 | 1-32 | 23637 | 23618 | 19 |
| val | 122776 | 122776 | 32 | 33-36 | 1799 | 1790 | 9 |
| test | 366171 | 366171 | 96 | 37-48 | 7292 | 7273 | 19 |

## Cache Contract

| Split | Event dtype | Event shape | Ellipse dtype | Ellipse shape | First index rows |
|---|---|---|---|---|---|
| train | `[('t', '<i8'), ('x', '<i8'), ('y', '<i8'), ('p', '<i8')]` | `(24967082,)` | `[('t', '<i8'), ('x', '<f8'), ('y', '<f8'), ('a', '<f8'), ('b', '<f8'), ('ang', '<f8')]` | `(968873,)` | `[[0, 468], [468, 1764], [1764, 3872]]` |
| val | `[('t', '<i8'), ('x', '<i8'), ('y', '<i8'), ('p', '<i8')]` | `(24972206,)` | `[('t', '<i8'), ('x', '<f8'), ('y', '<f8'), ('a', '<f8'), ('b', '<f8'), ('ang', '<f8')]` | `(122776,)` | `[[0, 851], [851, 2548], [2548, 5045]]` |
| test | `[('t', '<i8'), ('x', '<i8'), ('y', '<i8'), ('p', '<i8')]` | `(24986507,)` | `[('t', '<i8'), ('x', '<f8'), ('y', '<f8'), ('a', '<f8'), ('b', '<f8'), ('ang', '<f8')]` | `(366171,)` | `[[0, 676], [676, 2193], [2193, 4558]]` |

## Frame Recovery Check

| Split | Status | Example Frame Source | First Frame |
|---|---|---|---|
| train | ok | `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user1/left/session_1_0_1/frames` | `000001_1657710786177477.png` |
| val | ok | `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user33/left/session_1_0_1/frames` | `000001_1658285628730570.png` |
| test | ok | `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user37/left/session_1_0_1/frames` | `000001_1658305231950654.png` |

## Evaluation Inputs

- Motion labels: `/home/kjm26/project/PRJXR/HBTXR/analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv`
- Motion label rows: `366190`

| HBTXR Reference File | Exists | Rows |
|---|---:|---:|
| `HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv` | True | 366171 |
| `HBTXR_subject_independent_img64_patch4_test_sample_predictions.csv` | True | 366171 |
| `HBTXR_subject37_48_test_joined_motion_error.csv` | True | 360495 |

## Target Dataset Roots

| Model | Target Root | Exists Now |
|---|---|---:|
| EV-Eye | `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent` | True |
| EX-Gaze | `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent` | True |

## Execution Judgment

- Preflight passed for source split counts, cached event/ellipse availability, and raw frame recovery.
- The target EV-Eye/EX-Gaze derived dataset roots do not exist yet.
- Full export must write under `/home/kjm26/project/dataset/XR/EV_Eye/target_data`, which is outside the repo workspace.
- Main implementation should reuse the existing FACET session-range recovery pattern from `export_hbtxr_subject_independent_for_targets.py`.

## Next Concrete Steps

1. Add an EV-Eye/EX-Gaze export path that emits split-level metadata, frame-128 tensors/images, event-64 tensors, masks, and EX-Gaze JSON/HDF5 annotations.
2. Run a small export smoke test, for example first 32 samples per split.
3. Verify frame/event/label shapes and EX-Gaze patch extraction `8 x 2 x 16 x 16`.
4. After smoke passes, run full export under the target dataset roots.
5. Add model configs and launch EV-Eye on GPU0 and EX-Gaze on GPU1.

