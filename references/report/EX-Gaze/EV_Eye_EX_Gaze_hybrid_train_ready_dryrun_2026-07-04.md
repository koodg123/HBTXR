# EV-Eye and EX-Gaze Train-Ready Dry Run

Date: 2026-07-04

## Config Validation

| Experiment | GPU | Train | Val | Test | Batch | Workers | Epochs | Optimizer | Scheduler |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| EV_Eye_Hybrid_frame128_event64_subject_independent | 0 | 968873 | 122776 | 366171 | 32 | 4 | 70 | Adam lr=0.001 wd=1e-05 | StepLR step=10 gamma=0.7 |
| EX_Gaze_Hybrid_frame128_event64_subject_independent | 1 | 968873 | 122776 | 366171 | 32 | 4 | 70 | Adam lr=0.001 wd=1e-05 | StepLR step=10 gamma=0.7 |

## Live Loader Check

- Tag: `train_ready`
- Batch size: `4`
- Live mode: `True`

| Split | EV len | EX len | EV frame | EV event | EX patches |
|---|---:|---:|---|---|---|
| train | 968873 | 968873 | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` |
| val | 122776 | 122776 | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` |
| test | 366171 | 366171 | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` |

## Judgment

- Train-ready compact manifests exist for EV-Eye and EX-Gaze.
- The split counts match the HBTXR subject-independent split.
- Training hyperparameters are aligned to HBTXR: batch 32, workers 4, epochs 70, Adam, lr 1e-3, weight decay 1e-5, StepLR step 10 gamma 0.7.
- Live loader first-batch checks pass for train/val/test.
- Training has not been launched.

