# EV-Eye and EX-Gaze Hybrid Loader Smoke

Date: 2026-07-04

- Tag: `train_ready_smoke`
- Batch size: `4`
- Live cache mode: `True`
- EV-Eye root: `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent/train_ready_smoke`
- EX-Gaze root: `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent/train_ready_smoke`
- Source root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`

| Split | EV frame | EV mask | EV event | EX frame | EX volume | EX patches | EX regions |
|---|---|---|---|---|---|---|---|
| train | `[4, 1, 128, 128]` | `[4, 128, 128]` | `[4, 2, 64, 64]` | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` | `[4, 8, 4]` |
| val | `[4, 1, 128, 128]` | `[4, 128, 128]` | `[4, 2, 64, 64]` | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` | `[4, 8, 4]` |
| test | `[4, 1, 128, 128]` | `[4, 128, 128]` | `[4, 2, 64, 64]` | `[4, 1, 128, 128]` | `[4, 2, 64, 64]` | `[4, 8, 2, 16, 16]` | `[4, 8, 4]` |

## Judgment

- EV-Eye loader smoke passed for `frame`, `mask`, `event`, and `ellipse128` tensors.
- EX-Gaze loader smoke passed for `frame`, `input_volume`, `event_patches`, `pre_state`, `pupil`, and `sample_regions` tensors.
- This validates the package as a PyTorch-loadable contract. It does not yet validate the final model training loops.

