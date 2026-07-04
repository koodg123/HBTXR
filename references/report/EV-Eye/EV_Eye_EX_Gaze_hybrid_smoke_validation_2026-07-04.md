# EV-Eye and EX-Gaze Hybrid Export Validation

Date: 2026-07-04

- Tag: `smoke_32`
- Expected count per split: `32`
- EV-Eye root: `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent/smoke_32`
- EX-Gaze root: `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent/smoke_32`

| Split | EV Rows | Frame | Mask | Event | EX Ann Rows | EX Event | EX Patch | Region Range |
|---|---:|---|---|---|---:|---|---|---|
| train | 32 | `[128, 128] uint8` | `[128, 128] uint8` | `[2, 64, 64] uint16` | 32 | `[2, 64, 64] uint16` | `[8, 2, 16, 16] uint16` | 19..46 |
| val | 32 | `[128, 128] uint8` | `[128, 128] uint8` | `[2, 64, 64] uint16` | 32 | `[2, 64, 64] uint16` | `[8, 2, 16, 16] uint16` | 12..46 |
| test | 32 | `[128, 128] uint8` | `[128, 128] uint8` | `[2, 64, 64] uint16` | 32 | `[2, 64, 64] uint16` | `[8, 2, 16, 16] uint16` | 19..42 |

## Judgment

- Validation passed for row counts, frame shape, mask shape, event shape, EX-Gaze patch shape, and patch-region bounds.
- This validates the smoke package only. Full export still needs a separate run and validation pass.

