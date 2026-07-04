# EV-Eye Hybrid cu128 Training Launch

Date: 2026-07-04

## Summary

EV-Eye adapted hybrid training was launched on physical GPU0 using a project-local PyTorch CUDA 12.8 environment.

System driver, system CUDA directories, and system cuDNN libraries were not modified.

## Environment

| Item | Value |
|---|---|
| Virtual environment | `.facet-cu128-venv` |
| Python | 3.10.12 |
| PyTorch | `2.11.0+cu128` |
| CUDA wheel | `12.8` |
| cuDNN | `91900` |
| cuDNN enabled | `true` |
| Physical GPU | GPU0 |
| Runtime GPU mapping | `CUDA_VISIBLE_DEVICES=0`, script device `cuda:0` |

## Training Contract

| Item | Value |
|---|---|
| Config | `analysis/configs/EV_Eye_Hybrid_frame128_event64_subject_independent_train_ready.json` |
| Dataset root | `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent/train_ready` |
| Source root | `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent` |
| Train samples | `968873` |
| Val samples | `122776` |
| Batch size | `32` |
| Workers | `4` |
| Epochs | `70` |
| Optimizer | Adam |
| LR | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | StepLR, step size 10, gamma 0.7 |

## Validation Before Launch

- New venv CUDA check passed: GPU count `2`.
- cuDNN-enabled Conv2D/Conv3D smoke test passed.
- EV-Eye dry-run passed with batch 8/workers 0.
- EV-Eye dry-run passed with batch 32/workers 4.

## Launch Command

```bash
tmux new-session -d -s ev_eye_hybrid_gpu0 "bash references/report/EV-Eye/run_ev_eye_hybrid_train_ready_gpu0_2026-07-04.sh"
```

## Runtime Artifacts

| Artifact | Path |
|---|---|
| Launch script | `references/report/EV-Eye/run_ev_eye_hybrid_train_ready_gpu0_2026-07-04.sh` |
| Log file | `references/report/EV-Eye/logs/EV_Eye_Hybrid_frame128_event64_train_ready_gpu0_2026-07-04.log` |
| Output directory | `analysis/RESULTS/EV-Eye_Hybrid_frame128_event64_train_ready` |
| Metadata | `analysis/RESULTS/EV-Eye_Hybrid_frame128_event64_train_ready/run_metadata.json` |
| History | `analysis/RESULTS/EV-Eye_Hybrid_frame128_event64_train_ready/history.csv` |

## Initial Runtime Status

At launch verification, GPU0 showed an active `.facet-cu128-venv/bin/python` process with approximately 3.1 GiB memory and high utilization.

First epoch had not completed at the time of this note.

## Plan Progress

- [x] Diagnose current PyTorch/cuDNN mismatch.
- [x] Keep system driver/CUDA/cuDNN untouched.
- [x] Create project-local cu128 venv.
- [x] Validate cuDNN enabled Conv smoke test.
- [x] Validate EV-Eye dry-run.
- [x] Launch EV-Eye on GPU0.
- [ ] Monitor first epoch completion.
- [ ] Run test/evaluation package after training completes.
