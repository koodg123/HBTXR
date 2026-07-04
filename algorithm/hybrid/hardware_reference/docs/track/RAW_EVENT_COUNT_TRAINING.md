# Raw EV-Eye Event-Count Training

Date: 2026-06-10

## Objective

Train the paper-model software stack on `/home/kjm26/project/dataset/EV_Eye` without Frame/Event interpolation:

1. Stage1: frame-based training.
2. Stage2: hybrid event-frame training.
3. Event/frame alignment: raw frame source plus fixed-count event window.

## Current Prepared Artifacts

- Path config: `configs/external/paths/ev_eye_raw_paths.json`
- Stage1 config: `configs/external/mode1_stage1.yaml`
- Stage2 event-count config: `configs/external/mode1_stage2_raw_event_count.yaml`
- One-command wrapper: `scripts/external/run_raw_event_count_train.sh`
- CUDA-wait launcher: `scripts/external/run_raw_event_count_train_when_cuda_ready.sh`
- Contract checker: `scripts/external/check_raw_event_count_contract.py`
- Readiness checker: `scripts/external/check_raw_event_count_training_readiness.py`
- Result checker: `scripts/external/check_raw_event_count_training_result.py`
- CUDA/NVML diagnostic: `scripts/external/diagnose_cuda_nvml.py`
- NVIDIA device-node repair helper: `scripts/external/repair_nvidia_device_nodes_595.sh`
- Manifest root: `data/_internal/manifests/manifest1`

Manifest summary:

- `frame_source`: `original`
- `event_policy`: `fixed_count`
- `event_count_target`: `5000`
- `resize_policy`: `facet_square_direct`
- split counts: train `5929`, val `844`, test `2238`

## Verified Smoke Runs

Stage1 smoke:

- run root: `runs/raw_mode1_stage1_smoke_current_20260610_092642`
- result: train loss `304.4745`, val loss `291.9094`
- checkpoint: `runs/raw_mode1_stage1_smoke_current_20260610_092642/train/best_search_p10.pt`

Stage2 raw event-count smoke:

- run root: `runs/raw_mode1_stage2_event_count_config_smoke_20260610_092747`
- result: train loss `699.2955`, val loss `520.5239`
- checkpoint: `runs/raw_mode1_stage2_event_count_config_smoke_20260610_092747/train/best_track_p10.pt`

Wrapper contract-check smoke:

- run root: `runs/raw_mode1_stage2_contract_wrapper_smoke_20260610_093436`
- contract check: passed for train `5929`, val `844`, test `2238`
- result: train loss `783.1882`, val loss `520.8339`

Wrapper post-result-check smoke:

- run root: `runs/raw_mode1_stage2_postcheck_smoke_20260610_094414`
- checkpoint: `runs/raw_mode1_stage2_postcheck_smoke_20260610_094414/train/best_track_p10.pt`
- result checker: passed with `--allow-limited`

## Full Training Command

Run after CUDA/NVML is healthy:

```bash
sh scripts/external/run_raw_event_count_train.sh
```

The wrapper defaults to `--device cuda:0` and performs a CUDA/NVML preflight
before prepare/training. It requires both PyTorch CUDA visibility and
`nvidia-smi` success. If either check fails, it exits before starting a long CPU
run by mistake.

The wrapper also validates the raw event-count contract before training:

```bash
.venv/bin/python scripts/external/check_raw_event_count_contract.py
```

After a successful non-dry-run training command, the wrapper runs the result
checker automatically. Full runs must not contain limited-batch options. For
smoke/debug runs only, pass `--allow-limited-result-check`.

Check full readiness, including CUDA:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py --require-cuda
```

Collect non-destructive CUDA/NVML diagnostics:

```bash
.venv/bin/python scripts/external/diagnose_cuda_nvml.py
```

If driver/userspace versions are aligned but `/dev/nvidia0`, `/dev/nvidia1`,
and `/dev/nvidiactl` are missing, repair device nodes:

```bash
sh scripts/external/repair_nvidia_device_nodes_595.sh
```

After training, validate stage1/stage2 result artifacts:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_result.py
```

For smoke runs only, allow limited-batch histories:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_result.py --allow-limited \
  --stage1-run runs/raw_mode1_stage1_smoke_current_20260610_092642 \
  --stage2-run runs/raw_mode1_stage2_event_count_config_smoke_20260610_092747
```

Useful dry-run:

```bash
sh scripts/external/run_raw_event_count_train.sh --dry-run
```

Use existing manifests without rebuilding:

```bash
sh scripts/external/run_raw_event_count_train.sh --skip-prepare --device cuda:0
```

Wait for CUDA/NVML, then launch training:

```bash
sh scripts/external/run_raw_event_count_train_when_cuda_ready.sh --timeout-sec 21600 -- --skip-prepare
```

CPU is only intended for short smoke checks:

```bash
sh scripts/external/run_raw_event_count_train.sh --device cpu --skip-prepare --skip-stage1 \
  --stage1-checkpoint runs/raw_mode1_stage1_smoke_current_20260610_092642/train/best_search_p10.pt \
  --stage2-arg=--training.epochs=1 \
  --stage2-arg=--training.max_train_batches=1 \
  --stage2-arg=--training.max_val_batches=1
```

## Completed Full Training

CUDA/NVML readiness was restored on 2026-06-10, and full raw event-count
training completed with existing canonical/manifests:

```bash
sh scripts/external/run_raw_event_count_train.sh --skip-prepare
```

Stage1 full frame-based run:

- run root: `runs/raw_mode1_stage1_event_count_20260610_192838`
- checkpoint: `runs/raw_mode1_stage1_event_count_20260610_192838/train/best_search_p10.pt`
- export: `runs/raw_mode1_stage1_event_count_20260610_192838/train/export/student_export.pt`
- epochs: `14` with early stop
- best validation `metric_search_p10_pct`: `3.0828841947159678` at epoch `6`
- final validation loss: `72.48529473790583`

Stage2 full hybrid raw event-count run:

- run root: `runs/raw_mode1_stage2_event_count_20260610_193719`
- checkpoint: `runs/raw_mode1_stage2_event_count_20260610_193719/train/best_track_p10.pt`
- export: `runs/raw_mode1_stage2_event_count_20260610_193719/train/export/student_export.pt`
- epochs: `15` with early stop
- best validation `metric_track_p10_pct`: `7.2652742097962575` at epoch `1`
- final validation loss: `170.1914188456985`

Final result validation:

```bash
.venv/bin/python scripts/external/check_raw_event_count_training_result.py \
  --project-root /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software \
  --stage1-experiment-name raw_mode1_stage1_event_count \
  --stage2-experiment-name raw_mode1_stage2_event_count
```

Result:

```json
{
  "ok": true,
  "errors": []
}
```

Post-run CUDA readiness, checked outside the Codex sandbox because GPU device
nodes are hidden inside the default sandbox:

- `ready`: `true`
- `torch_cuda_available`: `true`
- `torch_cuda_device_count`: `2`
- `nvidia_smi_returncode`: `0`
- driver: `595.71.05`
- GPUs: two RTX 5080 devices

## Runtime Notes

The full run used `HBTXR_DISABLE_CUDNN=1`, set by
`scripts/external/run_raw_event_count_train.sh` unless the caller overrides it. This
is required on the current host because PyTorch `2.12.0+cu130` reports cuDNN
`92000`, but the shell environment also includes
`LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64`, and CUDA conv2d fails with:

```text
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

Disabling cuDNN lets PyTorch run CUDA conv kernels successfully on the RTX 5080
driver stack.

`scripts/external/run_prepare_and_train.sh` and
`scripts/external/check_raw_event_count_training_result.py` now select latest runs
only when the directory suffix exactly matches `YYYYMMDD_HHMMSS`. This prevents
smoke runs such as `raw_mode1_stage2_event_count_config_smoke_...` from being
mistaken for the full `raw_mode1_stage2_event_count_...` run.

## Stability Fix

Stage2 previously produced NaN gradients in `ellipse_gwd_loss` through `torch.linalg.eigh` backward. `mat_sqrt_2x2` now uses an analytic 2x2 SPD square-root formula, and `tests/test_losses_metrics.py` covers finite backward behavior for isotropic covariance.
