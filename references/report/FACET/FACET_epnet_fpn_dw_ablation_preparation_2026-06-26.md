# FACET EPNet fpn_dw Ablation Preparation

Date: 2026-06-26

## Summary

The reproduction plan requires the default full EPNet run with `mode: fpn_2d` and an additional `mode: fpn_dw` ablation for paper correspondence. The current active EPNet full run remains unchanged. This note records the prepared follow-up ablation path.

## Evidence in Code

`EPNet.py` supports both modes:

```text
mode: fpn_2d
mode: fpn_dw
```

The `fpn_dw` path replaces the 2D convolution projection/upsampling blocks with depthwise-separable variants:

```text
conv_dw
Upsample_dw
```

## Added Config

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml
```

Key settings:

```text
dataset root: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
train split: train
val split: val
batch_size: 32
model.type: EPNet
model.mode: fpn_dw
logger.name: EPNet_fpn_dw_full_unet
max_epochs: 70
```

## Added Wait-Safe Launcher

```text
references/report/FACET/run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh
```

Default behavior:

```text
wait interval: 3600 seconds
wait for DeanDataset_full_unet manifest/progress consistency
wait for baseline EPNet_full_unet max_epochs=70 completion marker
wait for GPU0 to have no active compute processes
then run DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml on GPU0
```

## Status Gate

`check_reproduction_status.py` now includes:

```text
Phase 4 EPNet fpn_dw ablation checkpoint
Phase 4 EPNet fpn_dw ablation completion
FACET_epnet_fpn_dw_reproduction_results_*.json
FACET_epnet_fpn_dw_table2_comparison_*.md
```

The prepared evaluation runner is:

```text
references/report/FACET/run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh
```

The prepared evaluation watcher is:

```text
references/report/FACET/watch_epnet_fpn_dw_checkpoints_and_evaluate_2026-06-26.sh
```

It checks every 3600 seconds by default and waits for both:

```text
EPNet_fpn_dw_full_unet checkpoint
EPNet_fpn_dw_full_unet max_epochs=70 completion marker
```

It selects the best metric-bearing checkpoint from:

```text
references/codebase/software/FACET/runs/logs/EPNet_fpn_dw_full_unet
```

and writes:

```text
FACET_epnet_fpn_dw_reproduction_results_2026-06-26.json
FACET_epnet_fpn_dw_table2_comparison_2026-06-26.md
```

## Runtime Note

No active training process was changed. This is preparation for the planned ablation after the current EPNet baseline run.

## Validation

Static checks:

```text
bash -n run_epnet_fpn_dw_full_unet_gpu0_after_baseline_2026-06-26.sh: passed
bash -n run_epnet_fpn_dw_checkpoint_evaluation_2026-06-26.sh: passed
check_reproduction_status.py py_compile: passed
DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml parse: passed
```

Model smoke:

```text
make_model(DavisEyeEllipse_EPNet_fpn_dw_full_unet.yaml model): passed
CPU forward input shape: [1, 2, 256, 256]
output keys: ['ab', 'hm', 'mask', 'reg', 'trig']
```

## Launch Registration

The wait-safe fpn_dw ablation sessions were registered after validation:

```text
facet_epnet_fpn_dw_gpu0_waiter
facet_epnet_fpn_dw_eval_watcher
```

Launch record:

```text
references/report/FACET/FACET_epnet_fpn_dw_waiter_launch_2026-06-26.md
```
