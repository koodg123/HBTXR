# FACET EPNet/HBTXR Data and Batch Consistency Note

Date: 2026-06-26

## Summary

EPNet/FACET and HBTXR-DeiT are not using different raw training data in the current full run. Both configs point to the same `DeanDataset_full_unet` root and the same `train`/`val` split names. The visible difference in epoch step count is caused by different per-device batch sizes.

## Evidence

Current full dataset:

```text
root: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
train samples: 1,165,260
val samples: 292,560
split rule: session-order split
```

Current EPNet full config:

```text
config: references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml
train root_path: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
train split: train
train batch_size: 32
drop_last: false
```

Current HBTXR full config:

```text
config: references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml
train root_path: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
train split: train
train batch_size: 4
drop_last: false
```

Expected steps per epoch:

```text
EPNet: ceil(1,165,260 / 32) = 36,415 steps
HBTXR: ceil(1,165,260 / 4)  = 291,315 steps
```

This matches the observed training progress totals. Therefore, the step-count difference is a batch-size effect, not a dataset split mismatch.

## Reproduction Interpretation

The current parallel run is valid for checking whether both models can train on the same generated full dataset and produce comparable validation metrics on the same validation split.

However, it is not fully controlled for optimizer update count:

```text
EPNet effective batch size: 32
HBTXR current effective batch size: 4
```

Because HBTXR uses a DeiT/Transformer-style backbone and consumes more GPU memory, the active run uses a smaller physical batch size. That reduces memory pressure but increases the number of optimizer steps per epoch.

## Added Support for Fair Effective Batch Comparison

`tools/train.py` now forwards `trainer.accumulate_grad_batches` from config to the Lightning Trainer.

New future-run config:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_effbs32.yaml
```

This config keeps HBTXR physical `batch_size: 4` and adds:

```text
trainer.accumulate_grad_batches: 8
```

So the HBTXR effective batch size becomes:

```text
4 * 8 = 32
```

This matches the EPNet full config's physical batch size of 32.

## Operational Decision

The active `DavisEyeEllipse_HBTXR_full_unet.yaml` run was not changed while training is in progress. It should remain the current same-data baseline unless it crashes or is intentionally stopped.

For a stricter EPNet-vs-HBTXR comparison after the current run, launch a separate HBTXR effective-batch run with:

```bash
FACET_DEVICES=1 FACET_DISABLE_CUDNN=1 \
  /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  tools/train.py -c DavisEyeEllipse_HBTXR_full_unet_effbs32.yaml
```

A wait-safe launcher was also added:

```text
references/report/FACET/run_hbtxr_full_unet_effbs32_gpu1_after_baseline_2026-06-26.sh
```

Default behavior:

```text
wait interval: 3600 seconds
wait for DeanDataset_full_unet manifest/progress consistency
wait for baseline HBTXR_full_unet max_epochs=70 completion marker
wait for GPU1 to have no active compute processes
then run DavisEyeEllipse_HBTXR_full_unet_effbs32.yaml on GPU1
```

The wait-safe launcher and evaluation watcher were registered as tmux sessions:

```text
facet_hbtxr_effbs32_gpu1_waiter
facet_hbtxr_effbs32_eval_watcher
```

Launch record:

```text
references/report/FACET/FACET_hbtxr_effbs32_waiter_launch_2026-06-26.md
```

The final comparison report should distinguish:

```text
HBTXR_full_unet: current same-data run, effective batch 4
HBTXR_full_unet_effbs32: future same-data fair effective-batch run, effective batch 32
```
