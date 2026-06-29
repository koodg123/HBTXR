# FACET Full Training Step Checkpoint Recovery

Date: 2026-06-26 11:05 KST

## Purpose

Full EPNet/FACET and HBTXR training are long-running jobs. HBTXR is especially risky because one epoch takes more than 14 hours at the current live throughput. The original full configs only saved checkpoints after validation at epoch boundaries, so a crash before the first epoch finished could lose many hours of work.

This change adds future-run step checkpoints without interrupting the currently running GPU0/GPU1 training jobs.

## Files Updated

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

## Change

Each full-training config now has a second `ModelCheckpoint` callback:

```text
every_n_train_steps: 5000
every_n_epochs: 0
monitor: null
save_top_k: 1
save_last: true
save_on_train_epoch_end: false
```

Step checkpoints are written under:

```text
EPNet:
references/codebase/software/FACET/runs/logs/EPNet_full_unet/step_checkpoints/checkpoints

HBTXR:
references/codebase/software/FACET/runs/logs/HBTXR_full_unet/step_checkpoints/checkpoints
```

The directory intentionally contains a `checkpoints` path component so the existing resume-capable launchers can discover these files with their current pattern:

```text
find "${RUN_ROOT}" -path '*/checkpoints/*.ckpt'
```

## Current-Run Impact

This does not modify the already running trainer instances:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
```

The change applies if a training session later restarts through the existing launcher/watchdog path. It is a future-run recovery guard, not a claim that current live training has produced step checkpoints.

## Validation

YAML parse check:

```text
DavisEyeEllipse_EPNet_full_unet.yaml callbacks 5 model_checkpoints 2
DavisEyeEllipse_HBTXR_full_unet.yaml callbacks 5 model_checkpoints 2
```

Callback instantiation check:

```text
DavisEyeEllipse_EPNet_full_unet.yaml ['ModelCheckpoint', 'ModelCheckpoint', 'TQDMProgressBar', 'Timer', 'LearningRateMonitor']
  val_mean_distance 3 1 0 None
  None 1 0 5000 /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/EPNet_full_unet/step_checkpoints/checkpoints

DavisEyeEllipse_HBTXR_full_unet.yaml ['ModelCheckpoint', 'ModelCheckpoint', 'TQDMProgressBar', 'Timer', 'LearningRateMonitor']
  val_mean_distance 3 1 0 None
  None 1 0 5000 /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/HBTXR_full_unet/step_checkpoints/checkpoints
```

Launcher syntax check:

```text
bash -n run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh: passed
bash -n run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh: passed
```

## Status

The FACET reproduction goal remains incomplete. This recovery guard only reduces future restart loss. Full completion still requires:

```text
full EPNet/FACET checkpoint and completion marker
full HBTXR checkpoint and completion marker
final EPNet paper comparison artifacts
final HBTXR-vs-EPNet comparison artifacts
```
