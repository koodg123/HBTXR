# HBTXR Experiment Execution Plan

Date: 2026-06-30

## Scope

This plan organizes the next HBTXR comparison experiments after aligning the
target training contract.

Shared contract:

- Dataset: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Split: train subjects 1-32, val subjects 33-36, test subjects 37-48
- Input resolution: 64x64 for comparable target runs
- Batch size: 32
- DataLoader workers: 4
- Epochs: 70
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 1e-5
- Scheduler: HBTXR-aligned `timm.scheduler.StepLRScheduler`

Device assignment is intentionally omitted from this planning document. Runtime
placement should be recorded in each run log.

## Current Baseline Context

The ongoing HBTXR img128 patch4 run should be left uninterrupted until it
finishes or reaches a user-approved stop point. The latest observed checkpoint
evidence showed completion through at least epoch 36 with a best recorded
`val_mean_distance` of 0.4787.

## Priority Order

1. Retina
   - Reason: smallest model and fastest expected feedback.
   - Purpose: confirm the exported HBTXR/Retina wrapper and bbox target
     formulation are viable.
   - Risk: upstream Retina has optional dependencies for spiking, wandb, and
     ONNX export; the HBTXR runner avoids those paths for `retina_ann`.

2. EPNet/FECET
   - Reason: strongest direct FACET-style CNN baseline for the same split and
     resolution.
   - Purpose: compare HBTXR against a FACET-native CNN/FPN baseline.

3. FACET TennSt
   - Reason: FACET-native temporal center-tracking baseline.
   - Purpose: quantify temporal-only center tracking under the same subject
     split.

4. ERVT
   - Reason: compact recurrent/event-transformer candidate.
   - Purpose: compare recurrent transformer behavior against HBTXR.

5. TENNs-Eye
   - Reason: external temporal spatiotemporal baseline with matching scheduler
     and optimizer settings.
   - Purpose: compare external temporal tracking under HBTXR export.

6. BRAT/CNN_GRU_base
   - Reason: heavier external sequence model.
   - Purpose: compare CNN-GRU sequence tracking against HBTXR.

7. TDTracker
   - Reason: highest measured compute load among prepared targets.
   - Purpose: run as a longer standalone tracking baseline.

## Expected Artifacts

For each run, record:

- exact command
- config path
- output/checkpoint directory
- start and end time
- best validation metric
- final validation metric
- any dependency/runtime issues

Recommended report naming:

- `references/report/HBTXR_<target>_training_status_<date>.md`
- `references/report/HBTXR_<target>_training_log_<date>.log`

## Validation Gates

Before treating a target as comparable:

1. Confirm train/val split matches the HBTXR subject-independent split.
2. Confirm input resolution is 64x64.
3. Confirm optimizer, learning rate, weight decay, scheduler, batch size, epoch
   count, and worker count match the shared contract.
4. Confirm the validation metric is computed in 64x64 pixel coordinates or is
   clearly converted to that scale.
5. Save checkpoint and metric evidence outside tracked Git artifacts.

## Known Caveats

- Retina, TDTracker, ERVT, TENNs-Eye, and BRAT are center or bbox tracking
  baselines. They do not natively predict FACET/HBTXR ellipse axes, angle, or
  mask outputs.
- FLOPs were measured using dummy inputs and `thop.profile`; sequence models
  depend strongly on sequence length.
- Generated checkpoints, event logs, and run directories must remain ignored and
  should not be committed.
