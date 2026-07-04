# FACET Full Training ETA and HBTXR Batch Risk - 2026-06-26

## Current Snapshot

Generated from the live training logs at approximately `2026-06-26 10:32 KST`.

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 22345 / 36415 | 61.36% | 11.44 it/s | 20:29 | 0 |
| HBTXR_full_unet | 0 | 1648 / 582630 | 0.28% | 10.41 it/s | 15:30:24 | 0 |

Both active training log tails were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

No matches were found in the HBTXR log slice after the latest restart marker.
The cumulative HBTXR log still contains the older `version_2` failure record, so
post-restart checks must slice from the latest `manifest found; starting HBTXR`
marker.

## ETA Estimate

These estimates are rough because validation time, checkpoint writing, and
learning-rate/runtime effects are not included.

EPNet:

```text
current epoch elapsed:    32:32
current epoch remaining:  20:29
estimated epoch time:     53:01
70-epoch train-only ETA:  ~61.9 hours
```

HBTXR:

```text
current epoch elapsed:    02:38
current epoch remaining:  15:30:24
estimated epoch time:     15:33:02
70-epoch train-only ETA:  ~45.4 days
```

HBTXR was restarted after a `ToFrameStack.normalize()` assertion in
`version_2`; the current estimate is based on the restarted `version_3` run.
The early rate is still warming up and may change.

## HBTXR Risk

The HBTXR full config currently uses:

```text
batch_size: 2
patch_size: 4
img_size: 256
output heatmap resolution: 64x64
```

The `patch_size: 4` setting is tied to the target heatmap resolution, so
increasing patch size directly is not a safe speed fix. It would change the
model output resolution and may break the loss/target shape contract.

The safer optimization candidate is a larger HBTXR batch size. Based on current
GPU memory usage, a larger batch may fit, but this must be tested with real
forward/backward passes before changing the live full run.

## Prepared Probe

Probe script:

```text
references/codebase/software/FACET/EvEye/utils/scripts/probe_hbtxr_batch_size.py
```

Wait script:

```text
references/report/FACET/run_hbtxr_batch_probe_gpu0_when_free_2026-06-26.sh
```

The wait script is fail-closed:

- If GPU0 has a compute app, it waits.
- If GPU state cannot be read, it treats GPU0 as busy.
- It does not start the probe while EPNet is actively training.

## Recommendation

Do not interrupt the current EPNet run. EPNet should reach the first full
validation/checkpoint much sooner than HBTXR.

For HBTXR, the current run is useful as a stability signal, but it is unlikely
to be a practical 70-epoch path at `batch_size: 2`. After EPNet reaches a
checkpoint or GPU0 becomes free, run the HBTXR batch-size probe. If a larger
batch is stable and improves samples/sec, restart HBTXR from scratch with the
larger batch before investing multiple days into the current HBTXR run.

## Remaining Reproduction Gates

The goal is still incomplete. Required gates remain:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.

## Update: 2026-06-26 10:38 KST

The `10:32 KST` ETA snapshot was superseded by an EPNet restart. EPNet
`version_3` crashed on an invalid transformed ellipse label:

```text
ValueError: cannot convert float NaN to integer
```

The dataset loader now skips target drawing for non-finite or non-positive-axis
ellipses and returns zero-filled targets with `close = 1`. HBTXR was restarted
as well so both models run against the same patched dataset code.

Fresh post-restart snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 396 / 36415 | 1.09% | 10.54 it/s | 56:58 | 0 |
| HBTXR_full_unet | 0 | 417 / 582630 | 0.07% | 9.85 it/s | 16:25:09 | 0 |

The HBTXR batch-size risk remains unchanged. The current run is primarily a
stability signal until the batch-size probe can run without interrupting EPNet.

## Update: 2026-06-26 10:44 KST

The HBTXR batch-size probe was run on GPU1 after stopping only the HBTXR
session. EPNet/FACET continued on GPU0.

Probe outcome:

```text
batch_size=4: ok, 22.28 samples/s in short probe, peak_reserved=10514 MiB
batch_size=5: ok but slower, 12.16 samples/s, peak_reserved=13718 MiB
batch_size=6: CUDA OOM
batch_size=8/12/16: CUDA OOM
```

Selected config:

```text
DavisEyeEllipse_HBTXR_full_unet.yaml
train batch_size: 4
val batch_size: 4
```

Fresh live ETA after restart:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 4620 / 36415 | 12.69% | 11.41 it/s | 46:25 | 0 |
| HBTXR_full_unet | 0 | 273 / 291315 | 0.09% | 5.24 it/s | 15:25:40 | 0 |

The larger HBTXR batch halves the step count, but the current live batch rate
also drops by roughly half. Therefore this change improves memory/runtime
confidence more than wall-clock ETA.

## Update: 2026-06-26 10:49 KST

Mixed precision was tested to see whether HBTXR could use a larger batch:

```text
bf16-mixed batch_size=4: ok but slower than fp32 batch_size=4
bf16-mixed batch_size=8/12/16: CUDA OOM
16-mixed batch_size=4: ok but slower than fp32 batch_size=4
16-mixed batch_size=8: CUDA OOM
```

`tools/train.py` now supports optional `trainer.precision` from YAML, but the
full HBTXR config keeps fp32 because the probes did not improve throughput or
memory headroom.

HBTXR was restarted with the selected fp32 `batch_size: 4` configuration.

Fresh live ETA:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 7619 / 36415 | 20.92% | 11.45 it/s | 41:55 | 0 |
| HBTXR_full_unet | 0 | 196 / 291315 | 0.07% | 5.15 it/s | 15:42:16 | 0 |
