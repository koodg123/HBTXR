# FACET HBTXR Batch-Size Probe - 2026-06-26

## Summary

HBTXR full training was temporarily stopped while it was still early in epoch 0
to test whether a larger batch size could reduce the wall-clock time of the
Phase 4B HBTXR comparison run. EPNet/FACET on GPU0 was left running.

The probe ran on GPU1 through tmux because direct sandbox Python could not
initialize CUDA, while tmux training processes had valid GPU access.

Result:

- `batch_size: 4` is the best stable candidate from the tested set.
- `batch_size: 5` is stable but slower in the short probe.
- `batch_size: 6+` hits CUDA OOM.
- `bf16-mixed` and `16-mixed` did not improve this model on the tested GPU.
- Live training with `batch_size: 4` is active, but current wall-clock epoch ETA
  is still about 15.5 hours because batch throughput drops to about 5.2 it/s.

## Probe Artifacts

Scripts and outputs:

```text
references/codebase/software/FACET/EvEye/utils/scripts/probe_hbtxr_batch_size.py
references/report/FACET/run_hbtxr_batch_probe_gpu1_now_2026-06-26.sh
references/report/FACET/HBTXR_batch_probe_gpu1_2026-06-26.log
references/report/FACET/HBTXR_batch_probe_gpu1_2026-06-26.json
references/report/FACET/HBTXR_batch_probe_gpu1_b5_b6_2026-06-26.json
references/report/FACET/HBTXR_batch_probe_gpu1_bf16_2026-06-26.json
references/report/FACET/HBTXR_batch_probe_gpu1_fp16_2026-06-26.json
```

`probe_hbtxr_batch_size.py` was patched to call `torch.cuda.set_device()` and
initialize the CUDA context before resetting peak memory stats. This makes
non-default GPU probes such as `cuda:1` work reliably in the tmux runtime.

`tools/train.py` was also patched to pass optional `trainer.precision` from the
YAML config into Lightning. The full HBTXR config does not currently enable
mixed precision because the measured probes were worse than fp32.

## Probe Results

Short probe results on GPU1:

| Batch size | Result | Completed steps | Samples/s | Peak allocated MiB | Peak reserved MiB |
|---:|---|---:|---:|---:|---:|
| 2 | ok | 3 | 7.08 | 5063 | 5584 |
| 4 | ok | 3 | 22.28 | 9999 | 10514 |
| 5 | ok | 3 | 12.16 | 12473 | 13718 |
| 6 | OOM | 0 | n/a | 13745 | 14868 |
| 8 | OOM | 0 | n/a | 13322 | 14282 |
| 12 | OOM | 0 | n/a | 14180 | 14276 |
| 16 | OOM | 0 | n/a | 11189 | 12786 |

The short probe favors `batch_size: 4`. `batch_size: 5` is not selected because
it was slower and much closer to the memory limit.

Mixed precision probes:

| Precision | Batch size | Result | Completed steps | Samples/s | Peak allocated MiB | Peak reserved MiB |
|---|---:|---|---:|---:|---:|---:|
| bf16-mixed | 4 | ok | 3 | 10.29 | 12084 | 13288 |
| bf16-mixed | 8 | OOM | 0 | n/a | 12606 | 14670 |
| bf16-mixed | 12 | OOM | 0 | n/a | 11301 | 14662 |
| bf16-mixed | 16 | OOM | 0 | n/a | 13500 | 14658 |
| 16-mixed | 4 | ok | 3 | 10.49 | 11700 | 12528 |
| 16-mixed | 8 | OOM | 0 | n/a | 14360 | 15064 |

The mixed precision probes are not selected. They did not permit a larger batch
and were slower than the fp32 `batch_size: 4` probe.

## Config Change

Modified config:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

Changed:

```text
train batch_size: 2 -> 4
val batch_size:   2 -> 4
```

EPNet/FACET baseline config was not changed.

## Restart Evidence

HBTXR was restarted after the config change:

```text
tmux: facet_hbtxr_full_gpu1
PID: 1460336
GPU1 memory: 11334 MiB
```

Latest restart-sliced log check:

```text
matches: {}
```

Checked needles:

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

Current live progress after restart:

```text
HBTXR_full_unet: epoch 0, 196 / 291315, 5.15 it/s, remaining 15:42:16
```

The reduced step count confirms that the batch-size change took effect.

## Interpretation

The batch-size change reduces the number of optimizer steps per epoch from
`582630` to `291315`. However, the live batch throughput also dropped from
about `10.3 it/s` at batch 2 to about `5.2 it/s` at batch 4, so the current
wall-clock epoch ETA is not materially improved.

Despite that, `batch_size: 4` keeps HBTXR within GPU memory and avoids the
very small-batch training setup. The run should be treated as the current
HBTXR stability/comparison path unless a more substantial model-side speed fix
is introduced later.

After mixed precision probes, HBTXR was restarted again with the selected fp32
`batch_size: 4` configuration:

```text
tmux: facet_hbtxr_full_gpu1
PID: 1483023
GPU1 memory: 11334 MiB
```

## Remaining Gates

The overall FACET reproduction goal remains incomplete:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.
