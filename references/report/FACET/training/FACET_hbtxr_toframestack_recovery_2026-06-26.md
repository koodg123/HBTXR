# FACET HBTXR ToFrameStack Recovery - 2026-06-26

## Summary

HBTXR full training on GPU1 stopped during epoch 0 after a DataLoader worker
raised `AssertionError: Data is invalid.` from `ToFrameStack.normalize()`.
EPNet/FACET on GPU0 stayed active and was not interrupted.

## Failure Evidence

Log:

```text
references/report/FACET/HBTXR_full_unet_gpu1_train_2026-06-26.log
```

Observed failure:

```text
AssertionError: Caught AssertionError in DataLoader worker process 0.
EvEye/utils/tonic/functional/ToFrameStack.py
AssertionError: Data is invalid.
```

Runtime impact:

```text
tmux: facet_hbtxr_full_gpu1 was absent
GPU1: no compute process, 18 MiB used
```

## Fix

Updated:

```text
references/codebase/software/FACET/EvEye/utils/tonic/functional/ToFrameStack.py
```

The `normalize()` helper now handles:

- empty timestamp arrays
- non-positive bin counts
- identical or reversed start/end timestamps
- NaN/inf values in timestamps or bounds

For degenerate same-timestamp slices, the function assigns all events to the
last valid interpolation interval instead of terminating the DataLoader.

Also updated:

```text
references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

The HBTXR launcher now sets:

```text
PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_full_unet_pycache
```

This isolates Python bytecode cache for the restarted run and reduces stale
cache ambiguity.

## Validation

Commands run from `/home/kjm26/project/PRJXR/HBTXR`:

```bash
PYTHONPYCACHEPREFIX=/tmp/facet_toframestack_pycache python3 -m py_compile \
  references/codebase/software/FACET/EvEye/utils/tonic/functional/ToFrameStack.py \
  references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py

bash -n references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

Direct `normalize()` smoke cases passed for identical timestamps and non-finite
timestamp bounds.

Dataset smoke loaded these train indices through `DavisEyeEllipseDataset.__getitem__()`:

```text
0
1
95780
95784
95785
100000
1165259
233478
52451
576778
513575
468106
292632
214947
1143716
```

All returned finite input arrays with shape:

```text
(2, 256, 256)
```

## Restart Evidence

HBTXR was restarted on GPU1:

```text
tmux new-session -d -s facet_hbtxr_full_gpu1 \
  'bash /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh'
```

Current restart evidence:

```text
tmux: facet_hbtxr_full_gpu1
PID: 1376090
GPU1 memory: 5920 MiB
Lightning run: HBTXR_full_unet/version_3
```

The last restart marker was found in the HBTXR log. The log slice after that
marker was checked for:

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

No matches were found after the restart marker.

Follow-up check at `2026-06-26 10:32 KST`:

```text
HBTXR_full_unet/version_3
epoch 0, 1648 / 582630 steps
GPU1 utilization: 94%
GPU1 memory: 5943 MiB
post-restart error-string matches: none
```

## Current Status

The overall FACET reproduction goal remains incomplete. HBTXR has restarted
successfully, but it has no full checkpoint or 70-epoch completion marker yet.
The active missing gates remain:

```text
Phase 4 full EPNet checkpoint
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```
