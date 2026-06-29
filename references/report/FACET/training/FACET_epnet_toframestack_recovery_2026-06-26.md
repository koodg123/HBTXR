# FACET EPNet ToFrameStack Recovery - 2026-06-26

## Summary

EPNet/FACET full training on GPU0 crashed during epoch 0 because a full
`DeanDataset_full_unet` event slice had identical start/end timestamps. The
training DataLoader called `to_frame_stack_numpy()` with `start_time ==
end_time`, and `normalize()` raised `AssertionError: Data is invalid.`

The crash was recovered by making `ToFrameStack.normalize()` tolerate valid
zero-duration event slices. EPNet full training was restarted on GPU0 while
HBTXR full training continued on GPU1.

## Patch

Updated file:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/EvEye/utils/tonic/functional/ToFrameStack.py
```

Behavior change:

- If `min >= max`, timestamps are mapped to the last valid interpolation
  interval instead of raising an assertion.
- `nearest` mode now masks out `t >= n_time_bins` to avoid invalid frame-stack
  indexing.

Rationale:

- Full generated event slices can contain real events whose timestamps are all
  identical.
- Dropping the whole training run for that case is too brittle.
- For the current EPNet config, `events_interpolation: causal_linear_ori` and
  `n_time_bins: 1`, mapping identical timestamps to `1 - eps` preserves a
  positive causal-linear accumulation weight.

## Validation

Commands run:

```bash
PYTHONPYCACHEPREFIX=/tmp/facet_toframestack_pycache python3 -m py_compile \
  references/codebase/software/FACET/EvEye/utils/tonic/functional/ToFrameStack.py \
  references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py

PYTHONPATH=. FACET_DISABLE_CUDNN=1 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  # direct DavisEyeEllipseDataset sample check

PYTHONPATH=. FACET_DISABLE_CUDNN=1 NO_ALBUMENTATIONS_UPDATE=1 \
  /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  # host DataLoader worker smoke, 3 shuffled batches
```

Results:

- Python syntax check passed.
- Synthetic identical-timestamp events produced a valid frame stack:
  `(1, 2, 260, 346)`, sum approximately `40.0`.
- Direct full train dataset sample checks passed for:
  `0`, `1`, `15920`, `15925`, `15926`, `50000`, `100000`, and `1165259`.
- Host DataLoader worker smoke passed for 3 shuffled batches:
  - input shape: `(32, 2, 256, 256)`
  - heatmap shape: `(32, 1, 64, 64)`

## Restart State

EPNet/FACET was restarted with:

```text
tmux session: facet_epnet_full_gpu0
script: /home/kjm26/project/PRJXR/HBTXR/references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
GPU: 0
```

HBTXR continued with:

```text
tmux session: facet_hbtxr_full_gpu1
GPU: 1
```

Current runtime evidence after restart:

```text
GPU0: NVIDIA GeForce RTX 5080, 87% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 5943 MiB used
```

Progress snapshot generated at `2026-06-26T01:06:17Z`:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 4379 / 36415 | 12.03% | 11.38 it/s | 46:54 | 0 |
| HBTXR_full_unet | 0 | 82032 / 582630 | 14.08% | 10.61 it/s | 13:06:09 | 0 |

## Remaining Gates

The FACET reproduction status remains incomplete. Remaining required gates:

- Phase 4 full EPNet checkpoint.
- Phase 4 full EPNet training completion marker.
- Phase 4B full HBTXR checkpoint.
- Phase 4B full HBTXR training completion marker.
- Final EPNet/FACET paper comparison artifacts.
- Final HBTXR-vs-EPNet comparison artifacts.

The watcher session `facet_full_eval_watcher` remains active and will evaluate
after both full trainings have completed and checkpoints are available.
