# FACET EPNet Invalid Ellipse Recovery - 2026-06-26

## Summary

EPNet full training on `DeanDataset_full_unet` crashed during epoch 0 because
one transformed ellipse label contained `NaN` values. The crash happened before
target generation, at the `math.ceil()` call used to compute the gaussian
radius.

The dataset loader now treats non-finite or non-positive-axis ellipses as
closed/invalid samples. In that case, the heatmap, mask, regression, axis, and
angle targets remain zero-filled, and `close` is set to `1`.

HBTXR was also restarted after the patch because it uses the same
`DavisEyeEllipseDataset` implementation.

## Failure Evidence

Log:

```text
references/report/FACET/EPNet_full_unet_gpu0_train_2026-06-26.log
```

Observed failure:

```text
ValueError: cannot convert float NaN to integer
```

Failing code path:

```text
EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py
radius = gaussian_radius((math.ceil(b), math.ceil(a)))
```

The crash occurred around:

```text
Epoch 0: 23270 / 36415
```

## Patch

Modified file:

```text
references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py
```

Patch behavior:

- Check `x, y, a, b, an` after downsampling.
- Require all label values to be finite.
- Require ellipse axes `a > 0` and `b > 0`.
- If invalid, set `close = 1` and return zero-filled targets.
- Only call `gaussian_radius()`, `draw_umich_gaussian()`, and `cv2.ellipse()`
  for valid ellipses.

Launcher update:

```text
references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
```

Added:

```text
PYTHONPYCACHEPREFIX=/tmp/facet_epnet_full_unet_pycache
```

## Validation

Syntax checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/facet_dataset_pycache python3 -m py_compile \
  references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py

bash -n references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
```

Dataset smoke:

```text
dataset smoke ok
len 1165260
checked 28 indices
```

The smoke covered fixed indices around previous failures and 20 deterministic
random training indices. It checked the following returned fields for finite
values:

```text
input, hm, ab, ang, trig, mask, reg, ellipse, center
```

## Restart Status

Both full training sessions were restarted after the patch:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
```

GPU process evidence after restart:

```text
GPU0: PID 1428589, python, 4214 MiB
GPU1: PID 1428417, python, 5920 MiB
```

Latest restart slices were checked for:

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

No matches were found after the latest restart markers.

Progress snapshot immediately after restart:

```text
EPNet_full_unet: epoch 0, 396 / 36415, 10.54 it/s
HBTXR_full_unet: epoch 0, 417 / 582630, 9.85 it/s
```

## Remaining Gates

This recovery does not complete the reproduction goal. Remaining gates are:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.
