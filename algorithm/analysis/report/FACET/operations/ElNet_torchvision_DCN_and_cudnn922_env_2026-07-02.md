# ElNet Torchvision DCN and cuDNN 9.22 Environment - 2026-07-02

## Summary

ElNet now supports a PyTorch/torchvision DCNv2-compatible implementation and can run with `torch.backends.cudnn.enabled=True` on this server.

## Changes

### Torchvision DCN wrapper

Added:

`references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/ElNet/torchvision_dcn.py`

This implements the original DCNv2-style `DCN` API using:

`torchvision.ops.deform_conv2d`

ElNet import selection now uses:

```bash
ELNET_DCN_IMPL=torchvision   # default
ELNET_DCN_IMPL=native        # native DCNv2 extension
ELNET_DCN_IMPL=conv2d        # non-deformable fallback
```

### cuDNN environment

The venv activation script now prepends:

```bash
/usr/lib/x86_64-linux-gnu
$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13/lib
```

Reason:

- venv bundled `nvidia-cudnn-cu13==9.20.0.48` caused `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` for ordinary `Conv2d`.
- system cuDNN `9.22` works with RTX 5080 and PyTorch `2.12.1+cu130`.

### Launchers

Updated:

`references/report/FACET/operations/run_brat_subject_independent_img64_gpu1_2026-07-02.sh`

Added:

`references/report/FACET/operations/run_elnet_subject_independent_img64_torchvision_dcn_gpu1_2026-07-02.sh`

Both launchers set the cuDNN 9.22 library path explicitly, so they do not depend on interactive activation.

## Verification

With `source .facet-train-venv/bin/activate`:

```text
torch 2.12.1+cu130 cuda 13.0 available True count 2
cudnn enabled True version 92200
conv ok (1, 27, 16, 16)
DCN module EvEye.model.DavisEyeEllipse.ElNet.torchvision_dcn
elnet ok {'hm': (1, 1, 16, 16), 'ab': (1, 2, 16, 16), 'ang': (1, 1, 16, 16), 'trig': (1, 2, 16, 16), 'reg': (1, 2, 16, 16), 'mask': (1, 1, 16, 16)}
```

## Notes

The torchvision DCN path is not the exact original DCNv2 CUDA kernel, but it is a deformable-convolution implementation maintained inside PyTorch/torchvision and avoids the custom extension build path. For exact original ElNet reproduction, use `ELNET_DCN_IMPL=native`; for stable training on this server, use `ELNET_DCN_IMPL=torchvision`.
