# ElNet DCNv2 Native Build - 2026-07-02

## Summary

ElNet original reproduction requires the native DCNv2 CUDA operator. The original `CharlesShang/DCNv2` source was obtained and vendored into:

`references/codebase/software/FACET/third_party/DCNv2`

The source was patched only for local build compatibility with PyTorch 2.12.1 + CUDA 13.0:

- expose package as `DCNv2.dcn_v2`
- build only the ElNet-required `dcn_v2_forward/backward` CUDA path
- exclude unused CPU and PSROI pooling build targets
- replace removed TH/THC APIs with ATen CUDA stream, cuBLAS, and CUDACachingAllocator
- force CUDA build when `CUDA_HOME` exists, because sandboxed build processes cannot see GPUs

## Environment

- Python: `.facet-train-venv`
- PyTorch: `2.12.1+cu130`
- Torch CUDA: `13.0`
- GPU: NVIDIA GeForce RTX 5080
- CUDA compiler wheel: `nvidia-cuda-nvcc==13.0.88`
- CUDA headers/runtime alignment:
  - `nvidia-cuda-cccl==13.0.85`
  - `nvidia-cuda-crt==13.0.88`
  - `nvidia-cuda-runtime==13.0.96`
  - `nvidia-nvvm==13.0.88`

The CUDA wheel runtime provides `libcudart.so.13` but not the linker symlink, so this symlink was added inside the venv:

`$VENV/lib/python3.10/site-packages/nvidia/cu13/lib/libcudart.so -> libcudart.so.13`

## Build Command

```bash
UV_CACHE_DIR=/tmp/uv-cache \
CUDA_HOME=/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/lib/python3.10/site-packages/nvidia/cu13 \
PATH=/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/lib/python3.10/site-packages/nvidia/cu13/bin:$PATH \
TORCH_CUDA_ARCH_LIST='12.0' \
uv pip install \
  --python /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  --no-build-isolation \
  --force-reinstall \
  /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/third_party/DCNv2
```

## Verification

Native DCNv2 import and CUDA forward/backward passed:

```text
torch 2.12.1+cu130 cuda 13.0 available True count 2
native _ext .../.facet-train-venv/lib/python3.10/site-packages/_ext.cpython-310-x86_64-linux-gnu.so
forward (2, 16, 16, 16) loss 0.07575025409460068
grad ok True True True
```

FACET ElNet integration also passed with native DCNv2:

```text
DCN module DCNv2.dcn_v2
{'hm': (1, 1, 16, 16), 'ab': (1, 2, 16, 16), 'ang': (1, 1, 16, 16), 'trig': (1, 2, 16, 16), 'reg': (1, 2, 16, 16), 'mask': (1, 1, 16, 16)}
```

## Runtime Caveat

This server still has a cuDNN sublibrary mismatch for ordinary `Conv2d` when cuDNN is enabled:

`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`

Therefore ElNet/DCNv2 smoke tests were run with:

```python
torch.backends.cudnn.enabled = False
```

DCNv2 itself is now native CUDA, not the previous Conv2d fallback. The cuDNN workaround affects regular convolution backend selection, not the DCNv2 custom CUDA extension.
