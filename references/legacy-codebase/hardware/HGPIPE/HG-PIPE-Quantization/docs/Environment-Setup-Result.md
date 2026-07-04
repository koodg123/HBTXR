# Environment Setup Result

## Date

2026-06-09 KST

## Virtual Environment

- Tool: `uv 0.11.19`
- Path: `.venv`
- Python: CPython 3.12.3

## GPU

- GPU: NVIDIA GeForce RTX 4070 Ti
- Driver CUDA reported by `nvidia-smi`: 13.2
- PyTorch CUDA runtime: 13.0

## Installed Core Packages

- `torch==2.12.0+cu130`
- `torchvision==0.27.0+cu130`
- `torchaudio==2.11.0+cu130`
- `timm==1.0.27`
- `numpy==2.4.6`
- `pandas==3.0.3`
- `tqdm==4.68.1`
- `pyyaml==6.0.3`
- `safetensors==0.7.0`

## Verification Commands

```bash
.venv/bin/python scripts/check_torch_gpu.py
.venv/bin/python -m hgpipe_quantization.eval.imagenet_eval --help
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m hgpipe_quantization.cli verify
```

## Verification Results

- `torch.cuda.is_available()`: `True`
- CUDA device: `NVIDIA GeForce RTX 4070 Ti`
- CUDA capability: `(8, 9)`
- GPU matmul smoke test: passed
- ImageNet evaluation CLI import/help: passed
- HG-PIPE quantization unit tests: 5 tests passed
- HG-PIPE reference quantization verification: `97/97 passed`, `0` mismatches

## ImageNet Dataset Status

The following common candidate paths were checked and no ImageNet directory was found:

- `/datasets/imagenet`
- `/home/user/datasets/imagenet`
- `/mnt/c/datasets/imagenet`
- `/data/imagenet`

Run evaluation after placing ImageNet in standard `ImageFolder` layout or by passing the actual path to `--data`.

## Accuracy Evaluation Command

```bash
cd /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization
.venv/bin/python -m hgpipe_quantization.eval.imagenet_eval \
  --data /path/to/imagenet \
  --models deit_tiny_patch16_224 deit_small_patch16_224 vit_tiny_patch16_224 \
  --precisions fp32 int8 int4 \
  --batch-size 128 \
  --workers 8 \
  --device cuda \
  --pretrained \
  --output reports/imagenet_accuracy.json
```
