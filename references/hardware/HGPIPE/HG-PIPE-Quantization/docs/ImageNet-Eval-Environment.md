# ImageNet Evaluation Environment

## Scope

This environment prepares PyTorch GPU validation for the models named in the HG-PIPE paper:

- `deit_tiny_patch16_224`
- `deit_small_patch16_224`
- `vit_tiny_patch16_224`

The paper directly evaluates DeiT-tiny and DeiT-small for HG-PIPE. ViT-tiny appears in the comparison table, so it is included in the registry for complete table coverage.

## Environment Setup

```bash
cd /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization
bash scripts/setup_uv_gpu.sh
```

The setup script creates `.venv` with `uv`, installs the package, installs CUDA PyTorch wheels from:

```text
https://download.pytorch.org/whl/cu130
```

and checks GPU visibility with:

```bash
.venv/bin/python scripts/check_torch_gpu.py
```

## ImageNet Layout

Use the standard torchvision `ImageFolder` layout:

```text
/datasets/imagenet/
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
    ...
```

The script also accepts a root where class folders are directly under `--data`.

## Accuracy Command

```bash
.venv/bin/python -m hgpipe_quantization.eval.imagenet_eval \
  --data /datasets/imagenet \
  --models deit_tiny_patch16_224 deit_small_patch16_224 vit_tiny_patch16_224 \
  --precisions fp32 int8 int4 \
  --batch-size 128 \
  --workers 8 \
  --device cuda \
  --pretrained \
  --output reports/imagenet_accuracy.json
```

## Precision Semantics

- `fp32`: baseline pretrained PyTorch/timm model.
- `int8`: fake-quantized PyTorch evaluation with 8-bit weight and activation simulation.
- `int4`: fake-quantized PyTorch evaluation with 4-bit weight and activation simulation.

This is a PyTorch-side validation environment. It is not a replacement for the original HG-PIPE QAT pipeline or hardware-exported weights.

## Known Limit

`확실하지 않음`: the repository does not include the original QAT checkpoints or the full calibration/training flow. The paper footnote states that QAT weights for DeiT-small or larger were not available, so DeiT-small 4-bit numbers from this PyTorch fake-quant run should be treated as an environment/prototype result, not a paper-equivalent reproduction.

## W4A8 And Report Provenance

The standard config now includes w4a8 in addition to fp32, int8, and int4. Each ImageNet row records evaluation_mode=timm_fake_quant, quantization_flow=fake_quant for quantized rows, paper_equivalent=false, dataset path/split metadata, the eval script name, and the command string. Completion audit treats these reports as PyTorch/timm fake-quant sanity evidence, not artifact-backed HG-PIPE paper-equivalent validation.
