# Artifact-Backed Graph Reconstruction

## Purpose

This path reconstructs HG-PIPE inference from the public ICCAD24-HG-PIPE integer artifacts instead of applying PyTorch fake quantization to a floating-point `timm` model.

The implementation reads these artifact classes from `../ICCAD24-HG-PIPE/case/refs`:

- Integer feature-map inputs and golden outputs.
- Static MatMul weights and biases.
- Dynamic attention QK/RV MatMul weight dumps.
- Quantization scalar triplets and lookup tables.
- LayerNorm affine weights, bias terms, and rsqrt tables.
- Softmax exp and segmented reciprocal tables.

## Implemented Graph Scope

`hgpipe_quantization.graph.ArtifactGraphRunner` now verifies:

- `patch_embed`
- `attn_0` through `attn_11`
- `mlp_0` through `mlp_11`
- `head`

The graph runner composes:

- Patch embedding integer MatMul, cls replacement, and right-shift scaling.
- Static MatMul as `x @ W.T + bias`.
- Dynamic head MatMul for attention QK and RV.
- Q/K/V/A table requantization.
- LayerNorm integer mean/variance, rsqrt lookup, affine, shift, and clamp.
- Softmax max subtraction, exp lookup, reciprocal-table lookup, multiply, shift, and clamp.
- Residual merge as `main + ((residual * RM + 2^(RS-1)) >> RS)`.
- Attention A head merge from head-major heads x T x 64 into O-projection input layout T x 192.

## Verification Result

Command:

```bash
.venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-graph --json reports/graph_verification.json
```

Result:

```text
graph_cases=268/268 passed mismatches=0
```

Breakdown:

| Kind | Cases |
|---|---:|
| graph_attention_block | 12 |
| graph_dynamic_matmul | 24 |
| graph_gelu_table | 12 |
| graph_head | 1 |
| graph_head_merge | 12 |
| graph_head_split | 24 |
| graph_head_transpose | 12 |
| graph_layernorm | 25 |
| graph_matmul | 73 |
| graph_mlp_block | 12 |
| graph_patch_embed | 1 |
| graph_requant_table | 48 |
| graph_softmax_table | 12 |

## Current Limitation

The runner is artifact-backed and bit-exact against the saved reference tensors. The internal attention O-projection bridge no longer reads attn_*_gen_o_matmul_input.txt as the source tensor. It regenerates that tensor from attn_*_aq_output by applying the recovered head-merge layout, then uses the saved artifact only as the verification target.

This is still not a general ImageNet inference model. The remaining bridge is outside the saved graph: converting arbitrary ImageNet images into the integer patch_embed_matmul_input.txt format and validating that preprocessing and calibration path against the original HG-PIPE data generation assumptions.

## Next Step

Connect image preprocessing to the integer patch embedding input format and add an ImageNet evaluation entrypoint for the artifact-backed graph. If original calibration scripts or image-to-patch quantization parameters are unavailable, that gap should be tracked separately from the now bit-exact artifact graph.
