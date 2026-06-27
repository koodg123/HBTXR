# HG-PIPE Codebase Quantization Inventory

## Evidence Roots

- Paper: `../Vision Transformer Acceleration with Hybrid-Grained Pipeline.pdf`
- Reference codebase: `../ICCAD24-HG-PIPE`
- HLS quantization kernels:
  - `../ICCAD24-HG-PIPE/src/quant.h`
  - `../ICCAD24-HG-PIPE/src/gelu.h`
  - `../ICCAD24-HG-PIPE/src/layernorm.h`
  - `../ICCAD24-HG-PIPE/src/softmax.h`
- Golden artifacts: `../ICCAD24-HG-PIPE/case/refs`
- Type/range statistics:
  - `../ICCAD24-HG-PIPE/statistics/type.npy`
  - `../ICCAD24-HG-PIPE/statistics/range.npy`

## Artifact Counts

The `case/refs` directory contains 1009 files. Relevant quantization artifact groups:

| Artifact group | Count | Use |
|---|---:|---|
| `*scalars.txt` | 122 | Shift/bias/bound/clamp scale factors |
| `*_q_table_m.txt` | 48 | Q/K/V/A ReQuant tables |
| `*_geluq_table_m.txt` | 12 | fused GeLU-ReQuant tables |
| `*_rsqrt_table_m.txt` | 25 | LayerNorm reciprocal-square-root tables |
| `*_exp_opp_table_m.txt` | 12 | Softmax inverse exponential tables |
| `*_recip_scaled_table_m_one.txt` | 12 | Softmax reciprocal segment 1 |
| `*_recip_scaled_table_m_two.txt` | 12 | Softmax reciprocal segment 2 |
| `*_input.txt` | 223 | golden inputs and intermediate tensors |
| `*_output.txt` | 222 | golden outputs and intermediate tensors |
| `*weight*.txt` | 196 | matmul/linear weights |
| `*bias.txt` | 74 | matmul/linear biases |

## HLS Formula Sources

### ReQuant

`src/quant.h` defines `b`, `s`, and `bound` from `scalars_init[0:3]`, then performs:

```text
cursor = (x + b) >> s
cursor = clamp(cursor, 0, bound)
output = table[cursor]
```

This is implemented in `hgpipe_quantization.ops.table_quantize`.

### LayerNorm

`src/layernorm.h` defines seven scalars:

```text
C_1_m, C_1_s, b, s1, bound, s2, clamp_bits
```

The HLS module has three passes:

1. buffer input and compute rounded integer mean;
2. compute variance sum and lookup `rsqrt_table`;
3. apply `(x - mean) * rsqrt * lnw + lnb`, shift by `s2`, then signed clamp.

This is implemented in `hgpipe_quantization.ops.layernorm_quantize`.

### Softmax

`src/softmax.h` defines fourteen scalars:

```text
b1, s1, bound1,
b2_one, s2_one, bound2_one, b3_one, s3_one,
b2_two, s2_two, bound2_two, b3_two, s3_two,
clamp_bits
```

The HLS module:

1. buffers rows and computes `max(row)`;
2. looks up inverse exponential by `(max - x + b1) >> s1`;
3. accumulates exponentials;
4. selects reciprocal table one or two based on `cursor_one > bound2_one`;
5. multiplies `exp_score * recip`, shifts with the selected segment scale, and unsigned clamps.

This is implemented in `hgpipe_quantization.ops.softmax_quantize`.

## Reconstruction Mapping

| Reference artifact pattern | Reconstructed case kind | Python implementation |
|---|---|---|
| `attn_*_{q,k,v,a}_q_scalars.txt` + `*_q_table_m.txt` + `*q_input/output.txt` | `requant_table` | `table_quantize` |
| `mlp_*_geluq_scalars.txt` + `*_geluq_table_m.txt` + `*_geluq_input/output.txt` | `gelu_requant_table` | `table_quantize` |
| `{attn,mlp}_*_lnq_*`, `head_lnq_*` | `layernorm_rsqrt_table` | `layernorm_quantize` |
| `attn_*_softmaxq_*` | `softmax_segmented_table` | `softmax_quantize` |

## Verified End-to-End Coverage

The CLI discovered 97 quantization cases:

| Case kind | Cases | Elements checked | Mismatches |
|---|---:|---:|---:|
| `requant_table` | 48 | 1,806,336 | 0 |
| `gelu_requant_table` | 12 | 1,806,336 | 0 |
| `layernorm_rsqrt_table` | 25 | 903,360 | 0 |
| `softmax_segmented_table` | 12 | 1,382,976 | 0 |

Total: 5,899,008 elements checked, 0 mismatches.

## Implementation Implications

- Scale factors in HG-PIPE are not single floating-point scale values. They are integer tuple parameters such as `(b, s, bound)` or segmented Softmax tuples that encode PoT-style index/scaling behavior.
- Quantization tables are first-class inference artifacts and should be exported with scalars, not regenerated inside HLS.
- Input/output statistics from `type.npy` and `range.npy` are useful for documenting bit width, signedness, and range coverage, while golden refs prove bit-exact behavior.
- End-to-end quantization validation should compare each reconstructed operator output against `case/refs/*_output.txt`; aggregate accuracy alone would be too weak for this hardware path.

## Uncertainty

`확실하지 않음`: the checkout does not include the full training/QAT/calibration code that originally generated the quantized model artifacts. The implemented package reconstructs and verifies the deployed inference quantization path encoded in the public HLS refs.
