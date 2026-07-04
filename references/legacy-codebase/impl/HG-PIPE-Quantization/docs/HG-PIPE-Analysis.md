# HG-PIPE Paper and Codebase Analysis

## Summary

HG-PIPE is a hybrid-grained FPGA pipeline for Vision Transformer inference. The key hardware idea is to keep the accelerator pipelined across blocks while also using fine-grained streaming inside blocks where data dependencies allow it. The quantization implementation is tightly coupled to that architecture: low-bit integer tensors reduce DSP and memory pressure, and non-linear or high-precision operations are moved into LUT-backed tables.

This reconstruction focuses on inference-time quantization artifacts available in `../ICCAD24-HG-PIPE`. QAT training and checkpoint export are not present in this checkout; therefore any claim about retraining flow is `확실하지 않음`.

## Paper-Level Quantization Findings

The paper frames GeLU, Softmax, LayerNorm, Exp, Rsqrt, Recip, and ReQuant as hardware-expensive operators when implemented with floating point or fixed-point multipliers. HG-PIPE reduces DSP pressure by:

- using low-bit quantization for tensor values and weights;
- treating ReQuant as a non-linear function that can be implemented by a lookup table;
- fusing GeLU and ReQuant into a single table where possible;
- using PoT-style table index computation so table lookup avoids high-precision multiplication;
- calibrating table ranges jointly so table entries match observed activation distributions;
- splitting Softmax reciprocal lookup into two table ranges because reciprocal has high dynamic range;
- inverting Exp lookup around `max - x` to preserve Softmax numerical stability.

## Codebase Structure

The reference codebase is organized as:

- `src/*.h`: reusable HLS modules for quantization, LayerNorm, Softmax, GeLU, matmul, reshaping, and wrappers.
- `case/*.cpp`: generated per-operator HLS test cases.
- `case/*.template`: templates filled by `step0_case_generation.py`.
- `case/refs/*.txt`: golden tensors, weights, tables, scalars, inputs, and outputs.
- `statistics/type.npy`: per-node signedness and bit width.
- `statistics/range.npy`: observed input/output/internal ranges.
- `statistics/print_statistics.py`: evidence that `.npy` statistics are intended as visible type/range sources.

`ICCAD24-HG-PIPE` is currently treated as read-only evidence. The reconstruction lives entirely in `HG-PIPE-Quantization`.

## Reconstructed Quantization Operators

### Table ReQuant

Evidence: `../ICCAD24-HG-PIPE/src/quant.h`

Formula:

```text
cursor = (x + b) >> s
cursor = clamp(cursor, 0, bound)
y = table[cursor]
```

This covers Q/K/V/A quantizers such as:

- `attn_0_q_q_scalars.txt`
- `attn_0_q_q_table_m.txt`
- `attn_0_qq_input.txt`
- `attn_0_qq_output.txt`

### GeLU-ReQuant

Evidence: `../ICCAD24-HG-PIPE/src/gelu.h`

The GeLU implementation uses the same table cursor formula as ReQuant. It reconstructs fused GeLU plus quantization artifacts such as `mlp_0_geluq_*`.

### LayerNorm Rsqrt Table

Evidence: `../ICCAD24-HG-PIPE/src/layernorm.h`

Integer sequence:

```text
mean = round(sum(x) * C_1_m / 2^C_1_s)
var_sum = sum((x - mean)^2)
rsqrt_cursor = clamp((var_sum + b) >> s1, 0, bound)
rsqrt = rsqrt_table[rsqrt_cursor]
affine = (x - mean) * rsqrt * lnw[c] + lnb[c]
shifted = affine >> s2
y = signed_clamp(shifted, clamp_bits)
```

This covers `attn_*_lnq`, `mlp_*_lnq`, and `head_lnq`.

### Softmax Segmented Table

Evidence: `../ICCAD24-HG-PIPE/src/softmax.h`

Integer sequence:

```text
max_val = max(row)
minus = max_val - x
exp_cursor = clamp((minus + b1) >> s1, 0, bound1)
exp_score = exp_table[exp_cursor]
acc = sum(exp_score)

cursor_one = (acc + b2_one) >> s2_one
if cursor_one > bound2_one:
    recip = recip_table_two[clamp((acc + b2_two) >> s2_two, 0, bound2_two)]
    rel = (exp_score * recip + b3_two) >> s3_two
else:
    recip = recip_table_one[clamp(cursor_one, 0, bound2_one)]
    rel = (exp_score * recip + b3_one) >> s3_one

y = unsigned_clamp(rel, clamp_bits)
```

The refs store three attention heads concatenated as `3 * T * T`, matching `case/ATTN.cpp.template`.

## Implemented Package

Implementation files:

- `hgpipe_quantization/artifacts.py`: parse refs and load `type.npy`/`range.npy`.
- `hgpipe_quantization/ops.py`: bit-exact integer kernels.
- `hgpipe_quantization/pipeline.py`: discover all supported quantization refs and verify outputs.
- `hgpipe_quantization/report.py`: write JSON/Markdown evidence reports.
- `hgpipe_quantization/cli.py`: `list` and `verify` commands.

## Verification Evidence

Commands executed from `HG-PIPE-Quantization`:

```bash
python3 -m unittest discover -s tests
python3 -m hgpipe_quantization.cli list --limit 12
python3 -m hgpipe_quantization.cli verify
```

Results:

- Unit/smoke tests: 5 tests passed.
- Discovered cases: 97.
- Full verification: 97/97 cases passed.
- Elements checked: 5,899,008.
- Total mismatches: 0.

Generated reports:

- `reports/verification.md`
- `reports/verification.json`

## Remaining Limits

- `확실하지 않음`: this checkout does not expose the full QAT/calibration training pipeline that originally produced the weights, scales, and tables.
- The implementation reconstructs deployed inference artifacts, not model training.
- Generated Python `__pycache__` directories may appear after running tests; they are not part of the design.
