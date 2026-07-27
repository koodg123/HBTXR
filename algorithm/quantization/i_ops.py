"""I-tier golden integer kernels (bit-exact HG-PIPE HLS equivalents).

Ported from the HG-PIPE reference for use as the integer-inference / verification
golden of the HBTXR quantized ViT. Pure-Python, list-based, arbitrary-precision int
(no overflow) — the ground truth the torch int8/int32 deployment kernels (ilayers)
are checked bit-exact against.

- ``table_quantize``: ReQuant / GeLU — cursor = (x + b) >> s, clamp, table lookup.
- ``layernorm_quantize``: integer LayerNorm (integer mean, rsqrt table, affine, clamp).
- ``softmax_quantize``: integer Softmax (max-subtract, exp table, dual reciprocal tables).
- ``requant``: dyadic requantization ``clamp((acc*mult + round) >> shift + zp)``.
- ``int_matmul`` / ``int_conv2d``: pure-int accumulation (weight×act and act×act).
- ``dyadic_params``: best ``(multiplier, shift)`` for a float scale ratio.

The torch deployment kernels (ilayers/int_functional.py) are checked bit-exact
against ``requant`` / ``int_matmul`` / ``int_conv2d`` here; this module stays the
pure-int golden.
"""
from __future__ import annotations

from quantization.scheme import clamp_int as clamp
from quantization.scheme import quantize_clamp


def dyadic_params(scale: float, *, shift_min: int = 1, shift_max: int = 31) -> tuple[int, int, float]:
    """Best ``(multiplier, shift, effective)`` with ``effective = multiplier / 2^shift``.

    Pure-int port of the HG-PIPE ``_dyadic_approx`` (references/): a float scale ratio
    becomes an integer multiply + right shift for the HW requant unit.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    best: tuple[float, int, int, float] | None = None
    for shift in range(shift_min, shift_max + 1):
        multiplier = max(1, round(scale * (1 << shift)))
        effective = multiplier / float(1 << shift)
        error = abs(effective - scale)
        if best is None or error < best[0]:
            best = (error, multiplier, shift, effective)
    assert best is not None
    return int(best[1]), int(best[2]), float(best[3])


def requant(acc: list[int], multiplier: int, shift: int, *, bits: int = 8, signed: bool = True,
            zero_point: int = 0) -> list[int]:
    """Dyadic requant: ``clamp((a*multiplier + round) >> shift + zp)`` per element.

    ``round = 2^(shift-1)`` gives round-half-up before the arithmetic right shift —
    the standard fixed-point requant a hardware multiplier+shifter performs.
    """
    if shift < 0:
        raise ValueError("shift must be non-negative")
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    return [quantize_clamp(((a * multiplier + rnd) >> shift) + zero_point, bits, signed) for a in acc]


def int_matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    """Pure-int ``[M,K] @ [K,N] -> [M,N]`` accumulation (no overflow); the MAC golden."""
    if not a or not b:
        raise ValueError("int_matmul requires non-empty operands")
    m, k, n = len(a), len(b), len(b[0])
    if any(len(row) != k for row in a):
        raise ValueError("inner dimension mismatch")
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        ai = a[i]
        row = out[i]
        for kk in range(k):
            aik = ai[kk]
            if aik == 0:
                continue
            bk = b[kk]
            for j in range(n):
                row[j] += aik * bk[j]
    return out


def _pad_pair(padding: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """``padding`` as ``(ph, pw)`` — an int means the same amount on both axes."""
    if isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError(f"padding must be an int or a (ph, pw) pair, got {padding!r}")
        ph, pw = int(padding[0]), int(padding[1])
    else:
        ph = pw = int(padding)
    if ph < 0 or pw < 0:
        raise ValueError(f"padding must be non-negative, got {padding!r}")
    return ph, pw


def int_conv2d(inp: list[list[list[int]]], weight: list[list[list[list[int]]]], *,
               stride: int = 1, padding: int | tuple[int, int] = 0,
               pad_value: int = 0) -> list[list[list[int]]]:
    """Pure-int conv: ``inp[Cin,H,W]``, ``weight[Cout,Cin,kh,kw]`` -> ``[Cout,Ho,Wo]``.

    A general single-group, dilation-1 conv: the kernel may be rectangular, and the
    stride is free of the kernel size, so this is the golden for the non-overlapping
    PatchEmbed conv (kernel == stride == patch_size), for an overlapping strided conv
    (k=3/s=2), and for the 3x3/pad-1/stride-1 convs of the mask and heatmap heads alike.

    ``pad_value`` is deliberately explicit and has no "obvious" default beyond the
    unpadded case. In an asymmetrically quantized activation grid the integer that
    represents a real zero is the *zero-point*, not 0, so a caller padding an int
    activation must pass ``pad_value=zero_point``; padding with 0 would feed the real
    value ``-zp * s_x`` into every border tap. Only the operand's owner knows which it
    is, so this function does not guess.

    Arbitrary-precision Python ints throughout: the accumulator cannot overflow, which
    is what makes it usable as the reference the fixed-width kernels are checked against.
    """
    ph, pw = _pad_pair(padding)
    cin = len(inp)
    height, width = len(inp[0]), len(inp[0][0])
    if ph or pw:
        row_pad = [pad_value] * pw
        full_row = [pad_value] * (width + 2 * pw)
        inp = [
            [list(full_row) for _ in range(ph)]
            + [row_pad + list(row) + row_pad for row in channel]
            + [list(full_row) for _ in range(ph)]
            for channel in inp
        ]
        height, width = height + 2 * ph, width + 2 * pw
    cout = len(weight)
    kh, kw = len(weight[0][0]), len(weight[0][0][0])
    ho = (height - kh) // stride + 1
    wo = (width - kw) // stride + 1
    out = [[[0] * wo for _ in range(ho)] for _ in range(cout)]
    for oc in range(cout):
        w_oc = weight[oc]
        for oy in range(ho):
            iy0 = oy * stride
            for ox in range(wo):
                ix0 = ox * stride
                acc = 0
                for ic in range(cin):
                    w_ic = w_oc[ic]
                    in_ic = inp[ic]
                    for dy in range(kh):
                        in_row = in_ic[iy0 + dy]
                        w_row = w_ic[dy]
                        for dx in range(kw):
                            acc += in_row[ix0 + dx] * w_row[dx]
                out[oc][oy][ox] = acc
    return out


def table_quantize(inputs: list[int], scalars: list[int], table: list[int]) -> list[int]:
    """Reconstruct HG-PIPE Quant::do_quant and GeLU::do_gelu."""
    if len(scalars) != 3:
        raise ValueError(f"table quantization expects 3 scalars, got {len(scalars)}")
    b, s, bound = scalars
    return [table[clamp((x + b) >> s, 0, bound)] for x in inputs]


def layernorm_quantize(
    inputs: list[int],
    scalars: list[int],
    lnw: list[int],
    lnb: list[int],
    rsqrt_table: list[int],
) -> list[int]:
    """Reconstruct HG-PIPE Layernorm::do_layernorm for flattened row-major tensors."""
    if len(scalars) != 7:
        raise ValueError(f"layernorm expects 7 scalars, got {len(scalars)}")
    if not lnw:
        raise ValueError("layernorm requires lnw weights to infer channel count")

    c = len(lnw)
    if len(inputs) % c:
        raise ValueError(f"input length {len(inputs)} is not divisible by channel count {c}")

    c_1_m, c_1_s, b, s1, bound, s2, clamp_bits = scalars
    outputs: list[int] = []
    for row_start in range(0, len(inputs), c):
        row = inputs[row_start : row_start + c]
        acc = sum(row)
        mean_tmp = acc * c_1_m
        mean_tmp += 1 << (c_1_s - 1)
        mean = mean_tmp >> c_1_s

        var_sum = sum((x - mean) * (x - mean) for x in row)
        cursor = clamp((var_sum + b) >> s1, 0, bound)
        rsqrt = rsqrt_table[cursor]

        for idx, value in enumerate(row):
            affine = (value - mean) * rsqrt * lnw[idx] + lnb[idx]
            shifted = affine >> s2
            outputs.append(quantize_clamp(shifted, clamp_bits, signed=True))
    return outputs


def softmax_quantize(
    inputs: list[int],
    scalars: list[int],
    exp_table: list[int],
    recip_table_one: list[int],
    recip_table_two: list[int],
    *,
    tokens: int = 196,
    heads: int | None = None,
) -> list[int]:
    """Reconstruct HG-PIPE Softmax::do_softmax over concatenated attention heads."""
    if len(scalars) != 14:
        raise ValueError(f"softmax expects 14 scalars, got {len(scalars)}")
    if len(inputs) % (tokens * tokens):
        raise ValueError(f"input length {len(inputs)} is not a multiple of tokens*tokens={tokens * tokens}")

    inferred_heads = len(inputs) // (tokens * tokens)
    if heads is None:
        heads = inferred_heads
    if heads != inferred_heads:
        raise ValueError(f"heads={heads} does not match inferred heads={inferred_heads}")

    (
        b1,
        s1,
        bound1,
        b2_one,
        s2_one,
        bound2_one,
        b3_one,
        s3_one,
        b2_two,
        s2_two,
        bound2_two,
        b3_two,
        s3_two,
        clamp_bits,
    ) = scalars

    outputs: list[int] = []
    per_head = tokens * tokens
    for head in range(heads):
        head_base = head * per_head
        for row_idx in range(tokens):
            row = inputs[head_base + row_idx * tokens : head_base + (row_idx + 1) * tokens]
            max_val = max(row)
            exp_values: list[int] = []
            acc_val = 0
            for value in row:
                cursor1 = clamp((max_val - value + b1) >> s1, 0, bound1)
                exp_value = exp_table[cursor1]
                exp_values.append(exp_value)
                acc_val += exp_value

            cursor_one = (acc_val + b2_one) >> s2_one
            if cursor_one > bound2_one:
                cursor_two = clamp((acc_val + b2_two) >> s2_two, 0, bound2_two)
                recip = recip_table_two[cursor_two]
                b3, s3 = b3_two, s3_two
            else:
                cursor_one = clamp(cursor_one, 0, bound2_one)
                recip = recip_table_one[cursor_one]
                b3, s3 = b3_one, s3_one

            for exp_value in exp_values:
                rel = (exp_value * recip + b3) >> s3
                outputs.append(quantize_clamp(rel, clamp_bits, signed=False))
    return outputs


__all__ = [
    "table_quantize",
    "layernorm_quantize",
    "softmax_quantize",
    "dyadic_params",
    "requant",
    "int_matmul",
    "int_conv2d",
]
