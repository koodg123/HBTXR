"""Bit-exact integer kernels reconstructed from HG-PIPE HLS sources."""

from __future__ import annotations


def clamp(value: int, min_value: int, max_value: int) -> int:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def quantize_clamp(value: int, bits: int, signed: bool) -> int:
    if signed:
        return clamp(value, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return clamp(value, 0, (1 << bits) - 1)


def table_quantize(inputs: list[int], scalars: list[int], table: list[int]) -> list[int]:
    """Reconstruct Quant::do_quant and GeLU::do_gelu."""
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
    """Reconstruct Layernorm::do_layernorm for flattened row-major tensors."""
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
    """Reconstruct Softmax::do_softmax over concatenated attention heads."""
    if len(scalars) != 14:
        raise ValueError(f"softmax expects 14 scalars, got {len(scalars)}")
    if len(inputs) % (tokens * tokens):
        raise ValueError(f"input length {len(inputs)} is not a multiple of tokens*tokens={tokens*tokens}")

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
