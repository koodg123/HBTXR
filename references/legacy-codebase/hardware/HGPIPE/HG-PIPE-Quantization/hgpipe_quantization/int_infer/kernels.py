"""Torch integer kernels for HG-PIPE artifact inference."""

from __future__ import annotations

from typing import Iterable


def _torch():
    import torch

    return torch


def as_int_tensor(values: Iterable[int], *, device: str | None = None):
    torch = _torch()
    return torch.as_tensor(list(values), dtype=torch.int64, device=device)


def quantize_clamp_tensor(values, bits: int, *, signed: bool):
    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1
    return values.clamp(qmin, qmax)


def table_quantize_tensor(inputs, scalars: Iterable[int], table: Iterable[int]):
    torch = _torch()
    b, s, bound = [int(value) for value in scalars]
    table_tensor = torch.as_tensor(list(table), dtype=torch.int64, device=inputs.device)
    cursor = ((inputs.to(torch.int64) + b) >> s).clamp(0, bound)
    return table_tensor[cursor]


def static_matmul_tensor(inputs, weight, bias):
    x = inputs.to(dtype=weight.dtype).reshape(-1, weight.shape[1])
    return (x @ weight.T + bias.reshape(1, -1)).reshape(-1)


def dynamic_head_matmul_tensor(inputs, weight, *, heads: int, tokens: int, ci: int, co: int):
    x = inputs.reshape(heads, tokens, ci)
    w = weight.reshape(heads, co, ci)
    return _torch().cat([(x[head] @ w[head].T).reshape(-1) for head in range(heads)])


def split_heads_tensor(values, *, tokens: int, channels: int, heads: int):
    x = values.reshape(tokens, channels)
    channels_per_head = channels // heads
    return _torch().cat([x[:, head * channels_per_head : (head + 1) * channels_per_head].reshape(-1) for head in range(heads)])


def transpose_head_values_tensor(values, *, tokens: int, channels: int, heads: int):
    x = values.reshape(tokens, channels)
    channels_per_head = channels // heads
    return _torch().cat([x[:, head * channels_per_head : (head + 1) * channels_per_head].T.reshape(-1) for head in range(heads)])


def merge_heads_tensor(values, *, tokens: int, channels: int, heads: int):
    channels_per_head = channels // heads
    return values.reshape(heads, tokens, channels_per_head).permute(1, 0, 2).reshape(-1)


def residual_merge_tensor(main_values, residual_values, scalars: Iterable[int]):
    rm, rs = [int(value) for value in list(scalars)[:2]]
    return main_values.to(dtype=residual_values.dtype) + ((residual_values * rm + (1 << (rs - 1))) >> rs)


def layernorm_quantize_tensor(inputs, scalars: Iterable[int], lnw, lnb, rsqrt_table):
    torch = _torch()
    c_1_m, c_1_s, b, s1, bound, s2, clamp_bits = [int(value) for value in scalars]
    c = int(lnw.numel())
    rows = inputs.reshape(-1, c).to(torch.int64)
    table = rsqrt_table.to(torch.int64)
    outputs = []
    for row in rows:
        acc = torch.sum(row)
        mean = ((acc * c_1_m) + (1 << (c_1_s - 1))) >> c_1_s
        diff = row - mean
        var_sum = torch.sum(diff * diff)
        cursor = int(torch.clamp((var_sum + b) >> s1, 0, bound).item())
        rsqrt = table[cursor]
        affine = diff * rsqrt * lnw.to(torch.int64) + lnb.to(torch.int64)
        outputs.append(quantize_clamp_tensor(affine >> s2, clamp_bits, signed=True))
    return torch.cat(outputs)


def softmax_quantize_tensor(
    inputs,
    scalars: Iterable[int],
    exp_table,
    recip_table_one,
    recip_table_two,
    *,
    tokens: int,
    heads: int,
):
    torch = _torch()
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
    ) = [int(value) for value in scalars]

    exp_table = exp_table.to(torch.int64)
    recip_table_one = recip_table_one.to(torch.int64)
    recip_table_two = recip_table_two.to(torch.int64)
    rows = inputs.reshape(heads, tokens, tokens).to(torch.int64)
    outputs = []
    for head in range(heads):
        for row in rows[head]:
            max_value = torch.max(row)
            cursor1 = ((max_value - row + b1) >> s1).clamp(0, bound1)
            exp_values = exp_table[cursor1]
            acc_value = torch.sum(exp_values)
            cursor_one = (acc_value + b2_one) >> s2_one
            if int(cursor_one.item()) > bound2_one:
                cursor_two = int(torch.clamp((acc_value + b2_two) >> s2_two, 0, bound2_two).item())
                recip = recip_table_two[cursor_two]
                b3, s3 = b3_two, s3_two
            else:
                recip = recip_table_one[int(torch.clamp(cursor_one, 0, bound2_one).item())]
                b3, s3 = b3_one, s3_one
            outputs.append(quantize_clamp_tensor((exp_values * recip + b3) >> s3, clamp_bits, signed=False))
    return torch.cat(outputs)

