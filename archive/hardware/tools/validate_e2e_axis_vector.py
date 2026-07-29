#!/usr/bin/env python3
"""Spec-driven software reference for the reduced E2E AXI vector gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "refs" / "e2e_axis_vector_spec.json"
DEFAULT_HGPIPE_CONTRACT = Path(__file__).resolve().parents[1] / "refs" / "hgpipe_lut_math_contract.json"


@dataclass(frozen=True)
class ReducedE2EConfig:
    height: int = 256
    width: int = 256
    patch: int = 16
    embed: int = 192
    state: int = 6
    blocks: int = 1
    active_tokens: int = 4
    patch_grid_h: int = 1
    patch_grid_w: int = 4
    heads: int = 3
    head_dim: int = 64
    ff_dim: int = 32
    weight_bits: int = 4
    weight_int: int = 2
    acc_scale: int = 16
    group_channels: int = 6


@dataclass(frozen=True)
class E2EMathConfig:
    mlp_gelu: str = "compact_lut"
    attention_softmax: str = "compact_lut"
    layernorm: str = "compact_lut"
    geluq_input_scale: int = 16
    geluq_output_scale: int = 4
    softmax_input_scale: int = 16
    softmax_prob_scale: int = 4
    layernorm_input_scale: int = 16
    layernorm_output_scale: int = 4
    layernorm_bias_shift: int = 33
    hgpipe_contract: str = str(DEFAULT_HGPIPE_CONTRACT)


@dataclass(frozen=True)
class E2EVectorReference:
    expected_raw: list[int]
    runtime_state: int
    pooled: list[float]
    head_float: list[float]
    live_weight_bit: int
    nonzero_counts: dict[str, int]
    notes: list[str]


def require_int(entry: dict[str, Any], key: str) -> int:
    value = entry[key]
    if not isinstance(value, int):
        raise ValueError(f"{key} must be int in {entry}")
    return value


def load_spec(path: Path) -> tuple[ReducedE2EConfig, E2EMathConfig, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("format") != "hgtxr_reduced_e2e_axis_vector_spec_v1":
        raise ValueError(f"Unsupported E2E vector spec format in {path}")
    cfg = ReducedE2EConfig(**payload.get("config", {}))
    math_cfg = E2EMathConfig(**payload.get("math", {}))
    if math_cfg.mlp_gelu not in {"compact_lut", "hgpipe_geluq"}:
        raise ValueError(f"unsupported math.mlp_gelu: {math_cfg.mlp_gelu}")
    if math_cfg.attention_softmax not in {"compact_lut", "hgpipe_softmaxq"}:
        raise ValueError(f"unsupported math.attention_softmax: {math_cfg.attention_softmax}")
    if math_cfg.layernorm not in {"compact_lut", "hgpipe_lnq"}:
        raise ValueError(f"unsupported math.layernorm: {math_cfg.layernorm}")
    if (
        math_cfg.geluq_input_scale <= 0
        or math_cfg.geluq_output_scale <= 0
        or math_cfg.softmax_input_scale <= 0
        or math_cfg.softmax_prob_scale <= 0
        or math_cfg.layernorm_input_scale <= 0
        or math_cfg.layernorm_output_scale <= 0
    ):
        raise ValueError("math scales must be positive")
    pattern = payload["pattern"]
    if cfg.state != len(pattern["head_raw"]):
        raise ValueError("state must match head_raw length")
    if cfg.embed % cfg.group_channels != 0:
        raise ValueError("embed must be divisible by group_channels")
    if cfg.heads * cfg.head_dim != cfg.embed:
        raise ValueError("heads * head_dim must match embed")
    if cfg.blocks < 1:
        raise ValueError("blocks must be at least one")
    if cfg.patch_grid_h * cfg.patch_grid_w < cfg.active_tokens:
        raise ValueError("patch grid must cover active_tokens")
    if cfg.patch_grid_h * cfg.patch > cfg.height or cfg.patch_grid_w * cfg.patch > cfg.width:
        raise ValueError("patch grid exceeds configured frame size")
    validate_entries(cfg, pattern)
    return cfg, math_cfg, pattern


def validate_raw(raw: int, cfg: ReducedE2EConfig) -> None:
    low = -(1 << (cfg.weight_bits - 1))
    high = (1 << (cfg.weight_bits - 1)) - 1
    if raw < low or raw > high:
        raise ValueError(f"Q{cfg.weight_bits} raw value out of range: {raw}")


def validate_entries(cfg: ReducedE2EConfig, pattern: dict[str, Any]) -> None:
    for raw in pattern["patch_raw"] + pattern["head_raw"]:
        validate_raw(int(raw), cfg)
    for key in [
        "ln_gamma_raw",
        "ln_beta_raw",
        "qkv_diag_raw",
        "wo_group_raw",
        "mlp_w1_group_raw",
        "mlp_w2_diag_raw",
    ]:
        validate_raw(int(pattern.get(key, 0)), cfg)
    for entry in pattern.get("patch_extra_entries", []):
        channel = require_int(entry, "channel")
        raw = require_int(entry, "raw")
        if channel < 0 or channel >= cfg.embed:
            raise ValueError(f"patch channel out of range: {entry}")
        validate_raw(raw, cfg)
    for entry in pattern.get("qkv_extra_entries", []):
        in_ch = require_int(entry, "input")
        out_ch = require_int(entry, "output")
        if not (0 <= in_ch < cfg.embed and 0 <= out_ch < cfg.embed):
            raise ValueError(f"qkv entry out of range: {entry}")
        for key in ["q", "k", "v"]:
            validate_raw(require_int(entry, key), cfg)
    for entry in pattern.get("wo_extra_entries", []):
        in_ch = require_int(entry, "input")
        out_ch = require_int(entry, "output")
        if not (0 <= in_ch < cfg.embed and 0 <= out_ch < cfg.embed):
            raise ValueError(f"wo entry out of range: {entry}")
        validate_raw(require_int(entry, "raw"), cfg)
    for entry in pattern.get("mlp_w1_extra_entries", []):
        in_ch = require_int(entry, "input")
        hidden = require_int(entry, "hidden")
        if not (0 <= in_ch < cfg.embed and 0 <= hidden < cfg.ff_dim):
            raise ValueError(f"w1 entry out of range: {entry}")
        validate_raw(require_int(entry, "raw"), cfg)
    for entry in pattern.get("mlp_w2_extra_entries", []):
        hidden = require_int(entry, "hidden")
        out_ch = require_int(entry, "output")
        if not (0 <= hidden < cfg.ff_dim and 0 <= out_ch < cfg.embed):
            raise ValueError(f"w2 entry out of range: {entry}")
        validate_raw(require_int(entry, "raw"), cfg)
    for entry in pattern.get("head_extra_entries", []):
        out_idx = require_int(entry, "output")
        channel = require_int(entry, "channel")
        if not (0 <= out_idx < cfg.state and 0 <= channel < cfg.embed):
            raise ValueError(f"head entry out of range: {entry}")
        validate_raw(require_int(entry, "raw"), cfg)


def q4_to_float(raw: int, cfg: ReducedE2EConfig) -> float:
    mask = (1 << cfg.weight_bits) - 1
    raw &= mask
    sign = 1 << (cfg.weight_bits - 1)
    if raw & sign:
        raw -= 1 << cfg.weight_bits
    frac = cfg.weight_bits - cfg.weight_int
    return raw / float(1 << frac)


def q_data(value: float) -> float:
    # HgtxrDataT in the Q4W/Q8A config is ap_fixed<8,4,AP_TRN,AP_SAT>.
    clipped = max(-8.0, min(7.9375, value))
    return int(clipped * 16.0) / 16.0


def frame_pixel(y: int, x: int) -> float:
    return float((x + y) & 0xFF) / 128.0


def rsqrt_lut(variance_mean: float) -> float:
    table = [
        11.313708, 0.695971, 0.492126, 0.401842,
        0.347985, 0.311264, 0.284190, 0.263166,
        0.246063, 0.231908, 0.219682, 0.209100,
        0.199647, 0.191337, 0.183974, 0.177386,
        0.171444, 0.166047, 0.161117, 0.156591,
        0.152416, 0.148550, 0.144956, 0.141604,
        0.138470, 0.135531, 0.132770, 0.130171,
        0.127719, 0.125402, 0.123210, 0.125000,
    ]
    clipped = max(0.0078125, min(64.0, variance_mean))
    return table[max(0, min(31, int(clipped * 0.484375)))]


def exp_lut(shifted_score: float) -> float:
    table = [
        1.000000, 0.586646, 0.344154, 0.201897,
        0.118442, 0.069483, 0.040762, 0.023916,
        0.014025, 0.008230, 0.004827, 0.002831,
        0.001661, 0.000974, 0.000572, 0.000335,
    ]
    opposite_delta = -shifted_score if shifted_score < 0 else 0.0
    return table[max(0, min(15, int(opposite_delta * 1.875)))]


def gelu_lut(x: float) -> float:
    table = [
        -0.0001, -0.0020, -0.0151, -0.0715,
        -0.1543, -0.1588, -0.0701, 0.1614,
        0.5714, 1.1299, 1.7588, 2.3788,
        2.9952, 3.6038, 4.2020, 4.8000,
    ]
    scaled = q_data((q_data(x) + q_data(4.0)) * q_data(1.875))
    return q_data(table[max(0, min(15, int(scaled)))])


def hgpipe_cursor_table(value: int, b: int, s: int, bound: int, table: list[int]) -> int:
    cursor = max(0, min(bound, (value + b) >> s))
    return int(table[cursor])


def load_hgpipe_geluq_tables(path: Path) -> list[tuple[dict[str, int], list[int]]]:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "hgtxr.hgpipe_lut_math_contract.v1":
        raise ValueError(f"unsupported HG-PIPE contract schema in {path}")
    tables = []
    for layer in range(12):
        key = f"gelu_quantized_mlp{layer}"
        cursor = contract["cursor_contracts"][key]
        table = [int(x) for x in contract["tables"][key]]
        tables.append(({name: int(value) for name, value in cursor["scalars"].items()}, table))
    return tables


def load_hgpipe_softmax_tables(path: Path) -> list[dict[str, Any]]:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "hgtxr.hgpipe_lut_math_contract.v1":
        raise ValueError(f"unsupported HG-PIPE contract schema in {path}")
    tables = []
    for layer in range(12):
        key = f"softmax_attn{layer}"
        cursor = contract["cursor_contracts"][key]
        tables.append({
            "scalars": {name: int(value) for name, value in cursor["scalars"].items()},
            "exp": [int(x) for x in contract["tables"][f"{key}_exp"]],
            "recip_one": [int(x) for x in contract["tables"][f"{key}_recip_one"]],
            "recip_two": [int(x) for x in contract["tables"][f"{key}_recip_two"]],
        })
    return tables


def load_hgpipe_layernorm_tables(path: Path) -> dict[str, list[dict[str, Any]]]:
    contract = json.loads(path.read_text())
    if contract.get("schema") != "hgtxr.hgpipe_lut_math_contract.v1":
        raise ValueError(f"unsupported HG-PIPE contract schema in {path}")
    tables: dict[str, list[dict[str, Any]]] = {"attn": [], "mlp": []}
    for prefix in tables:
        for layer in range(12):
            key = f"layernorm_{prefix}{layer}"
            cursor = contract["cursor_contracts"][key]
            tables[prefix].append({
                "scalars": {name: int(value) for name, value in cursor["scalars"].items()},
                "rsqrt": [int(x) for x in contract["tables"][f"{key}_rsqrt"]],
            })
    return tables


def gelu_with_math(x: float,
                   block_idx: int,
                   math_cfg: E2EMathConfig,
                   geluq_tables: list[tuple[dict[str, int], list[int]]] | None) -> float:
    if math_cfg.mlp_gelu == "compact_lut":
        return gelu_lut(x)
    if geluq_tables is None:
        raise ValueError("HG-PIPE GeLUQ tables are required for hgpipe_geluq math")
    scalars, table = geluq_tables[block_idx % 12]
    raw = int(q_data(x) * float(math_cfg.geluq_input_scale))
    quantized = hgpipe_cursor_table(raw, scalars["b"], scalars["s"], scalars["bound"], table)
    return q_data(quantized / float(math_cfg.geluq_output_scale))


def softmax_probs_with_math(scores: list[float],
                            block_idx: int,
                            math_cfg: E2EMathConfig,
                            softmax_tables: list[dict[str, Any]] | None) -> list[float]:
    row_max = max(scores)
    if math_cfg.attention_softmax == "compact_lut":
        probs = [exp_lut(score - row_max) for score in scores]
        row_sum = sum(probs) or 1.0
        return [prob / row_sum for prob in probs]
    if softmax_tables is None:
        raise ValueError("HG-PIPE softmax tables are required for hgpipe_softmaxq math")
    entry = softmax_tables[block_idx % 12]
    scalars = entry["scalars"]
    exp_scores = []
    for score in scores:
        opposite = q_data(row_max - score)
        opposite_raw = max(0, int(opposite * float(math_cfg.softmax_input_scale)))
        exp_scores.append(
            hgpipe_cursor_table(
                opposite_raw,
                scalars["b1"],
                scalars["s1"],
                scalars["bound1"],
                entry["exp"],
            )
        )
    acc = sum(exp_scores) or 1
    recip_one_raw = (acc + scalars["b2_one"]) >> scalars["s2_one"]
    in_table_two = recip_one_raw > scalars["bound2_one"]
    if in_table_two:
        recip = hgpipe_cursor_table(
            acc,
            scalars["b2_two"],
            scalars["s2_two"],
            scalars["bound2_two"],
            entry["recip_two"],
        )
        rel_b = scalars["b3_two"]
        rel_s = scalars["s3_two"]
    else:
        recip = hgpipe_cursor_table(
            acc,
            scalars["b2_one"],
            scalars["s2_one"],
            scalars["bound2_one"],
            entry["recip_one"],
        )
        rel_b = scalars["b3_one"]
        rel_s = scalars["s3_one"]
    out = []
    for exp_score in exp_scores:
        rel = ((exp_score * recip) + rel_b) >> rel_s
        quantized = max(0, min((1 << scalars["clamp_bits"]) - 1, rel))
        out.append(q_data(quantized / float(math_cfg.softmax_prob_scale)))
    return out


def zeros2(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def set_matrix(matrix: list[list[float]], row: int, col: int, raw: int, cfg: ReducedE2EConfig) -> None:
    matrix[row][col] = q4_to_float(raw, cfg)


def build_weights(cfg: ReducedE2EConfig, pattern: dict[str, Any]) -> dict[str, Any]:
    patch = [0 for _ in range(cfg.embed)]
    for channel, raw in enumerate(pattern["patch_raw"]):
        patch[channel] = int(raw)
    for entry in pattern.get("patch_extra_entries", []):
        patch[entry["channel"]] = entry["raw"]

    q = zeros2(cfg.embed, cfg.embed)
    k = zeros2(cfg.embed, cfg.embed)
    v = zeros2(cfg.embed, cfg.embed)
    wo = zeros2(cfg.embed, cfg.embed)
    w1 = zeros2(cfg.embed, cfg.ff_dim)
    w2 = zeros2(cfg.ff_dim, cfg.embed)
    head = zeros2(cfg.state, cfg.embed)

    diag = int(pattern["qkv_diag_raw"])
    for c in range(cfg.embed):
        set_matrix(q, c, c, diag, cfg)
        set_matrix(k, c, c, diag, cfg)
        set_matrix(v, c, c, diag, cfg)
        set_matrix(wo, c, c % cfg.group_channels, int(pattern["wo_group_raw"]), cfg)
        set_matrix(w1, c, c % cfg.group_channels, int(pattern["mlp_w1_group_raw"]), cfg)
    for h in range(cfg.group_channels):
        set_matrix(w2, h, h, int(pattern["mlp_w2_diag_raw"]), cfg)
    for out_idx, raw in enumerate(pattern["head_raw"]):
        set_matrix(head, out_idx, out_idx, int(raw), cfg)

    for entry in pattern.get("qkv_extra_entries", []):
        row = entry["input"]
        col = entry["output"]
        set_matrix(q, row, col, entry["q"], cfg)
        set_matrix(k, row, col, entry["k"], cfg)
        set_matrix(v, row, col, entry["v"], cfg)
    for entry in pattern.get("wo_extra_entries", []):
        set_matrix(wo, entry["input"], entry["output"], entry["raw"], cfg)
    for entry in pattern.get("mlp_w1_extra_entries", []):
        set_matrix(w1, entry["input"], entry["hidden"], entry["raw"], cfg)
    for entry in pattern.get("mlp_w2_extra_entries", []):
        set_matrix(w2, entry["hidden"], entry["output"], entry["raw"], cfg)
    for entry in pattern.get("head_extra_entries", []):
        set_matrix(head, entry["output"], entry["channel"], entry["raw"], cfg)

    return {"patch": patch, "q": q, "k": k, "v": v, "wo": wo, "w1": w1, "w2": w2, "head": head}


def conv_patch_embedding(cfg: ReducedE2EConfig, weights: dict[str, Any]) -> list[list[float]]:
    tokens = zeros2(cfg.active_tokens, cfg.embed)
    for gy in range(cfg.patch_grid_h):
        for gx in range(cfg.patch_grid_w):
            token = gy * cfg.patch_grid_w + gx
            if token >= cfg.active_tokens:
                continue
            for c in range(cfg.embed):
                patch_weight = q4_to_float(weights["patch"][c], cfg)
                acc = 0.0
                for py in range(cfg.patch):
                    for px in range(cfg.patch):
                        y = gy * cfg.patch + py
                        x = gx * cfg.patch + px
                        acc += frame_pixel(y, x) * patch_weight
                pos = ((c & 15) - 8) / 64.0
                tokens[token][c] = acc / 32.0 + pos
    return tokens


def compact_layernorm(tokens: list[list[float]],
                      gamma_raw: int,
                      beta_raw: int,
                      cfg: ReducedE2EConfig) -> list[list[float]]:
    out = zeros2(cfg.active_tokens, cfg.embed)
    gamma = q4_to_float(gamma_raw, cfg)
    beta = q4_to_float(beta_raw, cfg)
    for t in range(cfg.active_tokens):
        mean = sum(tokens[t]) / float(cfg.embed)
        var = sum((x - mean) * (x - mean) for x in tokens[t]) / float(cfg.embed)
        inv_std = rsqrt_lut(var)
        for c in range(cfg.embed):
            out[t][c] = (tokens[t][c] - mean) * inv_std * gamma + beta
    return out


def layernorm_with_math(tokens: list[list[float]],
                        gamma_raw: int,
                        beta_raw: int,
                        cfg: ReducedE2EConfig,
                        math_cfg: E2EMathConfig,
                        lnq_tables: dict[str, list[dict[str, Any]]] | None,
                        block_idx: int,
                        mlp_layernorm: bool) -> list[list[float]]:
    if math_cfg.layernorm == "compact_lut":
        return compact_layernorm(tokens, gamma_raw, beta_raw, cfg)
    if lnq_tables is None:
        raise ValueError("HG-PIPE LayerNormQ tables are required for hgpipe_lnq math")
    prefix = "mlp" if mlp_layernorm else "attn"
    entry = lnq_tables[prefix][block_idx % 12]
    scalars = entry["scalars"]
    out = zeros2(cfg.active_tokens, cfg.embed)
    for t in range(cfg.active_tokens):
        raw_values = [
            int(q_data(tokens[t][c]) * float(math_cfg.layernorm_input_scale))
            for c in range(cfg.embed)
        ]
        sum_raw = sum(raw_values)
        mean_raw = (
            (sum_raw * scalars["C_1_m"]) +
            (1 << (scalars["C_1_s"] - 1))
        ) >> scalars["C_1_s"]
        variance_sum = sum((x - mean_raw) * (x - mean_raw) for x in raw_values)
        rsqrt = hgpipe_cursor_table(
            variance_sum,
            scalars["b"],
            scalars["s1"],
            scalars["bound"],
            entry["rsqrt"],
        )
        lnb = int(beta_raw) * (1 << math_cfg.layernorm_bias_shift)
        for c, raw in enumerate(raw_values):
            rel = ((raw - mean_raw) * rsqrt * int(gamma_raw) + lnb) >> scalars["s2"]
            quantized = max(-4, min(3, rel))
            out[t][c] = q_data(quantized / float(math_cfg.layernorm_output_scale))
    return out


def project_qkv(norm: list[list[float]], weights: dict[str, Any], cfg: ReducedE2EConfig) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    q = zeros2(cfg.active_tokens, cfg.embed)
    k = zeros2(cfg.active_tokens, cfg.embed)
    v = zeros2(cfg.active_tokens, cfg.embed)
    for t in range(cfg.active_tokens):
        for o in range(cfg.embed):
            q[t][o] = sum(norm[t][c] * weights["q"][c][o] for c in range(cfg.embed)) / cfg.acc_scale
            k[t][o] = sum(norm[t][c] * weights["k"][c][o] for c in range(cfg.embed)) / cfg.acc_scale
            v[t][o] = sum(norm[t][c] * weights["v"][c][o] for c in range(cfg.embed)) / cfg.acc_scale
    return q, k, v


def attention_core(q: list[list[float]],
                   k: list[list[float]],
                   v: list[list[float]],
                   cfg: ReducedE2EConfig,
                   math_cfg: E2EMathConfig,
                   softmax_tables: list[dict[str, Any]] | None,
                   block_idx: int) -> list[list[float]]:
    attn = zeros2(cfg.active_tokens, cfg.embed)
    for head in range(cfg.heads):
        base = head * cfg.head_dim
        for tq in range(cfg.active_tokens):
            scores = []
            for tk in range(cfg.active_tokens):
                acc = sum(q[tq][base + d] * k[tk][base + d] for d in range(cfg.head_dim))
                scores.append(acc / 8.0)
            probs = softmax_probs_with_math(scores, block_idx, math_cfg, softmax_tables)
            for d in range(cfg.head_dim):
                attn[tq][base + d] = sum(probs[tk] * v[tk][base + d] for tk in range(cfg.active_tokens))
    return attn


def output_projection(tokens: list[list[float]], attn: list[list[float]], weights: dict[str, Any], cfg: ReducedE2EConfig) -> None:
    for t in range(cfg.active_tokens):
        for o in range(cfg.embed):
            acc = sum(attn[t][c] * weights["wo"][c][o] for c in range(cfg.embed))
            tokens[t][o] += acc / cfg.acc_scale


def mlp_unit(tokens: list[list[float]],
             weights: dict[str, Any],
             gamma_raw: int,
             beta_raw: int,
             cfg: ReducedE2EConfig,
             math_cfg: E2EMathConfig,
             geluq_tables: list[tuple[dict[str, int], list[int]]] | None,
             lnq_tables: dict[str, list[dict[str, Any]]] | None,
             block_idx: int) -> None:
    norm = layernorm_with_math(tokens, gamma_raw, beta_raw, cfg, math_cfg, lnq_tables, block_idx, True)
    hidden = zeros2(cfg.active_tokens, cfg.ff_dim)
    for t in range(cfg.active_tokens):
        for h in range(cfg.ff_dim):
            acc = sum(norm[t][c] * weights["w1"][c][h] for c in range(cfg.embed))
            hidden[t][h] = gelu_with_math(acc / cfg.acc_scale, block_idx, math_cfg, geluq_tables)
        for c in range(cfg.embed):
            acc = sum(hidden[t][h] * weights["w2"][h][c] for h in range(cfg.ff_dim))
            tokens[t][c] += acc / cfg.acc_scale


def build_reference(cfg: ReducedE2EConfig, math_cfg: E2EMathConfig, pattern: dict[str, Any]) -> E2EVectorReference:
    weights = build_weights(cfg, pattern)
    tokens = conv_patch_embedding(cfg, weights)
    geluq_tables = (
        load_hgpipe_geluq_tables(Path(math_cfg.hgpipe_contract))
        if math_cfg.mlp_gelu == "hgpipe_geluq"
        else None
    )
    softmax_tables = (
        load_hgpipe_softmax_tables(Path(math_cfg.hgpipe_contract))
        if math_cfg.attention_softmax == "hgpipe_softmaxq"
        else None
    )
    lnq_tables = (
        load_hgpipe_layernorm_tables(Path(math_cfg.hgpipe_contract))
        if math_cfg.layernorm == "hgpipe_lnq"
        else None
    )
    gamma_raw = int(pattern.get("ln_gamma_raw", 0))
    beta_raw = int(pattern["ln_beta_raw"])
    for block_idx in range(cfg.blocks):
        norm1 = layernorm_with_math(
            tokens, gamma_raw, beta_raw, cfg, math_cfg, lnq_tables, block_idx, False)
        q, k, v = project_qkv(norm1, weights, cfg)
        attn = attention_core(q, k, v, cfg, math_cfg, softmax_tables, block_idx)
        output_projection(tokens, attn, weights, cfg)
        mlp_unit(tokens, weights, gamma_raw, beta_raw, cfg, math_cfg, geluq_tables, lnq_tables, block_idx)

    pooled = [
        sum(tokens[t][c] / float(cfg.active_tokens) for t in range(cfg.active_tokens))
        for c in range(cfg.embed)
    ]
    head_float = []
    for out_idx in range(cfg.state):
        acc = sum(pooled[c] * weights["head"][out_idx][c] for c in range(cfg.embed))
        head_float.append(acc / cfg.acc_scale)

    live_weight_bit = int(pattern["live_weight_bit"])
    axis_values = [head_float[0] + live_weight_bit] + head_float[1:]
    expected_raw = [int(value * 16.0) for value in axis_values]
    nonzero_counts = {
        "patch": sum(1 for raw in weights["patch"] if raw != 0),
        "q": sum(1 for row in weights["q"] for value in row if value != 0),
        "k": sum(1 for row in weights["k"] for value in row if value != 0),
        "v": sum(1 for row in weights["v"] for value in row if value != 0),
        "wo": sum(1 for row in weights["wo"] for value in row if value != 0),
        "w1": sum(1 for row in weights["w1"] for value in row if value != 0),
        "w2": sum(1 for row in weights["w2"] for value in row if value != 0),
        "head": sum(1 for row in weights["head"] for value in row if value != 0),
    }
    return E2EVectorReference(
        expected_raw=expected_raw,
        runtime_state=1 + live_weight_bit,
        pooled=pooled[:12],
        head_float=head_float,
        live_weight_bit=live_weight_bit,
        nonzero_counts=nonzero_counts,
        notes=[
            "Q4 raw values use ap_fixed<4,2> interpretation.",
            "Reduced dense gate mirrors HLS LayerNorm, QKV, HG-PIPE exp/GELU LUTs, WO, MLP, and head math.",
            "AXI writer emits int(value * 16) in the low 16 bits.",
            f"LayerNorm mode: {math_cfg.layernorm}.",
            f"MLP GeLU mode: {math_cfg.mlp_gelu}.",
            f"Attention softmax mode: {math_cfg.attention_softmax}.",
        ],
    )


def c_array(name: str, values: list[int]) -> str:
    body = ", ".join(str(v) for v in values)
    return f"static const int {name}[{len(values)}] = {{{body}}};"


def struct_array(name: str, ctype: str, entries: list[str]) -> list[str]:
    if entries:
        return [
            f"static const int {name}Count = {len(entries)};",
            f"static const {ctype} {name}[{len(entries)}] = {{",
            *[f"  {entry}," for entry in entries],
            "};",
        ]
    return [
        f"static const int {name}Count = 0;",
        f"static const {ctype} {name}[1] = {{{{0, 0, 0, 0, 0}}}};",
    ]


def emit_c_header(cfg: ReducedE2EConfig, pattern: dict[str, Any], ref: E2EVectorReference) -> str:
    lines = [
        "#pragma once",
        "",
        "// Generated from hardware/refs/e2e_axis_vector_spec.json by validate_e2e_axis_vector.py.",
        "// Keep this file synchronized with the spec before running the C++/Vitis gate.",
        "struct HgtxrPatchEntry { int channel; int raw; };",
        "struct HgtxrQkvEntry { int input; int output; int q; int k; int v; };",
        "struct HgtxrMatrixEntry { int row; int col; int raw; };",
        "",
        f"static const int kGroupChannels = {cfg.group_channels};",
        f"static const int kExpectedBlocks = {cfg.blocks};",
        f"static const int kExpectedActiveTokens = {cfg.active_tokens};",
        f"static const int kExpectedPatchGridH = {cfg.patch_grid_h};",
        f"static const int kExpectedPatchGridW = {cfg.patch_grid_w};",
        f"static const int kExpectedFfDim = {cfg.ff_dim};",
        c_array("kPatchRaw", [int(v) for v in pattern["patch_raw"]]),
        c_array("kHeadRaw", [int(v) for v in pattern["head_raw"]]),
        f"static const int kLnGammaRaw = {int(pattern.get('ln_gamma_raw', 0))};",
        f"static const int kLnBetaRaw = {int(pattern['ln_beta_raw'])};",
        f"static const int kQkvDiagRaw = {int(pattern['qkv_diag_raw'])};",
        f"static const int kWoGroupRaw = {int(pattern['wo_group_raw'])};",
        f"static const int kMlpW1GroupRaw = {int(pattern['mlp_w1_group_raw'])};",
        f"static const int kMlpW2DiagRaw = {int(pattern['mlp_w2_diag_raw'])};",
        f"static const int kExpectedRuntimeState = {ref.runtime_state};",
        c_array("kExpectedRaw", ref.expected_raw),
        "",
    ]
    lines += struct_array(
        "kPatchExtraEntries",
        "HgtxrPatchEntry",
        [f"{{{e['channel']}, {e['raw']}}}" for e in pattern.get("patch_extra_entries", [])],
    )
    lines += struct_array(
        "kQkvExtraEntries",
        "HgtxrQkvEntry",
        [f"{{{e['input']}, {e['output']}, {e['q']}, {e['k']}, {e['v']}}}" for e in pattern.get("qkv_extra_entries", [])],
    )
    lines += struct_array(
        "kWoExtraEntries",
        "HgtxrMatrixEntry",
        [f"{{{e['input']}, {e['output']}, {e['raw']}}}" for e in pattern.get("wo_extra_entries", [])],
    )
    lines += struct_array(
        "kMlpW1ExtraEntries",
        "HgtxrMatrixEntry",
        [f"{{{e['input']}, {e['hidden']}, {e['raw']}}}" for e in pattern.get("mlp_w1_extra_entries", [])],
    )
    lines += struct_array(
        "kMlpW2ExtraEntries",
        "HgtxrMatrixEntry",
        [f"{{{e['hidden']}, {e['output']}, {e['raw']}}}" for e in pattern.get("mlp_w2_extra_entries", [])],
    )
    lines += struct_array(
        "kHeadExtraEntries",
        "HgtxrMatrixEntry",
        [f"{{{e['output']}, {e['channel']}, {e['raw']}}}" for e in pattern.get("head_extra_entries", [])],
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--emit-c-header", "--emit-header", dest="emit_c_header", default="")
    parser.add_argument("--check-c-header", "--check-header", dest="check_c_header", default="")
    args = parser.parse_args()

    cfg, math_cfg, pattern = load_spec(Path(args.spec))
    ref = build_reference(cfg, math_cfg, pattern)
    header_text = emit_c_header(cfg, pattern, ref)
    payload = {**asdict(ref), "config": asdict(cfg), "math": asdict(math_cfg), "pattern": pattern}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + chr(10))
    if args.emit_c_header:
        out = Path(args.emit_c_header)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(header_text)
    if args.check_c_header:
        header_path = Path(args.check_c_header)
        if header_path.read_text() != header_text:
            print(f"C header is out of sync with {args.spec}: {header_path}", file=sys.stderr)
            return 1
    print(text)
    if len(ref.expected_raw) != cfg.state or ref.runtime_state != 1 + int(pattern["live_weight_bit"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
