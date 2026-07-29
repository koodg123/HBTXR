#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import pack_cyclic_weights as packer


def piecewise_rsqrt(x: np.ndarray) -> np.ndarray:
    y = np.full_like(x, 0.125, dtype=np.float32)
    y = np.where(x <= 16.0, 0.25, y)
    y = np.where(x <= 8.0, 0.375, y)
    y = np.where(x <= 4.0, 0.5, y)
    y = np.where(x <= 2.0, 0.75, y)
    y = np.where(x <= 1.0, 1.0, y)
    y = np.where(x <= 0.5, 1.625, y)
    y = np.where(x <= 0.25, 2.5, y)
    y = np.where(x <= 0.125, 3.375, y)
    y = np.where(x <= 0.0625, 4.0, y)
    y = np.where(x <= 0.0001, 32.0, y)
    return y.astype(np.float32)


def preln_exact(tokens: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = tokens.mean(axis=1, keepdims=True)
    var = ((tokens - mean) ** 2).mean(axis=1, keepdims=True)
    return (tokens - mean) / np.sqrt(var + eps) * gamma + beta


def preln_hls_approx(tokens: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = tokens.mean(axis=1, keepdims=True)
    var = ((tokens - mean) ** 2).mean(axis=1, keepdims=True)
    return (tokens - mean) * piecewise_rsqrt(var + eps) * gamma + beta


def make_payload(layout: packer.Layout) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for layer in range(layout.blocks):
        attn = 2 * layer
        mlp = 2 * layer + 1
        base = np.arange(layout.embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.norm.weight"] = 0.75 + base / (layout.embed_dim * 2.0) + layer * 0.05
        payload[f"backbone.blocks.{attn}.norm.bias"] = -0.125 + base / (layout.embed_dim * 8.0) + layer * 0.025
        payload[f"backbone.blocks.{attn}.qkv.weight"] = np.zeros((3 * layout.embed_dim, layout.embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{attn}.out.weight"] = np.eye(layout.embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = 1.25 + base / (layout.embed_dim * 3.0) + layer * 0.075
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = 0.25 - base / (layout.embed_dim * 6.0) + layer * 0.05
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = np.zeros((layout.hidden_dim, layout.embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = np.zeros((layout.embed_dim, layout.hidden_dim), dtype=np.float32)
    return payload


def norm_from_packed_blocks(
    blocks: list[np.ndarray],
    layout: packer.Layout,
    layer: int,
    norm2: bool,
    beta: bool,
) -> np.ndarray:
    channel_pair_blocks = layout.channel_tiles * layout.channel_tiles
    mlp_pair_blocks = 2 * layout.channel_tiles * layout.hidden_tiles
    offset_name = (
        "norm2_beta" if norm2 and beta else
        "norm2_gamma" if norm2 else
        "norm1_beta" if beta else
        "norm1_gamma"
    )
    values = np.zeros((layout.embed_dim,), dtype=np.float32)
    for output_tile in range(layout.channel_tiles):
        block_idx = layer * (channel_pair_blocks + mlp_pair_blocks) + output_tile * layout.channel_tiles
        start = layout.offsets[offset_name]
        stop = start + layout.tile_channels
        tile = blocks[block_idx][start:stop]
        c0 = output_tile * layout.tile_channels
        values[c0:c0 + layout.tile_channels] = tile[: max(0, min(layout.tile_channels, layout.embed_dim - c0))]
    return values


def run_validation(layout: packer.Layout, tokens: int, eps: float) -> dict[str, Any]:
    payload = make_payload(layout)
    fallbacks: list[dict[str, Any]] = []
    blocks, records = packer.build_s2_block_first_step_blocks(payload, layout, True, fallbacks)
    if fallbacks:
        raise AssertionError(f"unexpected fallbacks: {fallbacks}")

    token_grid = np.linspace(-1.75, 2.25, num=tokens * layout.embed_dim, dtype=np.float32)
    token_grid = token_grid.reshape(tokens, layout.embed_dim)

    per_layer: list[dict[str, Any]] = []
    max_norm_param_error = 0.0
    max_preln_abs_error = 0.0
    for layer in range(layout.blocks):
        attn = 2 * layer
        mlp = 2 * layer + 1
        expected_norm1_gamma = payload[f"backbone.blocks.{attn}.norm.weight"]
        expected_norm1_beta = payload[f"backbone.blocks.{attn}.norm.bias"]
        expected_norm2_gamma = payload[f"backbone.blocks.{mlp}.net.0.weight"]
        expected_norm2_beta = payload[f"backbone.blocks.{mlp}.net.0.bias"]
        actual_norm1_gamma = norm_from_packed_blocks(blocks, layout, layer, False, False)
        actual_norm1_beta = norm_from_packed_blocks(blocks, layout, layer, False, True)
        actual_norm2_gamma = norm_from_packed_blocks(blocks, layout, layer, True, False)
        actual_norm2_beta = norm_from_packed_blocks(blocks, layout, layer, True, True)

        norm_errors = [
            float(np.max(np.abs(actual_norm1_gamma - expected_norm1_gamma))),
            float(np.max(np.abs(actual_norm1_beta - expected_norm1_beta))),
            float(np.max(np.abs(actual_norm2_gamma - expected_norm2_gamma))),
            float(np.max(np.abs(actual_norm2_beta - expected_norm2_beta))),
        ]
        max_norm_param_error = max(max_norm_param_error, max(norm_errors))

        exact1 = preln_exact(token_grid, actual_norm1_gamma, actual_norm1_beta, eps)
        approx1 = preln_hls_approx(token_grid, actual_norm1_gamma, actual_norm1_beta, eps)
        exact2 = preln_exact(token_grid, actual_norm2_gamma, actual_norm2_beta, eps)
        approx2 = preln_hls_approx(token_grid, actual_norm2_gamma, actual_norm2_beta, eps)
        preln_error = max(
            float(np.max(np.abs(approx1 - exact1))),
            float(np.max(np.abs(approx2 - exact2))),
        )
        max_preln_abs_error = max(max_preln_abs_error, preln_error)
        per_layer.append({
            "layer": layer,
            "norm_param_max_abs_error": max(norm_errors),
            "preln_piecewise_max_abs_error": preln_error,
        })

    return {
        "status": "pass",
        "layout": asdict(layout),
        "tokens": tokens,
        "record_count": len(records),
        "block_count": len(blocks),
        "max_norm_param_error": max_norm_param_error,
        "max_preln_piecewise_abs_error": max_preln_abs_error,
        "per_layer": per_layer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--tile-channels", type=int, default=4)
    parser.add_argument("--tile-ff", type=int, default=4)
    parser.add_argument("--bus-width", type=int, default=64)
    parser.add_argument("--bit-width", type=int, default=16)
    parser.add_argument("--int-width", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--max-norm-param-error", type=float, default=0.0)
    parser.add_argument("--max-preln-abs-error", type=float, default=0.75)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    layout = packer.Layout(
        embed_dim=args.embed_dim,
        tile_channels=args.tile_channels,
        tile_ff=args.tile_ff,
        bus_width=args.bus_width,
        bit_width=args.bit_width,
        int_width=args.int_width,
        blocks=args.blocks,
        mlp_ratio=args.mlp_ratio,
    )
    result = run_validation(layout, args.tokens, args.eps)
    if result["max_norm_param_error"] > args.max_norm_param_error:
        result["status"] = "fail"
    if result["max_preln_piecewise_abs_error"] > args.max_preln_abs_error:
        result["status"] = "fail"

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
