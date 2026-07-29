#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

import pack_cyclic_weights as packer
import validate_s2_block_preln as preln


ARRAY_KEYS = ("output", "tokens", "out", "golden", "expected", "expected_output")


HGPIPE_GELU_TABLE = np.asarray([
    -0.0001, -0.0020, -0.0151, -0.0715,
    -0.1543, -0.1588, -0.0701, 0.1614,
    0.5714, 1.1299, 1.7588, 2.3788,
    2.9952, 3.6038, 4.2020, 4.8000,
], dtype=np.float32)

HGPIPE_EXP_TABLE = np.asarray([
    1.000000, 0.586646, 0.344154, 0.201897,
    0.118442, 0.069483, 0.040762, 0.023916,
    0.014025, 0.008230, 0.004827, 0.002831,
    0.001661, 0.000974, 0.000572, 0.000335,
], dtype=np.float32)


def hard_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.clip(x / 4.0 + 0.5, 0.0, 1.0).astype(np.float32)


def gelu_approx(x: np.ndarray, use_hgpipe_lut_math: bool = True) -> np.ndarray:
    if use_hgpipe_lut_math:
        cursor = np.clip(((x + 4.0) * 1.875).astype(np.int64), 0, 15)
        return HGPIPE_GELU_TABLE[cursor].astype(np.float32)
    return (x * hard_sigmoid(x)).astype(np.float32)


def exp_approx_nonpos(x: np.ndarray, use_hgpipe_lut_math: bool = True) -> np.ndarray:
    if use_hgpipe_lut_math:
        opposite_delta = np.where(x < 0.0, -x, 0.0)
        cursor = np.clip((opposite_delta * 1.875).astype(np.int64), 0, 15)
        return HGPIPE_EXP_TABLE[cursor].astype(np.float32)
    return np.where(x <= -8.0, 0.0, np.where(x >= 0.0, 1.0, np.maximum(1.0 + x / 8.0, 0.0))).astype(np.float32)


def apply_score_scale(scores: np.ndarray, score_scale_shift: int) -> np.ndarray:
    if score_scale_shift > 0:
        return (scores / float(1 << score_scale_shift)).astype(np.float32)
    if score_scale_shift < 0:
        return (scores * float(1 << (-score_scale_shift))).astype(np.float32)
    return scores.astype(np.float32)


def attention_approx(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    num_heads: int = 1,
    score_scale_shift: int = 0,
    use_hgpipe_lut_math: bool = True,
) -> np.ndarray:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes must match, got {q.shape}, {k.shape}, {v.shape}")
    if q.shape[1] % num_heads != 0:
        raise ValueError(f"embed dim {q.shape[1]} is not divisible by num_heads={num_heads}")
    head_dim = q.shape[1] // num_heads
    out = np.zeros_like(v, dtype=np.float32)
    for head in range(num_heads):
        c0 = head * head_dim
        c1 = c0 + head_dim
        scores = apply_score_scale(q[:, c0:c1] @ k[:, c0:c1].T, score_scale_shift)
        row_max = scores.max(axis=1, keepdims=True)
        probs_score = exp_approx_nonpos(scores - row_max, use_hgpipe_lut_math)
        row_sum = probs_score.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
        probs = probs_score / row_sum
        out[:, c0:c1] = (probs @ v[:, c0:c1]).astype(np.float32)
    return out.astype(np.float32)


def make_full_payload(layout: packer.Layout) -> dict[str, np.ndarray]:
    payload = preln.make_payload(layout)
    for layer in range(layout.blocks):
        attn = 2 * layer
        mlp = 2 * layer + 1
        qkv = np.arange(3 * layout.embed_dim * layout.embed_dim, dtype=np.float32)
        qkv = qkv.reshape(3 * layout.embed_dim, layout.embed_dim)
        qkv = ((qkv % 17.0) - 8.0) / 128.0 + layer * 0.002
        payload[f"backbone.blocks.{attn}.qkv.weight"] = qkv.astype(np.float32)
        wo = np.arange(layout.embed_dim * layout.embed_dim, dtype=np.float32).reshape(layout.embed_dim, layout.embed_dim)
        payload[f"backbone.blocks.{attn}.out.weight"] = (((wo % 13.0) - 6.0) / 96.0).astype(np.float32)
        w1 = np.arange(layout.hidden_dim * layout.embed_dim, dtype=np.float32).reshape(layout.hidden_dim, layout.embed_dim)
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = (((w1 % 19.0) - 9.0) / 160.0).astype(np.float32)
        w2 = np.arange(layout.embed_dim * layout.hidden_dim, dtype=np.float32).reshape(layout.embed_dim, layout.hidden_dim)
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = (((w2 % 23.0) - 11.0) / 192.0).astype(np.float32)
    return payload


def reconstruct_attention_matrix(blocks: list[np.ndarray], layout: packer.Layout, layer: int, name: str) -> np.ndarray:
    matrix = np.zeros((layout.embed_dim, layout.embed_dim), dtype=np.float32)
    channel_pair_blocks = layout.channel_tiles * layout.channel_tiles
    mlp_pair_blocks = 2 * layout.channel_tiles * layout.hidden_tiles
    offset = layout.offsets[name]
    for output_tile in range(layout.channel_tiles):
        for input_tile in range(layout.channel_tiles):
            block_idx = layer * (channel_pair_blocks + mlp_pair_blocks) + output_tile * layout.channel_tiles + input_tile
            tile = blocks[block_idx][offset:offset + layout.tile_channels * layout.tile_channels]
            tile = tile.reshape(layout.tile_channels, layout.tile_channels)
            i0 = input_tile * layout.tile_channels
            o0 = output_tile * layout.tile_channels
            i1 = min(i0 + layout.tile_channels, layout.embed_dim)
            o1 = min(o0 + layout.tile_channels, layout.embed_dim)
            matrix[i0:i1, o0:o1] = tile[:i1 - i0, :o1 - o0]
    return matrix


def reconstruct_w1(blocks: list[np.ndarray], layout: packer.Layout, layer: int) -> np.ndarray:
    matrix = np.zeros((layout.embed_dim, layout.hidden_dim), dtype=np.float32)
    channel_pair_blocks = layout.channel_tiles * layout.channel_tiles
    mlp_pair_blocks = 2 * layout.channel_tiles * layout.hidden_tiles
    layer_base = layer * (channel_pair_blocks + mlp_pair_blocks) + channel_pair_blocks
    offset = layout.offsets["w1"]
    for hidden_tile in range(layout.hidden_tiles):
        for input_tile in range(layout.channel_tiles):
            block_idx = layer_base + hidden_tile * layout.channel_tiles + input_tile
            tile = blocks[block_idx][offset:offset + layout.tile_channels * layout.tile_ff]
            tile = tile.reshape(layout.tile_channels, layout.tile_ff)
            i0 = input_tile * layout.tile_channels
            h0 = hidden_tile * layout.tile_ff
            i1 = min(i0 + layout.tile_channels, layout.embed_dim)
            h1 = min(h0 + layout.tile_ff, layout.hidden_dim)
            matrix[i0:i1, h0:h1] = tile[:i1 - i0, :h1 - h0]
    return matrix


def reconstruct_w2(blocks: list[np.ndarray], layout: packer.Layout, layer: int) -> np.ndarray:
    matrix = np.zeros((layout.hidden_dim, layout.embed_dim), dtype=np.float32)
    channel_pair_blocks = layout.channel_tiles * layout.channel_tiles
    mlp_pair_blocks = 2 * layout.channel_tiles * layout.hidden_tiles
    layer_base = layer * (channel_pair_blocks + mlp_pair_blocks) + channel_pair_blocks
    w2_base = layer_base + layout.hidden_tiles * layout.channel_tiles
    offset = layout.offsets["w2"]
    for output_tile in range(layout.channel_tiles):
        for hidden_tile in range(layout.hidden_tiles):
            block_idx = w2_base + output_tile * layout.hidden_tiles + hidden_tile
            tile = blocks[block_idx][offset:offset + layout.tile_ff * layout.tile_channels]
            tile = tile.reshape(layout.tile_ff, layout.tile_channels)
            h0 = hidden_tile * layout.tile_ff
            o0 = output_tile * layout.tile_channels
            h1 = min(h0 + layout.tile_ff, layout.hidden_dim)
            o1 = min(o0 + layout.tile_channels, layout.embed_dim)
            matrix[h0:h1, o0:o1] = tile[:h1 - h0, :o1 - o0]
    return matrix


def require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if value.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {value.shape}")
    return value.astype(np.float32)


def direct_linear(payload: dict[str, np.ndarray], layout: packer.Layout, layer: int, name: str, shape: tuple[int, int]) -> np.ndarray:
    key, value = packer.find_first(payload, packer.layer_candidates(layer, name))
    if value is None:
        raise KeyError(f"missing tensor for layer {layer} {name}; tried {packer.layer_candidates(layer, name)}")
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{key} must be 2D, got {matrix.shape}")
    return require_shape(name, matrix.T.astype(np.float32), shape)


def direct_qkv(payload: dict[str, np.ndarray], layout: packer.Layout, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wq, wk, wv, _source = packer.split_qkv(payload, layer, layout.embed_dim)
    if wq is not None and wk is not None and wv is not None:
        shape = (layout.embed_dim, layout.embed_dim)
        return (
            require_shape("wq", wq.T.astype(np.float32), shape),
            require_shape("wk", wk.T.astype(np.float32), shape),
            require_shape("wv", wv.T.astype(np.float32), shape),
        )
    return (
        direct_linear(payload, layout, layer, "wq", (layout.embed_dim, layout.embed_dim)),
        direct_linear(payload, layout, layer, "wk", (layout.embed_dim, layout.embed_dim)),
        direct_linear(payload, layout, layer, "wv", (layout.embed_dim, layout.embed_dim)),
    )


def direct_matrices(payload: dict[str, np.ndarray], layout: packer.Layout, layer: int) -> dict[str, np.ndarray]:
    wq, wk, wv = direct_qkv(payload, layout, layer)
    return {
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": direct_linear(payload, layout, layer, "wo", (layout.embed_dim, layout.embed_dim)),
        "w1": direct_linear(payload, layout, layer, "w1", (layout.embed_dim, layout.hidden_dim)),
        "w2": direct_linear(payload, layout, layer, "w2", (layout.hidden_dim, layout.embed_dim)),
    }


def fused_block(
    tokens: np.ndarray,
    matrices: dict[str, np.ndarray],
    norms: dict[str, np.ndarray],
    eps: float,
    *,
    num_heads: int = 1,
    score_scale_shift: int = 0,
    use_hgpipe_lut_math: bool = True,
) -> np.ndarray:
    x0 = tokens.astype(np.float32)
    n1 = preln.preln_hls_approx(x0, norms["norm1_gamma"], norms["norm1_beta"], eps)
    q = n1 @ matrices["wq"]
    k = n1 @ matrices["wk"]
    v = n1 @ matrices["wv"]
    attn = attention_approx(
        q,
        k,
        v,
        num_heads=num_heads,
        score_scale_shift=score_scale_shift,
        use_hgpipe_lut_math=use_hgpipe_lut_math,
    )
    x1 = x0 + attn @ matrices["wo"]
    n2 = preln.preln_hls_approx(x1, norms["norm2_gamma"], norms["norm2_beta"], eps)
    hidden = gelu_approx(n2 @ matrices["w1"], use_hgpipe_lut_math)
    return (x1 + hidden @ matrices["w2"]).astype(np.float32)


def load_array(path: Path, keys: tuple[str, ...] = ARRAY_KEYS) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        key = next((candidate for candidate in keys if candidate in data.files), data.files[0] if data.files else None)
        if key is None:
            raise ValueError(f"{path} has no arrays")
        return np.asarray(data[key], dtype=np.float32)
    if suffix == ".json":
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            selected = next((raw[candidate] for candidate in keys if candidate in raw), None)
            if selected is None:
                raise ValueError(f"{path} must contain one of {keys}")
            raw = selected
        return np.asarray(raw, dtype=np.float32)
    raise ValueError(f"unsupported array suffix: {path.suffix}")


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, value.astype(np.float32))
    elif suffix == ".npz":
        np.savez(path, output=value.astype(np.float32))
    elif suffix == ".json":
        path.write_text(json.dumps({"output": value.astype(np.float32).tolist()}, indent=2) + "\n")
    else:
        raise ValueError(f"unsupported output array suffix: {path.suffix}")


def make_token_grid(layout: packer.Layout, tokens: int, tokens_input: Path | None) -> np.ndarray:
    if tokens_input is None:
        return np.linspace(-1.5, 1.75, num=tokens * layout.embed_dim, dtype=np.float32).reshape(tokens, layout.embed_dim)
    token_grid = load_array(tokens_input, ("tokens", "input", "x"))
    if token_grid.ndim != 2 or token_grid.shape[1] != layout.embed_dim:
        raise ValueError(f"tokens input expected shape (N, {layout.embed_dim}), got {token_grid.shape}")
    return token_grid.astype(np.float32)


def compare_expected_output(expected: np.ndarray, layer_outputs: np.ndarray) -> tuple[str, float]:
    if expected.shape == layer_outputs.shape:
        return "all_layers", float(np.max(np.abs(expected - layer_outputs)))
    if expected.shape == layer_outputs[0].shape:
        return "layer0", float(np.max(np.abs(expected - layer_outputs[0])))
    raise ValueError(f"expected output shape {expected.shape} does not match {layer_outputs.shape} or {layer_outputs[0].shape}")


def packed_roundtrip_blocks(blocks: list[np.ndarray], layout: packer.Layout) -> tuple[list[np.ndarray], float]:
    elems = np.concatenate([block.reshape(-1) for block in blocks]).astype(np.float32)
    raw = packer.pack_lanes(packer.quantize_fixed(elems, layout), layout)
    restored = packer.unpack_lanes(raw, layout)[: elems.size].astype(np.float32)
    max_quant_error = float(np.max(np.abs(restored - elems))) if elems.size else 0.0
    out: list[np.ndarray] = []
    cursor = 0
    for block in blocks:
        size = block.size
        out.append(restored[cursor:cursor + size].reshape(block.shape).astype(np.float32))
        cursor += size
    return out, max_quant_error


def run_validation(
    layout: packer.Layout,
    tokens: int,
    eps: float,
    input_path: Path | None = None,
    tokens_input: Path | None = None,
    expected_output: Path | None = None,
    out_output: Path | None = None,
    *,
    num_heads: int = 1,
    score_scale_shift: int = 0,
    use_hgpipe_lut_math: bool = True,
    packed_roundtrip: bool = False,
) -> dict[str, Any]:
    payload = packer.load_tensor_map(input_path) if input_path else make_full_payload(layout)
    fallbacks: list[dict[str, Any]] = []
    blocks, records = packer.build_s2_block_first_step_blocks(payload, layout, True, fallbacks)
    if fallbacks:
        raise AssertionError(f"unexpected fallbacks: {fallbacks}")
    max_packed_roundtrip_error = 0.0
    if packed_roundtrip:
        blocks, max_packed_roundtrip_error = packed_roundtrip_blocks(blocks, layout)
    token_grid = make_token_grid(layout, tokens, tokens_input)

    per_layer: list[dict[str, Any]] = []
    layer_outputs: list[np.ndarray] = []
    max_matrix_error = 0.0
    max_output_error = 0.0
    for layer in range(layout.blocks):
        packed = {
            "wq": reconstruct_attention_matrix(blocks, layout, layer, "wq"),
            "wk": reconstruct_attention_matrix(blocks, layout, layer, "wk"),
            "wv": reconstruct_attention_matrix(blocks, layout, layer, "wv"),
            "wo": reconstruct_attention_matrix(blocks, layout, layer, "wo"),
            "w1": reconstruct_w1(blocks, layout, layer),
            "w2": reconstruct_w2(blocks, layout, layer),
        }
        direct = direct_matrices(payload, layout, layer)
        matrix_error = max(float(np.max(np.abs(packed[name] - direct[name]))) for name in packed)
        norms = {
            "norm1_gamma": preln.norm_from_packed_blocks(blocks, layout, layer, False, False),
            "norm1_beta": preln.norm_from_packed_blocks(blocks, layout, layer, False, True),
            "norm2_gamma": preln.norm_from_packed_blocks(blocks, layout, layer, True, False),
            "norm2_beta": preln.norm_from_packed_blocks(blocks, layout, layer, True, True),
        }
        packed_out = fused_block(
            token_grid,
            packed,
            norms,
            eps,
            num_heads=num_heads,
            score_scale_shift=score_scale_shift,
            use_hgpipe_lut_math=use_hgpipe_lut_math,
        )
        direct_out = fused_block(
            token_grid,
            direct,
            norms,
            eps,
            num_heads=num_heads,
            score_scale_shift=score_scale_shift,
            use_hgpipe_lut_math=use_hgpipe_lut_math,
        )
        output_error = float(np.max(np.abs(packed_out - direct_out)))
        max_matrix_error = max(max_matrix_error, matrix_error)
        max_output_error = max(max_output_error, output_error)
        layer_outputs.append(packed_out)
        per_layer.append({
            "layer": layer,
            "matrix_max_abs_error": matrix_error,
            "output_max_abs_error": output_error,
        })
    output_stack = np.stack(layer_outputs).astype(np.float32)
    expected_mode = None
    max_expected_output_error = None
    if expected_output is not None:
        expected = load_array(expected_output)
        expected_mode, max_expected_output_error = compare_expected_output(expected, output_stack)
    if out_output is not None:
        save_array(out_output, output_stack)
    return {
        "status": "pass",
        "input": str(input_path) if input_path else None,
        "used_synthetic_payload": input_path is None,
        "tokens_input": str(tokens_input) if tokens_input else None,
        "expected_output": str(expected_output) if expected_output else None,
        "expected_output_compare": expected_mode,
        "out_output": str(out_output) if out_output else None,
        "layout": asdict(layout),
        "tokens": int(token_grid.shape[0]),
        "attention": {
            "num_heads": int(num_heads),
            "head_dim": int(layout.embed_dim // num_heads) if num_heads else None,
            "score_scale_shift": int(score_scale_shift),
            "use_hgpipe_lut_math": bool(use_hgpipe_lut_math),
        },
        "packed_roundtrip": bool(packed_roundtrip),
        "max_packed_roundtrip_error": max_packed_roundtrip_error,
        "record_count": len(records),
        "block_count": len(blocks),
        "max_matrix_error": max_matrix_error,
        "max_output_error": max_output_error,
        "max_expected_output_error": max_expected_output_error,
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
    parser.add_argument("--input", type=Path, help="Optional external weight mapping as .npz, .json, .pt, or .pth")
    parser.add_argument("--tokens-input", type=Path, help="Optional token input as .npy, .npz, or .json")
    parser.add_argument("--expected-output", type=Path, help="Optional golden output as .npy, .npz, or .json")
    parser.add_argument("--out-output", type=Path, help="Optional generated layer outputs as .npy, .npz, or .json")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--score-scale-shift", type=int, default=0)
    parser.add_argument("--use-hgpipe-lut-math", type=int, choices=[0, 1], default=1)
    parser.add_argument("--packed-roundtrip", action="store_true", help="Quantize, pack, and unpack blocks before validation to mirror the HLS AXI weight view")
    parser.add_argument("--max-packed-roundtrip-error", type=float, default=0.0)
    parser.add_argument("--max-matrix-error", type=float, default=0.0)
    parser.add_argument("--max-output-error", type=float, default=1e-6)
    parser.add_argument("--max-expected-output-error", type=float, default=1e-6)
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
    result = run_validation(
        layout,
        args.tokens,
        args.eps,
        args.input,
        args.tokens_input,
        args.expected_output,
        args.out_output,
        num_heads=args.num_heads,
        score_scale_shift=args.score_scale_shift,
        use_hgpipe_lut_math=bool(args.use_hgpipe_lut_math),
        packed_roundtrip=args.packed_roundtrip,
    )
    expected_error = result["max_expected_output_error"]
    if (
        result["max_matrix_error"] > args.max_matrix_error
        or result["max_output_error"] > args.max_output_error
        or (expected_error is not None and expected_error > args.max_expected_output_error)
    ):
        result["status"] = "fail"
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
