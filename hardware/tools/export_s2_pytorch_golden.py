#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import _bootstrap  # noqa: F401
import pack_cyclic_weights as packer
import validate_s2_block_full as s2full
import validate_s2_block_preln as preln
from software.config import load_config
from software.model import build_model


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        obj = obj["model"]
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a state_dict-like checkpoint")
    return {str(k): v.detach().cpu() for k, v in obj.items() if hasattr(v, "detach")}


def load_model(config: Path, checkpoint: Path) -> torch.nn.Module:
    model = build_model(load_config(config))
    state = load_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected[:10]}")
    backbone_missing = [key for key in missing if key.startswith("backbone.")]
    if backbone_missing:
        raise ValueError(f"missing backbone checkpoint keys: {backbone_missing[:10]}")
    model.eval()
    return model


def torch_independent_outputs(model: torch.nn.Module, tokens: np.ndarray, layers: int) -> np.ndarray:
    x0 = torch.from_numpy(tokens.astype(np.float32)).unsqueeze(0)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for layer in range(layers):
            attn = model.backbone.blocks[2 * layer]
            mlp = model.backbone.blocks[2 * layer + 1]
            y = mlp(attn(x0)).squeeze(0).cpu().numpy().astype(np.float32)
            outs.append(y)
    return np.stack(outs).astype(np.float32)


def torch_sequential_outputs(model: torch.nn.Module, tokens: np.ndarray, layers: int) -> np.ndarray:
    x = torch.from_numpy(tokens.astype(np.float32)).unsqueeze(0)
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for layer in range(layers):
            x = model.backbone.blocks[2 * layer](x)
            x = model.backbone.blocks[2 * layer + 1](x)
            outs.append(x.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(outs).astype(np.float32)


def s2_approx_outputs(
    checkpoint: Path,
    layout: packer.Layout,
    token_grid: np.ndarray,
    eps: float,
    *,
    num_heads: int,
    score_scale_shift: int,
    use_hgpipe_lut_math: bool,
    packed_roundtrip: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = packer.load_tensor_map(checkpoint)
    fallbacks: list[dict[str, Any]] = []
    blocks, records = packer.build_s2_block_first_step_blocks(payload, layout, True, fallbacks)
    if fallbacks:
        raise AssertionError(f"unexpected fallbacks: {fallbacks}")
    max_packed_roundtrip_error = 0.0
    if packed_roundtrip:
        blocks, max_packed_roundtrip_error = s2full.packed_roundtrip_blocks(blocks, layout)

    outs: list[np.ndarray] = []
    per_layer: list[dict[str, Any]] = []
    for layer in range(layout.blocks):
        matrices = {
            "wq": s2full.reconstruct_attention_matrix(blocks, layout, layer, "wq"),
            "wk": s2full.reconstruct_attention_matrix(blocks, layout, layer, "wk"),
            "wv": s2full.reconstruct_attention_matrix(blocks, layout, layer, "wv"),
            "wo": s2full.reconstruct_attention_matrix(blocks, layout, layer, "wo"),
            "w1": s2full.reconstruct_w1(blocks, layout, layer),
            "w2": s2full.reconstruct_w2(blocks, layout, layer),
        }
        norms = {
            "norm1_gamma": preln.norm_from_packed_blocks(blocks, layout, layer, False, False),
            "norm1_beta": preln.norm_from_packed_blocks(blocks, layout, layer, False, True),
            "norm2_gamma": preln.norm_from_packed_blocks(blocks, layout, layer, True, False),
            "norm2_beta": preln.norm_from_packed_blocks(blocks, layout, layer, True, True),
        }
        out = s2full.fused_block(
            token_grid,
            matrices,
            norms,
            eps,
            num_heads=num_heads,
            score_scale_shift=score_scale_shift,
            use_hgpipe_lut_math=use_hgpipe_lut_math,
        )
        outs.append(out)
        per_layer.append({"layer": layer, "approx_abs_max": float(np.max(np.abs(out)))})
    return np.stack(outs).astype(np.float32), {
        "record_count": len(records),
        "block_count": len(blocks),
        "fallback_count": len(fallbacks),
        "packed_roundtrip": bool(packed_roundtrip),
        "max_packed_roundtrip_error": max_packed_roundtrip_error,
        "attention": {
            "num_heads": int(num_heads),
            "head_dim": int(layout.embed_dim // num_heads) if num_heads else None,
            "score_scale_shift": int(score_scale_shift),
            "use_hgpipe_lut_math": bool(use_hgpipe_lut_math),
        },
        "per_layer_approx": per_layer,
    }


def error_stats(name: str, lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    diff = lhs.astype(np.float64) - rhs.astype(np.float64)
    return {
        "name": name,
        "shape": list(lhs.shape),
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def save_outputs(path: Path, *, tokens: np.ndarray, exact_independent: np.ndarray, exact_sequential: np.ndarray, approx_independent: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        tokens=tokens.astype(np.float32),
        exact_independent=exact_independent.astype(np.float32),
        exact_sequential=exact_sequential.astype(np.float32),
        approx_independent=approx_independent.astype(np.float32),
        output=exact_independent.astype(np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("software/configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("hardware/refs/weights/software_initial_weights.pt"))
    parser.add_argument("--tokens-input", type=Path)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--tile-channels", type=int, default=32)
    parser.add_argument("--tile-ff", type=int, default=32)
    parser.add_argument("--bus-width", type=int, default=128)
    parser.add_argument("--bit-width", type=int, default=16)
    parser.add_argument("--int-width", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--score-scale-shift", type=int, default=3)
    parser.add_argument("--use-hgpipe-lut-math", type=int, choices=[0, 1], default=1)
    parser.add_argument("--packed-roundtrip", action="store_true")
    parser.add_argument("--out-npz", type=Path, default=Path("docs/resources/s2_pytorch_golden_software_initial_2026_06_06.npz"))
    parser.add_argument("--report", type=Path, default=Path("docs/resources/s2_pytorch_golden_software_initial_2026_06_06.json"))
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
    token_grid = s2full.make_token_grid(layout, args.tokens, args.tokens_input)
    model = load_model(args.config, args.checkpoint)
    exact_independent = torch_independent_outputs(model, token_grid, layout.blocks)
    exact_sequential = torch_sequential_outputs(model, token_grid, layout.blocks)
    approx_independent, approx_meta = s2_approx_outputs(
        args.checkpoint,
        layout,
        token_grid,
        args.eps,
        num_heads=args.num_heads,
        score_scale_shift=args.score_scale_shift,
        use_hgpipe_lut_math=bool(args.use_hgpipe_lut_math),
        packed_roundtrip=args.packed_roundtrip,
    )
    save_outputs(args.out_npz, tokens=token_grid, exact_independent=exact_independent, exact_sequential=exact_sequential, approx_independent=approx_independent)

    result = {
        "status": "pass",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "out_npz": str(args.out_npz),
        "layout": asdict(layout),
        "tokens": int(token_grid.shape[0]),
        "contract": "exact PyTorch per-layer block outputs plus current S2 approximation outputs; exact_independent applies one attention+MLP pair independently to the same token input per layer",
        "approximation_limitations": [
            "S2 approximation omits Linear biases",
            "S2 approximation uses HLS compact LUT GELU/softmax approximations instead of exact PyTorch math",
            "S2 approximation uses HLS piecewise pre-LN rsqrt instead of exact LayerNorm rsqrt",
        ],
        "approx_meta": approx_meta,
        "compare": [
            error_stats("approx_independent_vs_exact_independent", approx_independent, exact_independent),
            error_stats("exact_independent_vs_exact_sequential", exact_independent, exact_sequential),
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.report.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
