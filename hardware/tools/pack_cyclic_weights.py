#!/usr/bin/env python3
"""Pack HGTXR cyclic Transformer tile-local weights for the optional AXI port.

The current HLS S1 ABI consumes one tile-local Transformer block per cyclic
stage invocation. This tool maps software/global tensors into that tile-local
layout, quantizes to signed fixed-point, and packs lanes into little-endian AXI
words compatible with HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Layout:
    embed_dim: int = 192
    tile_channels: int = 32
    tile_ff: int = 32
    bus_width: int = 128
    bit_width: int = 16
    int_width: int = 6
    blocks: int = 6
    mlp_ratio: int = 4

    @property
    def lanes(self) -> int:
        return self.bus_width // self.bit_width

    @property
    def frac_width(self) -> int:
        return self.bit_width - self.int_width

    @property
    def channel_tiles(self) -> int:
        return math.ceil(self.embed_dim / self.tile_channels)

    @property
    def hidden_dim(self) -> int:
        return self.embed_dim * self.mlp_ratio

    @property
    def hidden_tiles(self) -> int:
        return math.ceil(self.hidden_dim / self.tile_ff)

    @property
    def offsets(self) -> dict[str, int]:
        tc = self.tile_channels
        tf = self.tile_ff
        out: dict[str, int] = {}
        cur = 0
        for name in ("norm1_gamma", "norm1_beta", "norm2_gamma", "norm2_beta"):
            out[name] = cur
            cur += tc
        for name in ("wq", "wk", "wv", "wo"):
            out[name] = cur
            cur += tc * tc
        out["w1"] = cur
        cur += tc * tf
        out["w2"] = cur
        cur += tf * tc
        out["tile_elems"] = cur
        out["tile_words"] = (cur + self.lanes - 1) // self.lanes
        out["total_words"] = out["tile_words"] * self.blocks
        return out


DIRECT_KEYS = {
    "norm1_gamma": ("norm1_gamma", "attn_norm_gamma"),
    "norm1_beta": ("norm1_beta", "attn_norm_beta"),
    "norm2_gamma": ("norm2_gamma", "mlp_norm_gamma"),
    "norm2_beta": ("norm2_beta", "mlp_norm_beta"),
    "wq": ("wq", "q_weight"),
    "wk": ("wk", "k_weight"),
    "wv": ("wv", "v_weight"),
    "wo": ("wo", "out_weight", "proj_weight"),
    "w1": ("w1", "fc1_weight", "mlp1_weight"),
    "w2": ("w2", "fc2_weight", "mlp2_weight"),
}


def load_tensor_map(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("JSON weight input must be an object mapping names to arrays")
        return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}
    if suffix in (".pt", ".pth"):
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Loading .pt/.pth requires torch; export to .npz or install torch") from exc
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise ValueError("torch checkpoint must be a state_dict-like mapping")
        result = {}
        for key, value in obj.items():
            if hasattr(value, "detach"):
                result[key] = value.detach().cpu().numpy().astype(np.float32)
        return result
    raise ValueError(f"unsupported weight input suffix: {path.suffix}")


def find_first(tensors: dict[str, np.ndarray], candidates: list[str]) -> tuple[str | None, np.ndarray | None]:
    for key in candidates:
        if key in tensors:
            return key, tensors[key]
    return None, None


def layer_candidates(layer: int, name: str) -> list[str]:
    attn = 2 * layer
    mlp = 2 * layer + 1
    direct = [f"blocks.{layer}.{key}" for key in DIRECT_KEYS.get(name, (name,))]
    direct += [f"backbone.blocks.{layer}.{key}" for key in DIRECT_KEYS.get(name, (name,))]
    if name == "norm1_gamma":
        return [f"backbone.blocks.{attn}.norm.weight", f"blocks.{attn}.norm.weight"] + direct
    if name == "norm1_beta":
        return [f"backbone.blocks.{attn}.norm.bias", f"blocks.{attn}.norm.bias"] + direct
    if name == "norm2_gamma":
        return [f"backbone.blocks.{mlp}.net.0.weight", f"blocks.{mlp}.net.0.weight"] + direct
    if name == "norm2_beta":
        return [f"backbone.blocks.{mlp}.net.0.bias", f"blocks.{mlp}.net.0.bias"] + direct
    if name == "wo":
        return [f"backbone.blocks.{attn}.out.weight", f"blocks.{attn}.out.weight"] + direct
    if name == "w1":
        return [f"backbone.blocks.{mlp}.net.1.weight", f"blocks.{mlp}.net.1.weight"] + direct
    if name == "w2":
        return [f"backbone.blocks.{mlp}.net.4.weight", f"blocks.{mlp}.net.4.weight"] + direct
    return direct


def split_qkv(tensors: dict[str, np.ndarray], layer: int, embed_dim: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    attn = 2 * layer
    candidates = [
        f"backbone.blocks.{attn}.qkv.weight",
        f"blocks.{attn}.qkv.weight",
        f"backbone.blocks.{layer}.qkv.weight",
        f"blocks.{layer}.qkv.weight",
        "qkv.weight",
        "qkv_weight",
    ]
    key, value = find_first(tensors, candidates)
    if value is None:
        return None, None, None, None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != 3 * embed_dim:
        raise ValueError(f"{key} expected first dim {3 * embed_dim}, got {arr.shape}")
    return arr[0:embed_dim], arr[embed_dim:2 * embed_dim], arr[2 * embed_dim:3 * embed_dim], key


def vector_tile(value: np.ndarray | None, start: int, size: int, fallback: float) -> np.ndarray:
    if value is None:
        return np.full((size,), fallback, dtype=np.float32)
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.full((size,), fallback, dtype=np.float32)
    end = min(start + size, flat.shape[0])
    if end > start:
        out[: end - start] = flat[start:end]
    return out


def linear_tile(value: np.ndarray | None, in_start: int, out_start: int, in_size: int, out_size: int, *, transpose_from_torch: bool, identity: bool) -> np.ndarray:
    out = np.zeros((in_size, out_size), dtype=np.float32)
    if identity:
        for i in range(min(in_size, out_size)):
            out[i, i] = 1.0
    if value is None:
        return out
    arr = np.asarray(value, dtype=np.float32)
    if transpose_from_torch:
        if arr.ndim != 2:
            raise ValueError(f"linear tensor must be 2D, got {arr.shape}")
        arr = arr.T
    in_end = min(in_start + in_size, arr.shape[0])
    out_end = min(out_start + out_size, arr.shape[1])
    if in_end > in_start and out_end > out_start:
        out[: in_end - in_start, : out_end - out_start] = arr[in_start:in_end, out_start:out_end]
    return out


def tensor_for_block(tensors: dict[str, np.ndarray], layer: int, name: str, layout: Layout, channel_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    tc = layout.tile_channels
    tf = layout.tile_ff
    c0 = channel_tile * tc

    if name in ("wq", "wk", "wv"):
        split = split_qkv(tensors, layer, layout.embed_dim)
        idx = {"wq": 0, "wk": 1, "wv": 2}[name]
        value = split[idx]
        source = split[3]
        if value is not None:
            return linear_tile(value, c0, c0, tc, tc, transpose_from_torch=True, identity=False)
        key, direct = find_first(tensors, layer_candidates(layer, name))
        if direct is not None:
            return linear_tile(direct, c0, c0, tc, tc, transpose_from_torch=True, identity=False)
        source = key
    else:
        key, value = find_first(tensors, layer_candidates(layer, name))
        source = key
        if value is not None:
            if name in ("norm1_gamma", "norm2_gamma"):
                return vector_tile(value, c0, tc, 1.0)
            if name in ("norm1_beta", "norm2_beta"):
                return vector_tile(value, c0, tc, 0.0)
            if name == "wo":
                return linear_tile(value, c0, c0, tc, tc, transpose_from_torch=True, identity=False)
            if name == "w1":
                return linear_tile(value, c0, c0, tc, tf, transpose_from_torch=True, identity=False)
            if name == "w2":
                return linear_tile(value, c0, c0, tf, tc, transpose_from_torch=True, identity=False)

    if strict:
        raise KeyError(f"missing tensor for layer {layer} {name}; tried {layer_candidates(layer, name)}")
    fallback = "ones" if name in ("norm1_gamma", "norm2_gamma") else "zeros"
    if name in ("wq", "wk", "wv", "wo", "w1", "w2"):
        fallback = "identity" if name in ("wq", "wk", "wv", "wo") else "partial_identity"
    fallbacks.append({"layer": layer, "tensor": name, "fallback": fallback, "source": source})
    if name in ("norm1_gamma", "norm2_gamma"):
        return np.ones((tc,), dtype=np.float32)
    if name in ("norm1_beta", "norm2_beta"):
        return np.zeros((tc,), dtype=np.float32)
    return linear_tile(None, 0, 0, tc if name != "w2" else tf, tf if name in ("w1",) else tc, transpose_from_torch=False, identity=True)


def quantize_fixed(values: np.ndarray, layout: Layout) -> np.ndarray:
    scale = 1 << layout.frac_width
    min_q = -(1 << (layout.bit_width - 1))
    max_q = (1 << (layout.bit_width - 1)) - 1
    quant = np.rint(values.astype(np.float64) * scale)
    quant = np.clip(quant, min_q, max_q).astype(np.int64)
    mask = (1 << layout.bit_width) - 1
    return (quant & mask).astype(np.uint64)


def pack_lanes(q: np.ndarray, layout: Layout) -> bytes:
    lanes = layout.lanes
    mask = (1 << layout.bit_width) - 1
    padded = np.zeros(((len(q) + lanes - 1) // lanes) * lanes, dtype=np.uint64)
    padded[: len(q)] = q
    out = bytearray()
    for offset in range(0, len(padded), lanes):
        word = 0
        for lane, raw in enumerate(padded[offset: offset + lanes]):
            word |= (int(raw) & mask) << (lane * layout.bit_width)
        out.extend(int(word).to_bytes(layout.bus_width // 8, byteorder="little", signed=False))
    return bytes(out)


def unpack_lanes(blob: bytes, layout: Layout) -> np.ndarray:
    word_bytes = layout.bus_width // 8
    mask = (1 << layout.bit_width) - 1
    sign_bit = 1 << (layout.bit_width - 1)
    out = []
    for offset in range(0, len(blob), word_bytes):
        word = int.from_bytes(blob[offset: offset + word_bytes], byteorder="little", signed=False)
        for lane in range(layout.lanes):
            raw = (word >> (lane * layout.bit_width)) & mask
            if raw & sign_bit:
                raw -= 1 << layout.bit_width
            out.append(raw / float(1 << layout.frac_width))
    return np.asarray(out, dtype=np.float32)


def build_block(tensors: dict[str, np.ndarray], layer: int, layout: Layout, channel_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    offsets = layout.offsets
    elems = np.zeros((offsets["tile_elems"],), dtype=np.float32)
    for name in ("norm1_gamma", "norm1_beta", "norm2_gamma", "norm2_beta", "wq", "wk", "wv", "wo", "w1", "w2"):
        tile = tensor_for_block(tensors, layer, name, layout, channel_tile, strict, fallbacks).reshape(-1)
        start = offsets[name]
        elems[start:start + tile.size] = tile
    return elems


def tensor_for_pair(tensors: dict[str, np.ndarray], layer: int, name: str, layout: Layout, input_tile: int, output_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    tc = layout.tile_channels
    tf = layout.tile_ff
    in0 = input_tile * tc
    out0 = output_tile * tc

    if name in ("norm1_gamma", "norm1_beta", "norm2_gamma", "norm2_beta"):
        # Norm and beta are output-channel local. They are duplicated across
        # input tiles so a later HLS schedule can keep one uniform block shape.
        return tensor_for_block(tensors, layer, name, layout, output_tile, strict, fallbacks)

    if name in ("wq", "wk", "wv"):
        split = split_qkv(tensors, layer, layout.embed_dim)
        value = split[{"wq": 0, "wk": 1, "wv": 2}[name]]
        source = split[3]
        if value is not None:
            return linear_tile(value, in0, out0, tc, tc, transpose_from_torch=True, identity=False)
        key, direct = find_first(tensors, layer_candidates(layer, name))
        if direct is not None:
            return linear_tile(direct, in0, out0, tc, tc, transpose_from_torch=True, identity=False)
        source = key
    else:
        key, value = find_first(tensors, layer_candidates(layer, name))
        source = key
        if value is not None:
            if name == "wo":
                return linear_tile(value, in0, out0, tc, tc, transpose_from_torch=True, identity=False)
            if name == "w1":
                # Current S2 host contract uses output_tile as the FF tile index.
                return linear_tile(value, in0, output_tile * tf, tc, tf, transpose_from_torch=True, identity=False)
            if name == "w2":
                # Current S2 host contract uses input_tile as the FF tile index.
                return linear_tile(value, input_tile * tf, out0, tf, tc, transpose_from_torch=True, identity=False)

    if strict:
        raise KeyError(f"missing tensor for layer {layer} {name}; input_tile={input_tile} output_tile={output_tile}")
    fallback = "zeros"
    fallbacks.append({
        "layer": layer,
        "tensor": name,
        "input_tile": input_tile,
        "output_tile": output_tile,
        "fallback": fallback,
        "source": source,
    })
    if name == "w1":
        return np.zeros((tc, tf), dtype=np.float32)
    if name == "w2":
        return np.zeros((tf, tc), dtype=np.float32)
    return np.zeros((tc, tc), dtype=np.float32)


def build_pair_block(tensors: dict[str, np.ndarray], layer: int, layout: Layout, input_tile: int, output_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    offsets = layout.offsets
    elems = np.zeros((offsets["tile_elems"],), dtype=np.float32)
    for name in ("norm1_gamma", "norm1_beta", "norm2_gamma", "norm2_beta", "wq", "wk", "wv", "wo", "w1", "w2"):
        tile = tensor_for_pair(tensors, layer, name, layout, input_tile, output_tile, strict, fallbacks).reshape(-1)
        start = offsets[name]
        elems[start:start + tile.size] = tile
    return elems


def build_s2_channel_pair_blocks(tensors: dict[str, np.ndarray], layout: Layout, strict: bool, fallbacks: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, int]]]:
    blocks: list[np.ndarray] = []
    records: list[dict[str, int]] = []
    for layer in range(layout.blocks):
        for output_tile in range(layout.channel_tiles):
            for input_tile in range(layout.channel_tiles):
                records.append({
                    "block_index": len(blocks),
                    "layer": layer,
                    "input_tile": input_tile,
                    "output_tile": output_tile,
                    "input_channel_base": input_tile * layout.tile_channels,
                    "output_channel_base": output_tile * layout.tile_channels,
                })
                blocks.append(build_pair_block(tensors, layer, layout, input_tile, output_tile, strict, fallbacks))
    return blocks, records



def build_s2_mlp_w1_block(tensors: dict[str, np.ndarray], layer: int, layout: Layout, input_tile: int, hidden_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    offsets = layout.offsets
    elems = np.zeros((offsets["tile_elems"],), dtype=np.float32)
    tile = tensor_for_pair(tensors, layer, "w1", layout, input_tile, hidden_tile, strict, fallbacks).reshape(-1)
    elems[offsets["w1"]: offsets["w1"] + tile.size] = tile
    return elems


def build_s2_mlp_w2_block(tensors: dict[str, np.ndarray], layer: int, layout: Layout, hidden_tile: int, output_tile: int, strict: bool, fallbacks: list[dict[str, Any]]) -> np.ndarray:
    offsets = layout.offsets
    elems = np.zeros((offsets["tile_elems"],), dtype=np.float32)
    tile = tensor_for_pair(tensors, layer, "w2", layout, hidden_tile, output_tile, strict, fallbacks).reshape(-1)
    elems[offsets["w2"]: offsets["w2"] + tile.size] = tile
    return elems


def build_s2_mlp_hidden_pair_blocks(tensors: dict[str, np.ndarray], layout: Layout, strict: bool, fallbacks: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, int | str]]]:
    blocks: list[np.ndarray] = []
    records: list[dict[str, int | str]] = []
    for layer in range(layout.blocks):
        for hidden_tile in range(layout.hidden_tiles):
            for input_tile in range(layout.channel_tiles):
                records.append({
                    "block_index": len(blocks),
                    "phase": "w1",
                    "layer": layer,
                    "input_tile": input_tile,
                    "hidden_tile": hidden_tile,
                    "input_channel_base": input_tile * layout.tile_channels,
                    "hidden_channel_base": hidden_tile * layout.tile_ff,
                })
                blocks.append(build_s2_mlp_w1_block(tensors, layer, layout, input_tile, hidden_tile, strict, fallbacks))
        for output_tile in range(layout.channel_tiles):
            for hidden_tile in range(layout.hidden_tiles):
                records.append({
                    "block_index": len(blocks),
                    "phase": "w2",
                    "layer": layer,
                    "output_tile": output_tile,
                    "hidden_tile": hidden_tile,
                    "output_channel_base": output_tile * layout.tile_channels,
                    "hidden_channel_base": hidden_tile * layout.tile_ff,
                })
                blocks.append(build_s2_mlp_w2_block(tensors, layer, layout, hidden_tile, output_tile, strict, fallbacks))
    return blocks, records



def build_s2_block_first_step_blocks(tensors: dict[str, np.ndarray], layout: Layout, strict: bool, fallbacks: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, int | str]]]:
    blocks: list[np.ndarray] = []
    records: list[dict[str, int | str]] = []
    for layer in range(layout.blocks):
        for output_tile in range(layout.channel_tiles):
            for input_tile in range(layout.channel_tiles):
                records.append({
                    "block_index": len(blocks),
                    "phase": "attention",
                    "layer": layer,
                    "input_tile": input_tile,
                    "output_tile": output_tile,
                    "input_channel_base": input_tile * layout.tile_channels,
                    "output_channel_base": output_tile * layout.tile_channels,
                })
                blocks.append(build_pair_block(tensors, layer, layout, input_tile, output_tile, strict, fallbacks))
        for hidden_tile in range(layout.hidden_tiles):
            for input_tile in range(layout.channel_tiles):
                records.append({
                    "block_index": len(blocks),
                    "phase": "w1",
                    "layer": layer,
                    "input_tile": input_tile,
                    "hidden_tile": hidden_tile,
                    "input_channel_base": input_tile * layout.tile_channels,
                    "hidden_channel_base": hidden_tile * layout.tile_ff,
                })
                blocks.append(build_s2_mlp_w1_block(tensors, layer, layout, input_tile, hidden_tile, strict, fallbacks))
        for output_tile in range(layout.channel_tiles):
            for hidden_tile in range(layout.hidden_tiles):
                records.append({
                    "block_index": len(blocks),
                    "phase": "w2",
                    "layer": layer,
                    "output_tile": output_tile,
                    "hidden_tile": hidden_tile,
                    "output_channel_base": output_tile * layout.tile_channels,
                    "hidden_channel_base": hidden_tile * layout.tile_ff,
                })
                blocks.append(build_s2_mlp_w2_block(tensors, layer, layout, hidden_tile, output_tile, strict, fallbacks))
    return blocks, records


def write_outputs(raw: bytes, elems: np.ndarray, layout: Layout, out_bin: Path, out_npz: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(raw)
    word_bytes = layout.bus_width // 8
    words_u64 = np.frombuffer(raw, dtype="<u8").reshape((-1, word_bytes // 8))
    np.savez(out_npz, words_u64=words_u64, float_elems=elems.astype(np.float32))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def self_test() -> None:
    layout = Layout(embed_dim=8, tile_channels=4, tile_ff=4, bus_width=64, bit_width=16, int_width=6, blocks=2, mlp_ratio=1)
    tensors = {}
    for layer in range(layout.blocks):
        attn = 2 * layer
        mlp = 2 * layer + 1
        tensors[f"backbone.blocks.{attn}.norm.weight"] = np.arange(8, dtype=np.float32) / 8 + 1
        tensors[f"backbone.blocks.{attn}.norm.bias"] = np.arange(8, dtype=np.float32) / 16
        tensors[f"backbone.blocks.{attn}.qkv.weight"] = np.arange(3 * 8 * 8, dtype=np.float32).reshape(24, 8) / 128
        tensors[f"backbone.blocks.{attn}.out.weight"] = np.eye(8, dtype=np.float32)
        tensors[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(8, dtype=np.float32)
        tensors[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(8, dtype=np.float32)
        tensors[f"backbone.blocks.{mlp}.net.1.weight"] = np.arange(8 * 8, dtype=np.float32).reshape(8, 8) / 64
        tensors[f"backbone.blocks.{mlp}.net.4.weight"] = np.arange(8 * 8, dtype=np.float32).reshape(8, 8) / 32
    fallbacks: list[dict[str, Any]] = []
    block0 = build_block(tensors, 0, layout, 0, True, fallbacks)
    block1 = build_block(tensors, 0, layout, 1, True, fallbacks)
    s2_blocks, s2_records = build_s2_channel_pair_blocks(tensors, layout, True, fallbacks)
    s2_mlp_blocks, s2_mlp_records = build_s2_mlp_hidden_pair_blocks(tensors, layout, True, fallbacks)
    s2_block_blocks, s2_block_records = build_s2_block_first_step_blocks(tensors, layout, True, fallbacks)
    elems = np.concatenate([block0, block1] + s2_blocks + s2_mlp_blocks + s2_block_blocks)
    raw = pack_lanes(quantize_fixed(elems, layout), layout)
    restored = unpack_lanes(raw, layout)[: elems.size]
    max_err = float(np.max(np.abs(restored - elems)))
    assert max_err <= 1.0 / (1 << layout.frac_width), max_err
    assert len(s2_records) == layout.blocks * layout.channel_tiles * layout.channel_tiles
    assert len(s2_mlp_records) == layout.blocks * 2 * layout.channel_tiles * layout.hidden_tiles
    assert len(s2_block_records) == layout.blocks * (layout.channel_tiles * layout.channel_tiles + 2 * layout.channel_tiles * layout.hidden_tiles)
    assert not fallbacks
    print(json.dumps({"self_test": "pass", "max_abs_error": max_err, "words": len(raw) // (layout.bus_width // 8), "s2_records": len(s2_records), "s2_mlp_records": len(s2_mlp_records), "s2_block_records": len(s2_block_records)}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Weight mapping as .npz, .json, .pt, or .pth")
    parser.add_argument("--out-bin", type=Path, default=Path("hardware/refs/weights/cyclic_weights_s1.bin"))
    parser.add_argument("--out-npz", type=Path, default=Path("hardware/refs/weights/cyclic_weights_s1.npz"))
    parser.add_argument("--manifest", type=Path, default=Path("hardware/refs/weights/cyclic_weights_s1_manifest.json"))
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--tile-channels", type=int, default=32)
    parser.add_argument("--tile-ff", type=int, default=32)
    parser.add_argument("--bus-width", type=int, default=128)
    parser.add_argument("--bit-width", type=int, default=16)
    parser.add_argument("--int-width", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--channel-tile", type=int, default=0, help="Diagonal channel tile index used by current S1 ABI")
    parser.add_argument("--layout-mode", choices=["s1_diagonal", "s2_channel_pairs", "s2_mlp_hidden_pairs", "s2_block_first_step"], default="s1_diagonal")
    parser.add_argument("--strict", action="store_true", help="Fail instead of writing documented fallback weights")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --self-test is used")

    layout = Layout(
        embed_dim=args.embed_dim,
        tile_channels=args.tile_channels,
        tile_ff=args.tile_ff,
        bus_width=args.bus_width,
        bit_width=args.bit_width,
        int_width=args.int_width,
        blocks=args.blocks,
        mlp_ratio=args.mlp_ratio,
    )
    if layout.bus_width % layout.bit_width != 0:
        raise ValueError("bus width must be divisible by bit width")
    if not (0 <= args.channel_tile < layout.channel_tiles):
        raise ValueError(f"channel tile must be in [0, {layout.channel_tiles})")

    tensors = load_tensor_map(args.input)
    fallbacks: list[dict[str, Any]] = []
    if args.layout_mode == "s1_diagonal":
        block_records = [
            {
                "block_index": layer,
                "layer": layer,
                "input_tile": args.channel_tile,
                "output_tile": args.channel_tile,
                "input_channel_base": args.channel_tile * layout.tile_channels,
                "output_channel_base": args.channel_tile * layout.tile_channels,
            }
            for layer in range(layout.blocks)
        ]
        blocks = [build_block(tensors, layer, layout, args.channel_tile, args.strict, fallbacks) for layer in range(layout.blocks)]
        limitation = "S1 diagonal tile-local channel slice; current HLS consumes this layout"
    elif args.layout_mode == "s2_channel_pairs":
        blocks, block_records = build_s2_channel_pair_blocks(tensors, layout, args.strict, fallbacks)
        limitation = "S2 channel-pair schedule for HLS cross-channel QKV/WO accumulation"
    elif args.layout_mode == "s2_mlp_hidden_pairs":
        blocks, block_records = build_s2_mlp_hidden_pair_blocks(tensors, layout, args.strict, fallbacks)
        limitation = "S2 MLP hidden-pair schedule for explicit mlp_ratio hidden expansion over W1 and W2"
    else:
        blocks, block_records = build_s2_block_first_step_blocks(tensors, layout, args.strict, fallbacks)
        limitation = "S2 fused block first-step schedule: attention channel-pairs followed by MLP hidden-pairs per layer"

    elems = np.concatenate(blocks).astype(np.float32)
    raw = pack_lanes(quantize_fixed(elems, layout), layout)
    manifest = {
        "schema_version": "1.1",
        "input": str(args.input),
        "layout_mode": args.layout_mode,
        "layout": asdict(layout),
        "offsets": layout.offsets,
        "channel_tile": args.channel_tile if args.layout_mode == "s1_diagonal" else None,
        "channel_base": args.channel_tile * layout.tile_channels if args.layout_mode == "s1_diagonal" else None,
        "channel_tiles": layout.channel_tiles,
        "hidden_dim": layout.hidden_dim,
        "hidden_tiles": layout.hidden_tiles,
        "block_count": len(blocks),
        "block_records": block_records,
        "block_word_stride": layout.offsets["tile_words"],
        "binary": str(args.out_bin),
        "npz": str(args.out_npz),
        "element_count": int(elems.size),
        "word_count": len(raw) // (layout.bus_width // 8),
        "fallback_count": len(fallbacks),
        "fallbacks": fallbacks,
        "limitation": limitation,
    }
    write_outputs(raw, elems, layout, args.out_bin, args.out_npz, args.manifest, manifest)
    print(json.dumps({"output": str(args.out_bin), "manifest": str(args.manifest), "word_count": manifest["word_count"], "fallback_count": len(fallbacks)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
