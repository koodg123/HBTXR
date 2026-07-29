#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

HARDWARE_ROOT = Path(__file__).resolve().parents[1]
PYNQ_PARENT = HARDWARE_ROOT / "pynq"
if str(PYNQ_PARENT) not in sys.path:
    sys.path.insert(0, str(PYNQ_PARENT))

from hgtxr import e2e_m_axi_weights as weights

DEFAULT_PREFIX = HARDWARE_ROOT / "refs" / "weights" / "e2e_m_axi_active196_b6_ff768_q4_u32"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(bin_path: Path, arr: np.ndarray) -> dict[str, Any]:
    required_u32 = weights.REQUIRED_WEIGHT_WORDS_256B * weights.U32_PER_256B_WORD
    return {
        "name": "e2e_m_axi_active196_b6_ff768_q4_u32",
        "format": "raw-little-endian-uint32",
        "binary": str(bin_path),
        "sha256": sha256_file(bin_path),
        "bytes": bin_path.stat().st_size,
        "dtype": "uint32",
        "shape": [int(arr.size)],
        "capacity_u32": int(arr.size),
        "required_u32": int(required_u32),
        "tail_zero_start_u32": int(required_u32),
        "layout": {
            "bus_width_bits": 256,
            "weight_bits": weights.WEIGHT_BITS,
            "u32_per_256b_word": weights.U32_PER_256B_WORD,
            "q4_lanes_per_256b_word": weights.WEIGHT_LANES_PER_256B,
            "embed": weights.EMBED,
            "patch": weights.PATCH,
            "blocks": weights.BLOCKS,
            "ff_dim": weights.FF_DIM,
            "state": weights.STATE,
            "e2e_weight_words_256b": weights.E2E_WEIGHT_WORDS_256B,
            "required_weight_words_256b": weights.REQUIRED_WEIGHT_WORDS_256B,
            "patch_weight_elem_base": weights.PATCH_WEIGHT_ELEM_BASE,
            "block_weight_elem_base": weights.BLOCK_WEIGHT_ELEM_BASE,
            "head_weight_elem_base": weights.HEAD_WEIGHT_ELEM_BASE,
        },
        "expected_runtime_state": weights.EXPECTED_RUNTIME_STATE,
        "expected_raw": list(weights.EXPECTED_RAW),
        "known_offset_checks": {
            "patch_channel0_elem0": weights.get_q4_weight_raw(arr, weights.PATCH_WEIGHT_ELEM_BASE),
            "patch_channel2_elem0": weights.get_q4_weight_raw(
                arr, weights.PATCH_WEIGHT_ELEM_BASE + 2 * weights.PATCH_ELEMS
            ),
            "head_diag1": weights.get_q4_weight_raw(arr, weights.HEAD_WEIGHT_ELEM_BASE + weights.EMBED + 1),
            "block0_wq_diag0": weights.get_q4_weight_raw(arr, weights.BLOCK_WEIGHT_ELEM_BASE + weights.BLOCK_WQ),
        },
        "provenance": {
            "source": "hardware/pynq/hgtxr/e2e_m_axi_weights.py",
            "mirrors": "hardware/hls/tb/tb_hgtxr_e2e_m_axi_top.cpp::init_q4_vector_weights",
            "golden_header": "hardware/hls/tb/e2e_axis_vector_active196_b6_ff768_golden.hpp",
        },
    }


def export_weights(prefix: Path = DEFAULT_PREFIX) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = prefix.with_suffix(".bin")
    manifest_path = prefix.with_name(prefix.name + "_manifest.json")
    arr = weights.build_active196_b6_ff768_weights()
    arr.astype("<u4", copy=False).tofile(bin_path)
    manifest = build_manifest(bin_path, arr)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return bin_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export A2 E2E m_axi CSim-mirrored packed Q4 weights.")
    parser.add_argument("--prefix", type=Path, default=DEFAULT_PREFIX)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bin_path, manifest_path = export_weights(args.prefix)
    print(json.dumps({"binary": str(bin_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
