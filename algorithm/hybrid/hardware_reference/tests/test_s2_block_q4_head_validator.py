from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_validate_s2_block_full_q4_head_aligned_roundtrip(tmp_path: Path):
    weights = tmp_path / "weights_q4_head.npz"
    tokens = tmp_path / "tokens_q4_head.npz"
    report = tmp_path / "s2_block_full_q4_head_validation.json"
    payload = {}
    embed_dim = 12
    hidden_dim = 24
    for layer in range(2):
        attn = 2 * layer
        mlp = 2 * layer + 1
        qkv = np.arange(3 * embed_dim * embed_dim, dtype=np.float32).reshape(
            3 * embed_dim, embed_dim
        )
        payload[f"backbone.blocks.{attn}.norm.weight"] = np.ones(
            embed_dim, dtype=np.float32
        )
        payload[f"backbone.blocks.{attn}.norm.bias"] = np.zeros(
            embed_dim, dtype=np.float32
        )
        payload[f"backbone.blocks.{attn}.qkv.weight"] = ((qkv % 13.0) - 6.0) / 16.0
        payload[f"backbone.blocks.{attn}.out.weight"] = (
            np.eye(embed_dim, dtype=np.float32) * 0.5
        )
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(
            embed_dim, dtype=np.float32
        )
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(
            embed_dim, dtype=np.float32
        )
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = (
            np.arange(hidden_dim * embed_dim, dtype=np.float32).reshape(
                hidden_dim, embed_dim
            )
            / 512.0
        )
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = (
            np.arange(embed_dim * hidden_dim, dtype=np.float32).reshape(
                embed_dim, hidden_dim
            )
            / 768.0
        )
    np.savez(weights, **payload)
    np.savez(
        tokens,
        tokens=np.linspace(-0.5, 0.5, num=4 * embed_dim, dtype=np.float32).reshape(
            4, embed_dim
        ),
    )

    result = subprocess.run(
        [
            sys.executable,
            "hardware/tools/validate_s2_block_full.py",
            "--input",
            str(weights),
            "--tokens-input",
            str(tokens),
            "--out",
            str(report),
            "--embed-dim",
            "12",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--bit-width",
            "4",
            "--int-width",
            "2",
            "--blocks",
            "2",
            "--mlp-ratio",
            "2",
            "--num-heads",
            "3",
            "--score-scale-shift",
            "1",
            "--use-hgpipe-lut-math",
            "1",
            "--packed-roundtrip",
            "--max-packed-roundtrip-error",
            "0.126",
            "--max-matrix-error",
            "0.126",
            "--max-output-error",
            "100.0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "pass"
    assert data["packed_roundtrip"] is True
    assert data["attention"] == {
        "num_heads": 3,
        "head_dim": 4,
        "score_scale_shift": 1,
        "use_hgpipe_lut_math": True,
    }
    assert data["max_packed_roundtrip_error"] <= 0.126
    assert json.loads(report.read_text()) == data
