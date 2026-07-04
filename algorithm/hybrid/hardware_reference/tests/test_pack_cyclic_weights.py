from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_pack_cyclic_weights_self_test():
    result = subprocess.run(
        [sys.executable, "hardware/tools/pack_cyclic_weights.py", "--self-test"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["self_test"] == "pass"


def test_pack_cyclic_weights_npz_identity_fallback(tmp_path: Path):
    weights = tmp_path / "weights.npz"
    np.savez(
        weights,
        **{
            "backbone.blocks.0.norm.weight": np.ones(8, dtype=np.float32),
            "backbone.blocks.0.norm.bias": np.zeros(8, dtype=np.float32),
            "backbone.blocks.0.qkv.weight": np.zeros((24, 8), dtype=np.float32),
            "backbone.blocks.0.out.weight": np.eye(8, dtype=np.float32),
            "backbone.blocks.1.net.0.weight": np.ones(8, dtype=np.float32),
            "backbone.blocks.1.net.0.bias": np.zeros(8, dtype=np.float32),
            "backbone.blocks.1.net.1.weight": np.eye(8, dtype=np.float32),
            "backbone.blocks.1.net.4.weight": np.eye(8, dtype=np.float32),
        },
    )
    out_bin = tmp_path / "cyclic.bin"
    out_npz = tmp_path / "cyclic.npz"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "hardware/tools/pack_cyclic_weights.py",
            "--input",
            str(weights),
            "--out-bin",
            str(out_bin),
            "--out-npz",
            str(out_npz),
            "--manifest",
            str(manifest),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(manifest.read_text())
    assert payload["word_count"] == payload["offsets"]["total_words"]
    assert payload["fallback_count"] > 0
    assert out_bin.stat().st_size == payload["word_count"] * 8
    packed = np.load(out_npz)
    assert packed["words_u64"].shape[0] == payload["word_count"]


def test_pack_cyclic_weights_s2_channel_pairs(tmp_path: Path):
    weights = tmp_path / "weights_s2.npz"
    payload = {}
    embed_dim = 8
    hidden_dim = 8
    for layer in range(2):
        attn = 2 * layer
        mlp = 2 * layer + 1
        payload[f"backbone.blocks.{attn}.norm.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.norm.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.qkv.weight"] = np.zeros((3 * embed_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{attn}.out.weight"] = np.eye(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = np.zeros((hidden_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = np.zeros((embed_dim, hidden_dim), dtype=np.float32)
    np.savez(weights, **payload)

    out_bin = tmp_path / "cyclic_s2.bin"
    out_npz = tmp_path / "cyclic_s2.npz"
    manifest = tmp_path / "manifest_s2.json"
    subprocess.run(
        [
            sys.executable,
            "hardware/tools/pack_cyclic_weights.py",
            "--input",
            str(weights),
            "--layout-mode",
            "s2_channel_pairs",
            "--strict",
            "--out-bin",
            str(out_bin),
            "--out-npz",
            str(out_npz),
            "--manifest",
            str(manifest),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(manifest.read_text())
    assert data["layout_mode"] == "s2_channel_pairs"
    assert data["block_count"] == 8
    assert data["word_count"] == data["block_count"] * data["block_word_stride"]
    assert data["fallback_count"] == 0
    assert data["block_records"][0]["input_tile"] == 0
    assert data["block_records"][-1]["output_tile"] == 1




def test_pack_cyclic_weights_s2_mlp_hidden_pairs(tmp_path: Path):
    weights = tmp_path / "weights_s2_mlp.npz"
    payload = {}
    embed_dim = 8
    hidden_dim = 16
    for layer in range(2):
        attn = 2 * layer
        mlp = 2 * layer + 1
        payload[f"backbone.blocks.{attn}.norm.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.norm.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.qkv.weight"] = np.zeros((3 * embed_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{attn}.out.weight"] = np.eye(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = np.zeros((hidden_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = np.zeros((embed_dim, hidden_dim), dtype=np.float32)
    np.savez(weights, **payload)

    out_bin = tmp_path / "cyclic_s2_mlp.bin"
    out_npz = tmp_path / "cyclic_s2_mlp.npz"
    manifest = tmp_path / "manifest_s2_mlp.json"
    subprocess.run(
        [
            sys.executable,
            "hardware/tools/pack_cyclic_weights.py",
            "--input",
            str(weights),
            "--layout-mode",
            "s2_mlp_hidden_pairs",
            "--strict",
            "--out-bin",
            str(out_bin),
            "--out-npz",
            str(out_npz),
            "--manifest",
            str(manifest),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
            "--mlp-ratio",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(manifest.read_text())
    assert data["layout_mode"] == "s2_mlp_hidden_pairs"
    assert data["hidden_tiles"] == 4
    assert data["block_count"] == 32
    assert data["word_count"] == data["block_count"] * data["block_word_stride"]
    assert data["fallback_count"] == 0
    assert data["block_records"][0]["phase"] == "w1"
    assert data["block_records"][8]["phase"] == "w2"
    assert data["block_records"][16]["phase"] == "w1"
    assert data["block_records"][-1]["phase"] == "w2"



def test_pack_cyclic_weights_s2_block_first_step(tmp_path: Path):
    weights = tmp_path / "weights_s2_block.npz"
    payload = {}
    embed_dim = 8
    hidden_dim = 16
    for layer in range(2):
        attn = 2 * layer
        mlp = 2 * layer + 1
        payload[f"backbone.blocks.{attn}.norm.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.norm.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{attn}.qkv.weight"] = np.zeros((3 * embed_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{attn}.out.weight"] = np.eye(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = np.zeros((hidden_dim, embed_dim), dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = np.zeros((embed_dim, hidden_dim), dtype=np.float32)
    np.savez(weights, **payload)

    out_bin = tmp_path / "cyclic_s2_block.bin"
    out_npz = tmp_path / "cyclic_s2_block.npz"
    manifest = tmp_path / "manifest_s2_block.json"
    subprocess.run(
        [
            sys.executable,
            "hardware/tools/pack_cyclic_weights.py",
            "--input",
            str(weights),
            "--layout-mode",
            "s2_block_first_step",
            "--strict",
            "--out-bin",
            str(out_bin),
            "--out-npz",
            str(out_npz),
            "--manifest",
            str(manifest),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
            "--mlp-ratio",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(manifest.read_text())
    assert data["layout_mode"] == "s2_block_first_step"
    assert data["hidden_tiles"] == 4
    assert data["block_count"] == 40
    assert data["word_count"] == data["block_count"] * data["block_word_stride"]
    assert data["fallback_count"] == 0
    assert data["block_records"][0]["phase"] == "attention"
    assert data["block_records"][4]["phase"] == "w1"
    assert data["block_records"][20]["phase"] == "attention"
    assert data["block_records"][24]["phase"] == "w1"
    assert data["block_records"][-1]["phase"] == "w2"

def test_validate_s2_block_preln(tmp_path: Path):
    out = tmp_path / "s2_block_preln_validation.json"
    result = subprocess.run(
        [
            sys.executable,
            "hardware/tools/validate_s2_block_preln.py",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "pass"
    assert data["block_count"] == 40
    assert data["record_count"] == 40
    assert data["max_norm_param_error"] == 0.0
    assert data["max_preln_piecewise_abs_error"] <= 0.01
    assert json.loads(out.read_text()) == data

def test_validate_s2_block_full(tmp_path: Path):
    out = tmp_path / "s2_block_full_validation.json"
    result = subprocess.run(
        [
            sys.executable,
            "hardware/tools/validate_s2_block_full.py",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "pass"
    assert data["block_count"] == 40
    assert data["record_count"] == 40
    assert data["max_matrix_error"] == 0.0
    assert data["max_output_error"] == 0.0
    assert json.loads(out.read_text()) == data



def test_validate_s2_block_full_external_input(tmp_path: Path):
    weights = tmp_path / "weights_external.npz"
    tokens = tmp_path / "tokens.npz"
    first_out = tmp_path / "first_outputs.npz"
    second_out = tmp_path / "second_outputs.npz"
    report = tmp_path / "s2_block_full_external_validation.json"
    payload = {}
    embed_dim = 8
    hidden_dim = 16
    for layer in range(2):
        attn = 2 * layer
        mlp = 2 * layer + 1
        qkv = np.arange(3 * embed_dim * embed_dim, dtype=np.float32).reshape(3 * embed_dim, embed_dim)
        payload[f"backbone.blocks.{attn}.norm.weight"] = np.ones(embed_dim, dtype=np.float32) + layer * 0.01
        payload[f"backbone.blocks.{attn}.norm.bias"] = np.zeros(embed_dim, dtype=np.float32) + layer * 0.001
        payload[f"backbone.blocks.{attn}.qkv.weight"] = ((qkv % 11.0) - 5.0) / 128.0
        payload[f"backbone.blocks.{attn}.out.weight"] = np.eye(embed_dim, dtype=np.float32) * (0.5 + layer * 0.1)
        payload[f"backbone.blocks.{mlp}.net.0.weight"] = np.ones(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.0.bias"] = np.zeros(embed_dim, dtype=np.float32)
        payload[f"backbone.blocks.{mlp}.net.1.weight"] = np.arange(hidden_dim * embed_dim, dtype=np.float32).reshape(hidden_dim, embed_dim) / 256.0
        payload[f"backbone.blocks.{mlp}.net.4.weight"] = np.arange(embed_dim * hidden_dim, dtype=np.float32).reshape(embed_dim, hidden_dim) / 512.0
    np.savez(weights, **payload)
    np.savez(tokens, tokens=np.linspace(-0.25, 0.75, num=4 * embed_dim, dtype=np.float32).reshape(4, embed_dim))

    subprocess.run(
        [
            sys.executable,
            "hardware/tools/validate_s2_block_full.py",
            "--input",
            str(weights),
            "--tokens-input",
            str(tokens),
            "--out-output",
            str(first_out),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
            "--mlp-ratio",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "hardware/tools/validate_s2_block_full.py",
            "--input",
            str(weights),
            "--tokens-input",
            str(tokens),
            "--expected-output",
            str(first_out),
            "--out-output",
            str(second_out),
            "--out",
            str(report),
            "--embed-dim",
            "8",
            "--tile-channels",
            "4",
            "--tile-ff",
            "4",
            "--bus-width",
            "64",
            "--blocks",
            "2",
            "--mlp-ratio",
            "2",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    assert data["status"] == "pass"
    assert data["used_synthetic_payload"] is False
    assert data["expected_output_compare"] == "all_layers"
    assert data["max_expected_output_error"] == 0.0
    assert np.load(second_out)["output"].shape == (2, 4, 8)
    assert json.loads(report.read_text()) == data
