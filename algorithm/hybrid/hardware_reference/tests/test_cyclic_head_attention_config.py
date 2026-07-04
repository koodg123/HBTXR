from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def test_s2_attention_configs_are_head_aligned():
    for rel in [
        "hardware/configs/zcu104_cyclic_s2_attn_first_step_defines.h",
        "hardware/configs/zcu104_cyclic_s2_block_first_step_defines.h",
        "hardware/configs/zcu104_cyclic_s2_block_q4w8a_defines.h",
    ]:
        text = _read(rel)
        assert "#define HGTXR_TILE_CHANNELS 64" in text
        assert "#define HGTXR_CYCLIC_HEADS 3" in text
        assert "#define HGTXR_CYCLIC_HEAD_DIM 64" in text
        assert "#define HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT 3" in text
        assert "#define HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION 1" in text


def test_s2_attention_call_uses_configured_score_scale():
    top = _read("hardware/hls/src/hgtxr_top.cpp")
    assert "HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT" in top
    assert "Head-aligned S2 attention requires HGTXR_TILE_CHANNELS == HGTXR_CYCLIC_HEAD_DIM" in top

    params = _read("hardware/hls/include/hgtxr_cyclic_transformer_params.hpp")
    assert "#define HGTXR_CYCLIC_HEADS 3" in params
    assert "#define HGTXR_CYCLIC_HEAD_DIM (HGTXR_CYCLIC_MODEL_DIM / HGTXR_CYCLIC_HEADS)" in params


def test_head_aligned_attention_fixed_point_vector_compare(tmp_path: Path):
    exe = tmp_path / "tb_cyclic_head_attention"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I/tools/Xilinx/Vitis_HLS/2023.2/include",
            "-Ihardware/hls/include",
            "-include",
            "hardware/configs/zcu104_cyclic_s2_block_q4w8a_defines.h",
            "hardware/hls/tb/tb_cyclic_head_attention.cpp",
            "-o",
            str(exe),
        ],
        cwd=ROOT,
        check=True,
    )
    result = subprocess.run(
        [str(exe)],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "head_attention_score_shift=3" in result.stdout
    assert "head attention fixed-point vector comparison passed" in result.stdout

