from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_s2_block_vector_cxx_compare(tmp_path: Path):
    exe = tmp_path / "tb_cyclic_s2_block_vector"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-I/tools/Xilinx/Vitis_HLS/2023.2/include",
            "-Ihardware/hls/include",
            "-include",
            "hardware/configs/zcu104_cyclic_s2_block_q4w8a_defines.h",
            "hardware/hls/tb/tb_cyclic_s2_block_vector.cpp",
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
    assert "s2_block_vector_global_max_abs_diff=0.06233680" in result.stdout
    assert "S2 block vector comparison passed" in result.stdout
