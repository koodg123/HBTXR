from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class E2EAxisCase:
    name: str
    spec: Path
    header: Path
    extra_defines: tuple[str, ...] = ()


CASES = [
    E2EAxisCase(
        name="reduced",
        spec=Path("hardware/refs/e2e_axis_vector_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_golden.hpp"),
    ),
    E2EAxisCase(
        name="active8",
        spec=Path("hardware/refs/e2e_axis_vector_active8_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active8_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE8_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active16",
        spec=Path("hardware/refs/e2e_axis_vector_active16_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active16_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE16_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active16_ff64",
        spec=Path("hardware/refs/e2e_axis_vector_active16_ff64_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active16_ff64_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE16_FF64_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active16_ff128",
        spec=Path("hardware/refs/e2e_axis_vector_active16_ff128_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active16_ff128_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE16_FF128_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active16_ff256",
        spec=Path("hardware/refs/e2e_axis_vector_active16_ff256_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active16_ff256_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE16_FF256_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active16_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active16_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active16_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE16_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active32_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active32_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active32_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE32_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active64_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active64_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active64_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE64_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active128_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active128_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active128_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE128_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active196_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active196_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active196_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE196_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="active196_b6_ff768",
        spec=Path("hardware/refs/e2e_axis_vector_active196_b6_ff768_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_active196_b6_ff768_golden.hpp"),
        extra_defines=("-DHGTXR_E2E_USE_ACTIVE196_B6_FF768_GOLDEN=1",),
    ),
    E2EAxisCase(
        name="hgpipe_math",
        spec=Path("hardware/refs/e2e_axis_vector_hgpipe_math_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_hgpipe_math_golden.hpp"),
        extra_defines=(
            "-DHGTXR_E2E_USE_HGPIPE_MATH_GOLDEN=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4",
        ),
    ),
    E2EAxisCase(
        name="hgpipe_math_lnq",
        spec=Path("hardware/refs/e2e_axis_vector_hgpipe_math_lnq_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp"),
        extra_defines=(
            "-DHGTXR_E2E_USE_HGPIPE_MATH_LNQ_GOLDEN=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT=33",
        ),
    ),
    E2EAxisCase(
        name="hgpipe_math_lnq_active8",
        spec=Path("hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp"),
        extra_defines=(
            "-DHGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE8_GOLDEN=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT=33",
        ),
    ),
    E2EAxisCase(
        name="hgpipe_math_lnq_active16",
        spec=Path("hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active16_spec.json"),
        header=Path("hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active16_golden.hpp"),
        extra_defines=(
            "-DHGTXR_E2E_USE_HGPIPE_MATH_LNQ_ACTIVE16_GOLDEN=1",
            "-DHGTXR_E2E_USE_HGPIPE_INT_GELUQ=1",
            "-DHGTXR_E2E_HGPIPE_GELUQ_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_GELUQ_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_SOFTMAX_PROB_SCALE=4",
            "-DHGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ=1",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_INPUT_SCALE=16",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_OUTPUT_SCALE=4",
            "-DHGTXR_E2E_HGPIPE_LAYERNORM_BIAS_SHIFT=33",
        ),
    ),
]


def build_reference(tmp_path: Path, case: E2EAxisCase) -> dict:
    json_out = tmp_path / f"e2e_axis_vector_{case.name}_ref.json"
    result = subprocess.run(
        [
            "python3",
            "hardware/tools/validate_e2e_axis_vector.py",
            "--spec",
            str(case.spec),
            "--json-out",
            str(json_out),
            "--check-header",
            str(case.header),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    payload = json.loads(json_out.read_text())
    assert "ap_fixed<4,2>" in result.stdout
    assert "Reduced dense gate mirrors HLS" in result.stdout
    return payload


def compile_defines(payload: dict, case: E2EAxisCase) -> list[str]:
    cfg = payload["config"]
    return [
        *case.extra_defines,
        f"-DHGTXR_E2E_BLOCKS={cfg['blocks']}",
        f"-DHGTXR_E2E_ACTIVE_TOKENS={cfg['active_tokens']}",
        f"-DHGTXR_E2E_PATCH_GRID_H={cfg['patch_grid_h']}",
        f"-DHGTXR_E2E_PATCH_GRID_W={cfg['patch_grid_w']}",
        f"-DHGTXR_E2E_FF_DIM={cfg['ff_dim']}",
    ]


def test_e2e_axis_sw_reference(tmp_path: Path):
    for case in CASES:
        payload = build_reference(tmp_path, case)
        assert len(payload["expected_raw"]) == payload["config"]["state"]
        assert payload["runtime_state"] == 1 + payload["pattern"]["live_weight_bit"]
        assert payload["nonzero_counts"]["patch"] > len(payload["pattern"]["patch_raw"])
        assert payload["nonzero_counts"]["head"] > len(payload["pattern"]["head_raw"])
        assert payload["nonzero_counts"]["q"] > payload["config"]["embed"]
        assert payload["nonzero_counts"]["wo"] > payload["config"]["embed"]


def test_e2e_axis_vector_cxx_compare(tmp_path: Path):
    for case in CASES:
        payload = build_reference(tmp_path, case)
        exe = tmp_path / f"tb_hgtxr_e2e_axis_top_{case.name}"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-I/tools/Xilinx/Vitis_HLS/2023.2/include",
                "-Ihardware/hls/include",
                "-include",
                "hardware/configs/zcu104_e2e_q4w8a_defines.h",
                *compile_defines(payload, case),
                "hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp",
                "hardware/hls/src/hgtxr_e2e_axis_top.cpp",
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
        expected = payload["expected_raw"]
        runtime_state = payload["runtime_state"]
        assert f"runtime_state={runtime_state} count={len(expected)} last=1 failures=0" in result.stdout
        for idx, value in enumerate(expected):
            last = 1 if idx == len(expected) - 1 else 0
            assert f"e2e_axis_out[{idx}]={value} last={last}" in result.stdout
        assert "E2E AXIS vector comparison passed" in result.stdout
