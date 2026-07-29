from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_runtime_mode_board_latency_gate as gate_tool  # noqa: E402


def result_payload(profile: str, latency_ms: float) -> dict:
    runtime_state = 0 if profile == "search" else 1
    return {
        "status": "pass",
        "variant": "runtime-mode-par32",
        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit",
        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh",
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "weights_mode": "zero",
        "mode_profile": profile,
        "expected_runtime_state": runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": True,
        "out_state": [0, 0, 0, 0, 0, 0],
        "out_raw": [0, 0, 0, 0, 0, 0],
        "expected_out_raw": None,
        "output_match": True,
        "repeat": 1,
        "latency_ms_samples": [latency_ms],
        "latency_ms_summary": {"count": 1, "max": latency_ms},
        "accelerator_latency_ms_samples": [latency_ms],
        "accelerator_latency_ms_summary": {"count": 1, "max": latency_ms},
    }


def par32_rom_result_payload(profile: str, latency_ms: float) -> dict:
    runtime_state = 0 if profile == "search" else 1
    expected = [-1169, -1169, -1169, -1169, -1169, -1133] if profile == "search" else [-235, -235, -235, -235, -235, -226]
    return {
        "status": "pass",
        "variant": "par32-rom-compute-300",
        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.bit",
        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.hwh",
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "weights_mode": "zero",
        "mode_profile": profile,
        "expected_runtime_state": runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": True,
        "out_state": [float(value) / 16.0 for value in expected],
        "out_raw": expected,
        "expected_out_raw": expected,
        "output_match": True,
        "repeat": 1,
        "latency_ms_samples": [latency_ms],
        "latency_ms_summary": {"count": 1, "max": latency_ms},
        "accelerator_latency_ms_samples": [latency_ms],
        "accelerator_latency_ms_summary": {"count": 1, "max": latency_ms},
    }


def par32_prefetchall4_result_payload(profile: str, latency_ms: float) -> dict:
    runtime_state = 0 if profile == "search" else 1
    expected = [-1169, -1169, -1169, -1169, -1169, -1125] if profile == "search" else [-235, -235, -235, -235, -235, -239]
    return {
        "status": "pass",
        "variant": "par32-prefetchall4-300",
        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit",
        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.hwh",
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "weights_mode": "zero",
        "mode_profile": profile,
        "expected_runtime_state": runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": True,
        "out_state": [float(value) / 16.0 for value in expected],
        "out_raw": expected,
        "expected_out_raw": expected,
        "output_match": True,
        "repeat": 1,
        "latency_ms_samples": [latency_ms],
        "latency_ms_summary": {"count": 1, "max": latency_ms},
        "accelerator_latency_ms_samples": [latency_ms],
        "accelerator_latency_ms_summary": {"count": 1, "max": latency_ms},
    }


class RuntimeModeBoardLatencyGateTests(unittest.TestCase):
    def test_gate_reports_missing_without_board_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = gate_tool.build_gate(Path(tmp))
        self.assertEqual(gate["status"], "missing")
        self.assertEqual(gate["summary"]["cases_missing"], 2)

    def test_par32_rom_compute_gate_reports_missing_without_board_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = gate_tool.build_gate(
                Path(tmp),
                cases_cfg=gate_tool.PAR32_ROM_COMPUTE_CASES,
                target=gate_tool.PROFILE_SET_TARGETS["par32-rom-compute-300"],
                profile_set="par32-rom-compute-300",
            )
        self.assertEqual(gate["status"], "missing")
        self.assertEqual(gate["profile_set"], "par32-rom-compute-300")
        self.assertEqual(gate["summary"]["cases_missing"], 2)

    def test_gate_passes_when_search_and_track_results_meet_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "hardware" / "pynq" / "hgtxr"
            out_dir.mkdir(parents=True)
            (out_dir / "e2e_axis_dma_runtime_mode_par32_search_file_smoke.json").write_text(
                json.dumps(result_payload("search", 3.5))
            )
            (out_dir / "e2e_axis_dma_runtime_mode_par32_track_file_smoke.json").write_text(
                json.dumps(result_payload("track", 0.8))
            )
            gate = gate_tool.build_gate(root)
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["summary"]["cases_pass"], 2)

    def test_par32_rom_compute_gate_passes_with_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "hardware" / "pynq" / "hgtxr"
            out_dir.mkdir(parents=True)
            (out_dir / "e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json").write_text(
                json.dumps(par32_rom_result_payload("search", 3.5))
            )
            (out_dir / "e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json").write_text(
                json.dumps(par32_rom_result_payload("track", 0.8))
            )
            gate = gate_tool.build_gate(
                root,
                cases_cfg=gate_tool.PAR32_ROM_COMPUTE_CASES,
                target=gate_tool.PROFILE_SET_TARGETS["par32-rom-compute-300"],
                profile_set="par32-rom-compute-300",
            )
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["summary"]["cases_pass"], 2)

    def test_gate_fails_when_track_exceeds_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "hardware" / "pynq" / "hgtxr"
            out_dir.mkdir(parents=True)
            (out_dir / "e2e_axis_dma_runtime_mode_par32_search_file_smoke.json").write_text(
                json.dumps(result_payload("search", 3.5))
            )
            (out_dir / "e2e_axis_dma_runtime_mode_par32_track_file_smoke.json").write_text(
                json.dumps(result_payload("track", 1.2))
            )
            gate = gate_tool.build_gate(root)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["summary"]["cases_fail"], 1)

    def test_par32_prefetchall4_gate_fails_when_track_exceeds_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "hardware" / "pynq" / "hgtxr"
            out_dir.mkdir(parents=True)
            (out_dir / "e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json").write_text(
                json.dumps(par32_prefetchall4_result_payload("search", 3.5))
            )
            (out_dir / "e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json").write_text(
                json.dumps(par32_prefetchall4_result_payload("track", 1.2))
            )
            gate = gate_tool.build_gate(
                root,
                cases_cfg=gate_tool.PAR32_PREFETCHALL4_CASES,
                target=gate_tool.PROFILE_SET_TARGETS["par32-prefetchall4-300"],
                profile_set="par32-prefetchall4-300",
            )
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["summary"]["cases_fail"], 1)


if __name__ == "__main__":
    unittest.main()
