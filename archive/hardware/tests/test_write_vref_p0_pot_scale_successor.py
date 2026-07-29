#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_vref_p0_pot_scale_successor as successor_tool  # noqa: E402


class WriteVrefP0PotScaleSuccessorTests(unittest.TestCase):
    def test_build_package_writes_successor_spec_and_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_out = tmp_path / "successor_spec.json"
            header_out = tmp_path / "successor_golden.hpp"
            package = successor_tool.build_package(
                ROOT.parent,
                spec_out=spec_out,
                header_out=header_out,
            )

            self.assertEqual(package["status"], "pass")
            self.assertEqual(package["candidate"], "softmax_input_x2")
            self.assertTrue(spec_out.exists())
            self.assertTrue(header_out.exists())
            spec_payload = json.loads(spec_out.read_text())
            self.assertEqual(spec_payload["math"]["softmax_input_scale"], 32)
            self.assertEqual(package["math"]["softmax_input_scale"], 32)
            self.assertIn("HGTXR_E2E_SCALE", package["csim"]["env"])
            self.assertEqual(package["csim"]["env"]["HGTXR_E2E_SCALE"], "custom")
            self.assertEqual(package["csynth"]["env"]["HGTXR_E2E_SCALE"], "custom")
            self.assertTrue(package["csim"]["project_name"].endswith("_csim"))
            self.assertTrue(package["csynth"]["project_name"].endswith("_csynth"))
            self.assertEqual(package["ip_package"]["env"]["HGTXR_E2E_RESOURCE_POLICY"], "dsp_mixed_stream")
            self.assertIn("package_e2e_axis_ip.tcl", package["ip_package"]["command"])
            self.assertIn("build_e2e_axis_dma_bitstream.tcl", package["vivado_overlay"]["command"])
            self.assertIn("LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9", package["vivado_overlay"]["command"])
            self.assertIn("HGTXR_E2E_USE_VREF_P0_HGPIPE_LNQ_ACTIVE16_SOFTMAX_INPUT_X2_GOLDEN", package["custom_scale_flags"])
            self.assertIn("kExpectedRaw", header_out.read_text())

    def test_cli_writes_manifest_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_out = tmp_path / "successor.json"
            markdown_out = tmp_path / "successor.md"
            spec_out = tmp_path / "successor_spec.json"
            header_out = tmp_path / "successor_golden.hpp"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = successor_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--spec-out",
                        str(spec_out),
                        "--header-out",
                        str(header_out),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertIn("HGTXR_E2E_CUSTOM_SCALE_FLAGS", payload["csim"]["command"])
            self.assertIn("run_e2e_q4w8a_csynth.tcl", payload["csynth"]["command"])
            self.assertIn("package_e2e_axis_ip.tcl", payload["ip_package"]["command"])
            self.assertIn("vivado_overlay_result", payload)
            self.assertIn("PoT Scale Successor", markdown_out.read_text())

    def test_csim_tcl_supports_custom_scale_mode(self) -> None:
        text = (ROOT / "vivado" / "scripts" / "run_e2e_q4w8a_csim.tcl").read_text()
        self.assertIn('$e2e_scale eq "custom"', text)
        self.assertIn("HGTXR_E2E_CUSTOM_SCALE_FLAGS", text)

    def test_csynth_tcl_supports_custom_scale_mode(self) -> None:
        text = (ROOT / "vivado" / "scripts" / "run_e2e_q4w8a_csynth.tcl").read_text()
        self.assertIn('$e2e_scale eq "custom"', text)
        self.assertIn("HGTXR_E2E_CUSTOM_SCALE_FLAGS", text)

    def test_c3b_projection_flags_uram_over_threshold(self) -> None:
        projection = successor_tool.build_c3b_projection(
            {
                "status": "pass",
                "latency_cycles": 498485,
                "estimated_clock_ns": 4.069,
                "resources": {
                    "bram_18k": 64,
                    "dsp": 128,
                    "ff": 19664,
                    "lut": 43236,
                    "uram": 88,
                },
            }
        )

        self.assertEqual(projection["status"], "not_promotable")
        self.assertIn("uram_lte_c3b", projection["failures"])
        self.assertIn("routed_timing_available", projection["pending"])

    def test_c3b_projection_accepts_dsp_mixed_stream_uram(self) -> None:
        projection = successor_tool.build_c3b_projection(
            {
                "status": "pass",
                "latency_cycles": 498485,
                "estimated_clock_ns": 4.069,
                "resources": {
                    "bram_18k": 144,
                    "dsp": 128,
                    "ff": 19664,
                    "lut": 43236,
                    "uram": 32,
                },
            }
        )

        self.assertEqual(projection["status"], "not_promotable")
        self.assertEqual(projection["failures"], [])
        self.assertEqual(projection["pending"], ["routed_timing_available", "physical_smoke_available"])

    def test_c3b_projection_accepts_routed_timing_for_dsp_mixed_stream(self) -> None:
        projection = successor_tool.build_c3b_projection(
            {
                "status": "pass",
                "latency_cycles": 498485,
                "estimated_clock_ns": 4.069,
                "resources": {
                    "bram_18k": 144,
                    "dsp": 128,
                    "ff": 19664,
                    "lut": 43236,
                    "uram": 32,
                },
            },
            {"status": "pass", "timing": {"wns_ns": 4.497}},
        )

        self.assertEqual(projection["status"], "not_promotable")
        self.assertEqual(projection["failures"], [])
        self.assertEqual(projection["pending"], ["physical_smoke_available"])

    def test_c3b_projection_promotes_with_routed_timing_and_physical_smoke(self) -> None:
        projection = successor_tool.build_c3b_projection(
            {
                "status": "pass",
                "latency_cycles": 498485,
                "estimated_clock_ns": 4.069,
                "resources": {
                    "bram_18k": 144,
                    "dsp": 128,
                    "ff": 19664,
                    "lut": 43236,
                    "uram": 32,
                },
            },
            {"status": "pass", "timing": {"wns_ns": 4.497}},
            {"status": "pass", "result_json": "pynq/hgtxr/vref_smoke.json", "errors": []},
        )

        self.assertEqual(projection["status"], "promotable")
        self.assertEqual(projection["failures"], [])
        self.assertEqual(projection["pending"], [])

    def test_c3b_projection_fails_invalid_physical_smoke(self) -> None:
        projection = successor_tool.build_c3b_projection(
            {
                "status": "pass",
                "latency_cycles": 498485,
                "estimated_clock_ns": 4.069,
                "resources": {
                    "bram_18k": 144,
                    "dsp": 128,
                    "ff": 19664,
                    "lut": 43236,
                    "uram": 32,
                },
            },
            {"status": "pass", "timing": {"wns_ns": 4.497}},
            {"status": "fail", "result_json": "pynq/hgtxr/vref_smoke.json", "errors": ["out_raw mismatch"]},
        )

        self.assertEqual(projection["status"], "not_promotable")
        self.assertIn("physical_smoke_available", projection["failures"])
        self.assertEqual(projection["pending"], [])

    def test_read_successor_physical_smoke_result_validates_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hardware = Path(tmp) / "hardware"
            result_path = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "variant": "vref-p0-softmax-input-x2-dsp-mixed-stream",
                        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
                        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
                        "dma_in_name": "axi_dma_in",
                        "dma_out_name": "axi_dma_out",
                        "weights_mode": "file",
                        "expected_runtime_state": 2,
                        "runtime_state": 2,
                        "runtime_match": True,
                        "out_state": [3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
                        "out_raw": [58, -51, 42, -28, 36, -41],
                        "expected_out_raw": [58, -51, 42, -28, 36, -41],
                        "output_match": True,
                    }
                )
            )

            result = successor_tool.read_successor_physical_smoke_result(hardware)

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["result_json"], str(result_path))
        self.assertEqual(result["errors"], [])

    def test_vivado_report_parsers_match_routed_report_shape(self) -> None:
        timing = successor_tool.parse_timing_summary(
            """
| Design Timing Summary
    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints      WHS(ns)      THS(ns)  THS Failing Endpoints
      4.497        0.000                      0                49844        0.010        0.000                      0
"""
        )
        route = successor_tool.parse_route_status(
            """
       # of routable nets..................... :       19829 :
           # of fully routed nets............. :       19829 :
       # of nets with routing errors.......... :           0 :
"""
        )

        self.assertEqual(timing["wns_ns"], 4.497)
        self.assertEqual(timing["whs_ns"], 0.010)
        self.assertEqual(route["fully_routed_nets"], 19829)
        self.assertEqual(route["routing_error_nets"], 0)


if __name__ == "__main__":
    unittest.main()
