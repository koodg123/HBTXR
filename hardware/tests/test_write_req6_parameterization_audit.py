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

import write_req6_parameterization_audit as audit_tool  # noqa: E402


class Req6ParameterizationAuditTests(unittest.TestCase):
    def populate_minimal_tree(self, root: Path) -> Path:
        hgtxr = root / "HGTXR"
        hardware = hgtxr / "hardware"
        for rel, source_rel in [
            ("configs/zcu104_e2e_q4w8a_defines.h", "configs/zcu104_e2e_q4w8a_defines.h"),
            ("hls/include/hgtxr_cyclic_transformer_params.hpp", "hls/include/hgtxr_cyclic_transformer_params.hpp"),
            ("hls/include/hgtxr_e2e_vit.hpp", "hls/include/hgtxr_e2e_vit.hpp"),
            ("hls/src/hgtxr_e2e_axis_top.cpp", "hls/src/hgtxr_e2e_axis_top.cpp"),
            ("hls/src/hgtxr_e2e_m_axi_top.cpp", "hls/src/hgtxr_e2e_m_axi_top.cpp"),
            ("vivado/scripts/run_e2e_q4w8a_csim.tcl", "vivado/scripts/run_e2e_q4w8a_csim.tcl"),
            ("vivado/scripts/run_e2e_q4w8a_csynth.tcl", "vivado/scripts/run_e2e_q4w8a_csynth.tcl"),
            ("configs/sweeps/zcu104_cyclic_transformer_sweep.yaml", "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml"),
        ]:
            dst = hardware / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text((ROOT / source_rel).read_text(errors="replace"))
        return hgtxr

    def test_current_workspace_passes(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(audit["check_count"], 71)
        self.assertEqual(audit["knobs"]["config_macros"]["HGTXR_PARALLELISM_FACTOR"], 8)
        self.assertEqual(audit["knobs"]["config_macros"]["HGTXR_BUS_WIDTH"], 256)
        self.assertEqual(audit["knobs"]["config_macros"]["HGTXR_BIT_WIDTH"], 8)
        self.assertEqual(audit["knobs"]["legality_macros"]["HGTXR_WEIGHT_LANES"], 64)
        self.assertEqual(audit["knobs"]["sweep_values"]["parallelism_factor"], [1, 2, 4, 8, 16, 32])
        self.assertEqual(
            audit["knobs"]["parallelism_extensions"]["C3b_PAR16"]["status"],
            "validated_resource_matrix",
        )
        self.assertEqual(
            audit["knobs"]["parallelism_extensions"]["PAR32"]["status"],
            "exploratory_not_default",
        )
        names = {check["name"] for check in audit["checks"]}
        self.assertIn("tiling_to_tile_tokens", names)
        self.assertIn("axis_ports_use_fifo_depth", names)
        self.assertIn("sweep_covers_parallelism_factor", names)
        self.assertIn("parallelism_extension_c3b_par16_validated", names)
        self.assertIn("parallelism_extension_c3b_par16_evidence_resource_matrix", names)
        self.assertIn("parallelism_extension_par32_exploratory_not_default", names)
        self.assertIn("parallelism_extension_par32_requires_fresh_reports", names)
        self.assertIn("csim_tcl_supports_par16_par32", names)
        self.assertIn("csynth_tcl_supports_par16_par32", names)
        self.assertIn("legal_bus_width_weight_divisible", names)
        self.assertIn("legal_dense_parallelism_divides_embed", names)
        self.assertIn("legal_weight_lanes_divide_dense_parallelism", names)
        self.assertIn("e2e_static_assert_head_dim_covers_embed", names)
        self.assertIn("e2e_static_assert_dense_par_divides_weight_lanes", names)
        self.assertIn("e2e_static_assert_axis_width_matches_cyclic_axi", names)

    def test_missing_config_knob_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hgtxr = self.populate_minimal_tree(Path(tmp))
            hardware = hgtxr / "hardware"
            config = hardware / "configs/zcu104_e2e_q4w8a_defines.h"
            config.write_text(config.read_text().replace("#define HGTXR_FIFO_DEPTH 128\n", ""))

            audit = audit_tool.build_audit(hgtxr)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("config_defines_HGTXR_FIFO_DEPTH", failed)

    def test_missing_parallelism_extension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hgtxr = self.populate_minimal_tree(Path(tmp))
            sweep = hgtxr / "hardware/configs/sweeps/zcu104_cyclic_transformer_sweep.yaml"
            text = sweep.read_text()
            text = text.replace('      status: "exploratory_not_default"\n', '      status: "promoted_default"\n')
            sweep.write_text(text)

            audit = audit_tool.build_audit(hgtxr)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("parallelism_extension_par32_exploratory_not_default", failed)

    def test_illegal_bus_width_fails_legality_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hgtxr = self.populate_minimal_tree(Path(tmp))
            config = hgtxr / "hardware/configs/zcu104_e2e_q4w8a_defines.h"
            config.write_text(config.read_text().replace("#define HGTXR_BUS_WIDTH 256\n", "#define HGTXR_BUS_WIDTH 250\n"))

            audit = audit_tool.build_audit(hgtxr)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("legal_bus_width_byte_aligned", failed)
            self.assertIn("legal_bus_width_data_divisible", failed)
            self.assertIn("legal_bus_width_weight_divisible", failed)

    def test_illegal_parallelism_fails_legality_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hgtxr = self.populate_minimal_tree(Path(tmp))
            config = hgtxr / "hardware/configs/zcu104_e2e_q4w8a_defines.h"
            text = config.read_text()
            text = text.replace("#define HGTXR_PARALLELISM_FACTOR 8\n", "#define HGTXR_PARALLELISM_FACTOR 7\n")
            text = text.replace("#define HGTXR_E2E_DENSE_PAR 8\n", "#define HGTXR_E2E_DENSE_PAR 7\n")
            config.write_text(text)

            audit = audit_tool.build_audit(hgtxr)

            self.assertEqual(audit["status"], "fail")
            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("legal_dense_parallelism_divides_embed", failed)
            self.assertIn("legal_dense_parallelism_divides_ff_dim", failed)
            self.assertIn("legal_weight_lanes_divide_dense_parallelism", failed)

    def test_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "req6.json"
            md_out = Path(tmp) / "req6.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(md_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertIn("Req6 Parameterization Audit", md_out.read_text())
            self.assertIn("HGTXR_PARALLELISM_FACTOR", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
