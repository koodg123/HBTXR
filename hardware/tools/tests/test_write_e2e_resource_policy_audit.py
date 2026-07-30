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

import write_e2e_resource_policy_audit as audit_tool  # noqa: E402


class WriteE2EResourcePolicyAuditTests(unittest.TestCase):
    def test_build_audit_passes_on_current_workspace(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(audit["check_count"], 36)
        self.assertGreaterEqual(audit["observed"]["bind_op_dsp_count"], 2)
        self.assertGreaterEqual(audit["observed"]["rmu_smu_bind_op_dsp_count"], 2)
        self.assertGreaterEqual(audit["observed"]["rmu_smu_dsp_helper_count"], 2)
        self.assertGreaterEqual(audit["observed"]["rmu_smu_dsp_acc_helper_count"], 2)
        self.assertGreaterEqual(audit["observed"]["bind_storage_lutram_count"], 3)
        self.assertGreaterEqual(audit["observed"]["cyclic_weight_uram_count"], 6)
        self.assertGreaterEqual(audit["observed"]["cyclic_large_temp_uram_count"], 4)
        self.assertGreaterEqual(audit["observed"]["cyclic_small_tile_lutram_count"], 8)
        self.assertEqual(audit["observed"]["c3b"]["parallelism"], 16)
        self.assertEqual(audit["observed"]["c3b"]["memory_banks"], 16)
        self.assertGreater(audit["observed"]["c3b"]["dsp"], 0)
        self.assertGreater(audit["observed"]["c3b"]["uram"], 0)
        self.assertEqual(audit["observed"]["c3b_csynth"]["resources"]["dsp"], audit["observed"]["c3b"]["dsp"])
        self.assertEqual(audit["observed"]["c3b_csynth"]["resources"]["uram"], audit["observed"]["c3b"]["uram"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "resource_policy.json"
            markdown_out = Path(tmp) / "resource_policy.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--root",
                        str(ROOT.parent),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertIn("HGTXR E2E Resource Policy Audit", markdown_out.read_text())
            self.assertIn("c3b_dsp_increased_vs_a1", markdown_out.read_text())
            self.assertIn("rmu_projection_uses_dsp_helper", markdown_out.read_text())
            self.assertIn("smu_relation_uses_dsp_helper", markdown_out.read_text())
            self.assertIn("cyclic_weight_tiles_uram_pragmas", markdown_out.read_text())
            self.assertIn("c3b_csynth_resources_match_matrix", markdown_out.read_text())
            self.assertIn("c3b_csynth_lut_lte_threshold", markdown_out.read_text())
            self.assertIn("c3b_csynth_latency_lte_threshold", markdown_out.read_text())
            self.assertIn("c3b_routed_wns_gte_threshold", markdown_out.read_text())

    def test_matrix_resource_drift_fails_against_csynth_xml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "matrix.json"
            matrix = json.loads((ROOT / "generated/signoff/e2e_resource_matrix_2026_06_10.json").read_text())
            for row in matrix["rows"]:
                if row["id"] == "C3b":
                    row["hls"]["resources"]["dsp"] = 1
                    row["hls"]["latency_cycles"] = 1
                    break
            matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")

            audit = audit_tool.build_audit(ROOT.parent, matrix_path)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertEqual(audit["status"], "fail")
            self.assertIn("c3b_csynth_resources_match_matrix", failed)
            self.assertIn("c3b_csynth_latency_matches_matrix", failed)

    def test_c3b_threshold_drift_fails(self) -> None:
        old_lut = audit_tool.C3B_LUT_MAX
        old_latency = audit_tool.C3B_LATENCY_MAX
        old_wns = audit_tool.C3B_WNS_MIN
        try:
            audit_tool.C3B_LUT_MAX = 1
            audit_tool.C3B_LATENCY_MAX = 1
            audit_tool.C3B_WNS_MIN = 99.0
            audit = audit_tool.build_audit(ROOT.parent)
        finally:
            audit_tool.C3B_LUT_MAX = old_lut
            audit_tool.C3B_LATENCY_MAX = old_latency
            audit_tool.C3B_WNS_MIN = old_wns

        failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
        self.assertEqual(audit["status"], "fail")
        self.assertIn("c3b_csynth_lut_lte_threshold", failed)
        self.assertIn("c3b_csynth_latency_lte_threshold", failed)
        self.assertIn("c3b_routed_wns_gte_threshold", failed)


if __name__ == "__main__":
    unittest.main()
