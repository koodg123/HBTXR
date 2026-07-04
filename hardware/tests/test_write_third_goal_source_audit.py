#!/usr/bin/env python3
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

import write_third_goal_source_audit as audit_tool  # noqa: E402


class ThirdGoalSourceAuditTests(unittest.TestCase):
    def write(self, base: Path, rel: str, text: str = "# HGTXR ZCU104 C3b VREF E2E\n") -> None:
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        hgtxr = Path(tmp.name) / "XR-VIT" / "HGTXR"
        hardware = hgtxr / "hardware"
        for group, base_name, rel in audit_tool.required_sources():
            base = hardware if base_name == "hardware" else hgtxr
            self.write(base, rel, f"# {group} HGTXR ZCU104 C3b VREF E2E HANDOVER\n")
        return tmp, hgtxr, hardware

    def test_build_audit_passes_when_required_sources_exist(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["required_count"], len(audit_tool.required_sources()))
        self.assertEqual(audit["missing_required"], [])
        self.assertIn("handover", audit["group_summary"])
        required_paths = {entry["relative_path"] for entry in audit["sources"] if entry.get("required")}
        self.assertIn("generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_blocker_closure_readiness_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_blocker_closure_readiness_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/final_unblock_intake_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_unblock_intake_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/final_unblock_commands_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_unblock_commands_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/final_unblock_candidate_audit_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_unblock_candidate_audit_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/final_operator_handoff_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_operator_handoff_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/final_operator_handoff_validation_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/final_operator_handoff_validation_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/xr_vits_unblock_packet_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/xr_vits_unblock_packet_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/xr_vits_replacement_policy_preview_2026_06_10.json", required_paths)
        self.assertIn("generated/signoff/xr_vits_replacement_policy_preview_2026_06_10.md", required_paths)
        self.assertIn("generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/p2_vit_scale_calibration_report_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/p2_vit_scale_calibration_report_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/req6_parameterization_audit_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/req6_parameterization_audit_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/req1_environment_audit_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/req1_environment_audit_2026_06_16.md", required_paths)
        self.assertIn("generated/signoff/req9_deit_image_reference_audit_2026_06_16.json", required_paths)
        self.assertIn("generated/signoff/req9_deit_image_reference_audit_2026_06_16.md", required_paths)

    def test_build_audit_reports_missing_handover(self) -> None:
        tmp, hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        (hgtxr / "docs/track/HANDOVER.md").unlink()

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "partial")
        self.assertTrue(any(path.endswith("docs/track/HANDOVER.md") for path in audit["missing_required"]))

    def test_write_outputs(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        audit = audit_tool.build_audit(hardware)
        json_out = hardware / "generated/signoff/source.json"
        md_out = hardware / "generated/signoff/source.md"

        audit_tool.write_outputs(audit, json_out, md_out)

        self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
        self.assertIn("Third Goal Source Audit", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
