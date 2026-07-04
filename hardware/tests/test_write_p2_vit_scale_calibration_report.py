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

import write_p2_vit_scale_calibration_report as report_tool  # noqa: E402


class P2VitScaleCalibrationReportTests(unittest.TestCase):
    def write_json(self, hardware: Path, rel: str, payload: dict[str, object]) -> None:
        path = hardware / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def make_hardware(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        hardware = Path(tmp.name) / "HGTXR" / "hardware"
        self.write_json(
            hardware,
            "generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.json",
            {
                "status": "pass",
                "precision": {"weight_bits": 4, "activation_bits": 8},
                "pass_count": 6,
                "fail_count": 0,
                "csim_logs": [{}, {}, {}],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/vref_p0_pot_scale_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 14,
                "pass_count": 14,
                "fail_count": 0,
                "safety": {
                    "executes_hls": False,
                    "executes_vivado": False,
                    "writes_hls_source": False,
                    "overwrites_c3b_artifacts": False,
                },
            },
        )
        specs = [
            {
                "spec": f"spec_{idx}.json",
                "recommended_candidate": "current",
                "recommendation": "keep_current_scales",
                "candidate_count": 15,
                "pass_count": 15,
                "fail_count": 0,
                "exact_match_count": 1,
                "baseline_expected_raw": [32, -13, 26, -6, 14, -11],
            }
            for idx in range(3)
        ]
        self.write_json(
            hardware,
            "generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.json",
            {
                "status": "pass",
                "spec_count": 3,
                "specs": specs,
                "summary": {
                    "specs_recommending_current": 3,
                    "total_candidate_count": 45,
                    "total_fail_count": 0,
                    "next_action": "keep current PoT scales for C3b; use non-current rows only as explicit successor experiments",
                },
                "policy": {"promotion_gate": "candidate must regenerate golden header and pass CSim before HLS macro promotion"},
                "safety": {
                    "executes_hls": False,
                    "executes_vivado": False,
                    "writes_hls_source": False,
                    "overwrites_c3b_artifacts": False,
                },
            },
        )
        return tmp, hardware

    def test_build_report_passes_with_current_scale_decision(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)

        report = report_tool.build_report(hardware)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["policy"]["decision"], "keep_current_pot_scales_for_c3b")
        self.assertEqual(report["precision"]["weight_bits"], 4)
        self.assertEqual(report["precision"]["activation_bits"], 8)
        self.assertEqual(report["sweep_summary"]["total_candidate_count"], 45)
        self.assertEqual(report["fail_count"], 0)

    def test_bad_req5_fails_report(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)
        self.write_json(
            hardware,
            "generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.json",
            {"status": "fail", "precision": {"weight_bits": 4, "activation_bits": 8}, "pass_count": 0, "fail_count": 1, "csim_logs": []},
        )

        report = report_tool.build_report(hardware)

        self.assertEqual(report["status"], "fail")
        self.assertIn("req5_q4q8_pass", report["failed_checks"])

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)
        json_out = hardware / "generated/signoff/p2.json"
        md_out = hardware / "generated/signoff/p2.md"

        code = report_tool.main(["--root", str(hardware), "--json-out", str(json_out), "--markdown-out", str(md_out)])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
        self.assertIn("P2-ViT Scale Calibration Report", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
