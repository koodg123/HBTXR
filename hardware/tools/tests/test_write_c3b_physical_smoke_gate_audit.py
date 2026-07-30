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

import write_c3b_physical_smoke_gate_audit as audit_tool  # noqa: E402


class C3bPhysicalSmokeGateAuditTests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        hgtxr = Path(tmp.name) / "XR-VIT" / "HGTXR"
        hardware = hgtxr / "hardware"
        bundle_dir = hardware / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text((ROOT / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle/BUNDLE_MANIFEST.json").read_text())
        tar_src = ROOT / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        tar_dst = hardware / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        tar_dst.parent.mkdir(parents=True, exist_ok=True)
        tar_dst.write_bytes(tar_src.read_bytes())
        for path in sorted((ROOT / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle").rglob("*")):
            if path.is_file():
                rel = path.relative_to(ROOT / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle")
                dst = bundle_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(path.read_bytes())
        self.write_json(
            hardware / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json",
            {
                "status": "pass",
                "preset": "axis-c3b-mem16",
                "canonical_result_path": str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                "board_steps": ["run"],
                "host_steps": ["import"],
            },
        )
        return tmp, hgtxr, hardware

    def test_missing_canonical_result_blocks_gate(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "blocked_missing_canonical_physical_smoke_result")
        self.assertTrue(audit["ready_for_board"])
        self.assertFalse(audit["physical_smoke_pass"])
        self.assertEqual(audit["canonical_result"]["status"], "missing")
        self.assertFalse(audit["canonical_validation"]["exists"])
        self.assertEqual(audit["canonical_validation"]["status"], "missing")
        self.assertEqual(audit["bundle"]["status"], "pass")
        self.assertEqual(audit["session"]["status"], "pass")
        self.assertIn("C3b AXIS/DMA physical smoke result", audit["remaining_blockers"])

    def test_valid_canonical_result_passes_gate(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        self.write_json(
            hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
            {
                "status": "pass",
                "variant": "c3b-mem16",
                "weights_mode": "file",
                "expected_runtime_state": 2,
                "runtime_state": 2,
                "runtime_match": True,
                "expected_out_raw": [32, -13, 26, -6, 14, -11],
                "out_raw": [32, -13, 26, -6, 14, -11],
                "output_match": True,
                "dma_in_name": "dma_in",
                "dma_out_name": "dma_out",
                "bitfile": "hgtxr_e2e_axis_dma_c3b_mem16.bit",
                "hwhfile": "hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                "out_state": [32, -13, 26, -6, 14, -11],
            },
        )
        self.write_json(
            hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
            {
                "status": "pass",
                "preset": "axis-c3b-mem16",
                "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
                "errors": [],
            },
        )

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass")
        self.assertTrue(audit["ready_for_board"])
        self.assertTrue(audit["physical_smoke_pass"])
        self.assertEqual(audit["canonical_result"]["status"], "pass")
        self.assertEqual(audit["canonical_validation"]["status"], "pass")
        self.assertEqual(audit["canonical_validation"]["preset"], "axis-c3b-mem16")
        self.assertEqual(audit["remaining_blockers"], [])

    def test_failing_canonical_validation_blocks_gate(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        self.write_json(
            hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
            {
                "status": "pass",
                "variant": "c3b-mem16",
                "weights_mode": "file",
                "expected_runtime_state": 2,
                "runtime_state": 2,
                "runtime_match": True,
                "expected_out_raw": [32, -13, 26, -6, 14, -11],
                "out_raw": [32, -13, 26, -6, 14, -11],
                "output_match": True,
                "dma_in_name": "dma_in",
                "dma_out_name": "dma_out",
                "bitfile": "hgtxr_e2e_axis_dma_c3b_mem16.bit",
                "hwhfile": "hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                "out_state": [32, -13, 26, -6, 14, -11],
            },
        )
        self.write_json(
            hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
            {
                "status": "fail",
                "preset": "axis-c3b-mem16",
                "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
                "errors": ["runtime mismatch"],
            },
        )

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "blocked_missing_canonical_physical_smoke_result")
        self.assertFalse(audit["physical_smoke_pass"])
        self.assertEqual(audit["canonical_result"]["status"], "fail")
        self.assertEqual(audit["canonical_validation"]["status"], "fail")
        self.assertIn("validation status", "; ".join(audit["canonical_result"]["errors"]))
        self.assertIn("validation errors not empty", "; ".join(audit["canonical_result"]["errors"]))

    def test_write_outputs(self) -> None:
        tmp, _hgtxr, hardware = self.make_root()
        self.addCleanup(tmp.cleanup)
        audit = audit_tool.build_audit(hardware)
        json_out = hardware / "generated/signoff/c3b.json"
        md_out = hardware / "generated/signoff/c3b.md"

        audit_tool.write_outputs(audit, json_out, md_out)

        self.assertEqual(json.loads(json_out.read_text())["status"], "blocked_missing_canonical_physical_smoke_result")
        self.assertIn("C3b Physical Smoke Gate Audit", md_out.read_text())
        self.assertIn("canonical_validation_status", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
