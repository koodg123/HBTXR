#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_req5_q4q8_swhw_match_audit as audit_tool  # noqa: E402


class Req5Q4Q8SwHwMatchAuditTests(unittest.TestCase):
    def write(self, root: Path, rel: str, text: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, root: Path, rel: str, payload: dict[str, object]) -> None:
        self.write(root, rel, json.dumps(payload) + "\n")

    def make_hardware(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        hardware = Path(tmp.name) / "HGTXR" / "hardware"
        self.write(
            hardware,
            "configs/zcu104_e2e_q4w8a_defines.h",
            "#define HGTXR_WEIGHT_BIT_WIDTH 4\n#define HGTXR_BIT_WIDTH 8\n",
        )
        tb_text = "HGTXR_E2E_STRICT_GOLDEN init_q4_vector_weights FAIL: e2e output FAIL: e2e_m_axi output"
        self.write(hardware, "hls/tb/tb_hgtxr_e2e_axis_top.cpp", tb_text)
        self.write(hardware, "hls/tb/tb_hgtxr_e2e_m_axi_top.cpp", tb_text)
        binary = hardware / "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(bytes(range(32)))
        self.write_json(
            hardware,
            "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            {
                "binary": str(binary),
                "dtype": "uint32",
                "bytes": binary.stat().st_size,
                "required_u32": 32,
                "tail_zero_start_u32": 32,
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "expected_raw": [32, -13, 26, -6, 14, -11],
                "expected_runtime_state": 2,
                "known_offset_checks": {
                    "patch_channel0_elem0": 1,
                    "patch_channel2_elem0": -1,
                },
                "layout": {
                    "weight_bits": 4,
                    "bus_width_bits": 256,
                },
                "provenance": {"source": "fixture"},
            },
        )
        for _, rel_path, _ in audit_tool.CSIM_LOGS:
            self.write(
                hardware,
                rel_path,
                "\n".join(
                    [
                        "e2e_axis_out[0]=58 last=0",
                        "e2e_axis_out[1]=-51 last=0",
                        "e2e_axis_out[2]=42 last=0",
                        "e2e_axis_out[3]=-28 last=0",
                        "e2e_axis_out[4]=36 last=0",
                        "e2e_axis_out[5]=-41 last=1",
                        "runtime_state=2 count=6 last=1 failures=0",
                        "E2E AXIS vector comparison passed",
                        "INFO: [SIM 1] CSim done with 0 errors.",
                    ]
                )
                + "\n",
            )
        return tmp, hardware

    def test_build_audit_passes_with_manifest_and_csim_logs(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["precision"]["weight_bits"], 4)
        self.assertEqual(audit["precision"]["activation_bits"], 8)
        self.assertEqual(audit["manifest"]["status"], "pass")
        self.assertTrue(audit["manifest"]["binary_sha256_matches_manifest"])
        self.assertTrue(audit["manifest"]["binary_bytes_matches_manifest"])
        self.assertTrue(audit["manifest"]["expected_c3b_raw_ok"])
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(len(audit["csim_logs"]), 3)

    def test_bad_weight_binary_sha_fails(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)
        manifest_path = hardware / "refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n")

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "fail")
        self.assertIn("packed_weight_manifest", audit["failed_checks"])
        self.assertIn("packed_weight_sha256_matches_manifest", audit["failed_checks"])

    def test_bad_csim_log_fails(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)
        self.write(
            hardware,
            audit_tool.CSIM_LOGS[0][1],
            "runtime_state=2 count=6 last=1 failures=1\nFAIL: e2e output[0]\n",
        )

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "fail")
        self.assertIn("c3b_axis_csim_strict_csim", audit["failed_checks"])

    def test_cli_writes_outputs(self) -> None:
        tmp, hardware = self.make_hardware()
        self.addCleanup(tmp.cleanup)
        json_out = hardware / "generated/signoff/req5.json"
        md_out = hardware / "generated/signoff/req5.md"

        code = audit_tool.main(["--root", str(hardware), "--json-out", str(json_out), "--markdown-out", str(md_out)])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
        self.assertIn("Req5 Q4/Q8 SW-HW Match Audit", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
