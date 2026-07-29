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

import write_hgpipe_operator_audit as audit_tool  # noqa: E402


class HgpipeOperatorAuditTests(unittest.TestCase):
    def test_build_operator_audit_passes_on_repo_contract(self) -> None:
        audit = audit_tool.build_operator_audit(ROOT)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(
            {item["operator"]: item["status"] for item in audit["operators"]},
            {
                "LayerNorm": "pass",
                "GeLU": "pass",
                "Softmax": "pass",
                "Quantization": "pass",
            },
        )
        self.assertEqual(audit["contract_summary"]["total_ref_checks"], 97)
        self.assertGreater(audit["contract_summary"]["total_checked_samples"], 0)
        self.assertEqual(audit["property_summary"]["status"], "pass")
        self.assertGreater(audit["property_summary"]["check_count"], 0)
        self.assertEqual(audit["property_summary"]["fail_count"], 0)
        self.assertTrue(all(item["property_fail_count"] == 0 for item in audit["operators"]))
        self.assertFalse(audit["safety"]["executes_hls"])

    def test_write_outputs_creates_json_and_markdown(self) -> None:
        audit = audit_tool.build_operator_audit(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "audit.json"
            md_out = Path(tmp) / "audit.md"

            audit_tool.write_outputs(audit, json_out, md_out)

            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            markdown = md_out.read_text()
            self.assertIn("HG-PIPE Operator Audit", markdown)
            self.assertIn("Property Checks", markdown)


if __name__ == "__main__":
    unittest.main()
