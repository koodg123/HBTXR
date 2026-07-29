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

import write_c3b_smoke_result_contract as contract_tool  # noqa: E402


class WriteC3bSmokeResultContractTests(unittest.TestCase):
    def test_build_contract_records_required_fields_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"

            contract = contract_tool.build_contract(root)

            self.assertEqual(contract["status"], "pass")
            self.assertEqual(contract["preset"], "axis-c3b-mem16")
            self.assertEqual(contract["required_fields"]["variant"], "c3b-mem16")
            self.assertEqual(contract["required_fields"]["out_raw"], [32, -13, 26, -6, 14, -11])
            self.assertIn("e2e_axis_dma_c3b_mem16_file_smoke.json", contract["canonical_result_path"])
            self.assertIn("--dry-run", contract["validation"]["dry_run_import_command"])
            self.assertNotIn("--dry-run", contract["validation"]["active_import_command"])
            self.assertFalse(contract["safety"]["writes_canonical_inputs"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            json_out = root / "generated" / "contract.json"
            markdown_out = root / "generated" / "contract.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = contract_tool.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            text = markdown_out.read_text()
            self.assertIn("HGTXR C3b Board Smoke Result Contract", text)
            self.assertIn("Dry-Run Import", text)
            self.assertIn("Active Import", text)


if __name__ == "__main__":
    unittest.main()
