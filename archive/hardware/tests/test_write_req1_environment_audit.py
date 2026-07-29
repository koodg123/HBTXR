#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_req1_environment_audit as audit_tool  # noqa: E402


class Req1EnvironmentAuditTests(unittest.TestCase):
    def test_current_workspace_passes(self) -> None:
        audit = audit_tool.build_audit(ROOT.parent)

        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fail_count"], 0)
        self.assertEqual(audit["check_count"], 13)
        self.assertEqual(audit["environment"]["os_release"]["ID"], "ubuntu")
        self.assertEqual(audit["policy"]["xilinx_root"], "/tools/Xilinx")

    def test_legacy_wsl_path_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp) / "mnt" / "c" / "project" / "PRJXR" / "XR-VIT" / "HGTXR"
            (root / "hardware").mkdir(parents=True)

            audit = audit_tool.build_audit(root)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertEqual(audit["status"], "fail")
            self.assertIn("hardware_root_expected_prefix", failed)
            self.assertIn("no_legacy_wsl_or_xilinx_paths", failed)

    def test_wsl_kernel_marker_fails(self) -> None:
        with mock.patch.object(audit_tool, "read_text") as read_text:
            read_text.side_effect = lambda path: (
                'ID="ubuntu"\nVERSION_ID="22.04"\n'
                if str(path) == "/etc/os-release"
                else "Linux version 5.15.0-microsoft-standard-WSL2"
            )

            audit = audit_tool.build_audit(ROOT.parent)

            failed = {check["name"] for check in audit["checks"] if check["status"] == "fail"}
            self.assertIn("not_wsl_kernel", failed)

    def test_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_out = Path(tmp) / "req1.json"
            md_out = Path(tmp) / "req1.md"
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
            self.assertIn("Req1 Environment Audit", md_out.read_text())
            self.assertIn("/tools/Xilinx", md_out.read_text())


if __name__ == "__main__":
    unittest.main()
