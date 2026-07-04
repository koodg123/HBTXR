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

import audit_xr_vits_candidates as audit_tool  # noqa: E402


class AuditXrVitsCandidatesTests(unittest.TestCase):
    def test_build_audit_recommends_zcu104_cyclic_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT"
            xr_accel = root / "XR_Accel"
            (xr_accel / "configs" / "targets").mkdir(parents=True)
            (xr_accel / "configs" / "targets" / "zcu104.json").write_text("{}")
            (xr_accel / "configs" / "designs").mkdir(parents=True)
            (xr_accel / "configs" / "designs" / "deit_tiny_cyclic_zcu104.json").write_text("{}")
            (xr_accel / "workspace" / "hardware" / "src").mkdir(parents=True)
            for name in ["layernorm.h", "gelu.h", "softmax.h", "quant.h", "cyclic_vit.h"]:
                (xr_accel / "workspace" / "hardware" / "src" / name).write_text("// hls")
            (root / "ViT_Accel").mkdir(parents=True)

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "candidate-found")
            self.assertFalse(audit["approved_replacement"])
            self.assertEqual(audit["recommendation"]["role"], "candidate-xr-accel")

    def test_build_audit_does_not_approve_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT"
            (root / "ViT_Accel" / "configs" / "targets").mkdir(parents=True)
            (root / "ViT_Accel" / "configs" / "targets" / "zcu104.json").write_text("{}")

            audit = audit_tool.build_audit(root)

            self.assertEqual(audit["status"], "candidate-found")
            self.assertFalse(audit["approved_replacement"])
            self.assertIn("Do not treat", audit["policy"])

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT"
            (root / "XR_Accel" / "configs" / "targets").mkdir(parents=True)
            (root / "XR_Accel" / "configs" / "targets" / "zcu104.json").write_text("{}")
            json_out = Path(tmp) / "audit.json"
            markdown_out = Path(tmp) / "audit.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = audit_tool.main(
                    [
                        "--xr-vit-root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(json_out.read_text())["status"], "candidate-found")
            self.assertIn("XR-VITs Candidate Audit", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
