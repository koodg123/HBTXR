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

import write_xr_vits_unblock_packet as packet_tool  # noqa: E402


class WriteXrVitsUnblockPacketTests(unittest.TestCase):
    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        base = Path(tmp.name) / "XR-VIT"
        root = base / "HGTXR"
        docs = root / "docs" / "resources"
        docs.mkdir(parents=True)
        requested = base.parent / "XR-VITs"
        replacement = base / "XR_Accel"
        replacement.mkdir(parents=True)
        candidate = docs / "xr_vits_candidate_audit_2026_06_10.json"
        template = docs / "xr_vits_replacement_policy.template.json"
        policy = docs / "xr_vits_replacement_policy.json"
        candidate.write_text(
            json.dumps(
                {
                    "status": "candidate-found",
                    "requested_path": str(requested),
                    "approved_replacement": False,
                    "recommendation": {
                        "role": "candidate-xr-accel",
                        "path": str(replacement),
                        "score": 99,
                        "reason": "highest score for ZCU104/cyclic/DeiT/HLS evidence",
                    },
                    "candidates": [
                        {
                            "path": str(replacement),
                            "role": "candidate-xr-accel",
                            "score": 99,
                            "hls_file_count": 97,
                            "readme_exists": True,
                        }
                    ],
                }
            )
        )
        template.write_text(
            json.dumps(
                {
                    "approved_replacement": False,
                    "requested_path": str(requested),
                    "replacement_path": str(replacement),
                }
            )
        )
        return tmp, root, requested, candidate, policy

    def test_pending_when_no_exact_path_or_active_policy(self) -> None:
        tmp, root, requested, candidate, policy = self.make_root()
        self.addCleanup(tmp.cleanup)

        packet = packet_tool.build_packet(
            root=root,
            requested_path=requested,
            candidate_audit_path=candidate,
            policy_template_path=root / "docs" / "resources" / "xr_vits_replacement_policy.template.json",
            active_policy_path=policy,
        )

        self.assertEqual(packet["status"], "pending-user-choice")
        self.assertFalse(packet["requested_path_exists"])
        self.assertFalse(packet["active_policy_exists"])
        self.assertIn("approve-xr-accel-replacement", [option["id"] for option in packet["options"]])

    def test_exact_restored_wins_over_policy(self) -> None:
        tmp, root, requested, candidate, policy = self.make_root()
        self.addCleanup(tmp.cleanup)
        requested.mkdir(parents=True)

        packet = packet_tool.build_packet(
            root=root,
            requested_path=requested,
            candidate_audit_path=candidate,
            policy_template_path=root / "docs" / "resources" / "xr_vits_replacement_policy.template.json",
            active_policy_path=policy,
        )

        self.assertEqual(packet["status"], "exact-restored")
        self.assertTrue(packet["requested_path_exists"])

    def test_approved_policy_resolves_when_exact_missing(self) -> None:
        tmp, root, requested, candidate, policy = self.make_root()
        self.addCleanup(tmp.cleanup)
        policy.write_text(
            json.dumps(
                {
                    "approved_replacement": True,
                    "requested_path": str(requested),
                    "replacement_path": str(root.parent / "XR_Accel"),
                }
            )
        )

        packet = packet_tool.build_packet(
            root=root,
            requested_path=requested,
            candidate_audit_path=candidate,
            policy_template_path=root / "docs" / "resources" / "xr_vits_replacement_policy.template.json",
            active_policy_path=policy,
        )

        self.assertEqual(packet["status"], "replacement-approved")
        self.assertTrue(packet["active_policy_approved"])

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, root, _requested, candidate, policy = self.make_root()
        self.addCleanup(tmp.cleanup)
        json_out = root / "hardware" / "generated" / "packet.json"
        markdown_out = root / "hardware" / "generated" / "packet.md"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            code = packet_tool.main(
                [
                    "--root",
                    str(root),
                    "--candidate-audit",
                    str(candidate),
                    "--policy-template",
                    str(root / "docs" / "resources" / "xr_vits_replacement_policy.template.json"),
                    "--active-policy",
                    str(policy),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ]
            )

        self.assertEqual(code, 1)
        self.assertEqual(json.loads(json_out.read_text())["status"], "pending-user-choice")
        self.assertIn("HGTXR XR-VITs Unblock Packet", markdown_out.read_text())
        self.assertIn("create_xr_vits_replacement_policy.py", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
