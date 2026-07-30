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

import discover_c3b_smoke_candidates as discovery_tool  # noqa: E402
from validate_pynq_smoke_result import EXPECTED_RAW  # noqa: E402


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def smoke_payload(*, status: str = "pass") -> dict[str, object]:
    return {
        "status": status,
        "variant": "c3b-mem16",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "runtime_state": 2,
        "runtime_match": True,
        "expected_out_raw": EXPECTED_RAW,
        "out_raw": EXPECTED_RAW,
        "output_match": True,
        "out_state": EXPECTED_RAW,
        "dma_in_name": "axi_dma_0",
        "dma_out_name": "axi_dma_0",
        "bitfile": "/tmp/hgtxr_e2e_axis_dma_c3b_mem16.bit",
        "hwhfile": "/tmp/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
    }


class DiscoverC3bSmokeCandidatesTests(unittest.TestCase):
    def test_discovers_valid_candidate_and_import_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            write_json(candidate, smoke_payload())

            result = discovery_tool.build_discovery(root, [root / "incoming"], require_paths=True)

            self.assertEqual(result["status"], "found")
            self.assertEqual(result["pass_count"], 1)
            self.assertEqual(result["recommended_candidate"]["path"], str(candidate))
            self.assertIn("--dry-run-import-c3b-smoke", result["dry_run_import_command"])
            self.assertIn(str(candidate), result["dry_run_import_command"])
            self.assertIn("--import-c3b-smoke-json", result["import_command"])
            self.assertNotIn("--dry-run-import-c3b-smoke", result["import_command"])
            self.assertFalse(result["safety"]["writes_canonical_inputs"])

    def test_invalid_candidates_are_not_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            bad = root / "incoming" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            write_json(bad, smoke_payload(status="fail"))

            result = discovery_tool.build_discovery(root, [root / "incoming"], require_paths=True)

            self.assertEqual(result["status"], "missing")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["pass_count"], 0)
            self.assertIsNone(result["recommended_candidate"])

    def test_skips_generated_signoff_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            noise = root / "hardware" / "generated" / "signoff" / "c3b_board_smoke_readiness_2026_06_10.json"
            write_json(noise, {"status": "ready-for-board"})

            result = discovery_tool.build_discovery(root, [root / "hardware"], require_paths=True)

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["status"], "missing")

    def test_skips_docs_resource_metadata_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            resources = root / "docs" / "resources"
            for name in [
                "c3b_axis_dma_smoke_bundle_2026_06_10.json",
                "c3b_physical_smoke_gate_audit_2026_06_16.json",
                "c3b_smoke_candidate_discovery_2026_06_10.json",
                "c3b_smoke_result_contract_2026_06_10.json",
            ]:
                write_json(resources / name, {"status": "metadata", "variant": "c3b-mem16"})

            result = discovery_tool.build_discovery(root, [resources], require_paths=True)

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["status"], "missing")

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            json_out = root / "generated" / "discovery.json"
            markdown_out = root / "generated" / "discovery.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = discovery_tool.main(
                    [
                        "--root",
                        str(root),
                        "--scan-root",
                        str(root / "incoming"),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(json_out.read_text())["status"], "missing")
            self.assertIn("HGTXR C3b Smoke Candidate Discovery", markdown_out.read_text())

    def test_cli_markdown_includes_dry_run_import_command_for_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            json_out = root / "generated" / "discovery.json"
            markdown_out = root / "generated" / "discovery.md"
            write_json(candidate, smoke_payload())

            code = discovery_tool.main(
                [
                    "--root",
                    str(root),
                    "--scan-root",
                    str(root / "incoming"),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ]
            )

            payload = json.loads(json_out.read_text())
            markdown = markdown_out.read_text()
            self.assertEqual(code, 0)
            self.assertIn("--dry-run-import-c3b-smoke", payload["dry_run_import_command"])
            self.assertIn(str(candidate), payload["dry_run_import_command"])
            self.assertIn("Dry-run Import Command", markdown)
            self.assertIn("--dry-run-import-c3b-smoke", markdown)
            self.assertFalse((root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json").exists())


if __name__ == "__main__":
    unittest.main()
