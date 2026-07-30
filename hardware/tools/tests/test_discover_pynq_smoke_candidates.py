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

import discover_pynq_smoke_candidates as discovery_tool  # noqa: E402
from validate_pynq_smoke_result import EXPECTED_RAW, VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW  # noqa: E402

ARTIFACT_PREFIXES = {
    "c3b-mem16": "hgtxr_e2e_axis_dma_c3b_mem16",
    "vref-p0-softmax-input-x2-dsp-mixed-stream": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
    "vref-p0-softmax-input-x2-qkv-uram": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def smoke_payload(*, variant: str, out_raw: list[int], status: str = "pass") -> dict[str, object]:
    prefix = ARTIFACT_PREFIXES[variant]
    return {
        "status": status,
        "variant": variant,
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "runtime_state": 2,
        "runtime_match": True,
        "expected_out_raw": out_raw,
        "out_raw": out_raw,
        "output_match": True,
        "out_state": out_raw,
        "dma_in_name": "axi_dma_0",
        "dma_out_name": "axi_dma_0",
        "bitfile": f"/tmp/{prefix}.bit",
        "hwhfile": f"/tmp/{prefix}.hwh",
    }


class DiscoverPynqSmokeCandidatesTests(unittest.TestCase):
    def test_discovers_c3b_candidate_and_runner_import_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            write_json(candidate, smoke_payload(variant="c3b-mem16", out_raw=EXPECTED_RAW))

            result = discovery_tool.build_discovery(
                root,
                [root / "incoming"],
                preset="axis-c3b-mem16",
                require_paths=True,
            )

            self.assertEqual(result["status"], "found")
            self.assertEqual(result["pass_count"], 1)
            self.assertIn("--import-c3b-smoke-json", result["dry_run_import_command"])
            self.assertIn("--dry-run-import-c3b-smoke", result["dry_run_import_command"])
            self.assertIn(str(candidate), result["import_command"])
            self.assertFalse(result["safety"]["writes_canonical_inputs"])

    def test_discovers_vref_successor_candidate_and_runner_import_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            write_json(
                candidate,
                smoke_payload(
                    variant="vref-p0-softmax-input-x2-dsp-mixed-stream",
                    out_raw=VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
                ),
            )

            result = discovery_tool.build_discovery(
                root,
                [root / "incoming"],
                preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                require_paths=True,
            )

            self.assertEqual(result["status"], "found")
            self.assertEqual(result["pass_count"], 1)
            self.assertIn("--import-vref-successor-smoke-json", result["dry_run_import_command"])
            self.assertIn("--dry-run-import-vref-successor-smoke", result["dry_run_import_command"])
            self.assertIn(str(candidate), result["import_command"])

    def test_discovers_qkv_uram_candidate_and_runner_import_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            write_json(
                candidate,
                smoke_payload(
                    variant="vref-p0-softmax-input-x2-qkv-uram",
                    out_raw=VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
                ),
            )

            result = discovery_tool.build_discovery(
                root,
                [root / "incoming"],
                preset="axis-vref-p0-softmax-input-x2-qkv-uram",
                require_paths=True,
            )

            self.assertEqual(result["status"], "found")
            self.assertEqual(result["pass_count"], 1)
            self.assertIn("--import-qkv-uram-smoke-json", result["dry_run_import_command"])
            self.assertIn("--dry-run-import-qkv-uram-smoke", result["dry_run_import_command"])
            self.assertIn(str(candidate), result["import_command"])

    def test_invalid_candidate_not_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            bad = root / "incoming" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            write_json(
                bad,
                smoke_payload(
                    variant="vref-p0-softmax-input-x2-dsp-mixed-stream",
                    out_raw=VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
                    status="fail",
                ),
            )

            result = discovery_tool.build_discovery(
                root,
                [root / "incoming"],
                preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                require_paths=True,
            )

            self.assertEqual(result["status"], "missing")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["pass_count"], 0)
            self.assertIsNone(result["recommended_candidate"])

    def test_skips_session_and_validation_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            generated = root / "hardware" / "generated" / "pynq"
            for name in [
                "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.json",
                "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke_validation.json",
            ]:
                write_json(generated / name, {"status": "pass", "variant": "metadata"})

            result = discovery_tool.build_discovery(
                root,
                [generated],
                preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                require_paths=True,
            )

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["status"], "missing")

    def test_skips_docs_resource_gate_audit_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            resources = root / "docs" / "resources"
            for name in [
                "c3b_physical_smoke_gate_audit_2026_06_16.json",
                "vref_p0_qkv_uram_gate_audit_2026_06_16.json",
            ]:
                write_json(resources / name, {"status": "metadata", "variant": "c3b-mem16"})

            result = discovery_tool.build_discovery(
                root,
                [resources],
                preset="axis-c3b-mem16",
                require_paths=True,
            )

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["status"], "missing")

    def test_skips_generated_signoff_metadata_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            signoff = root / "hardware" / "generated" / "signoff"
            for name in [
                "c3b_physical_smoke_gate_audit_2026_06_16.json",
                "zcu104_c3b_smoke_remote_run_2026_06_10.json",
            ]:
                write_json(signoff / name, {"status": "metadata", "variant": "c3b-mem16"})

            result = discovery_tool.build_discovery(
                root,
                [signoff],
                preset="axis-c3b-mem16",
                require_paths=True,
            )

            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["status"], "missing")

    def test_cli_writes_vref_markdown_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            candidate = root / "incoming" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            json_out = root / "generated" / "discovery.json"
            markdown_out = root / "generated" / "discovery.md"
            write_json(
                candidate,
                smoke_payload(
                    variant="vref-p0-softmax-input-x2-dsp-mixed-stream",
                    out_raw=VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
                ),
            )
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = discovery_tool.main(
                    [
                        "--root",
                        str(root),
                        "--preset",
                        "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
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
            self.assertEqual(code, 0, stream.getvalue())
            self.assertIn("--dry-run-import-vref-successor-smoke", payload["dry_run_import_command"])
            self.assertIn("HGTXR PYNQ Smoke Candidate Discovery", markdown)
            self.assertFalse(
                (
                    root
                    / "hardware"
                    / "pynq"
                    / "hgtxr"
                    / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
