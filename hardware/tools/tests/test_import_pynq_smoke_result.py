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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import import_pynq_smoke_result as importer  # noqa: E402
from test_validate_pynq_smoke_result import (  # noqa: E402
    c3b_result,
    par32_prefetchall4_hybrid_result,
    qkv_uram_result,
    successor_result,
)


class ImportPynqSmokeResultTests(unittest.TestCase):
    def test_import_copies_valid_c3b_result_to_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(c3b_result()))

            result = importer.import_result(source, preset="axis-c3b-mem16", hgtxr_root=root)

            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["copied"])
            self.assertTrue(result["dest_is_default"])
            self.assertTrue(result["would_clear_current_gate"])
            self.assertFalse(result["dest_exists_before"])
            self.assertTrue(result["dest_exists_after"])
            self.assertEqual(len(result["source_sha256"]), 64)
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertEqual(result["dest_validation"]["dest_sha256"], result["source_sha256"])
            self.assertTrue(result["dest_validation"]["dest_matches_source"])
            self.assertEqual(result["payload_summary"]["variant"], "c3b-mem16")
            self.assertTrue(dest.exists())
            self.assertEqual(json.loads(dest.read_text())["variant"], "c3b-mem16")

    def test_import_copies_valid_vref_successor_result_to_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(successor_result()))

            result = importer.import_result(source, preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream", hgtxr_root=root)

            dest = (
                root
                / "hardware"
                / "pynq"
                / "hgtxr"
                / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["would_clear_current_gate"])
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertTrue(dest.exists())
            self.assertEqual(json.loads(dest.read_text())["variant"], "vref-p0-softmax-input-x2-dsp-mixed-stream")

    def test_import_copies_valid_qkv_uram_result_to_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(qkv_uram_result()))

            result = importer.import_result(source, preset="axis-vref-p0-softmax-input-x2-qkv-uram", hgtxr_root=root)

            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["would_clear_current_gate"])
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertTrue(dest.exists())
            self.assertEqual(json.loads(dest.read_text())["variant"], "vref-p0-softmax-input-x2-qkv-uram")

    def test_import_copies_valid_par32_prefetchall4_hybrid_result_to_default_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(par32_prefetchall4_hybrid_result()))

            result = importer.import_result(
                source,
                preset="axis-par32-prefetchall4-300-hybrid-10-90",
                hgtxr_root=root,
            )

            dest = (
                root
                / "hardware"
                / "pynq"
                / "hgtxr"
                / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json"
            )
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["would_clear_current_gate"])
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertEqual(result["payload_summary"]["hybrid_profile"], "search10-track90")
            self.assertEqual(result["payload_summary"]["search_track_invocation_distribution"], {"search": 1, "track": 9, "unknown": 0})
            self.assertTrue(dest.exists())
            self.assertEqual(json.loads(dest.read_text())["hybrid_profile"], "search10-track90")

    def test_import_rejects_invalid_result_without_overwriting_dest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            dest.parent.mkdir(parents=True)
            dest.write_text(json.dumps(c3b_result()))
            source = Path(tmp) / "bad.json"
            source.write_text(json.dumps(c3b_result(status="fail", runtime_state=0)))

            result = importer.import_result(source, preset="axis-c3b-mem16", hgtxr_root=root)

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["copied"])
            self.assertFalse(result["would_clear_current_gate"])
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertFalse(result["dest_validation"]["dest_matches_source"])
            self.assertEqual(json.loads(dest.read_text())["status"], "pass")

    def test_import_rejects_wrong_overlay_basename_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "wrong_overlay.json"
            source.write_text(
                json.dumps(
                    c3b_result(
                        bitfile="hgtxr/hgtxr_e2e_axis_dma.bit",
                        hwhfile="hgtxr/hgtxr_e2e_axis_dma.hwh",
                    )
                )
            )

            result = importer.import_result(source, preset="axis-c3b-mem16", hgtxr_root=root)

            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["copied"])
            self.assertFalse(result["would_clear_current_gate"])
            self.assertFalse(dest.exists())
            self.assertTrue(any("bitfile" in error for error in result["errors"]))
            self.assertTrue(any("hwhfile" in error for error in result["errors"]))

    def test_dry_run_validates_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(c3b_result()))

            result = importer.import_result(source, preset="axis-c3b-mem16", hgtxr_root=root, dry_run=True)

            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["dry_run"])
            self.assertFalse(result["copied"])
            self.assertFalse(result["would_clear_current_gate"])
            self.assertEqual(result["dest_validation"]["status"], "missing")
            self.assertFalse(dest.exists())

    def test_custom_dest_does_not_claim_current_gate_clearance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            custom_dest = Path(tmp) / "scratch" / "custom_smoke.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(c3b_result()))

            result = importer.import_result(source, preset="axis-c3b-mem16", hgtxr_root=root, dest=custom_dest)

            self.assertEqual(result["status"], "pass")
            self.assertFalse(result["dest_is_default"])
            self.assertTrue(result["copied"])
            self.assertTrue(custom_dest.exists())
            self.assertEqual(result["dest_validation"]["status"], "pass")
            self.assertTrue(result["dest_validation"]["dest_matches_source"])
            self.assertFalse(result["would_clear_current_gate"])

    def test_cli_writes_import_and_validation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "result.json"
            report = Path(tmp) / "import.json"
            validation = Path(tmp) / "validation.json"
            source.write_text(json.dumps(c3b_result()))
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = importer.main(
                    [
                        str(source),
                        "--preset",
                        "axis-c3b-mem16",
                        "--root",
                        str(root),
                        "--json-out",
                        str(report),
                        "--validation-out",
                        str(validation),
                    ]
                )

            self.assertEqual(code, 0, stream.getvalue())
            self.assertEqual(json.loads(report.read_text())["status"], "pass")
            self.assertEqual(json.loads(validation.read_text())["status"], "pass")
            self.assertEqual(len(json.loads(validation.read_text())["source_sha256"]), 64)
            self.assertTrue(json.loads(validation.read_text())["dest_is_default"])
            self.assertTrue(json.loads(validation.read_text())["would_clear_current_gate"])
            self.assertEqual(json.loads(validation.read_text())["dest_validation"]["status"], "pass")
            self.assertTrue(json.loads(validation.read_text())["dest_validation"]["dest_matches_source"])
            self.assertTrue(json.loads(validation.read_text())["copied"])

    def test_cli_dry_run_writes_reports_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            source = Path(tmp) / "result.json"
            report = Path(tmp) / "import.json"
            validation = Path(tmp) / "validation.json"
            source.write_text(json.dumps(c3b_result()))

            code = importer.main(
                [
                    str(source),
                    "--preset",
                    "axis-c3b-mem16",
                    "--root",
                    str(root),
                    "--dry-run",
                    "--json-out",
                    str(report),
                    "--validation-out",
                    str(validation),
                ]
            )

            dest = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(report.read_text())["dry_run"], True)
            self.assertEqual(json.loads(validation.read_text())["dry_run"], True)
            self.assertFalse(json.loads(validation.read_text())["copied"])
            self.assertFalse(json.loads(validation.read_text())["would_clear_current_gate"])
            self.assertEqual(json.loads(validation.read_text())["dest_validation"]["status"], "missing")
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
