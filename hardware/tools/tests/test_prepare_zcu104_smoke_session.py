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

import package_e2e_axis_dma_pynq_bundle as bundler  # noqa: E402
import prepare_zcu104_smoke_session as session_tool  # noqa: E402


class PrepareZcu104SmokeSessionTests(unittest.TestCase):
    def test_build_session_accepts_valid_c3b_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            root = Path(tmp) / "HGTXR"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")

            session = session_tool.build_session(
                bundle_dir=out_dir,
                tar_path=tar_path,
                variant="c3b-mem16",
                preset="axis-c3b-mem16",
                hgtxr_root=root,
            )

            self.assertEqual(session["status"], "pass")
            self.assertEqual(session["expected_runtime_state"], 2)
            self.assertEqual(session["expected_out_raw"], [32, -13, 26, -6, 14, -11])
            self.assertIn("./run_e2e_axis_dma_c3b_mem16_file_smoke.sh", session["board_steps"])
            self.assertIn("run_third_goal_final_signoff.py", session["host_steps"][0])
            self.assertIn("--import-c3b-smoke-json", session["host_steps"][0])
            self.assertIn("--dry-run-import-c3b-smoke", session["host_steps"][0])
            self.assertIn("--import-c3b-smoke-json", session["host_steps"][1])
            self.assertNotIn("--dry-run-import-c3b-smoke", session["host_steps"][1])

    def test_build_session_accepts_valid_vref_successor_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            root = Path(tmp) / "HGTXR"
            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-dsp-mixed-stream")

            session = session_tool.build_session(
                bundle_dir=out_dir,
                tar_path=tar_path,
                variant="vref-p0-softmax-input-x2-dsp-mixed-stream",
                preset="axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                hgtxr_root=root,
            )

            self.assertEqual(session["status"], "pass")
            self.assertEqual(session["expected_out_raw"], [58, -51, 42, -28, 36, -41])
            self.assertIn(
                "./run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
                session["board_steps"],
            )
            self.assertIn("--import-vref-successor-smoke-json", session["host_steps"][0])
            self.assertIn("--dry-run-import-vref-successor-smoke", session["host_steps"][0])
            self.assertIn("axis-vref-p0-softmax-input-x2-dsp-mixed-stream", session["host_steps"][2])

    def test_build_session_accepts_valid_qkv_uram_bundle_with_generic_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            root = tmp / "HGTXR"
            prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
            fake_hardware = tmp / "hardware"
            for rel in bundler.PACKAGE_FILES:
                path = fake_hardware / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n")
            for rel in bundler.WEIGHT_FILES:
                path = fake_hardware / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\x00\x00\x00\x00")
            for rel in bundler.TOOL_FILES:
                path = fake_hardware / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n")
            for suffix in [".bit", ".hwh"]:
                path = fake_hardware / "pynq" / "hgtxr" / f"{prefix}{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{suffix}\n")
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_hardware
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-qkv-uram")

            session = session_tool.build_session(
                bundle_dir=out_dir,
                tar_path=tar_path,
                variant="vref-p0-softmax-input-x2-qkv-uram",
                preset="axis-vref-p0-softmax-input-x2-qkv-uram",
                hgtxr_root=root,
            )

            self.assertEqual(session["status"], "pass")
            self.assertEqual(session["expected_out_raw"], [58, -51, 42, -28, 36, -41])
            self.assertIn("./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh", session["board_steps"])
            self.assertIn("tools/import_pynq_smoke_result.py", session["host_steps"][0])
            self.assertIn("--preset axis-vref-p0-softmax-input-x2-qkv-uram", session["host_steps"][0])
            self.assertIn("e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json", session["canonical_result_path"])

    def test_build_session_fails_when_tar_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            root = Path(tmp) / "HGTXR"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")
            tar_path.unlink()

            session = session_tool.build_session(
                bundle_dir=out_dir,
                tar_path=tar_path,
                variant="c3b-mem16",
                preset="axis-c3b-mem16",
                hgtxr_root=root,
            )

            self.assertEqual(session["status"], "fail")
            self.assertTrue(any("tar missing" in error for error in session["bundle_validation_errors"]))

    def test_cli_writes_json_and_markdown_runbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            json_out = Path(tmp) / "session.json"
            markdown_out = Path(tmp) / "session.md"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = session_tool.main(
                    [
                        "--bundle-dir",
                        str(out_dir),
                        "--tar",
                        str(tar_path),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0, stream.getvalue())
            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            self.assertIn("HGTXR ZCU104 Smoke Session", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
