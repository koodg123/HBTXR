#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_c3b_smoke_transfer_manifest as transfer_tool  # noqa: E402
from validate_pynq_bundle_package import sha256_file  # noqa: E402


class WriteC3bSmokeTransferManifestTests(unittest.TestCase):
    def make_bundle(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "XR-VIT" / "HGTXR"
        bundle_dir = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
        tar_path = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        session_path = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
        bundle_dir.mkdir(parents=True)
        required_files = [
            "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        ]
        file_entries = []
        for rel in required_files:
            path = bundle_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rel)
            file_entries.append(
                {
                    "bundle": rel,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "name": "hgtxr_e2e_axis_dma_smoke_bundle",
            "target": "ZCU104 PYNQ",
            "variant": "c3b-mem16",
            "artifact_prefix": "hgtxr_e2e_axis_dma_c3b_mem16",
            "command": "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "validation_command": "python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
            "files": file_entries,
        }
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        manifest["tar"] = {
            "path": str(tar_path),
            "bytes": tar_path.stat().st_size,
            "sha256": sha256_file(tar_path),
        }
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "c3b-mem16",
                    "preset": "axis-c3b-mem16",
                    "tar": manifest["tar"],
                    "expected_runtime_state": 2,
                    "expected_out_raw": [32, -13, 26, -6, 14, -11],
                    "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
                    "validation_json": "e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
                    "board_steps": [
                        "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
                        "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                        "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                    ],
                }
            )
        )
        return tmp, root, bundle_dir, tar_path, session_path

    def test_build_manifest_records_sha_and_board_commands(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)

        manifest = transfer_tool.build_manifest(
            session_path=session_path,
            bundle_dir=bundle_dir,
            tar_path=tar_path,
            hgtxr_root=root,
        )

        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(manifest["tar"]["sha256"], sha256_file(tar_path))
        self.assertIn("sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256", manifest["board_verify_commands"])
        self.assertIn("./run_e2e_axis_dma_c3b_mem16_file_smoke.sh", manifest["board_run_commands"])
        self.assertEqual(manifest["board_expected_outputs"]["expected_runtime_state"], 2)

    def test_detects_session_tar_sha_mismatch(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        session = json.loads(session_path.read_text())
        session["tar"]["sha256"] = "bad"
        session_path.write_text(json.dumps(session))

        manifest = transfer_tool.build_manifest(
            session_path=session_path,
            bundle_dir=bundle_dir,
            tar_path=tar_path,
            hgtxr_root=root,
        )

        self.assertEqual(manifest["status"], "fail")
        self.assertTrue(any("session tar sha256 mismatch" in error for error in manifest["bundle_validation_errors"]))

    def test_cli_writes_json_markdown_and_sha256(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        json_out = root / "generated" / "transfer.json"
        markdown_out = root / "generated" / "transfer.md"
        sha256_out = root / "generated" / "bundle.sha256"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            code = transfer_tool.main(
                [
                    "--root",
                    str(root),
                    "--session-json",
                    str(session_path),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--tar",
                    str(tar_path),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                    "--sha256-out",
                    str(sha256_out),
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
        self.assertIn("HGTXR C3b Smoke Transfer Manifest", markdown_out.read_text())
        self.assertIn(tar_path.name, sha256_out.read_text())


if __name__ == "__main__":
    unittest.main()
