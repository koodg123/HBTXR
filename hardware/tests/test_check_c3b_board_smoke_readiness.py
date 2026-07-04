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

import check_c3b_board_smoke_readiness as readiness_tool  # noqa: E402
import write_c3b_smoke_transfer_manifest as transfer_tool  # noqa: E402
from validate_pynq_bundle_package import sha256_file  # noqa: E402


class CheckC3bBoardSmokeReadinessTests(unittest.TestCase):
    def make_bundle(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, Path, Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "XR-VIT" / "HGTXR"
        bundle_dir = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
        tar_path = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
        session_path = root / "hardware" / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
        docs_resources = root / "docs" / "resources"
        transfer_path = docs_resources / "c3b_smoke_transfer_manifest_2026_06_10.json"
        sha256_path = docs_resources / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
        bundle_dir.mkdir(parents=True)
        docs_resources.mkdir(parents=True)
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
            file_entries.append({"bundle": rel, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "name": "hgtxr_e2e_axis_dma_smoke_bundle",
            "target": "ZCU104 PYNQ",
            "variant": "c3b-mem16",
            "artifact_prefix": "hgtxr_e2e_axis_dma_c3b_mem16",
            "command": "PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 --weights-mode file",
            "validation_command": "python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
            "expected_runtime_state": 2,
            "expected_out_raw": [32, -13, 26, -6, 14, -11],
            "files": file_entries,
        }
        (bundle_dir / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest))
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=bundle_dir.name)
        manifest["tar"] = {"path": str(tar_path), "name": tar_path.name, "bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path)}
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
                    "canonical_result_path": str(root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"),
                    "board_steps": [
                        "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
                        "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                        "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                    ],
                }
            )
        )
        transfer = transfer_tool.build_manifest(session_path=session_path, bundle_dir=bundle_dir, tar_path=tar_path, hgtxr_root=root)
        transfer_path.write_text(json.dumps(transfer, indent=2, sort_keys=True) + "\n")
        sha256_path.write_text(transfer["tar"]["sha256_line"] + "\n")
        return tmp, root, bundle_dir, tar_path, session_path, transfer_path

    def test_ready_for_board_when_physical_result_missing(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path, transfer_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        sha256_path = root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
        result_path = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"

        readiness = readiness_tool.build_readiness(
            transfer_manifest_path=transfer_path,
            session_path=session_path,
            bundle_dir=bundle_dir,
            tar_path=tar_path,
            sha256_file_path=sha256_path,
            result_path=result_path,
        )

        self.assertEqual(readiness["status"], "ready-for-board")
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["physical_result"]["status"], "missing")
        self.assertFalse(readiness["errors"])

    def test_blocks_on_corrupt_sha256_file(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path, transfer_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        sha256_path = root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
        sha256_path.write_text("bad  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n")

        readiness = readiness_tool.build_readiness(
            transfer_manifest_path=transfer_path,
            session_path=session_path,
            bundle_dir=bundle_dir,
            tar_path=tar_path,
            sha256_file_path=sha256_path,
            result_path=root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json",
        )

        self.assertEqual(readiness["status"], "blocked")
        self.assertTrue(any("sha256 file line" in error for error in readiness["errors"]))

    def test_complete_when_physical_result_validates(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path, transfer_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        result_path = root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
        result_path.parent.mkdir(parents=True)
        result_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "variant": "c3b-mem16",
                    "weights_mode": "file",
                    "expected_runtime_state": 2,
                    "runtime_state": 2,
                    "runtime_match": True,
                    "expected_out_raw": [32, -13, 26, -6, 14, -11],
                    "out_raw": [32, -13, 26, -6, 14, -11],
                    "out_state": [32, -13, 26, -6, 14, -11],
                    "output_match": True,
                    "dma_in_name": "axi_dma_0",
                    "dma_out_name": "axi_dma_1",
                    "bitfile": "hgtxr_e2e_axis_dma_c3b_mem16.bit",
                    "hwhfile": "hgtxr_e2e_axis_dma_c3b_mem16.hwh",
                }
            )
        )

        readiness = readiness_tool.build_readiness(
            transfer_manifest_path=transfer_path,
            session_path=session_path,
            bundle_dir=bundle_dir,
            tar_path=tar_path,
            sha256_file_path=root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
            result_path=result_path,
        )

        self.assertEqual(readiness["status"], "complete")
        self.assertTrue(readiness["ready"])

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, root, bundle_dir, tar_path, session_path, transfer_path = self.make_bundle()
        self.addCleanup(tmp.cleanup)
        json_out = root / "hardware" / "generated" / "readiness.json"
        markdown_out = root / "hardware" / "generated" / "readiness.md"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            code = readiness_tool.main(
                [
                    "--root",
                    str(root),
                    "--transfer-manifest",
                    str(transfer_path),
                    "--session-json",
                    str(session_path),
                    "--bundle-dir",
                    str(bundle_dir),
                    "--tar",
                    str(tar_path),
                    "--sha256-file",
                    str(root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(json_out.read_text())["status"], "ready-for-board")
        self.assertIn("HGTXR C3b Board Smoke Readiness", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
