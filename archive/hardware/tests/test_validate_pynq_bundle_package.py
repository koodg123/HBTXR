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

import package_e2e_axis_dma_pynq_bundle as bundler  # noqa: E402
import validate_pynq_bundle_package as validator  # noqa: E402


class ValidatePynqBundlePackageTests(unittest.TestCase):
    def make_fake_hardware_root(self, tmp: Path, prefix: str) -> Path:
        root = tmp / "hardware"
        for rel in bundler.PACKAGE_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n")
        for rel in bundler.WEIGHT_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00\x00\x00\x00")
        for rel in bundler.TOOL_FILES:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n")
        for suffix in [".bit", ".hwh"]:
            path = root / "pynq" / "hgtxr" / f"{prefix}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{suffix}\n")
        return root

    def test_validate_accepts_generated_c3b_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")

            errors = validator.validate_bundle(out_dir, tar_path, "c3b-mem16")

            self.assertEqual(errors, [])

    def test_validate_accepts_generated_vref_successor_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-dsp-mixed-stream")

            errors = validator.validate_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-dsp-mixed-stream")

            self.assertEqual(errors, [])

    def test_validate_accepts_generated_qkv_uram_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-qkv-uram")

            errors = validator.validate_bundle(out_dir, tar_path, "vref-p0-softmax-input-x2-qkv-uram")

            self.assertEqual(errors, [])

    def test_validate_accepts_generated_par32_rom_compute_search_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "par32-rom-compute-300-search")

            errors = validator.validate_bundle(out_dir, tar_path, "par32-rom-compute-300-search")

            self.assertEqual(errors, [])

    def test_validate_accepts_generated_par32_rom_compute_track_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "par32-rom-compute-300-track")

            errors = validator.validate_bundle(out_dir, tar_path, "par32-rom-compute-300-track")

            self.assertEqual(errors, [])

    def test_validate_accepts_generated_par32_prefetchall4_hybrid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            prefix = (
                "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
                "corefabric_normuram_compute_prefetchall4_300_mem16"
            )
            fake_root = self.make_fake_hardware_root(tmp, prefix)
            original_root = bundler.HARDWARE_ROOT
            bundler.HARDWARE_ROOT = fake_root
            self.addCleanup(setattr, bundler, "HARDWARE_ROOT", original_root)
            out_dir = tmp / "bundle"
            tar_path = tmp / "bundle.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "par32-prefetchall4-300-hybrid-10-90")

            errors = validator.validate_bundle(out_dir, tar_path, "par32-prefetchall4-300-hybrid-10-90")

            self.assertEqual(errors, [])

    def test_validate_rejects_tar_missing_validator_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            broken_tar = Path(tmp) / "broken.tar.gz"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")
            (out_dir / "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh").unlink()
            with tarfile.open(broken_tar, "w:gz") as tar:
                tar.add(out_dir, arcname=out_dir.name)

            errors = validator.validate_bundle(out_dir, broken_tar, "c3b-mem16")

            self.assertTrue(any("bundle file missing" in error for error in errors))
            self.assertTrue(any("tar missing files" in error for error in errors))

    def test_cli_writes_json_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            json_out = Path(tmp) / "validation.json"
            bundler.build_bundle(out_dir, tar_path, "c3b-mem16")
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = validator.main([
                    "--bundle-dir",
                    str(out_dir),
                    "--tar",
                    str(tar_path),
                    "--variant",
                    "c3b-mem16",
                    "--json-out",
                    str(json_out),
                ])

            self.assertEqual(code, 0, stream.getvalue())
            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["status"], "pass")


if __name__ == "__main__":
    unittest.main()
