from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import package_e2e_m_axi_pynq_bundle as bundler  # noqa: E402


class E2EMaxiPynqBundleTests(unittest.TestCase):
    def test_build_bundle_writes_expected_files_manifest_and_tar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "bundle"
            tar_path = Path(tmp) / "bundle.tar.gz"
            result = bundler.build_bundle(out_dir, tar_path)
            manifest = json.loads((out_dir / "BUNDLE_MANIFEST.json").read_text())

            self.assertEqual(result["out_dir"], str(out_dir))
            self.assertEqual(result["manifest"], str(out_dir / "BUNDLE_MANIFEST.json"))
            self.assertTrue(tar_path.exists())
            self.assertEqual(manifest["expected_runtime_state"], 2)
            self.assertEqual(manifest["expected_out_raw"], [32, -13, 26, -6, 14, -11])
            self.assertIn("--weights-mode file", manifest["command"])
            self.assertIn("--weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin", manifest["command"])

            bundle_files = {entry["bundle"] for entry in manifest["files"]}
            required = {
                "hgtxr/__init__.py",
                "hgtxr/hgtxr_overlay.py",
                "hgtxr/e2e_m_axi_overlay.py",
                "hgtxr/e2e_m_axi_weights.py",
                "hgtxr/run_e2e_m_axi_smoke.py",
                "hgtxr/hgtxr_e2e_m_axi.bit",
                "hgtxr/hgtxr_e2e_m_axi.hwh",
                "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
                "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
                "run_e2e_m_axi_file_smoke.sh",
            }
            self.assertTrue(required.issubset(bundle_files))
            self.assertFalse(any("__pycache__" in path for path in bundle_files))
            self.assertFalse(any(path in {"hgtxr/hgtxr.bit", "hgtxr/hgtxr.hwh"} for path in bundle_files))

            readme = (out_dir / "hgtxr" / "README.md").read_text()
            self.assertIn("weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin", readme)
            self.assertNotIn("hardware/refs/weights", readme)


if __name__ == "__main__":
    unittest.main()
