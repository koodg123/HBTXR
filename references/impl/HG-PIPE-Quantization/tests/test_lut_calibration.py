import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hgpipe_quantization.cli import main
from hgpipe_quantization.lut_calibration import apply_table, calibrate_gelu_requant, calibrate_requant, calibrate_softmax, cursor_for


class LutCalibrationTest(unittest.TestCase):
    def test_requant_builder_emits_hgpipe_cursor_contract(self):
        samples = np.arange(-16, 17, dtype=np.int64)

        payload = calibrate_requant(samples, entries=8, bits=3, signed=True, scale=0.5, max_iterations=2)

        self.assertEqual(payload["kind"], "requant_table")
        self.assertEqual(len(payload["scalars"]), 3)
        self.assertEqual(len(payload["tables"]["table"]), 8)
        self.assertFalse(payload["paper_equivalent"])
        cursor = cursor_for(samples, type("Params", (), {
            "offset": payload["index"]["offset"],
            "shift": payload["index"]["shift_scale"],
            "bound": payload["index"]["bound"],
        })())
        approx = apply_table(samples, type("Params", (), {
            "offset": payload["index"]["offset"],
            "shift": payload["index"]["shift_scale"],
            "bound": payload["index"]["bound"],
        })(), payload["tables"]["table"])
        self.assertEqual(cursor.shape, samples.shape)
        self.assertEqual(approx.shape, samples.shape)

    def test_gelu_requant_builder_fuses_activation_and_quantization(self):
        samples = np.arange(-8, 9, dtype=np.int64)

        payload = calibrate_gelu_requant(samples, entries=8, bits=3, signed=False, input_scale=0.1, output_scale=0.1)

        self.assertEqual(payload["kind"], "gelu_requant_table")
        self.assertEqual(payload["output_dtype"]["bits"], 3)
        self.assertFalse(payload["output_dtype"]["signed"])
        self.assertGreaterEqual(max(payload["tables"]["table"]), 1)

    def test_softmax_builder_emits_exp_and_segmented_recip_tables(self):
        rows = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.float32)

        payload = calibrate_softmax(rows, exp_entries=8, recip_entries=8, output_bits=3)

        self.assertEqual(payload["kind"], "softmax_segmented_table")
        self.assertEqual(len(payload["scalars"]), 14)
        self.assertEqual(len(payload["tables"]["exp_table"]), 8)
        self.assertEqual(len(payload["tables"]["recip_table_one"]), 8)
        self.assertEqual(len(payload["tables"]["recip_table_two"]), 8)
        self.assertEqual(payload["metrics"]["rows"], 2)

    def test_calibrate_lut_cli_writes_json_and_hgpipe_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_npy = root / "samples.npy"
            output_json = root / "lut.json"
            txt_dir = root / "txt"
            np.save(input_npy, np.arange(-16, 17, dtype=np.int64))

            code = main([
                "calibrate-lut-npy",
                "--kind",
                "requant",
                "--input-npy",
                str(input_npy),
                "--entries",
                "8",
                "--bits",
                "3",
                "--signed",
                "--scale",
                "0.5",
                "--json",
                str(output_json),
                "--txt-dir",
                str(txt_dir),
                "--stem",
                "unit",
            ])

            payload = json.loads(output_json.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(payload["schema"], "hgpipe_lut_calibration_v1")
            self.assertTrue((txt_dir / "unit_scalars.txt").exists())
            self.assertTrue((txt_dir / "unit_table_m.txt").exists())
            self.assertEqual(len(payload["txt_files"]), 2)


if __name__ == "__main__":
    unittest.main()
