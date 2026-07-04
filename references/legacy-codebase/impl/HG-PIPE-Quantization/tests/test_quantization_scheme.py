import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hgpipe_quantization.cli import main
from hgpipe_quantization.quantization_scheme import (
    calibrate_dyadic_scale_kl,
    hardware_lut_index,
    quantize_group_vector,
    quantize_nonlinear_lut,
)


class QuantizationSchemeTest(unittest.TestCase):
    def test_group_vector_quantization_uses_per_token_scales_for_activation(self):
        values = np.asarray([[1.0, -2.0, 3.0, -4.0], [8.0, -4.0, 2.0, -1.0]], dtype=np.float32)

        result = quantize_group_vector(values, tensor_role="activation", bits=3, group_size=2)

        self.assertEqual(result["granularity"], "per-token")
        self.assertEqual(result["quantized"].shape, values.shape)
        self.assertEqual(result["scales"].shape, (2, 2))
        self.assertTrue(np.all(result["quantized"] <= 3))
        self.assertTrue(np.all(result["quantized"] >= -4))

    def test_group_vector_quantization_uses_per_channel_scales_for_weights(self):
        weights = np.asarray([[1.0, -2.0, 3.0], [8.0, -4.0, 2.0]], dtype=np.float32)

        result = quantize_group_vector(weights, tensor_role="weight", bits=3)

        self.assertEqual(result["granularity"], "per-channel")
        self.assertEqual(result["quantized"].shape, weights.shape)
        self.assertEqual(result["scales"].shape, (2,))
        self.assertGreater(result["scales"][1], result["scales"][0])

    def test_dyadic_scale_calibration_returns_multiplier_shift_and_kl_metric(self):
        values = np.linspace(-2.0, 2.0, 257, dtype=np.float32)

        result = calibrate_dyadic_scale_kl(values, bits=4, signed=True, histogram_bins=64)

        self.assertEqual(result["method"], "kl_divergence")
        self.assertGreater(result["shift"], 0)
        self.assertGreater(result["multiplier"], 0)
        self.assertGreater(result["effective_scale"], 0.0)
        self.assertIn("kl_divergence", result)

    def test_nonlinear_lut_wrapper_supports_percentile_clipped_gelu(self):
        values = np.arange(-32, 33, dtype=np.int64)

        result = quantize_nonlinear_lut(
            values,
            op="gelu",
            entries=16,
            bits=3,
            percentile=90.0,
            input_scale=0.1,
            output_scale=0.1,
        )

        self.assertEqual(result["kind"], "gelu_requant_table")
        self.assertEqual(result["parameters"]["percentile"], 90.0)
        self.assertEqual(len(result["tables"]["table"]), 16)

    def test_hardware_lut_index_maps_input_to_bounded_entry_address(self):
        values = np.asarray([-16, -8, 0, 8, 16], dtype=np.int64)

        indices = hardware_lut_index(values, offset=16, shift=3, entries=8)

        np.testing.assert_array_equal(indices, np.asarray([0, 1, 2, 3, 4]))
        self.assertEqual(hardware_lut_index(np.asarray([100]), offset=16, shift=3, entries=8)[0], 7)

    def test_group_vector_cli_writes_quantized_tensor_and_scales(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_npy = root / "x.npy"
            quantized_npy = root / "xq.npy"
            scales_npy = root / "scales.npy"
            output_json = root / "report.json"
            np.save(input_npy, np.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32))

            code = main([
                "quantize-group-vector-npy",
                "--input-npy",
                str(input_npy),
                "--tensor-role",
                "activation",
                "--bits",
                "3",
                "--group-size",
                "2",
                "--quantized-npy",
                str(quantized_npy),
                "--scales-npy",
                str(scales_npy),
                "--json",
                str(output_json),
            ])

            self.assertEqual(code, 0)
            self.assertTrue(quantized_npy.exists())
            self.assertTrue(scales_npy.exists())
            self.assertEqual(json.loads(output_json.read_text())["granularity"], "per-token")

    def test_dyadic_scale_cli_writes_kl_calibration_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_npy = root / "linear.npy"
            output_json = root / "dyadic.json"
            np.save(input_npy, np.linspace(-2.0, 2.0, 257, dtype=np.float32))

            code = main([
                "calibrate-linear-dyadic-npy",
                "--input-npy",
                str(input_npy),
                "--bits",
                "4",
                "--json",
                str(output_json),
            ])

            payload = json.loads(output_json.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(payload["method"], "kl_divergence")
            self.assertEqual(payload["scale_type"], "dyadic")


if __name__ == "__main__":
    unittest.main()
