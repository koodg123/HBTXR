import unittest
from pathlib import Path

from hgpipe_quantization.artifacts import HgPipeSource
from hgpipe_quantization.ops import layernorm_quantize, quantize_clamp, softmax_quantize, table_quantize
from hgpipe_quantization.pipeline import discover_cases, verify_case


class TestQuantOps(unittest.TestCase):
    def test_table_quantize_uses_hgpipe_cursor_formula(self):
        self.assertEqual(table_quantize([-2, 0, 2, 8], [2, 1, 3], [10, 11, 12, 13]), [10, 11, 12, 13])

    def test_quantize_clamp_signed_and_unsigned(self):
        self.assertEqual(quantize_clamp(-9, 4, signed=True), -8)
        self.assertEqual(quantize_clamp(8, 4, signed=True), 7)
        self.assertEqual(quantize_clamp(-1, 3, signed=False), 0)
        self.assertEqual(quantize_clamp(9, 3, signed=False), 7)

    def test_layernorm_small_case_matches_hls_integer_steps(self):
        result = layernorm_quantize(
            inputs=[1, 3],
            scalars=[1, 1, 0, 0, 10, 0, 4],
            lnw=[1, 1],
            lnb=[0, 0],
            rsqrt_table=[1] * 11,
        )
        self.assertEqual(result, [-1, 1])

    def test_softmax_small_case_reconstructs_segmented_tables(self):
        result = softmax_quantize(
            inputs=[1, 0, 0, 1],
            scalars=[0, 0, 3, 0, 0, 10, 0, 0, 0, 0, 10, 0, 0, 3],
            exp_table=[4, 2, 1, 0],
            recip_table_one=[0, 1, 2, 3, 4, 5, 6],
            recip_table_two=[0],
            tokens=2,
            heads=1,
        )
        self.assertEqual(result, [7, 7, 7, 7])


class TestReferenceSmoke(unittest.TestCase):
    def test_reference_sample_cases_are_bit_exact(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        source = HgPipeSource.from_path(source_path)
        cases = {case.name: case for case in discover_cases(source)}
        for name in ["attn_0_qq", "attn_0_lnq", "attn_0_softmaxq", "mlp_0_geluq", "head_lnq"]:
            with self.subTest(name=name):
                result = verify_case(cases[name])
                self.assertTrue(result.passed, result)


if __name__ == "__main__":
    unittest.main()
