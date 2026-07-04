from pathlib import Path
import unittest

from hgpipe_quantization.fake_quant.graph_runner import FakeQuantGraphRunner
from hgpipe_quantization.int_infer import TorchIntGraphRunner
from hgpipe_quantization.run_result import compare_inference_results, make_inference_result


class RunResultTest(unittest.TestCase):
    def test_torch_int_and_fakequant_graph_final_logits_match(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        int_logits, _ = TorchIntGraphRunner(source_path).forward_from_patch_input()
        fake_logits, _ = FakeQuantGraphRunner(source_path).forward_from_patch_input()

        int_result = make_inference_result(
            runner="torch_int",
            output_name="head_logits",
            values=int_logits.detach().cpu().numpy().reshape(-1),
        )
        fake_result = make_inference_result(
            runner="fakequant_graph",
            output_name="head_logits",
            values=fake_logits.detach().cpu().numpy().reshape(-1),
        )
        comparison = compare_inference_results(fake_result.to_json(), int_result.to_json())

        self.assertEqual(int_result.numel, 1000)
        self.assertEqual(fake_result.numel, 1000)
        self.assertTrue(comparison.passed)
        self.assertEqual(comparison.mismatches, 0)
        self.assertTrue(comparison.top1_equal)

    def test_compare_inference_results_reports_mismatch(self):
        left = make_inference_result(runner="left", output_name="x", values=[1, 2]).to_json()
        right = make_inference_result(runner="right", output_name="x", values=[1, 3]).to_json()

        comparison = compare_inference_results(left, right)

        self.assertFalse(comparison.passed)
        self.assertEqual(comparison.mismatches, 1)
        self.assertEqual(comparison.max_abs_error, 1.0)


if __name__ == "__main__":
    unittest.main()
