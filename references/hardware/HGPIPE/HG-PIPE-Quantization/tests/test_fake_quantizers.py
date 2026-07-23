import unittest

import torch
from torch import nn

from hgpipe_quantization.fake_quant import AffineFakeQuantizer, HGTableFakeQuantizer, insert_output_fake_quantizers
from hgpipe_quantization.fake_quant.runner import FakeQuantRunner


class FakeQuantizerTest(unittest.TestCase):
    def test_hg_table_fake_quantizer_uses_lut_cursor_formula(self):
        quantizer = HGTableFakeQuantizer(scalars=[2, 1, 3], table=[10, 11, 12, 13])

        output = quantizer(torch.tensor([-2.0, 0.0, 2.0, 8.0]))

        torch.testing.assert_close(output, torch.tensor([10.0, 11.0, 12.0, 13.0]))

    def test_affine_fake_quantizer_dequantizes_to_float_domain(self):
        quantizer = AffineFakeQuantizer(scale=0.5, zero_point=0, qmin=-2, qmax=1)

        output = quantizer(torch.tensor([-2.0, -0.7, 0.2, 2.0]))

        torch.testing.assert_close(output, torch.tensor([-1.0, -0.5, 0.0, 0.5]))

    def test_insert_output_fake_quantizers_adds_module_after_target(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = Toy()
        with torch.no_grad():
            model.linear.weight.copy_(torch.eye(2))

        graph = insert_output_fake_quantizers(
            model,
            {"linear": lambda: AffineFakeQuantizer(scale=1.0, zero_point=0, qmin=0, qmax=1)},
        )
        output = graph(torch.tensor([[0.2, 2.0]]))

        self.assertTrue(any(name.endswith("fake_quant") for name, _ in graph.named_modules()))
        torch.testing.assert_close(output, torch.tensor([[0.0, 1.0]]))

    def test_fakequant_runner_verifies_lut_artifact_cases(self):
        from pathlib import Path

        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        results = FakeQuantRunner(source_path).verify_lut_cases()

        self.assertEqual(len(results), 60)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)


if __name__ == "__main__":
    unittest.main()

