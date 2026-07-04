from pathlib import Path
import unittest

from hgpipe_quantization.int_infer import TorchIntGraphRunner


class TorchIntRunnerTest(unittest.TestCase):
    def test_torch_int_runner_verifies_single_input_end_to_end_graph(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        results = TorchIntGraphRunner(source_path).verify_end_to_end()

        self.assertEqual(len(results), 293)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)


if __name__ == "__main__":
    unittest.main()

