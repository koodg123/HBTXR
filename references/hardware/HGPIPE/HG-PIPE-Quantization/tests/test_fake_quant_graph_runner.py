from pathlib import Path
import unittest

from hgpipe_quantization.compare import compare_trace_payloads
from hgpipe_quantization.fake_quant.graph_runner import FakeQuantGraphRunner
from hgpipe_quantization.int_infer import TorchIntCaseRunner


class FakeQuantGraphRunnerTest(unittest.TestCase):
    def test_fakequant_graph_runner_verifies_single_input_end_to_end_graph(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        results = FakeQuantGraphRunner(source_path).verify_end_to_end()

        self.assertEqual(len(results), 293)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)

    def test_fakequant_graph_traces_match_torch_int_lut_case_traces(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        fake_graph_traces = [trace.to_json() for trace in FakeQuantGraphRunner(source_path).trace_end_to_end()]
        int_case_traces = [trace.to_json() for trace in TorchIntCaseRunner(source_path).trace_lut_cases()]

        results = compare_trace_payloads(fake_graph_traces, int_case_traces)

        self.assertEqual(len(fake_graph_traces), 60)
        self.assertEqual(len(results), 60)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)


if __name__ == "__main__":
    unittest.main()
