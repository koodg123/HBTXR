import unittest

from hgpipe_quantization.compare import compare_trace_payloads
from hgpipe_quantization.fake_quant.runner import FakeQuantRunner
from hgpipe_quantization.int_infer import TorchIntCaseRunner


class TraceCompareTest(unittest.TestCase):
    def test_fakequant_and_torch_int_lut_traces_match(self):
        from pathlib import Path

        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        fake_traces = [trace.to_json() for trace in FakeQuantRunner(source_path).trace_lut_cases()]
        int_traces = [trace.to_json() for trace in TorchIntCaseRunner(source_path).trace_lut_cases()]

        results = compare_trace_payloads(fake_traces, int_traces)

        self.assertEqual(len(fake_traces), 60)
        self.assertEqual(len(int_traces), 60)
        self.assertEqual(len(results), 60)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)

    def test_compare_trace_payloads_reports_mismatch(self):
        left = [{"name": "x", "runner": "a", "values": [1.0, 2.0]}]
        right = [{"name": "x", "runner": "b", "values": [1.0, 3.0]}]

        result = compare_trace_payloads(left, right)[0]

        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)
        self.assertEqual(result.max_abs_error, 1.0)


if __name__ == "__main__":
    unittest.main()

