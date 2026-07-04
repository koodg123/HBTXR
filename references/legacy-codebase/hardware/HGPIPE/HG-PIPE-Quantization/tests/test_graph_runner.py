from pathlib import Path
import unittest

from hgpipe_quantization.artifacts import HgPipeSource
from hgpipe_quantization.graph import ArtifactGraphRunner


class GraphRunnerTest(unittest.TestCase):
    def test_artifact_graph_runner_verifies_reference_graph(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        source = HgPipeSource.from_path(source_path)
        results = ArtifactGraphRunner(source).verify_graph()

        self.assertEqual(len(results), 268)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)

    def test_artifact_graph_runner_verifies_single_input_end_to_end_graph(self):
        source_path = Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"
        source = HgPipeSource.from_path(source_path)
        results = ArtifactGraphRunner(source).verify_end_to_end()

        self.assertEqual(len(results), 293)
        self.assertTrue(all(result.passed for result in results))
        self.assertEqual(sum(result.mismatches for result in results), 0)


if __name__ == "__main__":
    unittest.main()
