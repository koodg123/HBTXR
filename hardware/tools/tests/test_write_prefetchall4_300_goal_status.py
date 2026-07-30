from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import write_prefetchall4_300_goal_status as status_tool  # noqa: E402


def board_payload(profile: str, accelerator_samples: list[float], board_samples: list[float]) -> dict:
    runtime_state = 0 if profile == "search" else 1
    return {
        "runtime_state": runtime_state,
        "repeat": len(accelerator_samples),
        "accelerator_latency_ms_samples": accelerator_samples,
        "accelerator_latency_ms_summary": status_tool.summarize(accelerator_samples),
        "latency_ms_samples": board_samples,
        "latency_ms_summary": status_tool.summarize(board_samples),
        "dma_bandwidth_Bps_samples": [100.0 if profile == "search" else 200.0],
        "dma_bandwidth_Bps_summary": status_tool.summarize([100.0 if profile == "search" else 200.0]),
        "aggregate_dma_effective_bandwidth_Bps_samples": [300.0 if profile == "search" else 400.0],
        "aggregate_dma_effective_bandwidth_Bps_summary": status_tool.summarize(
            [300.0 if profile == "search" else 400.0]
        ),
    }


class PrefetchAll4GoalStatusTests(unittest.TestCase):
    def test_build_board_measurements_reports_missing_without_payloads(self) -> None:
        measurements = status_tool.build_board_measurements({}, {})

        self.assertEqual(measurements["status"], "missing")
        self.assertEqual(measurements["hybrid_10_90"]["status"], "missing")
        self.assertIsNone(measurements["hybrid_10_90"]["accelerator_latency_ms_weighted_mean"])

    def test_build_board_measurements_computes_synthetic_10_90_hybrid(self) -> None:
        search = board_payload("search", [4.0, 5.0], [4.5, 5.5])
        track = board_payload("track", [1.0, 2.0], [1.5, 2.5])

        measurements = status_tool.build_board_measurements(search, track)
        hybrid = measurements["hybrid_10_90"]

        self.assertEqual(measurements["status"], "available")
        self.assertEqual(hybrid["status"], "synthetic-from-per-mode-samples")
        self.assertAlmostEqual(hybrid["accelerator_latency_ms_weighted_mean"], 1.8)
        self.assertAlmostEqual(hybrid["board_latency_ms_weighted_mean"], 2.3)
        self.assertAlmostEqual(hybrid["dma_bandwidth_Bps_weighted_mean"], 190.0)
        self.assertAlmostEqual(hybrid["aggregate_dma_effective_bandwidth_Bps_weighted_mean"], 390.0)
        self.assertAlmostEqual(hybrid["throughput_fps_from_accelerator_weighted_mean"], 1000.0 / 1.8)
        self.assertEqual(hybrid["worst_case_accelerator_latency_ms"], 5.0)
        self.assertEqual(hybrid["accelerator_latency_ms_summary"]["count"], 20)

    def test_build_board_measurements_prefers_measured_interleaved_hybrid(self) -> None:
        hybrid_payload = {
            "status": "pass",
            "repeat": 10,
            "search_track_invocation_distribution": {"search": 1, "track": 9, "unknown": 0},
            "accelerator_latency_ms_summary": {"count": 10, "mean": 1.07, "max": 3.5},
            "latency_ms_summary": {"count": 10, "mean": 1.3, "max": 4.0},
            "dma_bandwidth_Bps_summary": {"count": 10, "mean": 190.0},
            "aggregate_dma_effective_bandwidth_Bps_summary": {"count": 10, "mean": 390.0},
            "mode_summaries": {
                "search": {
                    "count": 1,
                    "expected_runtime_state": 0,
                    "accelerator_latency_ms_summary": {"count": 1, "mean": 3.5, "max": 3.5},
                },
                "track": {
                    "count": 9,
                    "expected_runtime_state": 1,
                    "accelerator_latency_ms_summary": {"count": 9, "mean": 0.8, "max": 0.8},
                },
            },
        }

        measurements = status_tool.build_board_measurements({}, {}, hybrid_payload)
        hybrid = measurements["hybrid_10_90"]

        self.assertEqual(measurements["status"], "available")
        self.assertEqual(measurements["search"]["source"], "hybrid-json")
        self.assertEqual(measurements["track"]["source"], "hybrid-json")
        self.assertEqual(hybrid["status"], "measured-interleaved-board-run")
        self.assertAlmostEqual(hybrid["search_ratio"], 0.1)
        self.assertAlmostEqual(hybrid["track_ratio"], 0.9)
        self.assertEqual(hybrid["accelerator_latency_ms_weighted_mean"], 1.07)
        self.assertEqual(hybrid["board_latency_ms_weighted_mean"], 1.3)
        self.assertEqual(hybrid["worst_case_accelerator_latency_ms"], 3.5)


if __name__ == "__main__":
    unittest.main()
