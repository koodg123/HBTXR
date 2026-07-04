from __future__ import annotations

import math

import torch

from hbtxr.loss.metrics import compute_metrics


def _batch() -> dict[str, torch.Tensor]:
    return {
        "cur_state": torch.tensor(
            [
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "annotation_quality": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "mask_valid": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        "closed_eye_flag": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        "valid_track": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    }


def test_track_p1_metric_uses_one_pixel_threshold_and_track_weights() -> None:
    outputs = {
        "track/state": torch.tensor(
            [
                [0.5, 0.0, 4.0, 4.0, 1.0, 0.0],
                [1.5, 0.0, 4.0, 4.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    metrics = compute_metrics(_batch(), outputs)

    assert math.isclose(float(metrics["metric_track_p1_pct"]), 50.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(metrics["metric_track_p5_pct"]), 100.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(metrics["metric_track_p10_pct"]), 100.0, rel_tol=0.0, abs_tol=1e-6)


def test_search_p1_metric_matches_search_threshold_pattern() -> None:
    outputs = {
        "search/state": torch.tensor(
            [
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
                [2.0, 0.0, 4.0, 4.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 4.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    }

    metrics = compute_metrics(_batch(), outputs)

    assert math.isclose(float(metrics["metric_search_p1_pct"]), 50.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(float(metrics["metric_search_p5_pct"]), 100.0, rel_tol=0.0, abs_tol=1e-6)
