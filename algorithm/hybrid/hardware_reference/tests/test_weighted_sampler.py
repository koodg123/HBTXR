from __future__ import annotations

import pytest
import torch

from hbtxr.data.loader import build_sample_weights_from_rows


def test_failure_bucket_weights_multiply_low_similarity_and_session() -> None:
    rows = [
        {"similarity_target": 0.05, "session_key": "user41/right/session_201"},
        {"similarity_target": 0.05, "session_key": "user01/right/session_001"},
        {"similarity_target": 0.5, "session_key": "user41/right/session_201"},
        {"similarity_target": 0.5, "session_key": "user01/right/session_001"},
    ]
    weights = build_sample_weights_from_rows(
        rows,
        {
            "low_similarity": {"enabled": True, "threshold": 0.1, "multiplier": 3.0},
            "session_contains": {"contains": "session_201", "multiplier": 2.0},
            "max_multiplier": 6.0,
        },
    )

    assert torch.allclose(weights, torch.tensor([6.0, 3.0, 2.0, 1.0], dtype=torch.double))


def test_failure_bucket_weights_clamp_to_min_weight() -> None:
    weights = build_sample_weights_from_rows(
        [{"similarity_target": 0.9}],
        {"base_weight": 0.0, "min_weight": 0.25},
    )

    assert weights.tolist() == [0.25]


def test_failure_bucket_weights_reject_all_zero_when_min_weight_zero() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        build_sample_weights_from_rows(
            [{"similarity_target": 0.9}],
            {"base_weight": 0.0, "min_weight": 0.0},
        )
