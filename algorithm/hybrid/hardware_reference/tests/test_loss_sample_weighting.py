from __future__ import annotations

import pytest
import torch

from hbtxr.loss.stage_common import build_loss_sample_weights


def _batch() -> dict:
    return {
        "annotation_quality": torch.ones(4, dtype=torch.float32),
        "similarity_target": torch.tensor([0.05, 0.05, 0.5, 0.5], dtype=torch.float32),
        "meta": [
            {"session_key": "user41/right/session_201"},
            {"session_key": "user01/right/session_001"},
            {"session_key": "user41/right/session_201"},
            {"session_key": "user01/right/session_001"},
        ],
    }


def test_track_sample_weights_multiply_low_similarity_and_session() -> None:
    weights = build_loss_sample_weights(
        _batch(),
        {
            "track_sample_weight": {
                "enabled": True,
                "low_similarity": {"enabled": True, "threshold": 0.1, "multiplier": 3.0},
                "session_contains": {"field": "session_key", "contains": "session_201", "multiplier": 2.0},
                "max_multiplier": 6.0,
            },
        },
    )

    assert weights is not None
    assert torch.allclose(weights, torch.tensor([6.0, 3.0, 2.0, 1.0]))


def test_track_sample_weights_disabled_by_default() -> None:
    assert build_loss_sample_weights(_batch(), {}) is None


def test_track_sample_weights_reject_all_zero() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        build_loss_sample_weights(
            _batch(),
            {"track_sample_weight": {"enabled": True, "base_weight": 0.0, "min_weight": 0.0}},
        )
