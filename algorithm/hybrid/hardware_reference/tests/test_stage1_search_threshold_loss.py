from __future__ import annotations

import torch

from hbtxr.loss.stage1 import compute_stage1_losses


def _batch() -> dict[str, torch.Tensor]:
    return {
        "annotation_quality": torch.ones(2, dtype=torch.float32),
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "mask_valid": torch.ones(2, dtype=torch.float32),
        "valid_track": torch.ones(2, dtype=torch.float32),
        "aux_target": torch.zeros(2, dtype=torch.long),
        "cur_state": torch.tensor(
            [
                [0.0, 0.0, 8.0, 4.0, 0.0, 1.0],
                [0.0, 0.0, 8.0, 4.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "pupil_search_target": torch.zeros((2, 7), dtype=torch.float32),
        "constraint_center": torch.zeros((2, 2), dtype=torch.float32),
    }


def test_stage1_search_threshold_losses_default_to_zero() -> None:
    batch = _batch()
    outputs = {
        "search/state": torch.tensor(
            [
                [12.0, 0.0, 8.0, 4.0, 0.0, 1.0],
                [4.0, 0.0, 8.0, 4.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    }

    logs = compute_stage1_losses(batch, outputs, {}, active_head="search")

    assert "loss_search_p10_soft_threshold" in logs
    assert "loss_search_p5_soft_threshold" in logs
    assert logs["loss_search_p10_soft_threshold"].item() == 0.0
    assert logs["loss_search_p5_soft_threshold"].item() == 0.0


def test_stage1_search_threshold_losses_penalize_metric_boundary() -> None:
    batch = _batch()
    outputs = {
        "search/state": torch.tensor(
            [
                [12.0, 0.0, 8.0, 4.0, 0.0, 1.0],
                [4.0, 0.0, 8.0, 4.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    }

    logs = compute_stage1_losses(
        batch,
        outputs,
        {
            "search_p10_soft_threshold_weight": 0.25,
            "search_p10_soft_threshold_margin_px": 10.0,
            "search_p10_soft_threshold_temperature_px": 2.0,
            "search_p5_soft_threshold_weight": 0.5,
            "search_p5_soft_threshold_margin_px": 5.0,
            "search_p5_soft_threshold_temperature_px": 1.0,
        },
        active_head="search",
    )

    errors = torch.tensor([12.0, 4.0], dtype=torch.float32)
    expected_p10 = 0.25 * torch.nn.functional.softplus((errors - 10.0) / 2.0).mean()
    expected_p5 = 0.5 * torch.nn.functional.softplus(errors - 5.0).mean()

    assert torch.isclose(logs["loss_search_p10_soft_threshold"], expected_p10)
    assert torch.isclose(logs["loss_search_p5_soft_threshold"], expected_p5)
    assert logs["loss_total"] >= logs["loss_search_p10_soft_threshold"] + logs["loss_search_p5_soft_threshold"]
