from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid.optim.lr_schedulers import build_lr_scheduler, list_lr_scheduler_names, metric_mode


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)


def test_supported_scheduler_names() -> None:
    assert list_lr_scheduler_names() == ["cosine", "none", "plateau", "step"]


def test_none_type_returns_no_scheduler() -> None:
    assert build_lr_scheduler(_optimizer(), {"scheduler": {"type": "none"}}, 10) is None


def test_missing_scheduler_config_defaults_to_none() -> None:
    assert build_lr_scheduler(_optimizer(), {}, 10) is None


def test_cosine_step_and_plateau_construct_expected_types() -> None:
    cosine = build_lr_scheduler(_optimizer(), {"scheduler": {"type": "cosine"}}, 10)
    step = build_lr_scheduler(_optimizer(), {"scheduler": {"type": "step"}}, 10)
    plateau = build_lr_scheduler(_optimizer(), {"scheduler": {"type": "plateau"}}, 10)
    assert isinstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert isinstance(step, torch.optim.lr_scheduler.StepLR)
    assert isinstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_metric_mode_direction() -> None:
    assert metric_mode("val_p10_acc") == "max"
    assert metric_mode("val_score") == "max"
    assert metric_mode("val_pct") == "max"
    assert metric_mode("val_mean_distance") == "min"


def test_external_scheduler_block_returns_none() -> None:
    scheduler = build_lr_scheduler(
        _optimizer(),
        {"scheduler": {"type": "cosine"}},
        10,
        optimizer_meta={"external_scheduler_allowed": False},
    )
    assert scheduler is None


def test_unknown_type_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_lr_scheduler(_optimizer(), {"scheduler": {"type": "mystery"}}, 10)
