from __future__ import annotations

import json

import pytest
import torch

from hbtxr.training.trainer import _ConsoleLogger, _apply_trainable_filter


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.track_head = torch.nn.Linear(4, 2)
        self.track_center_candidate_head = torch.nn.Linear(4, 3)


def test_trainable_filter_keeps_only_matching_parameters(tmp_path):
    model = TinyModel()
    output_dir = tmp_path / "run" / "train"
    logger = _ConsoleLogger({"console_log": {"enabled": False}})

    report = _apply_trainable_filter(
        model,
        {"trainable": {"include": ["track_head.*"]}},
        output_dir=output_dir,
        console_logger=logger,
    )

    assert report is not None
    assert all(param.requires_grad for _, param in model.track_head.named_parameters())
    assert not any(param.requires_grad for _, param in model.backbone.named_parameters())
    report_path = tmp_path / "run" / "hypers" / "trainable_filter.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["trainable_names"] == ["track_head.weight", "track_head.bias"]
    assert saved["frozen_tensor_count"] == 4


def test_trainable_filter_accepts_track_center_candidate_head_pattern(tmp_path):
    model = TinyModel()
    output_dir = tmp_path / "run" / "train"
    logger = _ConsoleLogger({"console_log": {"enabled": False}})

    report = _apply_trainable_filter(
        model,
        {"trainable": {"include": ["track_center_candidate_head.*"]}},
        output_dir=output_dir,
        console_logger=logger,
    )

    assert report is not None
    assert all(param.requires_grad for _, param in model.track_center_candidate_head.named_parameters())
    assert not any(param.requires_grad for _, param in model.backbone.named_parameters())
    assert not any(param.requires_grad for _, param in model.track_head.named_parameters())
    report_path = tmp_path / "run" / "hypers" / "trainable_filter.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["trainable_names"] == [
        "track_center_candidate_head.weight",
        "track_center_candidate_head.bias",
    ]
    assert saved["frozen_tensor_count"] == 4


def test_trainable_filter_rejects_empty_match(tmp_path):
    model = TinyModel()
    logger = _ConsoleLogger({"console_log": {"enabled": False}})

    with pytest.raises(ValueError, match="left no trainable parameters"):
        _apply_trainable_filter(
            model,
            {"trainable": {"include": ["missing_head.*"]}},
            output_dir=tmp_path / "train",
            console_logger=logger,
        )
