from __future__ import annotations

import torch
from torch import nn

from hbtxr.training.ensemble_teacher import EnsembleTeacherMember, WeightedOutputEnsembleTeacher


class _ToyTeacher(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, *_args, **_kwargs):
        return {
            "search/state": torch.full((2, 2), self.value),
            "search/pooled": torch.full((2, 3), self.value + 1.0),
        }


def test_weighted_output_ensemble_teacher_averages_common_float_outputs() -> None:
    teacher = WeightedOutputEnsembleTeacher(
        [
            EnsembleTeacherMember("a", _ToyTeacher(2.0), 0.75),
            EnsembleTeacherMember("b", _ToyTeacher(6.0), 0.25),
        ]
    )

    outputs = teacher({})

    assert torch.allclose(outputs["search/state"], torch.full((2, 2), 3.0))
    assert torch.allclose(outputs["search/pooled"], torch.full((2, 3), 4.0))
