from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EnsembleTeacherMember:
    label: str
    model: nn.Module
    weight: float


class WeightedOutputEnsembleTeacher(nn.Module):
    def __init__(self, members: list[EnsembleTeacherMember]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble teacher requires at least one member")
        total_weight = sum(float(member.weight) for member in members)
        if total_weight <= 0.0:
            raise ValueError("ensemble teacher weights must sum to a positive value")
        self.labels = [member.label for member in members]
        self.weights = [float(member.weight) / total_weight for member in members]
        self.models = nn.ModuleList([member.model for member in members])
        self.is_fixed_ensemble_teacher = True
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        outputs = [model(*args, **kwargs) for model in self.models]
        if not outputs:
            return {}
        common_keys = set(outputs[0])
        for output in outputs[1:]:
            common_keys &= set(output)

        ensembled: dict[str, torch.Tensor] = {}
        for key in sorted(common_keys):
            tensors = [output[key] for output in outputs]
            first = tensors[0]
            if (
                all(isinstance(tensor, torch.Tensor) for tensor in tensors)
                and torch.is_floating_point(first)
                and all(tuple(tensor.shape) == tuple(first.shape) for tensor in tensors[1:])
            ):
                acc = torch.zeros_like(first)
                for tensor, weight in zip(tensors, self.weights):
                    acc = acc + tensor * float(weight)
                ensembled[key] = acc
            elif isinstance(first, torch.Tensor):
                ensembled[key] = first
        return ensembled


__all__ = ["EnsembleTeacherMember", "WeightedOutputEnsembleTeacher"]
