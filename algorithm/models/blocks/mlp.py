"""Position-wise feed-forward (MLP) sublayer of the ViT TRB: FC1 -> GELU -> FC2.

Maps to the RMU (FC1/FC2) in the HBTXR accelerator. Hidden width defaults to
``mlp_ratio * dim`` at the call site.
"""
from __future__ import annotations

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        *,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x
