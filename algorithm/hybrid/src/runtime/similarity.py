from __future__ import annotations

import torch


def ellipse_similarity(prev_state: torch.Tensor, cur_state: torch.Tensor) -> torch.Tensor:
    prev_center = prev_state[..., 0:2]
    cur_center = cur_state[..., 0:2]
    prev_axes = torch.clamp(prev_state[..., 2:4], min=1e-3)
    cur_axes = torch.clamp(cur_state[..., 2:4], min=1e-3)

    center_dist = torch.linalg.norm(prev_center - cur_center, dim=-1)
    center_scale = torch.clamp(prev_axes.mean(dim=-1) + cur_axes.mean(dim=-1), min=1.0)
    axes_ratio = torch.abs(torch.log(cur_axes / prev_axes)).mean(dim=-1)
    angle_delta = 1.0 - torch.sum(prev_state[..., 4:6] * cur_state[..., 4:6], dim=-1).clamp(-1.0, 1.0)

    score = 1.0 - torch.clamp(0.5 * (center_dist / center_scale) + 0.35 * axes_ratio + 0.15 * angle_delta, min=0.0, max=1.0)
    return score.clamp(0.0, 1.0)
