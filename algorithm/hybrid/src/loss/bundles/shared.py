from __future__ import annotations

from typing import Dict

import torch

from src.loss.common import decode_track_state, normalize_uv, state6_to_xywht


def loss_device(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.device:
    for value in outputs.values():
        if torch.is_tensor(value):
            return value.device
    for value in batch.values():
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def zero_loss(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.zeros((), device=loss_device(batch, outputs))


def resolve_search_state(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "search/state" in outputs:
        return outputs["search/state"]
    pred = outputs["search/pupil"]
    return torch.cat([pred[:, :4], normalize_uv(pred[:, 4:6])], dim=-1)


def resolve_event_state(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "event/state" in outputs:
        return outputs["event/state"]
    pred = outputs["event/pupil"]
    return torch.cat([pred[:, :4], normalize_uv(pred[:, 4:6])], dim=-1)


def resolve_track_state(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "track/state" in outputs:
        return outputs["track/state"]
    return decode_track_state(batch["prev_state"], outputs["track/pupil"])


def pupil_bbox_targets(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "pupil_region_target" in batch:
        return batch["pupil_region_target"][:, :4]
    return state6_to_xywht(batch["cur_state"])[:, :4]


def pupil_obb_targets(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return state6_to_xywht(batch["cur_state"])


def hit_rate_percent(errors: torch.Tensor, threshold_px: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    hit = (errors <= threshold_px).float()
    if weights is None:
        return hit.mean() * 100.0
    w = weights.to(device=hit.device, dtype=hit.dtype).view(-1)
    return (hit * w).sum() / w.sum().clamp_min(1e-6) * 100.0


def sum_loss_logs(logs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(tuple(logs.values())).sum() if logs else zero_loss(batch, outputs)
