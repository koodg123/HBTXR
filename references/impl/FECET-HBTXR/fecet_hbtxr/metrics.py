from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def sample_quality(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["annotation_quality"].view(-1).to(dtype=torch.float32)


def geom_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    mask_valid = batch["mask_valid"].view(-1).to(dtype=torch.float32)
    closed = batch["closed_eye_flag"].view(-1).to(dtype=torch.float32)
    return mask_valid * (1.0 - closed)


def track_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return geom_mask(batch) * batch["valid_track"].view(-1).to(dtype=torch.float32)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if values.ndim > 1:
        values = values.mean(dim=tuple(range(1, values.ndim)))
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype).view(-1)
    denom = torch.clamp(weights.sum(), min=1e-6)
    return (values * weights).sum() / denom


def resolve_search_state(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "search/state" in outputs:
        return outputs["search/state"]
    pred = outputs["search/pupil"]
    return torch.cat([pred[:, :4], F.normalize(pred[:, 4:6], dim=-1, eps=1e-6)], dim=-1)


def resolve_event_state(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "event/state" in outputs:
        return outputs["event/state"]
    pred = outputs["event/pupil"]
    return torch.cat([pred[:, :4], F.normalize(pred[:, 4:6], dim=-1, eps=1e-6)], dim=-1)


def resolve_track_state(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "track/state" in outputs:
        return outputs["track/state"]
    prev_state = batch["prev_state"]
    pred = outputs["track/pupil"]
    cur = prev_state.clone()
    cur[:, 0:2] = prev_state[:, 0:2] + pred[:, 0:2]
    cur[:, 2:4] = prev_state[:, 2:4] * torch.exp(pred[:, 2:4])
    cur[:, 4:6] = F.normalize(prev_state[:, 4:6] + pred[:, 4:6], dim=-1, eps=1e-6)
    return cur


def hit_rate_percent(errors: torch.Tensor, threshold_px: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    hits = (errors <= threshold_px).to(dtype=torch.float32)
    return weighted_mean(hits, weights) * 100.0


def compute_metrics(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    quality = sample_quality(batch)
    geom = geom_mask(batch)
    track_geom = track_mask(batch)
    metrics: Dict[str, torch.Tensor] = {}

    if "search/pupil" in outputs:
        search_state = resolve_search_state(outputs)
        search_error = torch.linalg.norm(search_state[:, 0:2] - batch["cur_state"][:, 0:2], dim=-1)
        weights = quality * geom
        metrics["metric_search_center_px"] = weighted_mean(search_error, weights)
        metrics["metric_search_p10_pct"] = hit_rate_percent(search_error, 10.0, weights)
        metrics["metric_search_p5_pct"] = hit_rate_percent(search_error, 5.0, weights)

    if "event/pupil" in outputs:
        event_state = resolve_event_state(outputs)
        event_error = torch.linalg.norm(event_state[:, 0:2] - batch["cur_state"][:, 0:2], dim=-1)
        weights = quality * geom
        metrics["metric_event_center_px"] = weighted_mean(event_error, weights)
        metrics["metric_event_p10_pct"] = hit_rate_percent(event_error, 10.0, weights)

    if "track/pupil" in outputs:
        track_state = resolve_track_state(batch, outputs)
        track_error = torch.linalg.norm(track_state[:, 0:2] - batch["cur_state"][:, 0:2], dim=-1)
        weights = quality * track_geom
        metrics["metric_track_center_px"] = weighted_mean(track_error, weights)
        metrics["metric_track_p10_pct"] = hit_rate_percent(track_error, 10.0, weights)
        metrics["metric_track_p5_pct"] = hit_rate_percent(track_error, 5.0, weights)
        metrics["metric_track_quality_mean"] = torch.sigmoid(outputs["track/pupil"][:, 7]).mean()

    return metrics
