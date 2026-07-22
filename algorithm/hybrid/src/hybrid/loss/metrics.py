from __future__ import annotations

from typing import Dict

import torch

from hybrid.loss.bundles import (
    hit_rate_percent as _hit_rate_percent,
    resolve_event_state as _resolve_event_state,
    resolve_search_state as _resolve_search_state,
    resolve_track_state as _resolve_track_state,
)
from hybrid.loss.stage_common import resolve_sample_masks


def compute_metrics(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    quality, geom, track_geom = resolve_sample_masks(batch)
    logs: Dict[str, torch.Tensor] = {}
    if "search/pupil" in outputs or "search/state" in outputs:
        search_state = _resolve_search_state(outputs)
        search_error = torch.linalg.norm(search_state[:, :2] - batch["cur_state"][:, :2], dim=-1)
        logs.update(
            {
                "metric_search_center_px": (search_error * (quality * geom)).sum() / (quality * geom).sum().clamp_min(1e-6),
                "metric_search_p10_pct": _hit_rate_percent(search_error, 10.0, quality * geom),
                "metric_search_p5_pct": _hit_rate_percent(search_error, 5.0, quality * geom),
                "metric_search_p1_pct": _hit_rate_percent(search_error, 1.0, quality * geom),
            }
        )
    if "event/pupil" in outputs or "event/state" in outputs:
        event_state = _resolve_event_state(outputs)
        event_error = torch.linalg.norm(event_state[:, :2] - batch["cur_state"][:, :2], dim=-1)
        logs.update(
            {
                "metric_event_center_px": (event_error * (quality * geom)).sum() / (quality * geom).sum().clamp_min(1e-6),
                "metric_event_p10_pct": _hit_rate_percent(event_error, 10.0, quality * geom),
            }
        )
    if "track/pupil" in outputs or "track/state" in outputs:
        track_state = _resolve_track_state(batch, outputs)
        track_error = torch.linalg.norm(track_state[:, :2] - batch["cur_state"][:, :2], dim=-1)
        logs.update(
            {
                "metric_track_center_px": (track_error * (quality * track_geom)).sum() / (quality * track_geom).sum().clamp_min(1e-6),
                "metric_track_p10_pct": _hit_rate_percent(track_error, 10.0, quality * track_geom),
                "metric_track_p5_pct": _hit_rate_percent(track_error, 5.0, quality * track_geom),
                "metric_track_p1_pct": _hit_rate_percent(track_error, 1.0, quality * track_geom),
            }
        )
        if "track/pupil" in outputs:
            logs["metric_track_quality_mean"] = torch.sigmoid(outputs["track/pupil"][:, 7]).mean()
    return logs


__all__ = ["compute_metrics"]
