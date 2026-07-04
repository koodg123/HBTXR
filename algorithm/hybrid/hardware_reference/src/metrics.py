from __future__ import annotations

import torch


def center_error(pred_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred_state[..., :2] - target_state[..., :2], dim=-1)


def hit_rate(errors: torch.Tensor, threshold_px: float) -> torch.Tensor:
    return (errors <= float(threshold_px)).float().mean() * 100.0


def compute_metrics(batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = batch["target_state"]
    search_err = center_error(outputs["search/state"], target)
    track_err = center_error(outputs["track/state"], target)
    return {
        "metric/search_center_error": search_err.mean(),
        "metric/track_center_error": track_err.mean(),
        "metric/search_p10": hit_rate(search_err, 10.0),
        "metric/track_p10": hit_rate(track_err, 10.0),
    }

