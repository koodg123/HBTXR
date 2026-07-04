from __future__ import annotations

from typing import Dict

import torch

from src.loss.bundles import resolve_event_state as _resolve_event_state
from src.loss.bundles import resolve_search_state as _resolve_search_state
from src.loss.bundles import resolve_track_state as _resolve_track_state
from src.loss.bundles import sum_loss_logs as _sum_loss_logs
from src.loss.bundles import track_branch_losses as _track_branch_losses
from src.loss.bundles import zero_loss as _zero_loss
from src.loss.common import ellipse_gwd_loss
from src.loss.stage_common import (
    compute_aux_loss,
    compute_constraint_center_log,
    compute_eye_logs,
    compute_mask_losses,
    compute_pupil_branch_log_group,
    resolve_sample_masks,
)


def compute_stage2_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    active_head: str = "all",
) -> Dict[str, torch.Tensor]:
    quality, geom, track_geom = resolve_sample_masks(batch)
    geom_weights = quality * geom
    track_weights = quality * track_geom
    search_state = _resolve_search_state(outputs) if ("search/state" in outputs or "search/pupil" in outputs) else None
    event_state = _resolve_event_state(outputs) if ("event/state" in outputs or "event/pupil" in outputs) else None
    track_state = _resolve_track_state(batch, outputs) if ("track/state" in outputs or "track/pupil" in outputs) else None
    search_logs = compute_pupil_branch_log_group(
        batch,
        outputs,
        loss_cfg,
        branch="search",
        state=search_state,
        weights=geom_weights,
    )
    event_logs = compute_pupil_branch_log_group(
        batch,
        outputs,
        loss_cfg,
        branch="event",
        state=event_state,
        weights=geom_weights,
    )
    track_losses = _track_branch_losses(
        pred=outputs["track/pupil"],
        target=batch["pupil_track_target"],
        state=track_state,
        target_state=batch["cur_state"],
        quality=quality,
        track_geom=track_geom,
        valid_track=batch["valid_track"],
        loss_cfg=loss_cfg,
    ) if ("track/pupil" in outputs and track_state is not None) else {}
    eye_logs = compute_eye_logs(batch, outputs, loss_cfg, quality=quality, active_head=active_head)
    mask_loss, mask_coarse_loss = compute_mask_losses(batch, outputs, loss_cfg, weights=geom_weights, active_head=active_head)
    consistency = (
        ellipse_gwd_loss(search_state, track_state, track_weights) * float(loss_cfg.get("consistency_weight", 0.0))
        if (search_state is not None and track_state is not None and float(loss_cfg.get("consistency_weight", 0.0)) > 0.0)
        else _zero_loss(batch, outputs)
    )
    constraint = compute_constraint_center_log(
        batch,
        outputs,
        loss_cfg,
        primary_state=track_state,
        primary_weights=track_weights,
        fallback_state=search_state,
        fallback_weights=geom_weights,
    )
    aux_key = "event/aux" if "event/aux" in outputs else "search/aux" if "search/aux" in outputs else None
    aux_loss = compute_aux_loss(batch, outputs, loss_cfg, quality=quality, aux_key=aux_key)
    logs = {
        **eye_logs,
        "loss_mask": mask_loss,
        "loss_mask_coarse": mask_coarse_loss,
        "loss_constraint_center": constraint,
        "loss_consistency": consistency,
        "loss_aux": aux_loss,
        **search_logs,
        **event_logs,
        **{f"loss_{name}": value for name, value in track_losses.items()},
    }
    logs["loss_total"] = _sum_loss_logs(logs, batch, outputs)
    return logs


__all__ = ["compute_stage2_losses"]
