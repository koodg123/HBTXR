from __future__ import annotations

from typing import Dict

import torch

from common.losses.bundles import resolve_event_state as _resolve_event_state
from common.losses.bundles import resolve_search_state as _resolve_search_state
from common.losses.bundles import resolve_track_state as _resolve_track_state
from common.losses.bundles import sum_loss_logs as _sum_loss_logs
from common.losses.bundles import track_branch_losses as _track_branch_losses
from common.losses.bundles import track_center_candidate_losses as _track_center_candidate_losses
from common.losses.bundles import track_state_aux_losses as _track_state_aux_losses
from common.losses.bundles import track_center_heatmap_losses as _track_center_heatmap_losses
from common.losses.bundles import track_center_refine_losses as _track_center_refine_losses
from common.losses.bundles import track_state_simdr_losses as _track_state_simdr_losses
from common.losses.bundles import track_target_override_losses as _track_target_override_losses
from common.losses.bundles import zero_loss as _zero_loss
from common.losses.common import ellipse_gwd_loss
from common.losses.stage_common import (
    build_loss_sample_weights,
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
    track_sample_weights = build_loss_sample_weights(batch, loss_cfg)
    track_quality = quality if track_sample_weights is None else quality * track_sample_weights
    track_weights = track_quality * track_geom
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
        quality=track_quality,
        track_geom=track_geom,
        valid_track=batch["valid_track"],
        similarity_target=batch.get("similarity_target"),
        loss_cfg=loss_cfg,
    ) if ("track/pupil" in outputs and track_state is not None) else {}
    track_aux_losses = _track_state_aux_losses(
        state_aux=outputs["track/state_aux"],
        target_state=batch["cur_state"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
    ) if "track/state_aux" in outputs else {}
    track_simdr_losses = _track_state_simdr_losses(
        logits=outputs["track/state_simdr"],
        target_state=batch["cur_state"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
    ) if "track/state_simdr" in outputs else {}
    track_heatmap_losses = _track_center_heatmap_losses(
        logits=outputs["track/center_heatmap_logits"],
        offset=outputs["track/center_heatmap_offset"],
        target_state=batch["cur_state"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
    ) if "track/center_heatmap_logits" in outputs and "track/center_heatmap_offset" in outputs else {}
    track_refine_losses = _track_center_refine_losses(
        delta=outputs["track/center_refine_delta"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
    ) if "track/center_refine_delta" in outputs else {}
    track_candidate_losses = _track_center_candidate_losses(
        candidate_xy=outputs["track/center_candidate_xy"],
        candidate_logits=outputs["track/center_candidate_logits"],
        candidate_delta=outputs["track/center_candidate_delta"],
        target_state=batch["cur_state"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
    ) if (
        "track/center_candidate_xy" in outputs
        and "track/center_candidate_logits" in outputs
        and "track/center_candidate_delta" in outputs
    ) else {}
    track_override_losses = _track_target_override_losses(
        state=track_state,
        override_state=batch["track_target_override_state"],
        override_weight=batch["track_target_override_weight"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
        prefix="track_target_override",
    ) if (
        track_state is not None
        and "track_target_override_state" in batch
        and "track_target_override_weight" in batch
    ) else {}
    track_aux_override_losses = _track_target_override_losses(
        state=outputs["track/state_aux"],
        override_state=batch["track_target_override_state"],
        override_weight=batch["track_target_override_weight"],
        quality=track_quality,
        track_geom=track_geom,
        loss_cfg=loss_cfg,
        prefix="track_state_aux_target_override",
    ) if (
        "track/state_aux" in outputs
        and "track_target_override_state" in batch
        and "track_target_override_weight" in batch
    ) else {}
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
        **{f"loss_{name}": value for name, value in track_aux_losses.items()},
        **{f"loss_{name}": value for name, value in track_simdr_losses.items()},
        **{f"loss_{name}": value for name, value in track_heatmap_losses.items()},
        **{f"loss_{name}": value for name, value in track_refine_losses.items()},
        **{f"loss_{name}": value for name, value in track_candidate_losses.items()},
        **{f"loss_{name}": value for name, value in track_override_losses.items()},
        **{f"loss_{name}": value for name, value in track_aux_override_losses.items()},
    }
    logs["loss_total"] = _sum_loss_logs(logs, batch, outputs)
    return logs


__all__ = ["compute_stage2_losses"]
