from __future__ import annotations

from typing import Dict

import torch

from src.loss.bundles import center_soft_threshold_loss
from src.loss.bundles import pupil_center_candidate_losses as _pupil_center_candidate_losses
from src.loss.bundles import resolve_search_state as _resolve_search_state
from src.loss.bundles import sum_loss_logs as _sum_loss_logs
from src.loss.stage_common import (
    compute_aux_loss,
    compute_constraint_center_log,
    compute_eye_logs,
    compute_mask_losses,
    compute_pupil_branch_log_group,
    resolve_sample_masks,
)


def _search_soft_threshold_logs(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    search_state: torch.Tensor | None,
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    zero = weights.sum() * 0.0
    if search_state is None:
        return {
            "loss_search_p10_soft_threshold": zero,
            "loss_search_p5_soft_threshold": zero,
        }
    p10_weight = float(loss_cfg.get("search_p10_soft_threshold_weight", 0.0))
    p5_weight = float(loss_cfg.get("search_p5_soft_threshold_weight", 0.0))
    p10_margin = float(loss_cfg.get("search_p10_soft_threshold_margin_px", 10.0))
    p10_temperature = float(loss_cfg.get("search_p10_soft_threshold_temperature_px", 1.0))
    p5_margin = float(loss_cfg.get("search_p5_soft_threshold_margin_px", 5.0))
    p5_temperature = float(loss_cfg.get("search_p5_soft_threshold_temperature_px", p10_temperature))
    return {
        "loss_search_p10_soft_threshold": center_soft_threshold_loss(
            search_state,
            batch["cur_state"],
            weights,
            margin_px=p10_margin,
            temperature_px=p10_temperature,
        )
        * p10_weight,
        "loss_search_p5_soft_threshold": center_soft_threshold_loss(
            search_state,
            batch["cur_state"],
            weights,
            margin_px=p5_margin,
            temperature_px=p5_temperature,
        )
        * p5_weight,
    }


def compute_stage1_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    active_head: str = "all",
) -> Dict[str, torch.Tensor]:
    quality, geom, _ = resolve_sample_masks(batch)
    weights = quality * geom
    search_state = _resolve_search_state(outputs) if ("search/state" in outputs or "search/pupil" in outputs) else None
    search_logs = compute_pupil_branch_log_group(
        batch,
        outputs,
        loss_cfg,
        branch="search",
        state=search_state,
        weights=weights,
    )
    eye_logs = compute_eye_logs(batch, outputs, loss_cfg, quality=quality, active_head=active_head)
    mask_loss, mask_coarse_loss = compute_mask_losses(batch, outputs, loss_cfg, weights=weights, active_head=active_head)
    aux_loss = compute_aux_loss(batch, outputs, loss_cfg, quality=quality, aux_key="search/aux")
    search_candidate_losses = _pupil_center_candidate_losses(
        candidate_xy=outputs["search/center_candidate_xy"],
        candidate_logits=outputs["search/center_candidate_logits"],
        candidate_delta=outputs["search/center_candidate_delta"],
        target_state=batch["cur_state"],
        weights=weights,
        loss_cfg=loss_cfg,
    ) if (
        "search/center_candidate_xy" in outputs
        and "search/center_candidate_logits" in outputs
        and "search/center_candidate_delta" in outputs
    ) else {}
    constraint = compute_constraint_center_log(
        batch,
        outputs,
        loss_cfg,
        primary_state=search_state,
        primary_weights=weights,
    )
    logs = {
        **eye_logs,
        "loss_mask": mask_loss,
        "loss_mask_coarse": mask_coarse_loss,
        "loss_constraint_center": constraint,
        "loss_aux": aux_loss,
        **_search_soft_threshold_logs(batch, outputs, loss_cfg, search_state=search_state, weights=weights),
        **{f"loss_{name}": value for name, value in search_candidate_losses.items()},
        **search_logs,
    }
    logs["loss_total"] = _sum_loss_logs(logs, batch, outputs)
    return logs


__all__ = ["compute_stage1_losses"]
