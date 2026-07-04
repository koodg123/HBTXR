from __future__ import annotations

from typing import Dict

import torch

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
        **search_logs,
    }
    logs["loss_total"] = _sum_loss_logs(logs, batch, outputs)
    return logs


__all__ = ["compute_stage1_losses"]
