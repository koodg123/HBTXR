from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def decode_simdr_xy(logits: torch.Tensor, *, coordinate_max: float = 255.0) -> torch.Tensor:
    bins = max(2, int(logits.shape[-1]))
    coord_max = max(float(coordinate_max), 1.0e-6)
    grid = torch.arange(bins, device=logits.device, dtype=logits.dtype) * (coord_max / float(bins - 1))
    return (logits.softmax(dim=-1) * grid.view(1, 1, bins)).sum(dim=-1)


def decode_heatmap_xy(
    logits: torch.Tensor,
    offset: torch.Tensor | None = None,
    *,
    coordinate_max: float = 255.0,
) -> torch.Tensor:
    batch, height, width = logits.shape
    coord_max = max(float(coordinate_max), 1.0e-6)
    probs = logits.flatten(1).softmax(dim=-1)
    ys = torch.arange(height, device=logits.device, dtype=logits.dtype)
    xs = torch.arange(width, device=logits.device, dtype=logits.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    if offset is not None:
        off_x = offset[:, 0].flatten(1)
        off_y = offset[:, 1].flatten(1)
        x = (probs * (grid_x.flatten().view(1, -1) + 0.5 + off_x)).sum(dim=-1)
        y = (probs * (grid_y.flatten().view(1, -1) + 0.5 + off_y)).sum(dim=-1)
    else:
        x = (probs * (grid_x.flatten().view(1, -1) + 0.5)).sum(dim=-1)
        y = (probs * (grid_y.flatten().view(1, -1) + 0.5)).sum(dim=-1)
    return torch.stack([x * coord_max / float(width), y * coord_max / float(height)], dim=-1)


class TrackStateBranch:
    def __init__(
        self,
        *,
        prev_state_encoder: nn.Module,
        track_head: nn.Module | None,
        track_state_aux_head: nn.Module | None = None,
        track_state_simdr_head: nn.Module | None = None,
        track_center_heatmap_head: nn.Module | None = None,
        track_center_refine_head: nn.Module | None = None,
        track_center_candidate_head: nn.Module | None = None,
        apply_width_mask: Callable[[torch.Tensor, int], torch.Tensor],
        apply_track_mask: Callable[[torch.Tensor, int], torch.Tensor],
        decode_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        simdr_as_track_state: bool = False,
        simdr_coordinate_max: float = 255.0,
        simdr_blend: float = 1.0,
        heatmap_as_track_state: bool = False,
        heatmap_coordinate_max: float = 255.0,
        heatmap_blend: float = 1.0,
        refine_as_track_state: bool = False,
        refine_blend: float = 1.0,
        candidate_as_track_state: bool = False,
        candidate_blend: float = 1.0,
    ) -> None:
        self.prev_state_encoder = prev_state_encoder
        self.track_head = track_head
        self.track_state_aux_head = track_state_aux_head
        self.track_state_simdr_head = track_state_simdr_head
        self.track_center_heatmap_head = track_center_heatmap_head
        self.track_center_refine_head = track_center_refine_head
        self.track_center_candidate_head = track_center_candidate_head
        self.apply_width_mask = apply_width_mask
        self.apply_track_mask = apply_track_mask
        self.decode_state = decode_state
        self.simdr_as_track_state = bool(simdr_as_track_state)
        self.simdr_coordinate_max = float(simdr_coordinate_max)
        self.simdr_blend = min(1.0, max(0.0, float(simdr_blend)))
        self.heatmap_as_track_state = bool(heatmap_as_track_state)
        self.heatmap_coordinate_max = float(heatmap_coordinate_max)
        self.heatmap_blend = min(1.0, max(0.0, float(heatmap_blend)))
        self.refine_as_track_state = bool(refine_as_track_state)
        self.refine_blend = min(1.0, max(0.0, float(refine_blend)))
        self.candidate_as_track_state = bool(candidate_as_track_state)
        self.candidate_blend = min(1.0, max(0.0, float(candidate_blend)))

    @property
    def enabled(self) -> bool:
        return self.track_head is not None

    def forward(
        self,
        *,
        pooled: torch.Tensor,
        prev_state: torch.Tensor,
        active_dim: int,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        prev_feat = self.prev_state_encoder(prev_state)
        prev_feat = self.apply_width_mask(prev_feat, active_dim)
        fused = torch.cat([pooled, prev_feat], dim=-1)
        fused = self.apply_track_mask(fused, active_dim)
        outputs = {
            "track/fused": fused,
            "track/event_pooled": pooled,
            "track/prev_feat": prev_feat,
            "pruning/active_dim": torch.full((batch_size,), float(active_dim), device=device),
        }
        base_state = None
        current_state = None
        if self.track_head is not None:
            track_logits = self.track_head(fused)
            base_state = self.decode_state(prev_state, track_logits)
            current_state = base_state
            outputs["track/pupil"] = track_logits
            outputs["track/state"] = current_state
        if self.track_state_aux_head is not None:
            outputs["track/state_aux"] = self.track_state_aux_head(fused)
        if self.track_state_simdr_head is not None:
            simdr_logits = self.track_state_simdr_head(fused)
            outputs["track/state_simdr"] = simdr_logits
            outputs["track/state_simdr_xy"] = decode_simdr_xy(simdr_logits, coordinate_max=self.simdr_coordinate_max)
            if self.simdr_as_track_state and base_state is not None:
                source_state = current_state if current_state is not None else base_state
                refined_state = source_state.clone()
                simdr_xy = outputs["track/state_simdr_xy"]
                refined_state[:, :2] = (1.0 - self.simdr_blend) * source_state[:, :2] + self.simdr_blend * simdr_xy
                outputs["track/state_base"] = base_state
                outputs["track/state"] = refined_state
                current_state = refined_state
        if self.track_center_heatmap_head is not None:
            heatmap = self.track_center_heatmap_head(fused)
            outputs["track/center_heatmap_logits"] = heatmap["logits"]
            outputs["track/center_heatmap_offset"] = heatmap["offset"]
            outputs["track/center_heatmap_xy"] = decode_heatmap_xy(
                heatmap["logits"],
                heatmap["offset"],
                coordinate_max=self.heatmap_coordinate_max,
            )
            if self.heatmap_as_track_state and base_state is not None:
                source_state = current_state if current_state is not None else base_state
                refined_state = source_state.clone()
                heatmap_xy = outputs["track/center_heatmap_xy"]
                refined_state[:, :2] = (1.0 - self.heatmap_blend) * source_state[:, :2] + self.heatmap_blend * heatmap_xy
                outputs.setdefault("track/state_base", base_state)
                outputs["track/state"] = refined_state
                current_state = refined_state
        if self.track_center_refine_head is not None:
            refine = self.track_center_refine_head(fused)
            outputs["track/center_refine_delta"] = refine["delta"]
            outputs["track/center_refine_gate_logit"] = refine["gate_logit"]
            outputs["track/center_refine_gate"] = refine["gate"]
            if self.refine_as_track_state and current_state is not None:
                refined_state = current_state.clone()
                refined_state[:, :2] = current_state[:, :2] + self.refine_blend * refine["delta"]
                outputs.setdefault("track/state_base", base_state if base_state is not None else current_state)
                outputs["track/state_pre_refine"] = current_state
                outputs["track/state"] = refined_state
                current_state = refined_state
        if self.track_center_candidate_head is not None:
            candidate = self.track_center_candidate_head(fused)
            outputs["track/center_candidate_delta"] = candidate["delta"]
            outputs["track/center_candidate_logits"] = candidate["logits"]
            if current_state is not None:
                candidate_xy = current_state[:, None, :2] + candidate["delta"]
                outputs["track/center_candidate_xy"] = candidate_xy
                top_idx = candidate["logits"].argmax(dim=-1)
                gather_idx = top_idx.view(-1, 1, 1).expand(-1, 1, 2)
                selected_xy = candidate_xy.gather(1, gather_idx).squeeze(1)
                outputs["track/center_candidate_selected_xy"] = selected_xy
                if self.candidate_as_track_state:
                    source_state = current_state
                    refined_state = source_state.clone()
                    refined_state[:, :2] = (1.0 - self.candidate_blend) * source_state[:, :2] + self.candidate_blend * selected_xy
                    outputs.setdefault("track/state_base", base_state if base_state is not None else source_state)
                    outputs["track/state_pre_candidate"] = source_state
                    outputs["track/state"] = refined_state
                    current_state = refined_state
        return outputs
