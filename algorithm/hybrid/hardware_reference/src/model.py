from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .heads import MaskHead, SearchHead, TrackHead, mlp_head, normalize_state
from .modules.fusion import StateFusion
from .modules.hgpipe_backbone import HGPipeBackbone
from .modules.patch_embed import EventPatchEmbedding, FramePatchEmbedding
from .scheduler import TrackSearchSchedulerFSM


class ModalityAdapter(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class HGTXRTracker(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        depth: int = 6,
        event_cut_depth: int | None = None,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        input_size: tuple[int, int] = (256, 256),
        dropout: float = 0.0,
        aux_classes: int = 5,
        runtime_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = tuple(int(v) for v in input_size)
        self.depth = int(depth)
        self.event_cut_depth = int(event_cut_depth if event_cut_depth is not None else max(1, depth // 2))
        self.frame_embed = FramePatchEmbedding(embed_dim, patch_size)
        self.event_embed = EventPatchEmbedding(embed_dim, patch_size)
        self.frame_adapter = ModalityAdapter(embed_dim)
        self.event_adapter = ModalityAdapter(embed_dim)
        self.backbone = HGPipeBackbone(embed_dim, depth, num_heads, mlp_ratio, dropout)
        self.eye_head = mlp_head(embed_dim, 5)
        self.search_head = SearchHead(embed_dim)
        self.event_head = SearchHead(embed_dim)
        self.track_head = TrackHead(embed_dim * 2)
        self.mask_head = MaskHead(embed_dim, self.input_size)
        self.aux_head = mlp_head(embed_dim, aux_classes)
        self.fusion = StateFusion(embed_dim)
        self.scheduler = TrackSearchSchedulerFSM(**(runtime_cfg or {}))

    def encode_frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, grid = self.frame_embed(frame)
        return (*self.backbone(self.frame_adapter(tokens)), grid)

    def encode_event(self, event: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, grid = self.event_embed(event)
        return (*self.backbone(self.event_adapter(tokens), depth_limit=self.event_cut_depth), grid)

    def decode_track_state(self, prev_state: torch.Tensor, track_logits: torch.Tensor) -> torch.Tensor:
        delta = track_logits[..., :6]
        # Paper contract: event branch predicts a residual in runtime pupil-state
        # space relative to the scheduler-maintained decoded anchor state.
        raw = prev_state + delta
        raw = torch.cat([raw[..., :4], F.normalize(raw[..., 4:6], dim=-1, eps=1e-6)], dim=-1)
        return raw

    def forward_train(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frame_tokens, frame_pooled, frame_grid = self.encode_frame(batch["frame"])
        _event_tokens, event_pooled, _event_grid = self.encode_event(batch["event"])
        prev_state = batch.get("prev_state")
        if prev_state is None:
            prev_state = torch.zeros(frame_pooled.shape[0], 6, device=frame_pooled.device)
        search_logits = self.search_head(frame_pooled)
        event_logits = self.event_head(event_pooled)
        fused = self.fusion(frame_pooled, event_pooled, prev_state)
        track_logits = self.track_head(fused)
        return {
            "eye/logits": self.eye_head(frame_pooled),
            "search/pupil": search_logits,
            "search/state": normalize_state(search_logits),
            "event/pupil": event_logits,
            "event/residual": normalize_state(event_logits),
            "event/state": prev_state + normalize_state(event_logits),
            "track/pupil": track_logits,
            "track/residual": track_logits[..., :6],
            "track/state": self.decode_track_state(prev_state, track_logits),
            "mask/logits": self.mask_head(frame_tokens, frame_grid),
            "aux/logits": self.aux_head(frame_pooled),
        }

    def forward(self, frame: torch.Tensor, event: torch.Tensor, prev_state: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch = {"frame": frame, "event": event}
        if prev_state is not None:
            batch["prev_state"] = prev_state
        return self.forward_train(batch)


def build_model(cfg: dict[str, Any]) -> HGTXRTracker:
    model_cfg = dict(cfg.get("model", {}))
    model_cfg["runtime_cfg"] = cfg.get("runtime", {})
    if "input_size" in model_cfg:
        model_cfg["input_size"] = tuple(model_cfg["input_size"])
    return HGTXRTracker(**model_cfg)

