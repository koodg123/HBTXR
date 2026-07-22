"""Frame-only HBTXR model (search path, standalone direct detector).

Frame ``[B,1,H,W]`` -> Conv-F patch embed -> shared ViT (full depth) -> head
(default PupilBoxHead) -> oriented box -> g() -> pupil ellipse state. Direct
detection: no anchor is needed, so the model is independently trainable exactly
like a plain detector (box + optional aux mask loss). The primary head is chosen
by config (``build_head``) so it is swappable (point 1/8).

``DirectPupilDetector`` is the shared assembly reused by the event-only model
(models.event) with the event stem — the two modality baselines are symmetric,
differing only in the input stem, so there is no duplicated forward logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from models.backbones import ViTBackbone, ViTConfig, build_patch_embed
from models.geometry import box_to_state
from models.heads import ReliabilityHead, build_head


@dataclass
class DirectDetectorConfig:
    modality: str = "frame"          # "frame" (Conv-F) | "event" (Conv-E)
    embed_dim: int = 192
    patch_size: int = 16
    backbone: ViTConfig = field(default_factory=ViTConfig)
    head: str = "bbox"               # config-swappable primary head
    head_hidden_dim: int | None = None
    with_mask: bool = True
    with_roi: bool = True
    with_reliability: bool = True


class DirectPupilDetector(nn.Module):
    """Stem -> full-depth ViT -> primary head (+ aux) -> ellipse state.

    Shared by the frame-only and event-only baselines; ``config.modality`` picks
    Conv-F or Conv-E. Uses the full backbone depth (the early-exit cut point is a
    hybrid track-path optimization, not used by the standalone detectors).
    """

    def __init__(self, config: DirectDetectorConfig | None = None) -> None:
        super().__init__()
        cfg = config or DirectDetectorConfig()
        self.config = cfg
        dim = cfg.embed_dim
        self.patch_embed = build_patch_embed(cfg.modality, embed_dim=dim, patch_size=cfg.patch_size)
        # model.embed_dim is the single source of truth: keep the backbone width in
        # sync so setting embed_dim (without also setting backbone.embed_dim) is valid.
        backbone_cfg = cfg.backbone if cfg.backbone.embed_dim == dim else replace(cfg.backbone, embed_dim=dim)
        self.backbone = ViTBackbone(backbone_cfg)
        self.head = build_head(cfg.head, dim, hidden_dim=cfg.head_hidden_dim)
        self.mask_head = build_head("mask", dim) if cfg.with_mask else None
        self.roi_head = build_head("roi_guidance", dim) if cfg.with_roi else None
        self.reliability_head = ReliabilityHead(dim) if cfg.with_reliability else None

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, grid_hw = self.patch_embed(image)
        feats = self.backbone.forward_search(tokens)
        head_out = self.head(feats)
        out: dict[str, torch.Tensor] = {"head": head_out}
        if self.config.head == "bbox":
            out["box"] = head_out
            out["state"] = box_to_state(head_out)
        if self.mask_head is not None:
            out["mask"] = self.mask_head(feats, grid_hw=grid_hw)
        if self.roi_head is not None:
            out["eye_box"] = self.roi_head(feats)
        if self.reliability_head is not None:
            out["reliability"] = self.reliability_head(feats)
        return out


@dataclass
class FrameModelConfig(DirectDetectorConfig):
    modality: str = "frame"


class FrameModel(DirectPupilDetector):
    """Frame-only detector: Conv-F stem, full-depth ViT, box head (default)."""

    def __init__(self, config: FrameModelConfig | None = None) -> None:
        super().__init__(config or FrameModelConfig())


def build_frame_model(config: FrameModelConfig | None = None) -> FrameModel:
    return FrameModel(config)
