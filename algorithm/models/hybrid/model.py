"""Hybrid HBTXR model (paper full system): search + track on a shared backbone.

One shared ViT backbone processes both paths (paper Sec III-C):

- SEARCH (frame): Conv-F stem -> full-depth ViT (B_1:L) -> PupilBoxHead -> g() ->
  anchor state; plus g_eye ROI guidance and the h_search reliability head.
- TRACK (event): Conv-E stem -> early-exit ViT (B_1:c) -> PupilEllipseHead ->
  anchor-relative residual -> apply_residual(z, ds) -> updated state; h_track
  reliability head.

A CPU-side TrackSearchScheduler selects the mode over a stream (Sec III-C.3).

Training uses each branch on its own step (search_step / track_step); the
scheduler is inference-only. The backbone/stems may be initialized from a
frame-only checkpoint (optional, paper staged training). Heads are config-swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from models.backbones import EventPatchEmbed, FramePatchEmbed, ViTBackbone, ViTConfig
from models.geometry import apply_residual, box_to_state
from models.heads import EyeRegionHead, ReliabilityHead, build_head
from models.hybrid.scheduler import SchedulerConfig, TrackSearchScheduler


@dataclass
class HybridModelConfig:
    embed_dim: int = 192
    patch_size: int = 16
    backbone: ViTConfig = field(default_factory=ViTConfig)
    search_head: str = "bbox"       # config-swappable
    track_head: str = "ellipse"
    head_hidden_dim: int | None = None
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    with_roi: bool = True


class HybridModel(nn.Module):
    def __init__(self, config: HybridModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or HybridModelConfig()
        self.config = cfg
        dim = cfg.embed_dim
        # ONE shared backbone: full depth for search, early-exit (B_1:c) for track.
        # model.embed_dim is authoritative: keep the backbone width in sync with it.
        backbone_cfg = cfg.backbone if cfg.backbone.embed_dim == dim else replace(cfg.backbone, embed_dim=dim)
        self.backbone = ViTBackbone(backbone_cfg)
        self.frame_stem = FramePatchEmbed(embed_dim=dim, patch_size=cfg.patch_size)
        self.event_stem = EventPatchEmbed(embed_dim=dim, patch_size=cfg.patch_size)
        self.search_head = build_head(cfg.search_head, dim, hidden_dim=cfg.head_hidden_dim)
        self.track_head = build_head(cfg.track_head, dim, hidden_dim=cfg.head_hidden_dim)
        self.roi_head = EyeRegionHead(dim) if cfg.with_roi else None
        self.search_reliability = ReliabilityHead(dim)  # h_search
        self.track_reliability = ReliabilityHead(dim)   # h_track
        self.scheduler = TrackSearchScheduler(cfg.scheduler)

    def search_step(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        """Frame -> full-depth ViT -> box -> anchor state (+ reliability, ROI)."""
        tokens, _ = self.frame_stem(frame)
        feats = self.backbone.forward_search(tokens)
        box = self.search_head(feats)
        out = {"box": box, "state": box_to_state(box), "reliability": self.search_reliability(feats)}
        if self.roi_head is not None:
            out["eye_box"] = self.roi_head(feats)
        return out

    def track_step(self, event: torch.Tensor, anchor_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Event + anchor -> early-exit ViT -> residual -> updated state (+ reliability)."""
        tokens, _ = self.event_stem(event)
        feats = self.backbone.forward_track(tokens)
        residual = self.track_head(feats, anchor_state)
        return {
            "residual": residual,
            "state": apply_residual(anchor_state, residual),
            "reliability": self.track_reliability(feats),
        }

    def initial_scheduler_state(self):
        return self.scheduler.initial_state()

    @torch.no_grad()
    def run_step(self, sched_state, *, frame: torch.Tensor | None = None, event: torch.Tensor | None = None):
        """One runtime step: run the scheduler-selected branch, advance the FSM.

        Returns ``(branch_output, next_scheduler_state)``. Reliability is read as a
        scalar pair (runtime processes one stream window at a time).
        """
        mode = self.scheduler.select_mode(sched_state)
        if mode == "search":
            out = self.search_step(frame)
            rel = out["reliability"].flatten().tolist()[:2]
            nxt = self.scheduler.after_search(sched_state, rel, out["state"])
        else:
            out = self.track_step(event, sched_state.anchor_state)
            rel = out["reliability"].flatten().tolist()[:2]
            nxt = self.scheduler.after_track(sched_state, rel, out["state"])
        out["mode"] = mode
        return out, nxt


def build_hybrid_model(config: HybridModelConfig | None = None) -> HybridModel:
    return HybridModel(config)
