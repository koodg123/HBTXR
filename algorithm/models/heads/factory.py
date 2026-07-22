"""Head factory — build a head by name so a config swaps heads (point 1 / 8).

``build_head(name, embed_dim, **kwargs)`` maps a config string to a head module,
so a modality's head is chosen by editing ``config.head.*`` rather than code. The
paper defaults are ``search: bbox`` and ``track: ellipse``; ``heatmap`` etc. are
selectable alternatives, and the center/corner/yolo experimental heads (parked in
tmp) can be registered here later without touching the detectors.
"""
from __future__ import annotations

from torch import nn

from models.heads.bbox import PupilBoxHead
from models.heads.ellipse import PupilEllipseHead
from models.heads.heatmap import MultiCenterHeatmapHead, SingleCenterHeatmapHead
from models.heads.mask import PupilMaskHead
from models.heads.reliability import ReliabilityHead
from models.heads.roi_guidance import EyeRegionHead

HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "bbox": PupilBoxHead,
    "ellipse": PupilEllipseHead,
    "mask": PupilMaskHead,
    "reliability": ReliabilityHead,
    "roi_guidance": EyeRegionHead,
    # heatmap alternatives: "heatmap" aliases the faithful CenterNet multi-head
    # (= the parked HBTXR/EPNet head); "heatmap_single" is the minimal variant.
    "heatmap": MultiCenterHeatmapHead,
    "heatmap_multi": MultiCenterHeatmapHead,
    "heatmap_single": SingleCenterHeatmapHead,
}


def build_head(name: str, embed_dim: int, **kwargs) -> nn.Module:
    key = str(name).strip().lower()
    try:
        cls = HEAD_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(HEAD_REGISTRY))
        raise ValueError(f"unknown head {name!r}; known heads: {known}") from exc
    return cls(embed_dim, **kwargs)


def list_head_names() -> list[str]:
    return sorted(HEAD_REGISTRY)
