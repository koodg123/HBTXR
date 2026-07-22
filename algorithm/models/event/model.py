"""Event-only HBTXR model (standalone direct detector).

Event voxel ``[B,2,H,W]`` -> Conv-E stem -> shared ViT (full depth) -> head
(default PupilBoxHead, the SAME head as the frame-only model) -> g() -> pupil
ellipse state. Symmetric to the frame-only baseline (models.frame): it reuses the
exact DirectPupilDetector assembly, only the input stem differs (Conv-E vs
Conv-F). Direct detection means no anchor is required, so it is independently
trainable just like the frame model (no GT-anchor teacher-forcing needed). The
head is config-swappable via build_head.
"""
from __future__ import annotations

from dataclasses import dataclass

from models.frame.model import DirectDetectorConfig, DirectPupilDetector


@dataclass
class EventModelConfig(DirectDetectorConfig):
    modality: str = "event"


class EventModel(DirectPupilDetector):
    """Event-only detector: Conv-E stem, full-depth ViT, box head (default)."""

    def __init__(self, config: EventModelConfig | None = None) -> None:
        super().__init__(config or EventModelConfig())


def build_event_model(config: EventModelConfig | None = None) -> EventModel:
    return EventModel(config)
