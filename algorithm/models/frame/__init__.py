"""models.frame — Frame-only HBTXR model (search path, standalone).

Independently trainable direct pupil detector on frame input. Exposes the shared
DirectPupilDetector base (reused by models.event) and FrameModel.
"""
from models.frame.model import (
    DirectDetectorConfig,
    DirectPupilDetector,
    FrameModel,
    FrameModelConfig,
    build_frame_model,
)

__all__ = [
    "DirectPupilDetector",
    "DirectDetectorConfig",
    "FrameModel",
    "FrameModelConfig",
    "build_frame_model",
]
