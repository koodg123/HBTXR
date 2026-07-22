"""models.event — Event-only HBTXR model (standalone).

Independently trainable direct pupil detector on event-voxel input. Reuses the
shared DirectPupilDetector (models.frame) with the Conv-E stem; same box head as
frame-only. No anchor needed.
"""
from models.event.model import EventModel, EventModelConfig, build_event_model

__all__ = ["EventModel", "EventModelConfig", "build_event_model"]
