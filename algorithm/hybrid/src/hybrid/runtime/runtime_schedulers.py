from __future__ import annotations

from typing import Any

from hybrid.models.controller import TrackSearchSchedulerFSM

_RUNTIME_SCHEDULERS = {
    "fsm": TrackSearchSchedulerFSM,
    "track_search": TrackSearchSchedulerFSM,
    "track_search_fsm": TrackSearchSchedulerFSM,
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def list_runtime_scheduler_names() -> list[str]:
    return sorted(_RUNTIME_SCHEDULERS.keys())


def get_runtime_scheduler_class(name: str = "track_search_fsm"):
    normalized = _normalize_name(name)
    try:
        return _RUNTIME_SCHEDULERS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown runtime scheduler: {normalized}") from exc


def build_runtime_scheduler(name: str = "track_search_fsm", **kwargs: Any) -> TrackSearchSchedulerFSM:
    return get_runtime_scheduler_class(name)(**kwargs)


__all__ = [
    "TrackSearchSchedulerFSM",
    "build_runtime_scheduler",
    "get_runtime_scheduler_class",
    "list_runtime_scheduler_names",
]
