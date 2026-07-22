from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EVENT_GENERATION_BACKEND = "none"
SUPPORTED_EVENT_GENERATION_BACKENDS = ("none", "v2e")


def resolve_event_generation_backend(event_generation_backend: str | None) -> str:
    text = str(event_generation_backend or DEFAULT_EVENT_GENERATION_BACKEND).strip().lower()
    if text not in SUPPORTED_EVENT_GENERATION_BACKENDS:
        raise ValueError(f"Unsupported event_generation_backend: {event_generation_backend!r}")
    return text


def _ensure_optional_module(name: str, attrs: dict[str, Any] | None = None) -> None:
    if importlib.util.find_spec(name) is not None or name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module


def _ensure_v2e_optional_imports() -> None:
    # HBTXR uses v2e only for in-memory event generation. Optional viewer/output
    # modules should not block importing EventEmulator when their backends are absent.
    _ensure_optional_module("easygui")
    _ensure_optional_module("dv_processing")
    _ensure_optional_module("screeninfo", {"get_monitors": lambda: []})


class _V2eRuntime:
    def __init__(self, *, v2e_root: Path, device: str = "cpu", v2e_kwargs: dict[str, Any] | None = None) -> None:
        repo_root = Path(v2e_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"v2e root not found: {repo_root}")
        repo_text = str(repo_root)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        _ensure_v2e_optional_imports()
        try:
            from v2ecore.emulator import EventEmulator
        except Exception as exc:  # pragma: no cover - import path is environment-dependent
            raise RuntimeError(
                f"Failed to import v2e from {repo_root}. Check the repository path and its Python dependencies."
            ) from exc
        self.emulator_cls = EventEmulator
        self.device = str(device)
        self.v2e_kwargs = dict(v2e_kwargs or {})

    def generate_sequence_events(
        self,
        *,
        frames: np.ndarray,
        frame_timestamps_us: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        frame_stack = np.asarray(frames, dtype=np.uint8)
        timestamps = np.asarray(frame_timestamps_us, dtype=np.int64)
        if frame_stack.ndim != 3:
            raise ValueError(f"Expected frames with shape [N, H, W], got {frame_stack.shape}")
        if timestamps.shape[0] != frame_stack.shape[0]:
            raise ValueError(
                f"frame_timestamps_us length ({timestamps.shape[0]}) must match frame count ({frame_stack.shape[0]})"
            )
        if frame_stack.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.int64)
            return {
                "t": empty,
                "x": empty.astype(np.int16),
                "y": empty.astype(np.int16),
                "p": empty.astype(np.int8),
            }, np.zeros((0, 2), dtype=np.int64)
        timestamp_origin_us = int(timestamps[0])
        relative_timestamps_us = timestamps - timestamp_origin_us

        emulator = self.emulator_cls(
            output_folder=None,
            dvs_h5=None,
            dvs_aedat2=None,
            dvs_aedat4=None,
            dvs_text=None,
            output_width=int(frame_stack.shape[2]),
            output_height=int(frame_stack.shape[1]),
            device=self.device,
            **self.v2e_kwargs,
        )

        collected: list[np.ndarray] = []
        event_ranges: list[list[int]] = []
        total_events = 0
        for frame, timestamp_us in zip(frame_stack, relative_timestamps_us, strict=False):
            batch = emulator.generate_events(np.asarray(frame, dtype=np.uint8), float(timestamp_us) / 1_000_000.0)
            if batch is not None and len(batch) > 0:
                collected.append(np.asarray(batch))
                total_events += int(len(batch))
            event_ranges.append([event_ranges[-1][1] if event_ranges else 0, total_events])

        if not collected:
            empty = np.zeros((0,), dtype=np.int64)
            return {
                "t": empty,
                "x": empty.astype(np.int16),
                "y": empty.astype(np.int16),
                "p": empty.astype(np.int8),
            }, np.asarray(event_ranges, dtype=np.int64)

        merged = np.concatenate(collected, axis=0)
        event_t_relative_us = np.asarray(np.round(merged[:, 0] * 1_000_000.0), dtype=np.int64)
        return {
            "t": event_t_relative_us + timestamp_origin_us,
            "x": np.asarray(merged[:, 1], dtype=np.int16),
            "y": np.asarray(merged[:, 2], dtype=np.int16),
            "p": np.asarray(merged[:, 3], dtype=np.int8),
        }, np.asarray(event_ranges, dtype=np.int64)


def build_event_generation_runtime(
    *,
    event_generation_backend: str,
    v2e_root: Path | None = None,
    v2e_device: str = "cpu",
    v2e_kwargs: dict[str, Any] | None = None,
) -> Any | None:
    resolved_backend = resolve_event_generation_backend(event_generation_backend)
    if resolved_backend == "none":
        return None
    if v2e_root is None:
        raise ValueError("v2e_root is required for event_generation_backend='v2e'")
    return _V2eRuntime(v2e_root=Path(v2e_root).resolve(), device=str(v2e_device), v2e_kwargs=v2e_kwargs)


def generate_event_packets(
    *,
    event_generation_backend: str,
    frames: np.ndarray,
    frame_timestamps_us: np.ndarray,
    raw_events: dict[str, np.ndarray] | None = None,
    frame_plan: list[dict[str, Any]] | None = None,
    v2e_root: Path | None = None,
    v2e_device: str = "cpu",
    v2e_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    resolved_backend = resolve_event_generation_backend(event_generation_backend)
    if resolved_backend == "none":
        if raw_events is None or frame_plan is None:
            raise ValueError("raw_events and frame_plan are required when event_generation_backend='none'")
        from hybrid.preprocess.target_fps_build import build_target_event_packets

        return build_target_event_packets(raw_events, frame_plan)
    runtime = build_event_generation_runtime(
        event_generation_backend=resolved_backend,
        v2e_root=v2e_root,
        v2e_device=v2e_device,
        v2e_kwargs=v2e_kwargs,
    )
    if runtime is None:
        raise RuntimeError("Resolved event generation runtime unexpectedly missing")
    return runtime.generate_sequence_events(
        frames=np.asarray(frames, dtype=np.uint8),
        frame_timestamps_us=np.asarray(frame_timestamps_us, dtype=np.int64),
    )


__all__ = [
    "DEFAULT_EVENT_GENERATION_BACKEND",
    "SUPPORTED_EVENT_GENERATION_BACKENDS",
    "build_event_generation_runtime",
    "generate_event_packets",
    "resolve_event_generation_backend",
]
