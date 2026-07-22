from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.state6 import xywht_to_xyabuv


def import_h5py(required: bool = True):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError:
        if required:
            raise ModuleNotFoundError(
                "h5py is required to read session_store_format='h5'. "
                "Install h5py, or materialize target-fps stores as npz."
            )
        return None
    return h5py


def to_tensor_image(arr: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(arr.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def safe_text(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    text = str(value).strip()
    return text if text else str(default)


def resolve_xywh(row: dict[str, Any], *keys: str, default: list[float] | None = None) -> list[float]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return [float(v) for v in value]
    if default is None:
        raise KeyError(f"Missing any of keys: {keys}")
    return [float(v) for v in default]


def ellipse_to_bbox(ellipse_xywht: list[float]) -> list[float]:
    x, y, a, b, _ = [float(v) for v in ellipse_xywht]
    return [x - 0.5 * a, y - 0.5 * b, a, b]


def clip_xywh_to_sensor(
    xywh: tuple[float, float, float, float] | list[float],
    *,
    sensor_size_wh: tuple[int, int],
) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in xywh]
    max_w = max(1.0, float(sensor_size_wh[0]))
    max_h = max(1.0, float(sensor_size_wh[1]))
    w = min(max(1.0, w), max_w)
    h = min(max(1.0, h), max_h)
    x = min(max(0.0, x), max_w - w)
    y = min(max(0.0, y), max_h - h)
    return x, y, w, h


def constrain_xywh_to_parent(
    xywh: tuple[float, float, float, float] | list[float],
    *,
    parent_xywh: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in xywh]
    px, py, pw, ph = [float(v) for v in parent_xywh]
    if w >= pw or h >= ph:
        return px, py, pw, ph
    x = min(max(x, px), px + pw - w)
    y = min(max(y, py), py + ph - h)
    return x, y, w, h


def rasterize_bbox_mask(
    bbox_xywh: tuple[float, float, float, float] | list[float],
    *,
    sensor_size_wh: tuple[int, int],
) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    mask = np.zeros((int(sensor_size_wh[1]), int(sensor_size_wh[0])), dtype=np.uint8)
    x0 = max(0, min(int(sensor_size_wh[0]), int(np.floor(x))))
    y0 = max(0, min(int(sensor_size_wh[1]), int(np.floor(y))))
    x1 = max(x0, min(int(sensor_size_wh[0]), int(np.ceil(x + w))))
    y1 = max(y0, min(int(sensor_size_wh[1]), int(np.ceil(y + h))))
    mask[y0:y1, x0:x1] = 255
    return mask


def rasterize_ellipse_mask(
    ellipse_xywht: tuple[float, float, float, float, float] | list[float],
    *,
    sensor_size_wh: tuple[int, int],
) -> np.ndarray:
    cx, cy, w, h, theta = [float(v) for v in ellipse_xywht]
    canvas = Image.new("L", tuple(int(v) for v in sensor_size_wh), color=0)
    draw = ImageDraw.Draw(canvas)
    bbox = (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)
    draw.ellipse(bbox, fill=255)
    if abs(theta) > 1e-6:
        canvas = canvas.rotate(
            float(np.degrees(theta)),
            resample=Image.Resampling.BILINEAR,
            center=(cx, cy),
            fillcolor=0,
        )
    return np.asarray(canvas, dtype=np.uint8)


def extract_valid_event_points(
    events: dict[str, np.ndarray],
    *,
    selection: slice,
    sensor_size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(events["x"][selection], dtype=np.int64)
    ys = np.asarray(events["y"][selection], dtype=np.int64)
    valid = (xs >= 0) & (xs < int(sensor_size_wh[0])) & (ys >= 0) & (ys < int(sensor_size_wh[1]))
    return xs[valid], ys[valid]


def state6_from_annotation(annotation: dict[str, Any]) -> np.ndarray:
    if annotation.get("state_xyabuv") is not None:
        return np.asarray(annotation["state_xyabuv"], dtype=np.float32)
    ellipse = (
        annotation.get("pupil_ellipse_xywht_sensor")
        or annotation.get("ellipse_sensor_xywht")
        or annotation.get("ellipse_xywht")
        or annotation.get("ellipse_frame_xywht")
    )
    if ellipse is None:
        raise KeyError("Annotation is missing ellipse/state information")
    return np.asarray(xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)), dtype=np.float32)


def xywht_from_state6(state6: np.ndarray) -> np.ndarray:
    state6 = np.asarray(state6, dtype=np.float32)
    x, y, a, b, u, v = state6.tolist()
    theta = 0.5 * np.arctan2(u, v)
    return np.asarray([x, y, a, b, theta], dtype=np.float32)


def normalize_uv(state6: np.ndarray) -> np.ndarray:
    state6 = np.asarray(state6, dtype=np.float32).copy()
    norm = np.linalg.norm(state6[4:6])
    if norm < 1e-6:
        state6[4:6] = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        state6[4:6] = state6[4:6] / norm
    return state6
