from __future__ import annotations

import numpy as np

from .io import angle_to_uv, uv_to_angle, xyabuv_to_xywht, xywht_to_xyabuv


def ellipse_mask(mask_shape_hw: tuple[int, int], ellipse_xywht: list[float] | np.ndarray) -> np.ndarray:
    h, w = int(mask_shape_hw[0]), int(mask_shape_hw[1])
    x, y, a, b, theta = [float(v) for v in ellipse_xywht]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    dx = xx - x
    dy = yy - y
    xr = c * dx + s * dy
    yr = -s * dx + c * dy
    aa = max(a / 2.0, 1e-6)
    bb = max(b / 2.0, 1e-6)
    return ((xr / aa) ** 2 + (yr / bb) ** 2 <= 1.0).astype(np.uint8)


def compute_open_extent_from_binary_mask(mask: np.ndarray, ellipse_xywht: list[float] | np.ndarray) -> float:
    mask_bool = np.asarray(mask) > 0
    ell = ellipse_mask(mask_bool.shape, ellipse_xywht) > 0
    denom = float(ell.sum())
    if denom <= 1e-6:
        return 0.0
    return float(np.logical_and(mask_bool, ell).sum()) / denom


def wrap_pi(theta: np.ndarray | float) -> np.ndarray:
    return np.mod(theta, np.pi)


def canonicalize_xywht(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).copy()
    swap = arr[..., 2] < arr[..., 3]
    if np.any(swap):
        old_a = arr[..., 2].copy()
        arr[..., 2] = np.where(swap, arr[..., 3], arr[..., 2])
        arr[..., 3] = np.where(swap, old_a, arr[..., 3])
        arr[..., 4] = np.where(swap, arr[..., 4] + np.pi / 2.0, arr[..., 4])
    arr[..., 4] = wrap_pi(arr[..., 4])
    return arr


def bbox_anchor_to_state(anchor_xywhr: np.ndarray) -> np.ndarray:
    arr = np.asarray(anchor_xywhr, dtype=np.float32)
    x, y, w, h, r = [arr[..., i] for i in range(5)]
    theta = np.where(w >= h, r, r + np.pi / 2.0)
    xywht = np.stack([x, y, np.maximum(w, h) / 2.0, np.minimum(w, h) / 2.0, wrap_pi(theta)], axis=-1)
    return canonicalize_xywht(xywht)


__all__ = [
    "angle_to_uv",
    "compute_open_extent_from_binary_mask",
    "ellipse_mask",
    "wrap_pi",
    "canonicalize_xywht",
    "bbox_anchor_to_state",
    "uv_to_angle",
    "xyabuv_to_xywht",
    "xywht_to_xyabuv",
]

