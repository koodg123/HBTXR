from __future__ import annotations

import cv2
import numpy as np

from .io import angle_to_uv, uv_to_angle, xyabuv_to_xywht, xywht_to_xyabuv
from .transforms import (
    SpatialTransform,
    apply_transform_to_event,
    apply_transform_to_frame,
    apply_transform_to_mask,
    build_transform,
)


def ellipse_mask(mask_shape_hw: tuple[int, int], ellipse_xywht: list[float] | np.ndarray) -> np.ndarray:
    x, y, a, b, theta = [float(v) for v in ellipse_xywht]
    canvas = np.zeros((int(mask_shape_hw[0]), int(mask_shape_hw[1])), dtype=np.uint8)
    if a <= 0.0 or b <= 0.0:
        return canvas
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(a / 2.0))), max(1, int(round(b / 2.0))))
    angle_deg = float(theta) * 180.0 / np.pi
    return cv2.ellipse(canvas, center, axes, angle_deg, 0, 360, 1, -1)


def compute_open_extent_from_binary_mask(mask: np.ndarray, ellipse_xywht: list[float] | np.ndarray, *, interpolation_mode: bool = False) -> float:
    mask_bool = np.asarray(mask) > 0
    ellipse = ellipse_mask(mask_bool.shape, ellipse_xywht) > 0
    ellipse_area = float(ellipse.sum())
    if ellipse_area <= 1e-6:
        return 0.0
    if interpolation_mode:
        return float(mask_bool.sum()) / ellipse_area
    return float(np.logical_and(mask_bool, ellipse).sum()) / ellipse_area


__all__ = [
    "SpatialTransform",
    "angle_to_uv",
    "apply_transform_to_event",
    "apply_transform_to_frame",
    "apply_transform_to_mask",
    "build_transform",
    "compute_open_extent_from_binary_mask",
    "ellipse_mask",
    "uv_to_angle",
    "xyabuv_to_xywht",
    "xywht_to_xyabuv",
]
