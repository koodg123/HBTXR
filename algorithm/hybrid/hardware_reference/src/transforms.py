from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialTransform:
    source_xywh: tuple[float, float, float, float]
    target_size_wh: tuple[int, int]
    scale_x: float
    scale_y: float
    offset_x: float = 0.0
    offset_y: float = 0.0

    def point(self, xy: np.ndarray) -> np.ndarray:
        out = np.asarray(xy, dtype=np.float32).copy()
        out[..., 0] = (out[..., 0] - self.source_xywh[0]) * self.scale_x + self.offset_x
        out[..., 1] = (out[..., 1] - self.source_xywh[1]) * self.scale_y + self.offset_y
        return out

    def ellipse_xywht(self, ellipse: np.ndarray) -> np.ndarray:
        out = np.asarray(ellipse, dtype=np.float32).copy()
        out[..., 0:2] = self.point(out[..., 0:2])
        out[..., 2] *= self.scale_x
        out[..., 3] *= self.scale_y
        return out


def build_transform(source_xywh: tuple[float, float, float, float], target_size_wh: tuple[int, int]) -> SpatialTransform:
    sx = float(target_size_wh[0]) / max(float(source_xywh[2]), 1.0)
    sy = float(target_size_wh[1]) / max(float(source_xywh[3]), 1.0)
    return SpatialTransform(source_xywh=source_xywh, target_size_wh=target_size_wh, scale_x=sx, scale_y=sy)

