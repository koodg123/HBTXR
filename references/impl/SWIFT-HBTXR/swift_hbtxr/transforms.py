from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class SpatialTransform:
    policy: str
    matrix: np.ndarray
    source_xywh: tuple[float, float, float, float]
    target_size_wh: tuple[int, int]
    scale_xy: tuple[float, float]
    pad_xy: tuple[float, float]

    def point(self, xy: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(xy, dtype=np.float32))
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        hom = np.concatenate([points, ones], axis=1)
        mapped = hom @ self.matrix.T
        return mapped[:, :2]

    def bbox(self, xywh: list[float] | tuple[float, float, float, float]) -> list[float]:
        x, y, w, h = [float(v) for v in xywh]
        corners = np.asarray([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
        mapped = self.point(corners)
        min_xy = mapped.min(axis=0)
        max_xy = mapped.max(axis=0)
        wh = np.maximum(max_xy - min_xy, 1e-3)
        center = 0.5 * (min_xy + max_xy)
        return [float(center[0]), float(center[1]), float(wh[0]), float(wh[1])]

    def ellipse(self, xywht: list[float] | tuple[float, float, float, float, float]) -> list[float]:
        x, y, a, b, theta = [float(v) for v in xywht]
        center = self.point(np.asarray([[x, y]], dtype=np.float32))[0]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.asarray([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        axes = np.asarray([[a * a, 0.0], [0.0, b * b]], dtype=np.float32)
        covariance = rotation @ axes @ rotation.T
        linear = self.matrix[:2, :2].astype(np.float32)
        covariance_t = linear @ covariance @ linear.T
        evals, evecs = np.linalg.eigh(covariance_t)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 1e-6)
        evecs = evecs[:, order]
        new_a = float(np.sqrt(evals[0]))
        new_b = float(np.sqrt(evals[1]))
        new_theta = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
        return [float(center[0]), float(center[1]), new_a, new_b, new_theta]

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "matrix": self.matrix.tolist(),
            "source_xywh": [float(v) for v in self.source_xywh],
            "target_size_wh": [int(v) for v in self.target_size_wh],
            "scale_xy": [float(v) for v in self.scale_xy],
            "pad_xy": [float(v) for v in self.pad_xy],
        }


def _crop_with_padding(image: np.ndarray, source_xywh: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = source_xywh
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = int(np.ceil(x + w))
    y1 = int(np.ceil(y + h))
    if image.ndim == 3:
        output = np.zeros((max(1, y1 - y0), max(1, x1 - x0), image.shape[2]), dtype=image.dtype)
    else:
        output = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=image.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.shape[1], x1)
    src_y1 = min(image.shape[0], y1)
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    output[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return output


def _resize_image(arr: np.ndarray, size_wh: tuple[int, int], *, mode: str) -> np.ndarray:
    resample = Image.Resampling.BILINEAR if mode == "bilinear" else Image.Resampling.NEAREST
    return np.asarray(Image.fromarray(arr).resize(size_wh, resample=resample))


def _resize_tensor(arr: np.ndarray, size_wh: tuple[int, int], *, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size_wh[1], size_wh[0]), mode=mode, align_corners=False if mode != "nearest" else None)
    return resized.squeeze(0).cpu().numpy()


def _direct_transform(source_xywh: tuple[float, float, float, float], target_size_wh: tuple[int, int], *, policy: str) -> SpatialTransform:
    x, y, w, h = source_xywh
    sx = float(target_size_wh[0]) / max(float(w), 1.0)
    sy = float(target_size_wh[1]) / max(float(h), 1.0)
    matrix = np.asarray([[sx, 0.0, -x * sx], [0.0, sy, -y * sy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return SpatialTransform(policy=policy, matrix=matrix, source_xywh=source_xywh, target_size_wh=target_size_wh, scale_xy=(sx, sy), pad_xy=(0.0, 0.0))


def _letterbox_transform(source_xywh: tuple[float, float, float, float], target_size_wh: tuple[int, int]) -> SpatialTransform:
    x, y, w, h = source_xywh
    scale = min(float(target_size_wh[0]) / max(float(w), 1.0), float(target_size_wh[1]) / max(float(h), 1.0))
    pad_x = 0.5 * (float(target_size_wh[0]) - w * scale)
    pad_y = 0.5 * (float(target_size_wh[1]) - h * scale)
    matrix = np.asarray([[scale, 0.0, pad_x - x * scale], [0.0, scale, pad_y - y * scale], [0.0, 0.0, 1.0]], dtype=np.float32)
    return SpatialTransform(policy="letterbox_square", matrix=matrix, source_xywh=source_xywh, target_size_wh=target_size_wh, scale_xy=(scale, scale), pad_xy=(pad_x, pad_y))


def build_transform(
    *,
    resize_policy: str,
    source_shape_hw: tuple[int, int],
    roi_xywh: tuple[float, float, float, float],
    target_size_wh: tuple[int, int],
) -> SpatialTransform:
    if resize_policy == "facet_square_direct":
        return _direct_transform(roi_xywh, target_size_wh, policy=resize_policy)
    if resize_policy == "letterbox_square":
        return _letterbox_transform(roi_xywh, target_size_wh)
    if resize_policy == "sensor_full_square":
        full_xywh = (0.0, 0.0, float(source_shape_hw[1]), float(source_shape_hw[0]))
        return _direct_transform(full_xywh, target_size_wh, policy=resize_policy)
    raise ValueError(f"Unsupported resize policy: {resize_policy}")


def apply_transform_to_frame(frame: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    cropped = _crop_with_padding(frame, transform.source_xywh)
    if transform.policy == "letterbox_square":
        scale = transform.scale_xy[0]
        scaled_w = max(1, int(round(cropped.shape[1] * scale)))
        scaled_h = max(1, int(round(cropped.shape[0] * scale)))
        resized = _resize_image(cropped, (scaled_w, scaled_h), mode="bilinear")
        canvas = np.zeros((transform.target_size_wh[1], transform.target_size_wh[0]), dtype=resized.dtype)
        pad_x = int(round(transform.pad_xy[0]))
        pad_y = int(round(transform.pad_xy[1]))
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        return canvas
    return _resize_image(cropped, transform.target_size_wh, mode="bilinear")


def apply_transform_to_mask(mask: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    cropped = _crop_with_padding(mask, transform.source_xywh)
    if transform.policy == "letterbox_square":
        scale = transform.scale_xy[0]
        scaled_w = max(1, int(round(cropped.shape[1] * scale)))
        scaled_h = max(1, int(round(cropped.shape[0] * scale)))
        resized = _resize_image(cropped, (scaled_w, scaled_h), mode="nearest")
        canvas = np.zeros((transform.target_size_wh[1], transform.target_size_wh[0]), dtype=resized.dtype)
        pad_x = int(round(transform.pad_xy[0]))
        pad_y = int(round(transform.pad_xy[1]))
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        return canvas
    return _resize_image(cropped, transform.target_size_wh, mode="nearest")


def apply_transform_to_event(event: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    cropped_channels = [_crop_with_padding(event[channel], transform.source_xywh) for channel in range(event.shape[0])]
    stacked = np.stack(cropped_channels, axis=0).astype(np.float32)
    if transform.policy == "letterbox_square":
        scale = transform.scale_xy[0]
        scaled_w = max(1, int(round(stacked.shape[-1] * scale)))
        scaled_h = max(1, int(round(stacked.shape[-2] * scale)))
        resized = _resize_tensor(stacked, (scaled_w, scaled_h), mode="bilinear")
        canvas = np.zeros((stacked.shape[0], transform.target_size_wh[1], transform.target_size_wh[0]), dtype=np.float32)
        pad_x = int(round(transform.pad_xy[0]))
        pad_y = int(round(transform.pad_xy[1]))
        canvas[:, pad_y:pad_y + resized.shape[1], pad_x:pad_x + resized.shape[2]] = resized
        return canvas
    return _resize_tensor(stacked, transform.target_size_wh, mode="bilinear")
