from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import ellipse_mask
from .io import ensure_dir


def save_overlay(frame: np.ndarray, ellipse_xywht: list[float], output_path: str | Path) -> Path:
    gray = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    mask = ellipse_mask(gray.shape, ellipse_xywht) > 0
    rgb[mask, 0] = 255
    p = Path(output_path)
    ensure_dir(p.parent)
    Image.fromarray(rgb).save(p)
    return p

