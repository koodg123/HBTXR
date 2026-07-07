from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def frame_to_image(frame: np.ndarray) -> Image.Image:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


def event_to_image(event: np.ndarray) -> Image.Image:
    arr = np.asarray(event, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(f"Expected event shape [2,H,W], got {arr.shape}")
    neg = arr[0]
    pos = arr[1]
    neg = neg / max(float(np.max(np.abs(neg))), 1e-6)
    pos = pos / max(float(np.max(np.abs(pos))), 1e-6)
    rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(pos * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(neg * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb)


def save_panel(images: list[Image.Image], output_path: str | Path, *, labels: list[str] | None = None) -> None:
    if not images:
        return
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + (24 if labels else 0)
    canvas = Image.new("RGB", (width, height), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for idx, image in enumerate(images):
        canvas.paste(image, (x, 24 if labels else 0))
        if labels:
            draw.text((x + 4, 4), labels[idx], fill=(255, 255, 255))
        x += image.width
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
