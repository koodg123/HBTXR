from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .event_repr import build_event_frame
from .geometry import ellipse_mask
from .io import read_jsonl, xywht_to_xyabuv


def _load_gray(path: str | Path, size_hw: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize((int(size_hw[1]), int(size_hw[0])))
    return np.asarray(img, dtype=np.float32) / 255.0


def _state_from_row(row: dict[str, Any], size_hw: tuple[int, int]) -> np.ndarray:
    if "state6" in row:
        arr = np.asarray(row["state6"], dtype=np.float32)
        if arr.shape[-1] == 6:
            return arr
    ellipse = row.get("ellipse_xywht") or [size_hw[1] / 2.0, size_hw[0] / 2.0, 40.0, 24.0, 0.0]
    return xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32))


def make_synthetic_batch(batch_size: int = 2, *, input_size: tuple[int, int] = (256, 256)) -> dict[str, torch.Tensor]:
    h, w = int(input_size[0]), int(input_size[1])
    frame = torch.rand(batch_size, 1, h, w)
    event = torch.rand(batch_size, 2, h, w) * 0.1
    center = torch.tensor([w / 2.0, h / 2.0, 42.0, 26.0, 1.0, 0.0]).repeat(batch_size, 1)
    mask = torch.zeros(batch_size, 1, h, w)
    return {
        "frame": frame,
        "event": event,
        "prev_state": center.clone(),
        "target_state": center.clone(),
        "mask": mask,
        "quality": torch.ones(batch_size),
        "closed_eye_flag": torch.zeros(batch_size),
        "event_density": event.mean(dim=(1, 2, 3)),
    }


class HGTXRDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, input_size: tuple[int, int] = (256, 256)) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.input_size = tuple(int(v) for v in input_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        h, w = self.input_size
        if row.get("frame_path"):
            frame_np = _load_gray(row["frame_path"], self.input_size)
        else:
            frame_np = np.zeros((h, w), dtype=np.float32)
        event_np = build_event_frame(row, input_size=self.input_size)
        target = _state_from_row(row, self.input_size)
        prev = np.asarray(row.get("prev_state", target), dtype=np.float32)
        ellipse = row.get("ellipse_xywht")
        mask_np = ellipse_mask((h, w), ellipse).astype(np.float32) if ellipse else np.zeros((h, w), dtype=np.float32)
        return {
            "frame": torch.from_numpy(frame_np[None]),
            "event": torch.from_numpy(event_np),
            "prev_state": torch.from_numpy(prev.astype(np.float32)),
            "target_state": torch.from_numpy(target.astype(np.float32)),
            "mask": torch.from_numpy(mask_np[None]),
            "quality": torch.tensor(float(row.get("quality", 1.0)), dtype=torch.float32),
            "closed_eye_flag": torch.tensor(float(row.get("closed_eye_flag", 0.0)), dtype=torch.float32),
            "event_density": torch.tensor(float(event_np.mean()), dtype=torch.float32),
            "sample_id": str(row.get("sample_id", idx)),
            "meta": row.get("meta", {}),
        }

