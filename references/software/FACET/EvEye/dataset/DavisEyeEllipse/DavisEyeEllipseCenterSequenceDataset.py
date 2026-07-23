import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset


class DavisEyeEllipseCenterSequenceDataset(Dataset):
    """HBTXR ellipse-cache adapter for center-sequence models such as TennSt."""

    def __init__(
        self,
        root_path: Path | str,
        split="train",
        frames_per_segment=50,
        stride=50,
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear_ori",
        pupil_area=200,
        default_resolution=(64, 64),
        temporal_transform=True,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.frames_per_segment = int(frames_per_segment)
        self.stride = int(stride)
        self.temporal_transform = temporal_transform
        self.default_resolution = tuple(default_resolution)
        self.base_dataset = DavisEyeEllipseDataset(
            root_path=self.root_path,
            split=split,
            accumulate_mode="fixed_count",
            sensor_size=sensor_size,
            events_interpolation=events_interpolation,
            pupil_area=pupil_area,
            num_classes=1,
            default_resolution=list(default_resolution),
        )
        self.segments = self._build_segments()

    def _build_segments(self):
        progress_path = self.root_path / "progress_state.json"
        if not progress_path.exists():
            total = len(self.base_dataset)
            return [
                (start, start + self.frames_per_segment)
                for start in range(0, total - self.frames_per_segment + 1, self.stride)
            ]

        with progress_path.open() as f:
            progress = json.load(f)

        segments = []
        offset = 0
        for session in progress["session_summaries"]:
            if session["split"] != self.split:
                continue
            valid = int(session["valid"])
            for local_start in range(0, valid - self.frames_per_segment + 1, self.stride):
                start = offset + local_start
                segments.append((start, start + self.frames_per_segment))
            offset += valid
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        start, end = self.segments[index]
        frames = []
        centers = []
        openness = []
        for sample_index in range(start, end):
            sample = self.base_dataset[sample_index]
            frames.append(sample["input"])
            center64 = np.asarray(sample["center"], dtype=np.float32) * 4.0
            if int(sample["close"]) != 0:
                center64[:] = 0
            centers.append(center64 / np.asarray(self.default_resolution, dtype=np.float32))
            openness.append(1.0 - float(sample["close"]))

        event_frames = np.stack(frames, axis=1).astype(np.float32)
        labels = np.stack(centers, axis=1).astype(np.float32)
        closes = np.asarray(openness, dtype=np.float32)

        if self.split == "train" and self.temporal_transform and np.random.rand() > 0.5:
            event_frames = np.flip(event_frames, axis=(1,)).copy()
            labels = np.flip(labels, axis=(1,)).copy()
            closes = np.flip(closes, axis=(0,)).copy()

        return (
            torch.from_numpy(event_frames),
            torch.from_numpy(labels),
            torch.from_numpy(closes),
        )
