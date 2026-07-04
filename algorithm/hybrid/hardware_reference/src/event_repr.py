from __future__ import annotations

import numpy as np


def to_polarity_frame(
    events: np.ndarray,
    *,
    height: int,
    width: int,
    normalize: bool = True,
) -> np.ndarray:
    frame = np.zeros((2, int(height), int(width)), dtype=np.float32)
    if events.size == 0:
        return frame
    xs = np.clip(events[:, 0].astype(np.int64), 0, width - 1)
    ys = np.clip(events[:, 1].astype(np.int64), 0, height - 1)
    ps = (events[:, 3] > 0).astype(np.int64)
    np.add.at(frame, (ps, ys, xs), 1.0)
    if normalize:
        denom = max(float(frame.max()), 1.0)
        frame /= denom
    return frame


def build_event_frame(row: dict, *, input_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = int(input_size[0]), int(input_size[1])
    events = row.get("events")
    if events is None:
        return np.zeros((2, h, w), dtype=np.float32)
    return to_polarity_frame(np.asarray(events, dtype=np.float32), height=h, width=w)

