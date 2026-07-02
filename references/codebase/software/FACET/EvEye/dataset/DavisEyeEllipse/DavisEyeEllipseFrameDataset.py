import bisect
import json
import math
from functools import lru_cache
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import (
    draw_umich_gaussian,
    gaussian_radius,
)
from EvEye.dataset.DavisEyeEllipse.utils import cal_ellipse_area, convert_to_ellipse
from EvEye.utils.cache.MemmapCacheStructedEvents import load_memmap
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy


def natural_key(path: Path):
    parts = []
    for part in path.as_posix().replace("_", "/").replace(".", "/").split("/"):
        parts.append(int(part) if part.isdigit() else part)
    return parts


def parse_frame_timestamp(frame_path: Path) -> int:
    return int(frame_path.stem.split("_")[-1])


class DavisEyeEllipseFrameDataset(Dataset):
    def __init__(
        self,
        root_path: Path | str,
        raw_root: Path | str | None = None,
        split="train",
        pupil_area=200,
        num_classes=1,
        default_resolution=[128, 128],
        channels=1,
        progress_state_name="progress_state.json",
        use_cached_frame=False,
        cached_frame_name="cached_frame",
        use_cached_aps=False,
        use_cached_events=False,
        sensor_wh=(240, 160),
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.raw_root = Path(raw_root) if raw_root is not None else None
        self.split = split
        self.ellipse_path = self.root_path / self.split / "cached_ellipse"
        self.frame_path = self.root_path / self.split / cached_frame_name
        self.pupil_area = pupil_area
        self.num_classes = num_classes
        self.max_objs = 100
        self.default_resolution = default_resolution
        self.channels = channels
        self.use_cached_frame = use_cached_frame
        self.use_cached_aps = use_cached_aps
        self.use_cached_events = use_cached_events
        self.sensor_wh = tuple(sensor_wh)                          # crop (W, H) for event rendering
        self.ellipse_data_path = self.ellipse_path / "ellipses_batch_0.memmap"
        self.ellipse_info_path = self.ellipse_path / "ellipses_batch_info_0.txt"
        self.frame_data_path = self.frame_path / "frames_batch_0.memmap"
        self.frame_info_path = self.frame_path / "frames_batch_info_0.txt"
        if not self.use_cached_events and self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")

        self.ellipses = self._load_ellipses()
        self.transform = self.get_transforms()
        self.cached_frames = self._load_cached_frames() if self.use_cached_frame else None
        self.aps_paths = self._load_aps_paths() if self.use_cached_aps else None
        self.ev_mm, self.ev_idx = self._load_events() if self.use_cached_events else (None, None)
        self.num_samples = len(self.ellipses)
        if self.use_cached_aps:
            if len(self.aps_paths) != self.num_samples:
                raise ValueError(
                    f"cached_aps frame count mismatch for split={self.split}: "
                    f"aps={len(self.aps_paths)}, ellipses={self.num_samples}"
                )
            self.session_records = []
            self.cumulative_counts = np.array([], dtype=np.int64)
        elif self.use_cached_events:
            if len(self.ev_idx) != self.num_samples:
                raise ValueError(
                    f"cached_events frame count mismatch for split={self.split}: "
                    f"events={len(self.ev_idx)}, ellipses={self.num_samples}"
                )
            self.session_records = []
            self.cumulative_counts = np.array([], dtype=np.int64)
        elif self.use_cached_frame:
            if len(self.cached_frames) != self.num_samples:
                raise ValueError(
                    f"cached frame count mismatch for split={self.split}: "
                    f"frames={len(self.cached_frames)}, ellipses={self.num_samples}"
                )
            self.session_records = []
            self.cumulative_counts = np.array([], dtype=np.int64)
        else:
            if self.raw_root is None:
                raise ValueError("raw_root is required when use_cached_frame is false")
            self.session_records = self._load_session_records(progress_state_name)
            self.cumulative_counts = np.cumsum(
                [record["count"] for record in self.session_records], dtype=np.int64
            )
            total_records = int(self.cumulative_counts[-1]) if len(self.cumulative_counts) else 0
            if total_records != self.num_samples:
                raise ValueError(
                    f"session valid count mismatch for split={self.split}: "
                    f"sessions={total_records}, ellipses={self.num_samples}"
                )

    def _load_ellipses(self):
        if self.use_cached_aps or self.use_cached_events:          # crop dataset: plain .npy [t,x,y,a,b,ang] (crop coords)
            return np.load(self.root_path / self.split / "cached_ellipse" / "ellipse_records.npy")
        return load_memmap(self.ellipse_data_path, self.ellipse_info_path)

    def _load_aps_paths(self):
        """Ordered cached_aps PNG paths aligned 1:1 to ellipse_records (crop 240x160)."""
        fi = np.load(self.root_path / self.split / "labels_original" / "frame_index.npy")
        aps_root = self.root_path / self.split / "cached_aps"
        return [aps_root / str(r["key"]) / f"{int(r['idx']):06d}_{int(r['t'])}.png" for r in fi]

    _EV_DT = np.dtype([("t", "<i8"), ("x", "<i8"), ("y", "<i8"), ("p", "<i8")])

    def _load_events(self):
        """cached_data (single batch) memmap + per-frame [start,end] indices, aligned to ellipse_records."""
        d = self.root_path / self.split / "cached_data"
        idx = np.load(d / "events_indices_0.npy")
        n = int(idx[-1, 1]) if len(idx) else 0
        mm = np.memmap(d / "events_batch_0.memmap", dtype=self._EV_DT, mode="r", shape=(n,))
        return mm, idx

    def _render_event(self, ev):
        """crop events (t,x,y,p) -> 2-channel polarity frame (Hc, Wc, 2), 0..255 float (causal_linear, matches ref)."""
        w, h = self.sensor_wh
        if len(ev) == 0:
            return np.zeros((h, w, 2), np.float32)
        fs = to_frame_stack_numpy(ev, (w, h, 2), 1, "causal_linear",
                                  int(ev["t"][0]), int(ev["t"][-1]), 10).squeeze(0)  # (2, Hc, Wc)
        np.clip(fs, 0, 255, out=fs)
        return np.moveaxis(fs, 0, -1).astype(np.float32)           # (Hc, Wc, 2)

    def _load_cached_frames(self):
        return load_memmap(self.frame_data_path, self.frame_info_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ellipses"] = None
        state["cached_frames"] = None
        state["ev_mm"] = None
        return state

    def _ensure_memmaps(self):
        if self.ellipses is None:
            self.ellipses = self._load_ellipses()
        if self.use_cached_frame and self.cached_frames is None:
            self.cached_frames = self._load_cached_frames()
        if self.use_cached_events and self.ev_mm is None:
            self.ev_mm, self.ev_idx = self._load_events()

    def _load_session_records(self, progress_state_name: str):
        progress_path = self.root_path / progress_state_name
        if not progress_path.exists():
            raise FileNotFoundError(f"missing progress state: {progress_path}")
        with progress_path.open("r", encoding="utf-8") as fp:
            progress = json.load(fp)

        records = []
        ellipse_start = 0
        for summary in progress.get("session_summaries", []):
            if summary.get("split") != self.split:
                continue
            count = int(summary.get("valid", 0))
            if count <= 0:
                continue
            session_path = self._remap_session_path(Path(summary["session"]))
            records.append(
                {
                    "session_path": session_path,
                    "ellipse_start": ellipse_start,
                    "count": count,
                }
            )
            ellipse_start += count
        return records

    def _remap_session_path(self, session_path: Path) -> Path:
        parts = session_path.parts
        if "Data_davis" in parts:
            rel = Path(*parts[parts.index("Data_davis") + 1 :])
            session_path = self.raw_root / rel
        if not (session_path / "frames").exists():
            raise FileNotFoundError(f"missing frames directory for session: {session_path}")
        return session_path

    def get_transforms(self):
        if self.split == "train":
            return A.ReplayCompose(
                [
                    A.Resize(self.default_resolution[0], self.default_resolution[1]),
                    A.ShiftScaleRotate(
                        shift_limit=0.2,
                        scale_limit=0.2,
                        rotate_limit=15,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        p=1,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )
        return A.ReplayCompose(
            [A.Resize(self.default_resolution[0], self.default_resolution[1])]
        )

    @lru_cache(maxsize=64)
    def _session_frame_index(self, session_index: int):
        frames_dir = self.session_records[session_index]["session_path"] / "frames"
        frame_paths = sorted(frames_dir.glob("*.png"), key=natural_key)
        return {parse_frame_timestamp(frame_path): frame_path for frame_path in frame_paths}

    def _load_frame(self, frame_path: Path):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(f"failed to load frame: {frame_path}")
        return frame

    def transform_ellipse(self, ellipse, replay, image_shape):
        height, width = image_shape[:2]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.ellipse(canvas, ellipse, (255, 255, 255), -1)
        transformed_ellipse = A.ReplayCompose.replay(replay, image=canvas)["image"]
        gray = cv2.cvtColor(transformed_ellipse, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            raise ValueError("No contours found")
        cnt = contours[0]
        if len(cnt) < 5:
            raise ValueError("Not enough points to fit an ellipse")
        ellipse = cv2.fitEllipse(cnt)
        x, y = ellipse[0]
        a, b = ellipse[1]
        ang = ellipse[2]
        if a < b:
            a, b = b, a
            ang += 90
        while ang > 90 or ang < -90:
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
        x, y, a, b, ang = [round(val, 2) for val in [x, y, a, b, ang]]
        return ((x, y), (a, b), ang)

    def cal_trig(self, ang):
        ang = np.deg2rad(ang)
        return np.array([np.sin(2 * ang), np.cos(2 * ang)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        self._ensure_memmaps()
        ellipse = convert_to_ellipse(self.ellipses[index])
        if self.use_cached_aps:
            frame = self._load_frame(self.aps_paths[index])       # crop 240x160 grayscale PNG
            image_shape = frame.shape                              # (160, 240) -> Resize handles anisotropic remap
        elif self.use_cached_events:
            s, e = self.ev_idx[index]
            frame = self._render_event(self.ev_mm[int(s):int(e)])  # (160, 240, 2) polarity frame
            image_shape = frame.shape[:2]                          # (160, 240)
        elif self.use_cached_frame:
            frame = np.asarray(self.cached_frames[index])
            image_shape = (260, 346)
        else:
            session_index = bisect.bisect_right(self.cumulative_counts, index)
            timestamp = int(self.ellipses[index]["t"])
            frame_index = self._session_frame_index(session_index)
            frame_path = frame_index.get(timestamp)
            if frame_path is None:
                raise FileNotFoundError(
                    f"no frame with timestamp={timestamp} in "
                    f"{self.session_records[session_index]['session_path']}"
                )
            frame = self._load_frame(frame_path)
            image_shape = frame.shape

        transformed = self.transform(image=frame)
        frame = transformed["image"]
        replay = transformed["replay"]

        if (
            ellipse[0] != (0, 0)
            and cal_ellipse_area(ellipse[1][0], ellipse[1][1]) > self.pupil_area
        ):
            try:
                ellipse = self.transform_ellipse(ellipse, replay, image_shape=image_shape)
                close = 0
            except ValueError:
                ellipse = ((0, 0), (0, 0), 0)
                close = 1
        else:
            close = 1

        if self.use_cached_events:
            frame = frame.astype(np.float32) / 255.0
            frame = np.moveaxis(frame, -1, 0)                      # (H, W, 2) -> (2, H, W)
        elif self.channels == 1:
            frame = frame.astype(np.float32) / 255.0
            frame = np.expand_dims(frame, axis=0)
        else:
            frame = frame.astype(np.float32) / 255.0
            frame = np.repeat(np.expand_dims(frame, axis=0), 3, axis=0)

        down_ratio = 4
        output_height = self.default_resolution[0] // down_ratio
        output_width = self.default_resolution[1] // down_ratio
        x, y = ellipse[0]
        a, b = ellipse[1]
        an = ellipse[2]
        x, y, a, b = [round(val / down_ratio, 2) for val in [x, y, a, b]]
        label_values = np.array([x, y, a, b, an], dtype=np.float32)
        valid_ellipse = close == 0 and np.isfinite(label_values).all() and a > 0 and b > 0
        if valid_ellipse:
            x = np.clip(x, 0, output_width - 1)
            y = np.clip(y, 0, output_height - 1)
            a = np.clip(a, 0, output_width - 1)
            b = np.clip(b, 0, output_height - 1)
            label_values = np.array([x, y, a, b, an], dtype=np.float32)
            valid_ellipse = np.isfinite(label_values).all() and a > 0 and b > 0

        if not valid_ellipse:
            close = 1
            x = y = a = b = an = 0.0

        ellipse_downsampled = ((x, y), (a, b), an)
        ellipse_downsampled_tensor = torch.tensor([x, y, a, b, an], dtype=torch.float32)

        hm = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)
        ab = np.zeros((self.max_objs, 2), dtype=np.float32)
        ang = np.zeros((self.max_objs, 1), dtype=np.float32)
        trig = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        mask = np.zeros((self.num_classes, output_height, output_width), dtype=np.float32)

        center = np.array([x, y], dtype=np.float32)
        if valid_ellipse:
            radius = gaussian_radius((math.ceil(b), math.ceil(a)))
            radius = max(0, int(radius))
            center_int = center.astype(np.int32)
            draw_umich_gaussian(hm[0], center_int, radius)
            ab[0] = a * 1.0, b * 1.0
            ang[0] = an * 1.0 + 90
            trig[0] = self.cal_trig(an * 1.0 + 90)
            ind[0] = center_int[1] * output_width + center_int[0]
            reg[0] = center - center_int
            reg_mask[0] = 1
            cv2.ellipse(mask[0], ellipse_downsampled, 1, -1)

        return {
            "input": frame,
            "hm": hm,
            "reg_mask": reg_mask,
            "ind": ind,
            "ab": ab,
            "ang": ang,
            "trig": trig,
            "mask": mask,
            "reg": reg,
            "center": center,
            "close": close,
            "ellipse": ellipse_downsampled_tensor,
        }
