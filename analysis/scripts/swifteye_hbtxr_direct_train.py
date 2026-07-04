#!/usr/bin/env python3
"""Prepare Swift-Eye training directly from the HBTXR subject-independent cache.

This path intentionally does not export PNG/DOTA datasets. It keeps the
Swift-Eye detector and temporal-fusion heads built from the original MMRotate
configs and changes the input contract to HBTXR event cache -> 2 channel event
tensor -> 2 channel Swift-Eye backbone patch embedding.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
SWIFTEYE_ROOT = REPO_ROOT / "references/codebase/software/Swift-Eye/mmrotate"
FACET_ROOT = REPO_ROOT / "references/codebase/software/FACET"
DETECTOR_CONFIG = (
    SWIFTEYE_ROOT / "train_swift_eye/train_backbone_and_neck/swift_eye_config.py"
)
TEMPORAL_CONFIG = (
    SWIFTEYE_ROOT / "train_swift_eye/train_with_temporal_fusion_component/model_config.py"
)
DEFAULT_DATA_ROOT = Path(
    "/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis/results/Swift-Eye/direct_hbtxr_img64"
SENSOR_WIDTH = 346
SENSOR_HEIGHT = 260
DEFAULT_IMAGE_SIZE = 64


def add_project_paths() -> None:
    """Add local Swift-Eye/MMRotate and FACET helper paths."""
    for path in (SWIFTEYE_ROOT, FACET_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


add_project_paths()

from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.models.builder import MODELS  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402,F401


LOG = logging.getLogger("swifteye_hbtxr_direct")


@dataclass(frozen=True)
class DirectSwiftEyeConfig:
    """Configuration snapshot written beside run artifacts."""

    data_root: str
    output_root: str
    detector_config: str
    temporal_config: str
    input_height: int
    input_width: int
    pad_divisor: int
    event_mode: str
    event_weight: int
    input_channels: int
    backbone_patch_embed: str
    detector_anchor_scales: List[int]
    temporal_anchor_scales: List[int]
    temporal_search_shape: int
    temporal_template_shape: int
    split_train: str
    split_val: str
    split_test: str
    optimizer_detector: str
    optimizer_temporal: str
    lr_detector: float
    lr_temporal: float
    weight_decay_detector: float
    weight_decay_temporal: float
    temporal_pair_offset: int
    temporal_max_delta_us: int


def parse_shape_dtype(info_path: Path) -> Tuple[Tuple[int, ...], np.dtype]:
    """Read shape and dtype from a FACET memmap info file."""
    lines = info_path.read_text(encoding="utf-8").splitlines()
    shape_str = lines[0].split(": ", 1)[1]
    dtype_str = lines[1].split(": ", 1)[1]
    shape = tuple(int(num) for num in shape_str.strip("()").split(",") if num.strip())
    dtype = np.dtype(eval(dtype_str, {"__builtins__": {}}, {"np": np}))
    return shape, dtype


def load_memmap(data_path: Path, info_path: Path) -> np.memmap:
    """Open a FACET memmap in read-only mode."""
    shape, dtype = parse_shape_dtype(info_path)
    return np.memmap(data_path, dtype=dtype, mode="r", shape=shape)


@lru_cache(maxsize=256)
def cached_event_batch(data_root: str, batch_id: int) -> Tuple[np.memmap, np.ndarray]:
    """Load one cached event batch and its local sample index table."""
    base = Path(data_root)
    events = load_memmap(
        base / f"events_batch_{batch_id}.memmap",
        base / f"events_batch_info_{batch_id}.txt",
    )
    indices = np.load(base / f"events_indices_{batch_id}.npy", mmap_mode="r")
    return events, indices


@lru_cache(maxsize=8)
def event_index_metadata(data_root: str) -> Tuple[Tuple[int, ...], np.ndarray]:
    """Return ordered batch ids and cumulative sample counts for one split."""
    base = Path(data_root)
    index_paths = natsorted(base.glob("events_indices_*.npy"))
    if not index_paths:
        raise FileNotFoundError(f"No events_indices_*.npy under {base}")
    batch_ids = tuple(int(path.stem.split("_")[-1]) for path in index_paths)
    lengths = [int(np.load(path, mmap_mode="r").shape[0]) for path in index_paths]
    return batch_ids, np.cumsum(lengths)


@lru_cache(maxsize=8)
def cached_ellipse_records(ellipse_root: str) -> np.memmap:
    """Load split ellipse records."""
    base = Path(ellipse_root)
    record_path = base / "ellipse_records.npy"
    if record_path.exists():
        return np.load(record_path, mmap_mode="r")
    return load_memmap(base / "ellipses_batch_0.memmap", base / "ellipses_batch_info_0.txt")


def load_event_segment(index: int, data_root: Path) -> np.ndarray:
    """Load one structured event segment by global split index."""
    batch_ids, cumulative = event_index_metadata(str(data_root.resolve()))
    if index < 0 or index >= int(cumulative[-1]):
        raise IndexError(f"event index {index} is out of range for {data_root}")
    meta_pos = int(np.searchsorted(cumulative, index, side="right"))
    previous = int(cumulative[meta_pos - 1]) if meta_pos else 0
    batch_id = batch_ids[meta_pos]
    local_index = index - previous
    events, indices = cached_event_batch(str(data_root.resolve()), batch_id)
    start, end = indices[local_index]
    return events[int(start) : int(end)]


def normalize_angle_le90(angle_rad: float) -> float:
    """Normalize radians to MMRotate le90 interval."""
    return float((angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0)


def polygon_to_obb_le90(points: np.ndarray) -> torch.Tensor:
    """Convert four scaled rectangle points to a MMRotate le90 OBB."""
    pt1, pt2, pt3, _pt4 = points
    edge1 = float(np.linalg.norm(pt1 - pt2))
    edge2 = float(np.linalg.norm(pt2 - pt3))
    if edge1 >= edge2:
        width, height = edge1, edge2
        angle = math.atan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    else:
        width, height = edge2, edge1
        angle = math.atan2(float(pt3[1] - pt2[1]), float(pt3[0] - pt2[0]))
    center = (pt1 + pt3) * 0.5
    return torch.tensor(
        [center[0], center[1], width, height, normalize_angle_le90(angle)],
        dtype=torch.float32,
    )


def ellipse_record_to_obb(record: np.void, height: int, width: int) -> Tuple[torch.Tensor, bool]:
    """Convert one raw FACET ellipse record to image-space MMRotate OBB."""
    cx = float(record["x"])
    cy = float(record["y"])
    axis_w = float(record["a"])
    axis_h = float(record["b"])
    angle_rad = math.radians(float(record["ang"]))
    valid = (
        np.isfinite([cx, cy, axis_w, axis_h, angle_rad]).all()
        and axis_w > 0.0
        and axis_h > 0.0
        and (math.pi * axis_w * axis_h / 4.0) > 200.0
    )
    if not valid:
        return torch.zeros((5,), dtype=torch.float32), False
    half_w = axis_w / 2.0
    half_h = axis_h / 2.0
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    points = corners.dot(rotation.T)
    points[:, 0] += cx
    points[:, 1] += cy
    points[:, 0] *= width / float(SENSOR_WIDTH)
    points[:, 1] *= height / float(SENSOR_HEIGHT)
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))
    obb = polygon_to_obb_le90(points)
    valid = bool(torch.isfinite(obb).all() and obb[2] > 0.0 and obb[3] > 0.0)
    return obb, valid


def detector_cfg_for_img64(cfg_path: Path, args: argparse.Namespace) -> Config:
    """Return Swift-Eye detector config adapted for 2ch 64x64 event input."""
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.backbone.in_channels = 2
    cfg.model.rpn_head.anchor_generator.scales = args.detector_anchor_scales
    return cfg


def temporal_cfg_for_img64(cfg_path: Path, args: argparse.Namespace) -> Config:
    """Return Swift-Eye temporal config adapted for 2ch 64x64 event input."""
    cfg = Config.fromfile(str(cfg_path))
    cfg.backbone.in_channels = 2
    cfg.tracking_head.anchor_generator.scales = args.temporal_anchor_scales
    cfg.tracking_head.anchor_generator.search_shape = args.temporal_search_shape
    cfg.tracking_head.anchor_generator.template_shape = args.temporal_template_shape
    return cfg


def events_to_chw(
    events: np.ndarray,
    height: int,
    width: int,
    mode: str,
    weight: int,
) -> np.ndarray:
    """Convert one structured event segment into a 2xHxW float tensor."""
    if len(events) == 0:
        return np.zeros((2, height, width), dtype=np.float32)
    frame = to_frame_stack_numpy(
        events,
        (SENSOR_WIDTH, SENSOR_HEIGHT, 2),
        1,
        mode,
        int(events["t"][0]),
        int(events["t"][-1]),
        weight,
    ).squeeze(0)
    frame = np.moveaxis(frame, 0, -1).astype(np.float32)
    if (height, width) != (SENSOR_HEIGHT, SENSOR_WIDTH):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    frame = np.moveaxis(frame, -1, 0).astype(np.float32) / 255.0
    return np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)


def pad_tensor_chw(tensor: torch.Tensor, divisor: int) -> torch.Tensor:
    """Pad a BCHW tensor on bottom/right to a fixed divisor."""
    if divisor <= 1:
        return tensor
    height, width = tensor.shape[-2:]
    pad_h = int(math.ceil(height / divisor) * divisor - height)
    pad_w = int(math.ceil(width / divisor) * divisor - width)
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))


def make_img_meta(height: int, width: int, pad_h: int, pad_w: int, tag: str) -> Dict:
    """Create the minimal MMDetection image metadata dict."""
    return {
        "filename": tag,
        "ori_filename": tag,
        "ori_shape": (height, width, 2),
        "img_shape": (height, width, 2),
        "pad_shape": (pad_h, pad_w, 2),
        "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "flip": False,
        "flip_direction": None,
    }


class HBTXRDirectEventDataset(Dataset):
    """HBTXR event cache dataset for direct Swift-Eye detector training."""

    def __init__(
        self,
        root: Path,
        split: str,
        height: int,
        width: int,
        event_mode: str = "causal_linear",
        event_weight: int = 1,
        limit: Optional[int] = None,
        valid_only: bool = True,
    ) -> None:
        self.root = root
        self.split = split
        self.data_root = root / split / "cached_data"
        self.ellipse_root = root / split / "cached_ellipse"
        self.height = height
        self.width = width
        self.event_mode = event_mode
        self.event_weight = event_weight
        self.ellipses = cached_ellipse_records(str(self.ellipse_root.resolve()))
        total = len(self.ellipses)
        if valid_only:
            area = np.pi * self.ellipses["a"] * self.ellipses["b"] / 4.0
            finite = (
                np.isfinite(self.ellipses["x"])
                & np.isfinite(self.ellipses["y"])
                & np.isfinite(self.ellipses["a"])
                & np.isfinite(self.ellipses["b"])
                & np.isfinite(self.ellipses["ang"])
            )
            indices = np.where(finite & (self.ellipses["a"] > 0) & (self.ellipses["b"] > 0) & (area > 200.0))[0]
        else:
            indices = np.arange(total)
        if limit is not None:
            indices = indices[:limit]
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict:
        index = int(self.indices[item])
        events = load_event_segment(index, self.data_root)
        event_chw = events_to_chw(
            events,
            self.height,
            self.width,
            self.event_mode,
            self.event_weight,
        )
        obb, valid = ellipse_record_to_obb(self.ellipses[index], self.height, self.width)
        return {
            "events": torch.from_numpy(event_chw),
            "gt_bboxes": obb.reshape(1, 5) if valid else torch.zeros((0, 5), dtype=torch.float32),
            "gt_labels": torch.zeros((1,), dtype=torch.long) if valid else torch.zeros((0,), dtype=torch.long),
            "index": index,
            "timestamp": int(self.ellipses[index]["t"]),
            "split": self.split,
        }


class HBTXRDirectTemporalPairDataset(Dataset):
    """On-the-fly template/search pairs from the same HBTXR split."""

    def __init__(
        self,
        root: Path,
        split: str,
        height: int,
        width: int,
        pair_offset: int = 1,
        max_delta_us: int = 50000,
        event_mode: str = "causal_linear",
        event_weight: int = 1,
        limit: Optional[int] = None,
    ) -> None:
        self.base = HBTXRDirectEventDataset(
            root=root,
            split=split,
            height=height,
            width=width,
            event_mode=event_mode,
            event_weight=event_weight,
            limit=None,
            valid_only=False,
        )
        self.pair_offset = pair_offset
        t = self.base.ellipses["t"].astype(np.int64)
        area = np.pi * self.base.ellipses["a"] * self.base.ellipses["b"] / 4.0
        valid = (
            np.isfinite(self.base.ellipses["x"])
            & np.isfinite(self.base.ellipses["y"])
            & np.isfinite(self.base.ellipses["a"])
            & np.isfinite(self.base.ellipses["b"])
            & np.isfinite(self.base.ellipses["ang"])
            & (self.base.ellipses["a"] > 0)
            & (self.base.ellipses["b"] > 0)
            & (area > 200.0)
        )
        starts = np.arange(0, len(t) - pair_offset, dtype=np.int64)
        delta = t[starts + pair_offset] - t[starts]
        ok = valid[starts] & valid[starts + pair_offset] & (delta > 0) & (delta <= max_delta_us)
        self.starts = starts[ok]
        if limit is not None:
            self.starts = self.starts[:limit]

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, item: int) -> Dict:
        template_index = int(self.starts[item])
        search_index = template_index + self.pair_offset
        template = self.base[self._base_position(template_index)]
        search = self.base[self._base_position(search_index)]
        return {
            "template_events": template["events"],
            "search_events": search["events"],
            "template_bbox": template["gt_bboxes"][0],
            "search_bbox": search["gt_bboxes"][0],
            "template_index": template_index,
            "search_index": search_index,
        }

    def _base_position(self, index: int) -> int:
        # HBTXRDirectEventDataset with valid_only=False keeps identity indices.
        return index


def collate_detector(samples: Sequence[Dict]) -> Dict:
    """Collate direct detector samples into MMDetection forward_train inputs."""
    events = torch.stack([sample["events"] for sample in samples], dim=0)
    gt_bboxes = [sample["gt_bboxes"] for sample in samples]
    gt_labels = [sample["gt_labels"] for sample in samples]
    height, width = events.shape[-2:]
    padded_h = int(math.ceil(height / 32.0) * 32)
    padded_w = int(math.ceil(width / 32.0) * 32)
    img_metas = [
        make_img_meta(height, width, padded_h, padded_w, f"{sample['split']}_{sample['index']}")
        for sample in samples
    ]
    return {
        "events": events,
        "img_metas": img_metas,
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
    }


def collate_temporal(samples: Sequence[Dict]) -> Dict:
    """Collate direct temporal pair samples."""
    return {
        "template_events": torch.stack([sample["template_events"] for sample in samples], dim=0),
        "search_events": torch.stack([sample["search_events"] for sample in samples], dim=0),
        "template_bboxes": torch.stack([sample["template_bbox"] for sample in samples], dim=0),
        "search_bboxes": torch.stack([sample["search_bbox"] for sample in samples], dim=0),
        "template_indices": [sample["template_index"] for sample in samples],
        "search_indices": [sample["search_index"] for sample in samples],
    }


def parse_losses(losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, OrderedDict]:
    """MMDetection-style loss parser."""
    log_vars = OrderedDict()
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            log_vars[name] = value.mean()
        elif isinstance(value, list):
            log_vars[name] = sum(item.mean() for item in value)
        else:
            raise TypeError(f"{name} has unsupported loss type {type(value)!r}")
    loss = sum(value for key, value in log_vars.items() if "loss" in key)
    log_vars["loss"] = loss
    return loss, OrderedDict((key, float(value.detach().cpu())) for key, value in log_vars.items())


class DirectSwiftEyeDetector(nn.Module):
    """Swift-Eye detector with its first backbone projection changed to 2 channels."""

    def __init__(self, cfg_path: Path, args: argparse.Namespace) -> None:
        super().__init__()
        cfg = detector_cfg_for_img64(cfg_path, args)
        self.detector = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
        self.pad_divisor = args.pad_divisor

    def forward_train(self, batch: Dict) -> Tuple[torch.Tensor, OrderedDict]:
        img = batch["events"]
        img = pad_tensor_chw(img, self.pad_divisor)
        losses = self.detector.forward_train(
            img,
            batch["img_metas"],
            batch["gt_bboxes"],
            batch["gt_labels"],
        )
        return parse_losses(losses)


class DirectSwiftEyeTemporalFusion(nn.Module):
    """Swift-Eye temporal components with 2-channel backbone input and 64px crops."""

    def __init__(self, cfg_path: Path, args: argparse.Namespace) -> None:
        super().__init__()
        cfg = temporal_cfg_for_img64(cfg_path, args)
        self.backbone = MODELS.build(cfg.backbone)
        self.neck = MODELS.build(cfg.neck)
        self.correlation_head = MODELS.build(cfg.correlation_head)
        self.tracking_head = MODELS.build(cfg.tracking_head)
        self.pad_divisor = args.pad_divisor
        self.search_shape = args.temporal_search_shape
        self.template_shape = args.temporal_template_shape
        self.crop_stride = 4
        for module in (self.backbone, self.neck):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def get_features(self, events: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        img = pad_tensor_chw(events, self.pad_divisor)
        return self.neck(self.backbone(img))

    def crop_feature(
        self,
        feature: torch.Tensor,
        bbox: torch.Tensor,
        crop_shape: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, feat_h, feat_w = feature.shape
        center_x = torch.floor(bbox[0] / self.crop_stride).long()
        center_y = torch.floor(bbox[1] / self.crop_stride).long()
        half = crop_shape // 2
        x0 = torch.clamp(center_x - half, min=0, max=max(feat_w - crop_shape, 0))
        y0 = torch.clamp(center_y - half, min=0, max=max(feat_h - crop_shape, 0))
        crop = feature[:, y0 : y0 + crop_shape, x0 : x0 + crop_shape]
        return crop, torch.stack([x0, y0]).to(bbox.device)

    def forward_train(self, batch: Dict) -> Tuple[torch.Tensor, OrderedDict]:
        template_events = batch["template_events"]
        search_events = batch["search_events"]
        all_events = torch.cat([template_events, search_events], dim=0)
        features = self.get_features(all_events)[0]
        batch_size = template_events.shape[0]
        template_features = features[:batch_size]
        search_features = features[batch_size:]
        kernels = []
        searches = []
        gt_bboxes = []
        gt_labels = []
        img_metas = []
        for idx in range(batch_size):
            kernel, _ = self.crop_feature(
                template_features[idx],
                batch["template_bboxes"][idx],
                self.template_shape,
            )
            search, top_left = self.crop_feature(
                search_features[idx],
                batch["search_bboxes"][idx],
                self.search_shape,
            )
            local_bbox = batch["search_bboxes"][idx].clone()
            local_bbox[0] -= top_left[0].float() * self.crop_stride
            local_bbox[1] -= top_left[1].float() * self.crop_stride
            kernels.append(kernel)
            searches.append(search)
            gt_bboxes.append(local_bbox.reshape(1, 5))
            gt_labels.append(torch.zeros((1,), dtype=torch.long, device=local_bbox.device))
            crop_px = self.search_shape * self.crop_stride
            img_metas.append(make_img_meta(crop_px, crop_px, crop_px, crop_px, f"temporal_{idx}"))
        x = self.correlation_head(torch.stack(kernels, dim=0), torch.stack(searches, dim=0))
        losses = self.tracking_head.forward_train((x,), img_metas, gt_bboxes, gt_labels, None)
        return parse_losses(losses)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move tensor leaves in a batch dict."""
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            moved[key] = value
    return moved


def write_metadata(args: argparse.Namespace) -> Path:
    """Write a reproducible preparation metadata JSON."""
    args.output_root.mkdir(parents=True, exist_ok=True)
    metadata = DirectSwiftEyeConfig(
        data_root=str(args.data_root),
        output_root=str(args.output_root),
        detector_config=str(DETECTOR_CONFIG),
        temporal_config=str(TEMPORAL_CONFIG),
        input_height=args.input_height,
        input_width=args.input_width,
        pad_divisor=args.pad_divisor,
        event_mode=args.event_mode,
        event_weight=args.event_weight,
        input_channels=2,
        backbone_patch_embed="SwinTransformer.patch_embed.projection Conv2d(2, 96, kernel_size=4, stride=4)",
        detector_anchor_scales=args.detector_anchor_scales,
        temporal_anchor_scales=args.temporal_anchor_scales,
        temporal_search_shape=args.temporal_search_shape,
        temporal_template_shape=args.temporal_template_shape,
        split_train="subject 1-32",
        split_val="subject 33-36",
        split_test="subject 37-48",
        optimizer_detector="AdamW, matching original Swift-Eye detector config family",
        optimizer_temporal="Adam, matching original Swift-Eye temporal trainer family",
        lr_detector=args.lr_detector,
        lr_temporal=args.lr_temporal,
        weight_decay_detector=args.weight_decay_detector,
        weight_decay_temporal=args.weight_decay_temporal,
        temporal_pair_offset=args.temporal_pair_offset,
        temporal_max_delta_us=args.temporal_max_delta_us,
    )
    path = args.output_root / "direct_hbtxr_training_preparation.json"
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return path


def inspect_data(args: argparse.Namespace) -> None:
    """Print split and pair counts without building models."""
    summary = {}
    for split in args.splits:
        det = HBTXRDirectEventDataset(
            args.data_root,
            split,
            args.input_height,
            args.input_width,
            args.event_mode,
            args.event_weight,
            valid_only=True,
        )
        pairs = HBTXRDirectTemporalPairDataset(
            args.data_root,
            split,
            args.input_height,
            args.input_width,
            args.temporal_pair_offset,
            args.temporal_max_delta_us,
            args.event_mode,
            args.event_weight,
        )
        summary[split] = {
            "valid_detector_samples": len(det),
            "valid_temporal_pairs": len(pairs),
        }
    print(json.dumps(summary, indent=2))


def smoke_detector(args: argparse.Namespace) -> None:
    """Run a small detector forward/backward smoke test."""
    dataset = HBTXRDirectEventDataset(
        args.data_root,
        args.split,
        args.input_height,
        args.input_width,
        args.event_mode,
        args.event_weight,
        limit=args.limit,
        valid_only=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_detector)
    model = DirectSwiftEyeDetector(DETECTOR_CONFIG, args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_detector,
        weight_decay=args.weight_decay_detector,
    )
    model.train()
    for step, batch in enumerate(loader):
        if step >= args.dry_run_batches:
            break
        batch = move_batch_to_device(batch, args.device)
        loss, logs = model.forward_train(batch)
        if args.backward:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        LOG.info("detector step=%d logs=%s", step, dict(logs))


def smoke_temporal(args: argparse.Namespace) -> None:
    """Run a small temporal-fusion forward/backward smoke test."""
    dataset = HBTXRDirectTemporalPairDataset(
        args.data_root,
        args.split,
        args.input_height,
        args.input_width,
        args.temporal_pair_offset,
        args.temporal_max_delta_us,
        args.event_mode,
        args.event_weight,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_temporal)
    model = DirectSwiftEyeTemporalFusion(TEMPORAL_CONFIG, args).to(args.device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr_temporal,
        weight_decay=args.weight_decay_temporal,
    )
    model.train()
    for step, batch in enumerate(loader):
        if step >= args.dry_run_batches:
            break
        batch = move_batch_to_device(batch, args.device)
        loss, logs = model.forward_train(batch)
        if args.backward:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        LOG.info("temporal step=%d logs=%s", step, dict(logs))


def append_history(path: Path, row: Dict) -> None:
    """Append one JSONL history row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    extra: Dict,
) -> None:
    """Save a minimal PyTorch training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "extra": extra,
        },
        path,
    )


def train_detector(args: argparse.Namespace) -> None:
    """Train detector stage without launching MMRotate runners or DOTA data."""
    dataset = HBTXRDirectEventDataset(
        args.data_root,
        args.split,
        args.input_height,
        args.input_width,
        args.event_mode,
        args.event_weight,
        limit=args.limit,
        valid_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detector,
        drop_last=False,
    )
    model = DirectSwiftEyeDetector(DETECTOR_CONFIG, args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr_detector,
        weight_decay=args.weight_decay_detector,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )
    history_path = args.output_root / "detector_history.jsonl"
    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        iterator: Iterable = tqdm(loader, desc=f"detector epoch {epoch}", unit="batch")
        for batch in iterator:
            batch = move_batch_to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = model.forward_train(batch)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                row = {"stage": "detector", "epoch": epoch, "step": global_step, **dict(logs)}
                append_history(history_path, row)
                iterator.set_postfix(loss=f"{logs['loss']:.4f}")
        scheduler.step()
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                args.output_root / "checkpoints" / f"detector_epoch_{epoch:03d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                {"stage": "detector", "config": str(DETECTOR_CONFIG)},
            )


def train_temporal(args: argparse.Namespace) -> None:
    """Train temporal fusion stage directly from HBTXR event pairs."""
    dataset = HBTXRDirectTemporalPairDataset(
        args.data_root,
        args.split,
        args.input_height,
        args.input_width,
        args.temporal_pair_offset,
        args.temporal_max_delta_us,
        args.event_mode,
        args.event_weight,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_temporal,
        drop_last=False,
    )
    model = DirectSwiftEyeTemporalFusion(TEMPORAL_CONFIG, args).to(args.device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.lr_temporal,
        weight_decay=args.weight_decay_temporal,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )
    history_path = args.output_root / "temporal_history.jsonl"
    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        iterator: Iterable = tqdm(loader, desc=f"temporal epoch {epoch}", unit="batch")
        for batch in iterator:
            batch = move_batch_to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = model.forward_train(batch)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % args.log_interval == 0:
                row = {"stage": "temporal", "epoch": epoch, "step": global_step, **dict(logs)}
                append_history(history_path, row)
                iterator.set_postfix(loss=f"{logs['loss']:.4f}")
        scheduler.step()
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                args.output_root / "checkpoints" / f"temporal_epoch_{epoch:03d}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                {"stage": "temporal", "config": str(TEMPORAL_CONFIG)},
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "inspect",
            "smoke-detector",
            "smoke-temporal",
            "train-detector",
            "train-temporal",
            "write-metadata",
        ],
        default="inspect",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--input-height", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--input-width", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--pad-divisor", type=int, default=32)
    parser.add_argument("--event-mode", choices=["causal_linear", "nearest", "bilinear"], default="causal_linear")
    parser.add_argument("--event-weight", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run-batches", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--scheduler-step", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.7)
    parser.add_argument("--lr-detector", type=float, default=1e-4)
    parser.add_argument("--lr-temporal", type=float, default=1e-4)
    parser.add_argument("--weight-decay-detector", type=float, default=5e-2)
    parser.add_argument("--weight-decay-temporal", type=float, default=0.0)
    parser.add_argument("--detector-anchor-scales", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--temporal-anchor-scales", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--temporal-search-shape", type=int, default=15)
    parser.add_argument("--temporal-template-shape", type=int, default=7)
    parser.add_argument("--temporal-pair-offset", type=int, default=1)
    parser.add_argument("--temporal-max-delta-us", type=int, default=50000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.device = torch.device(args.device)
    if args.temporal_template_shape >= args.temporal_search_shape:
        raise ValueError("--temporal-template-shape must be smaller than --temporal-search-shape")
    if args.temporal_search_shape % 2 == 0 or args.temporal_template_shape % 2 == 0:
        raise ValueError("temporal crop shapes should be odd to keep a centered crop")
    min_feature_size = min(
        int(math.ceil(args.input_height / 4.0)),
        int(math.ceil(args.input_width / 4.0)),
    )
    if args.temporal_search_shape > min_feature_size:
        raise ValueError(
            f"--temporal-search-shape={args.temporal_search_shape} exceeds "
            f"P2 feature size {min_feature_size} for {args.input_height}x{args.input_width}"
        )
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    metadata_path = write_metadata(args)
    LOG.info("wrote preparation metadata: %s", metadata_path)
    if args.mode == "inspect":
        inspect_data(args)
    elif args.mode == "smoke-detector":
        smoke_detector(args)
    elif args.mode == "smoke-temporal":
        smoke_temporal(args)
    elif args.mode == "train-detector":
        train_detector(args)
    elif args.mode == "train-temporal":
        train_temporal(args)
    elif args.mode == "write-metadata":
        print(metadata_path)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
