from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from _bootstrap import PROJECT_ROOT
from _config import apply_config_overrides, build_dataset_kwargs, load_config, resolve_training_entry

from fecet_hbtxr.dataset import FECETHBTXRDataset


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in {1, 2, 3}:
        arr = np.moveaxis(arr, 0, -1)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max - arr_min < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - arr_min) / (arr_max - arr_min) * 255.0, 0.0, 255.0).astype(np.uint8)


def _event_to_rgb(event: np.ndarray) -> np.ndarray:
    pos = np.clip(event[1], 0.0, None)
    neg = np.clip(event[0], 0.0, None)
    pos_u8 = _normalize_to_uint8(pos)
    neg_u8 = _normalize_to_uint8(neg)
    zeros = np.zeros_like(pos_u8)
    return np.stack([pos_u8, zeros, neg_u8], axis=-1)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a sample from a FECET-HBTXR manifest")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "runs" / "visualize" / "sample.png"))
    parser.add_argument("--override", action="append", default=[])
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    cfg = apply_config_overrides(cfg, overrides=args.override)
    entry = resolve_training_entry(cfg, config_path=args.config, project_root=PROJECT_ROOT)
    manifest = args.manifest or entry["train_manifest"]
    dataset = FECETHBTXRDataset(str(manifest), **build_dataset_kwargs(cfg.get("data") or {}))
    sample = dataset[args.index]

    frame = _normalize_to_uint8(sample["frame"].numpy())
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    mask = _normalize_to_uint8(sample["mask_target"].numpy())
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.repeat(mask, 3, axis=-1)
    event = _event_to_rgb(sample["event"].numpy())

    frame_img = Image.fromarray(frame)
    event_img = Image.fromarray(event)
    mask_img = Image.fromarray(mask)
    canvas = Image.new("RGB", (frame_img.width * 3, frame_img.height), color=(16, 16, 16))
    canvas.paste(frame_img, (0, 0))
    canvas.paste(event_img, (frame_img.width, 0))
    canvas.paste(mask_img, (frame_img.width * 2, 0))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "manifest": str(Path(manifest).resolve()),
        "sample_id": sample["sample_id"],
        "output": str(output),
        "selected_event_count": int(sample["meta"]["selected_event_count"]),
    }


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
