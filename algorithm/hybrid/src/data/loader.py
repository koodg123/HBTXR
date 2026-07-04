from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config.runtime_config import build_dataset_kwargs, resolve_data_mode

from .dataset import EVEyeHBTXRDataset, Mode0Dataset, Mode1Dataset, Mode2Dataset


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.stack(values, dim=0)
        elif key in {"sample_id", "meta"}:
            collated[key] = values
        else:
            collated[key] = values
    return collated


def make_dataset_by_mode(manifest_path: str, cfg: dict[str, Any]) -> EVEyeHBTXRDataset:
    data_cfg = cfg.get("data") or {}
    dataset_kwargs = build_dataset_kwargs(data_cfg)
    mode = resolve_data_mode(data_cfg)
    dataset_cls = {"mode0": Mode0Dataset, "mode1": Mode1Dataset, "mode2": Mode2Dataset}[mode]
    return dataset_cls(manifest_path, **dataset_kwargs)


def _resolve_pin_memory(training_cfg: dict[str, Any]) -> bool:
    raw = training_cfg.get("pin_memory", "auto")
    if isinstance(raw, bool):
        return raw and torch.cuda.is_available()
    text = str(raw).strip().lower()
    if text in {"false", "0", "off", "no"}:
        return False
    if text in {"true", "1", "on", "yes"}:
        return torch.cuda.is_available()
    device_spec = str(training_cfg.get("device", "cpu")).strip().lower()
    return torch.cuda.is_available() and ("cuda" in device_spec or device_spec == "multi-gpu")


def make_loader_by_mode(manifest_path: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = make_dataset_by_mode(manifest_path, cfg)
    training_cfg = cfg.get("training") or {}
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=bool(shuffle),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=_resolve_pin_memory(training_cfg),
        collate_fn=collate_samples,
    )


def make_loader(manifest_path: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return make_loader_by_mode(manifest_path, cfg, shuffle)
