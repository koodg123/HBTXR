from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def make_dataset_by_mode(manifest_path: str, cfg: dict[str, Any], *, training: bool = False) -> EVEyeHBTXRDataset:
    data_cfg = cfg.get("data") or {}
    dataset_kwargs = build_dataset_kwargs(data_cfg)
    augmentation = dict(dataset_kwargs.get("augmentation") or {})
    if not training:
        augmentation["enabled"] = False
    dataset_kwargs["augmentation"] = augmentation
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_float(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _multiply_if_text_contains(weight: float, row: dict[str, Any], cfg: dict[str, Any]) -> float:
    field = str(cfg.get("field", "session_key"))
    needle = str(cfg.get("contains", "")).strip()
    if not needle:
        return weight
    text = str(row.get(field, ""))
    if needle in text:
        return weight * max(0.0, _as_float(cfg.get("multiplier"), 1.0))
    return weight


def _sample_weight_for_row(row: dict[str, Any], sampler_cfg: dict[str, Any]) -> float:
    min_weight = max(0.0, _as_float(sampler_cfg.get("min_weight"), 1.0e-6))
    weight = max(min_weight, _as_float(sampler_cfg.get("base_weight"), 1.0))

    low_sim_cfg = dict(sampler_cfg.get("low_similarity") or {})
    if _as_bool(low_sim_cfg.get("enabled"), default=False):
        field = str(low_sim_cfg.get("field", "similarity_target"))
        similarity = _row_float(row, field)
        threshold = _as_float(low_sim_cfg.get("threshold"), 0.1)
        if similarity is not None and similarity <= threshold:
            weight *= max(0.0, _as_float(low_sim_cfg.get("multiplier"), 1.0))

    session_cfg = sampler_cfg.get("session_contains")
    if isinstance(session_cfg, str):
        session_cfg = {"contains": session_cfg}
    if isinstance(session_cfg, dict):
        weight = _multiply_if_text_contains(weight, row, session_cfg)

    for text_cfg in sampler_cfg.get("text_contains") or []:
        if isinstance(text_cfg, dict):
            weight = _multiply_if_text_contains(weight, row, text_cfg)

    max_multiplier = sampler_cfg.get("max_multiplier")
    if max_multiplier is not None:
        weight = min(weight, max(0.0, _as_float(max_multiplier, weight)))
    return max(min_weight, weight)


def build_sample_weights_from_rows(rows: list[dict[str, Any]], sampler_cfg: dict[str, Any]) -> torch.DoubleTensor:
    weights = [_sample_weight_for_row(row, sampler_cfg) for row in rows]
    if not weights:
        raise ValueError("weighted sampler requested for empty dataset")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("weighted sampler produced all-zero weights")
    return torch.as_tensor(weights, dtype=torch.double)


def _make_weighted_sampler(dataset: EVEyeHBTXRDataset, training_cfg: dict[str, Any]) -> WeightedRandomSampler | None:
    sampler_cfg = dict(training_cfg.get("sampler") or {})
    sampler_type = str(sampler_cfg.get("type", "weighted_failure_bucket")).strip().lower()
    if not _as_bool(sampler_cfg.get("enabled"), default=False):
        return None
    if sampler_type not in {"weighted_failure_bucket", "weighted", "weighted_random"}:
        raise ValueError(f"Unsupported training.sampler.type: {sampler_type}")

    rows = list(getattr(dataset, "rows", []))
    weights = build_sample_weights_from_rows(rows, sampler_cfg)
    num_samples = int(sampler_cfg.get("num_samples") or len(rows))
    if num_samples <= 0:
        raise ValueError(f"training.sampler.num_samples must be positive, got {num_samples}")
    generator = torch.Generator()
    generator.manual_seed(int(training_cfg.get("seed", 42)))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=_as_bool(sampler_cfg.get("replacement"), default=True),
        generator=generator,
    )


def make_loader_by_mode(manifest_path: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    dataset = make_dataset_by_mode(manifest_path, cfg, training=bool(shuffle))
    training_cfg = cfg.get("training") or {}
    sampler = _make_weighted_sampler(dataset, training_cfg) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=_resolve_pin_memory(training_cfg),
        collate_fn=collate_samples,
    )


def make_loader(manifest_path: str, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    return make_loader_by_mode(manifest_path, cfg, shuffle)
