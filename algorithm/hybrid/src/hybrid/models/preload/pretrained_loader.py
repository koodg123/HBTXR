from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _extract_state_dict(payload: Any, source: str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        if all(torch.is_tensor(value) for value in payload.values()):
            return {str(key): value for key, value in payload.items()}
        preferred_keys = []
        if source == "deit":
            preferred_keys.extend(["model", "state_dict"])
        preferred_keys.extend(["state_dict", "model", "student", "teacher", "net", "module"])
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(item) for item in value.values()):
                return {str(inner_key): inner_value for inner_key, inner_value in value.items() if torch.is_tensor(inner_value)}
    raise ValueError("Could not locate a tensor state_dict in checkpoint payload")


def _normalize_key(key: str) -> str:
    normalized = str(key)
    for prefix in ("module.", "model.", "student.", "teacher.", "net."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def _slice_tensor_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor | None:
    if tensor.ndim != len(target_shape):
        return None
    if any(source_dim < target_dim for source_dim, target_dim in zip(tensor.shape, target_shape)):
        return None
    slices = tuple(slice(0, int(target_dim)) for target_dim in target_shape)
    return tensor[slices].clone()


def load_pretrained_weights(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    source: str = "auto",
    strict_shape: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    source_state = _extract_state_dict(payload, str(source).strip().lower())
    normalized_source_state = {_normalize_key(key): value for key, value in source_state.items()}
    mapper = getattr(model, "map_pretrained_state_dict", None)
    if callable(mapper):
        mapped_state = mapper(normalized_source_state, source=str(source).strip().lower())
        if isinstance(mapped_state, dict):
            normalized_source_state = {str(key): value for key, value in mapped_state.items() if torch.is_tensor(value)}
    target_state = model.state_dict()

    loadable: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []
    unmatched_source: list[str] = []
    partially_loaded: list[dict[str, Any]] = []

    for key, tensor in normalized_source_state.items():
        if key not in target_state:
            unmatched_source.append(key)
            continue
        if tuple(target_state[key].shape) != tuple(tensor.shape):
            sliced = None if strict_shape else _slice_tensor_to_shape(tensor, tuple(target_state[key].shape))
            if sliced is not None:
                loadable[key] = sliced
                partially_loaded.append(
                    {
                        "key": key,
                        "source_shape": list(tensor.shape),
                        "target_shape": list(target_state[key].shape),
                        "method": "prefix_slice",
                    }
                )
                continue
            skipped.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state[key].shape),
                }
            )
            if strict_shape:
                continue
            continue
        loadable[key] = tensor

    missing_target = sorted(set(target_state.keys()) - set(loadable.keys()))
    incompatible = model.load_state_dict(loadable, strict=False)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "source": str(source),
        "loaded_keys": sorted(loadable.keys()),
        "loaded_count": len(loadable),
        "partially_loaded": partially_loaded,
        "skipped": skipped,
        "missing_target_keys": missing_target,
        "unexpected_keys": sorted(getattr(incompatible, "unexpected_keys", [])),
        "missing_keys_after_load": sorted(getattr(incompatible, "missing_keys", [])),
        "unmatched_source_keys": sorted(unmatched_source),
        "strict_shape": bool(strict_shape),
    }
    return report


def write_pretrained_report(path: str | Path, report: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
