"""Model factory: build a reimplemented HBTXR model from a config block.

Maps the ``model:`` block of a modality config (see configs/modality/*/*.yaml) to
one of the reimplemented models (models.frame / event / hybrid / mask). Replaces
the old detector/tracker registry, which is parked in models/tmp.

    from engine.model_factory import make_model
    model = make_model(config["model"])   # e.g. {"target": "models.frame.FrameModel", ...}
"""
from __future__ import annotations

import dataclasses
from typing import Any

from torch import nn

from models.backbones import ViTConfig
from models.event import EventModel, EventModelConfig
from models.frame import FrameModel, FrameModelConfig
from models.hybrid import HybridModel, HybridModelConfig
from models.hybrid.scheduler import SchedulerConfig
from models.mask import MaskModel, MaskModelConfig

# accept either the modality name or the target class name (case-insensitive)
MODEL_REGISTRY: dict[str, tuple[type[nn.Module], type]] = {
    "frame": (FrameModel, FrameModelConfig),
    "framemodel": (FrameModel, FrameModelConfig),
    "event": (EventModel, EventModelConfig),
    "eventmodel": (EventModel, EventModelConfig),
    "hybrid": (HybridModel, HybridModelConfig),
    "hybridmodel": (HybridModel, HybridModelConfig),
    "mask": (MaskModel, MaskModelConfig),
    "maskmodel": (MaskModel, MaskModelConfig),
}

# nested sub-config dataclasses keyed by the config field that carries them
_NESTED: dict[str, type] = {"backbone": ViTConfig, "scheduler": SchedulerConfig}


def _resolve_key(cfg: dict[str, Any]) -> str:
    target = str(cfg.get("target", "")).strip()
    if target:
        return target.rsplit(".", 1)[-1].lower()
    return str(cfg.get("name", "")).strip().lower()


def _build_nested(nested_cls: type, value: dict[str, Any]):
    allowed = {f.name for f in dataclasses.fields(nested_cls)}
    return nested_cls(**{k: v for k, v in value.items() if k in allowed})


def _build_config(config_cls: type, cfg: dict[str, Any]):
    fields = {f.name for f in dataclasses.fields(config_cls)}
    kwargs: dict[str, Any] = {}
    for key, value in cfg.items():
        if key in ("target", "name") or key not in fields:
            continue
        if key in _NESTED and isinstance(value, dict):
            kwargs[key] = _build_nested(_NESTED[key], value)
        else:
            kwargs[key] = value
    return config_cls(**kwargs)


def make_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a model from its config dict (the ``model:`` block)."""
    key = _resolve_key(cfg)
    if key not in MODEL_REGISTRY:
        known = ", ".join(sorted(set(MODEL_REGISTRY)))
        raise ValueError(f"unknown model {key!r}; known: {known}")
    model_cls, config_cls = MODEL_REGISTRY[key]
    return model_cls(_build_config(config_cls, cfg))


def list_model_names() -> list[str]:
    return sorted({name for name in MODEL_REGISTRY if not name.endswith("model")})
