"""Checkpoint save/load for the reimplemented models (P5.7).

A minimal, framework-free checkpoint format: ``{"state_dict": ..., "meta": {...}}``.
``load_checkpoint`` also accepts a bare ``state_dict`` or common wrapper keys
(``state_dict`` / ``model`` / ``model_state_dict``) so external checkpoints load too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

_STATE_KEYS = ("state_dict", "model_state_dict", "model")


def save_checkpoint(model: nn.Module, path: str | Path, *, meta: dict[str, Any] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": dict(meta or {})}, out)
    return out


def load_checkpoint(model: nn.Module, path: str | Path, *, map_location: str = "cpu", strict: bool = True) -> nn.Module:
    payload = torch.load(str(path), map_location=map_location)
    state_dict = payload
    if isinstance(payload, dict):
        for key in _STATE_KEYS:
            if key in payload and isinstance(payload[key], dict):
                state_dict = payload[key]
                break
    model.load_state_dict(state_dict, strict=strict)
    return model


__all__ = ["save_checkpoint", "load_checkpoint"]
