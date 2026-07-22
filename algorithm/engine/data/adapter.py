"""Batch adapter: HBTXR sample contract -> reimplemented-model I/O contract (P5.5).

The HBTXR dataset (``dataset.hbtxr``) emits a rich sample dict designed for the
old detector/tracker (frame, event, mask_target, cur_state/prev_state as 6D
``state6`` = (x, y, a, b, u, v), pupil_search_target, ...). The reimplemented
models consume a smaller, explicit contract:

- direct frame / event detector -> ``model(image)`` with targets ``box`` (5D
  oriented box), ``mask``, ``reliability`` (2D), ``eye_box`` (4D).
- hybrid -> ``search_step(frame)`` / ``track_step(event, anchor_state)`` with a
  search ``box`` target and a track ``residual`` target relative to the GT anchor.
- mask -> ``model(image)`` with a ``mask`` target only.

``adapt_batch`` runs on an already-collated batch (batch dim present) so every
conversion is a vectorized tensor op. The geometry is owned by ``models.geometry``
(state canonicalization + ``state_to_box``) and ``utils.state6`` (the 6D<->5D
angle codec); this module only routes fields per modality. ``sample_id`` / ``meta``
are carried through unchanged for logging.
"""
from __future__ import annotations

from typing import Any

import torch

from models.geometry import canonicalize, state_to_box
from utils.state6 import xyabuv_to_xywht

_DIRECT = {"frame", "event"}


def _state6_to_state5(state6: torch.Tensor) -> torch.Tensor:
    """(B, 6) state6 (x, y, a, b, u, v) -> canonical (B, 5) state (x, y, a, b, theta)."""
    return canonicalize(xyabuv_to_xywht(state6))


def _reliability_target(batch: dict[str, Any]) -> torch.Tensor:
    """(B, 2) = (confidence, annotation quality), both in [0, 1] for the BCE head."""
    search_target = batch["pupil_search_target"]              # (B, 7) = state6 + confidence
    confidence = search_target[..., 6:7]                      # (B, 1)
    quality = batch["annotation_quality"].reshape(-1, 1)      # (B, 1)
    return torch.cat([confidence, quality], dim=-1).clamp(0.0, 1.0)


def resolve_modality(cfg: dict[str, Any]) -> str:
    """Infer the training modality from a ``model:`` config block (target or name)."""
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    target = str(model_cfg.get("target", "")).strip() or str(model_cfg.get("name", "")).strip()
    token = target.rsplit(".", 1)[-1].lower()
    for modality in ("frame", "event", "hybrid", "mask"):
        if modality in token:
            return modality
    return "frame"


def adapt_batch(batch: dict[str, Any], modality: str) -> dict[str, Any]:
    """Map one collated HBTXR batch to the reimplemented-model I/O contract."""
    modality = str(modality).lower()
    cur5 = _state6_to_state5(batch["cur_state"])
    out: dict[str, Any] = {}

    if modality in _DIRECT:
        out["image"] = batch["frame"] if modality == "frame" else batch["event"]
        out["box"] = state_to_box(cur5)
        out["state"] = cur5
        out["mask"] = batch["mask_target"]
        out["reliability"] = _reliability_target(batch)
        out["eye_box"] = batch["eye_target"][..., :4]
    elif modality == "mask":
        out["image"] = batch["frame"]
        out["mask"] = batch["mask_target"]
    elif modality == "hybrid":
        prev5 = _state6_to_state5(batch["prev_state"])
        out["frame"] = batch["frame"]
        out["event"] = batch["event"]
        out["box"] = state_to_box(cur5)          # search head target
        out["anchor_state"] = prev5              # GT anchor (teacher forcing)
        out["residual"] = cur5 - prev5           # additive residual target: Pi(prev + d) == cur
        out["state"] = cur5
        out["reliability"] = _reliability_target(batch)
        out["eye_box"] = batch["eye_target"][..., :4]
    else:
        raise ValueError(f"unknown modality {modality!r}; expected frame|event|hybrid|mask")

    for passthrough in ("sample_id", "meta"):
        if passthrough in batch:
            out[passthrough] = batch[passthrough]
    return out


__all__ = ["adapt_batch", "resolve_modality"]
