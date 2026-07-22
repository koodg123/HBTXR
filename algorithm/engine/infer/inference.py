"""Inference for the reimplemented HBTXR models (P5.7).

Two modes:

- ``predict_batch``: one forward per batch, modality-aware. Direct frame/event ->
  ``model(image)``; hybrid -> ``search_step`` (frame) or ``track_step`` (event +
  anchor); mask -> logit map. Returns the model output dict (state / box / mask).
- ``stream_infer``: the paper runtime path — drive ``model.run_step`` over an ordered
  stream so the CPU-side scheduler (search vs track FSM) selects the branch per step
  and carries the anchor state forward. Yields ``{mode, state}`` per step.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


@torch.no_grad()
def predict_batch(model: nn.Module, batch: dict[str, Any], modality: str) -> dict[str, Any]:
    """One forward pass; returns the model output dict for the batch."""
    modality = str(modality).lower()
    if modality.startswith("hybrid"):
        if "frame" in batch:
            return model.search_step(batch["frame"])
        anchor = batch.get("anchor_state")
        if anchor is None and "state" in batch:
            anchor = batch["state"]
        return model.track_step(batch["event"], anchor)
    outputs = model(batch["image"])
    if torch.is_tensor(outputs):
        return {"mask": outputs}
    return outputs


@torch.no_grad()
def stream_infer(model: nn.Module, steps: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the scheduler FSM over an ordered stream of ``{frame?, event?}`` steps.

    Requires a hybrid model (``run_step`` / ``initial_scheduler_state``). Returns a
    per-step list of ``{mode, state}`` with the selected branch mode.
    """
    sched_state = model.initial_scheduler_state()
    results: list[dict[str, Any]] = []
    for step in steps:
        out, sched_state = model.run_step(
            sched_state,
            frame=step.get("frame"),
            event=step.get("event"),
        )
        results.append({"mode": out.get("mode"), "state": out.get("state")})
    return results


__all__ = ["predict_batch", "stream_infer"]
