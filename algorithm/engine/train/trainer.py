"""Training loop for the reimplemented HBTXR models (P5.5).

A small modality-aware trainer that drives the new models (models.frame / event /
hybrid / mask) built by engine.model_factory:

- direct detectors (frame / event) and the mask model: one forward on the input
  image + compute_losses on the head outputs.
- hybrid: the search and track branches are supervised on their own steps
  (the runtime scheduler is inference-only); their losses are summed.

Optimizer / LR-scheduler come from common.optim / common.schedulers. This is the
core training step + fit loop; data loading (dataset.*) and richer callbacks
(engine.callback / logger) are wired by the entrypoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from engine.train.losses import compute_losses


@dataclass
class TrainConfig:
    modality: str = "frame"          # frame | event | hybrid | mask
    epochs: int = 1
    device: str = "cpu"
    loss_weights: dict[str, float] | None = None


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainConfig | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.model = model.to(self.config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.is_hybrid = self.config.modality.lower().startswith("hybrid")

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        dev = self.config.device
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def training_step(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        losses = self._hybrid_losses(batch) if self.is_hybrid else self._direct_losses(batch)
        total = losses["total"]
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return losses

    def _direct_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        outputs = self.model(batch["image"])
        if torch.is_tensor(outputs):  # mask model returns a bare logit map
            outputs = {"mask": outputs}
        return compute_losses(outputs, batch, self.config.loss_weights)

    def _hybrid_losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        merged: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=self.config.device)
        if "frame" in batch:
            out_s = self.model.search_step(batch["frame"])
            ls = compute_losses(out_s, batch, self.config.loss_weights)
            total = total + ls["total"]
            merged.update({f"search_{k}": v for k, v in ls.items() if k != "total"})
        if "event" in batch:
            anchor = batch.get("anchor_state")
            if anchor is None and "state" in batch:
                anchor = batch["state"]  # GT-anchor teacher forcing (anchor_source=gt)
            out_t = self.model.track_step(batch["event"], anchor)
            lt = compute_losses(out_t, batch, self.config.loss_weights)
            total = total + lt["total"]
            merged.update({f"track_{k}": v for k, v in lt.items() if k != "total"})
        merged["total"] = total
        return merged

    def fit(self, dataloader: Iterable[dict[str, Any]]) -> None:
        for _ in range(self.config.epochs):
            self.model.train()
            for batch in dataloader:
                self.training_step(batch)
            if self.scheduler is not None:
                self.scheduler.step()
