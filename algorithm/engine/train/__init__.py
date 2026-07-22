"""engine.train — training losses and loop for the reimplemented HBTXR models.

- losses.compute_losses: box / ellipse / mask / reliability terms for the new heads.
- Trainer / TrainConfig: modality-aware training step + fit loop (direct detectors
  vs the hybrid search/track branches).
"""
from engine.train.losses import compute_losses
from engine.train.trainer import Trainer, TrainConfig

__all__ = ["compute_losses", "Trainer", "TrainConfig"]
