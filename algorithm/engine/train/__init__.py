"""engine.train — training losses and loop for the reimplemented HBTXR models.

- losses.compute_losses: box / ellipse / mask / reliability terms for the new heads.
- Trainer / TrainConfig: modality-aware training step + fit loop (direct detectors
  vs the hybrid search/track branches).
"""
from engine.train.losses import compute_losses
from engine.train.trainer import Trainer, TrainConfig

__all__ = ["compute_losses", "Trainer", "TrainConfig", "run_train"]


def run_train(*args, **kwargs):
    """Lazy proxy to engine.train.entrypoint.run_train (avoids importing torch data
    stack at package import time)."""
    from engine.train.entrypoint import run_train as _run_train

    return _run_train(*args, **kwargs)
