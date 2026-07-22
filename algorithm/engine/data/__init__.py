"""engine.data — feed HBTXR dataset samples to the reimplemented models.

- adapter.adapt_batch: map a collated HBTXR sample batch to the model I/O contract
  (image / frame / event / box / residual / state / anchor_state / reliability / mask).
- factory.build_dataloader: HBTXR data pipeline (dataset.hbtxr.loader) wrapped with
  the adapter, returning an AdaptedLoader that Trainer.fit consumes directly.
"""
from engine.data.adapter import adapt_batch, resolve_modality
from engine.data.factory import AdaptedLoader, build_dataloader

__all__ = ["adapt_batch", "resolve_modality", "AdaptedLoader", "build_dataloader"]
