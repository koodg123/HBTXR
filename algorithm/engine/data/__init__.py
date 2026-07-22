"""engine.data — feed HBTXR dataset samples to the reimplemented models.

- adapter.adapt_batch: map a collated HBTXR sample batch to the model I/O contract
  (image / frame / event / box / residual / state / anchor_state / reliability / mask).
- factory.build_dataloader: HBTXR data pipeline (dataset.hbtxr.loader) wrapped with
  the adapter, returning an AdaptedLoader that Trainer.fit consumes directly.

``adapter`` only needs torch + the geometry/state codecs, so it is imported eagerly;
``factory`` pulls the full dataset pipeline (PIL / cv2 / ...), so it is exposed lazily
to keep ``from engine.data.adapter import ...`` (and the contract mapping) importable
without the data-pipeline dependencies installed.
"""
from engine.data.adapter import adapt_batch, resolve_modality

__all__ = ["adapt_batch", "resolve_modality", "AdaptedLoader", "build_dataloader"]


def __getattr__(name: str):
    if name in ("AdaptedLoader", "build_dataloader"):
        from engine.data import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
