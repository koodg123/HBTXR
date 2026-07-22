"""Data-loading factory for the reimplemented models (P5.5).

Reuses the existing HBTXR data pipeline (``dataset.hbtxr.loader.make_loader_by_mode``:
mode resolution, weighted sampler, collate) and layers the model-input adapter on
top so the reimplemented ``engine.train.Trainer`` receives batches in its contract
(image / frame / event / box / residual / state / anchor_state / reliability / mask).

``build_dataloader(manifest_path, cfg, shuffle=...)`` returns an ``AdaptedLoader`` —
an ``Iterable[dict]`` that is a drop-in for ``Trainer.fit``. The training modality
is taken from ``cfg['model']`` unless passed explicitly.
"""
from __future__ import annotations

from typing import Any, Iterator

from torch.utils.data import DataLoader

from dataset.hbtxr.loader import make_dataset_by_mode, make_loader_by_mode

from engine.data.adapter import adapt_batch, resolve_modality


class AdaptedLoader:
    """Wrap a DataLoader and adapt each collated batch to the model I/O contract."""

    def __init__(self, base_loader: DataLoader, modality: str) -> None:
        self.base_loader = base_loader
        self.modality = str(modality).lower()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch in self.base_loader:
            yield adapt_batch(batch, self.modality)

    def __len__(self) -> int:
        return len(self.base_loader)


def build_dataloader(
    manifest_path: str,
    cfg: dict[str, Any],
    *,
    shuffle: bool,
    modality: str | None = None,
) -> AdaptedLoader:
    """Build a model-ready loader: HBTXR pipeline + collate + per-batch adapter."""
    resolved_modality = str(modality).lower() if modality else resolve_modality(cfg)
    base_loader = make_loader_by_mode(manifest_path, cfg, shuffle=shuffle)
    return AdaptedLoader(base_loader, resolved_modality)


__all__ = ["AdaptedLoader", "build_dataloader", "make_dataset_by_mode"]
