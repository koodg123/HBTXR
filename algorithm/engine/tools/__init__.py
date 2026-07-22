"""engine.tools — small launcher utilities (config loading, checkpoints).

``load_config`` is torch-free and imported eagerly; the checkpoint helpers pull in
torch, so they are exposed lazily to keep ``from engine.tools.load_config import
load_config`` (and config-only tooling) importable without torch installed.
"""
from engine.tools.load_config import load_config

__all__ = ["load_config", "save_checkpoint", "load_checkpoint"]


def __getattr__(name: str):
    if name in ("save_checkpoint", "load_checkpoint"):
        from engine.tools import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
