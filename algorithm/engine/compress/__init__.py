"""engine.compress — model compression for the reimplemented models.

- pruning.prune_model / remove_pruning / sparsity_report: native torch pruning.
- entrypoint.run_compress: load config + checkpoint, prune, report, save.
"""
from engine.compress.pruning import prune_model, remove_pruning, sparsity_report

__all__ = ["prune_model", "remove_pruning", "sparsity_report", "run_compress"]


def run_compress(*args, **kwargs):
    """Lazy proxy to engine.compress.entrypoint.run_compress."""
    from engine.compress.entrypoint import run_compress as _run_compress

    return _run_compress(*args, **kwargs)
