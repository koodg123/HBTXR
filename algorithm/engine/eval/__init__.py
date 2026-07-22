"""engine.eval — evaluation loop + metrics for the reimplemented models.

- evaluator.evaluate: average per-term losses + contract metrics over a dataloader.
- metrics: center_distance, mask_iou (contract-native).
- entrypoint.run_eval: load config + checkpoint and evaluate on val/test.
"""
from engine.eval.evaluator import evaluate
from engine.eval.metrics import center_distance, mask_iou

__all__ = ["evaluate", "center_distance", "mask_iou", "run_eval"]


def run_eval(*args, **kwargs):
    """Lazy proxy to engine.eval.entrypoint.run_eval."""
    from engine.eval.entrypoint import run_eval as _run_eval

    return _run_eval(*args, **kwargs)
