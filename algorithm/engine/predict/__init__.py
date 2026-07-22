"""engine.predict — prediction == inference for the reimplemented models.

The paper has a single detect/predict path, so prediction reuses engine.infer:
``predict_batch`` for a single forward and ``run_predict`` (= run_infer) as the CLI.
"""
from engine.infer.inference import predict_batch

__all__ = ["predict_batch", "run_predict"]


def run_predict(*args, **kwargs):
    """Lazy proxy to engine.infer.entrypoint.run_infer (predict == infer)."""
    from engine.infer.entrypoint import run_infer as _run_infer

    return _run_infer(*args, **kwargs)
