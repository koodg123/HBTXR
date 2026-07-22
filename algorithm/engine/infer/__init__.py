"""engine.infer — inference for the reimplemented models.

- inference.predict_batch: modality-aware single forward pass.
- inference.stream_infer: hybrid runtime FSM streaming over an ordered stream.
- entrypoint.run_infer: load config + checkpoint, run over a test manifest, write states.
"""
from engine.infer.inference import predict_batch, stream_infer

__all__ = ["predict_batch", "stream_infer", "run_infer"]


def run_infer(*args, **kwargs):
    """Lazy proxy to engine.infer.entrypoint.run_infer."""
    from engine.infer.entrypoint import run_infer as _run_infer

    return _run_infer(*args, **kwargs)
