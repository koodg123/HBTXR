"""engine.distill — knowledge distillation for the reimplemented models.

- distiller.distillation_loss: contract-native teacher<-student agreement loss.
- distiller.DistillTrainer / DistillConfig: task + distill training loop.
- entrypoint.run_distill: build frozen teacher + student and distill.
"""
from engine.distill.distiller import DistillConfig, DistillTrainer, distillation_loss

__all__ = ["distillation_loss", "DistillConfig", "DistillTrainer", "run_distill"]


def run_distill(*args, **kwargs):
    """Lazy proxy to engine.distill.entrypoint.run_distill."""
    from engine.distill.entrypoint import run_distill as _run_distill

    return _run_distill(*args, **kwargs)
