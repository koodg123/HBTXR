from __future__ import annotations

from . import assigners, bundles, common, distillation, metrics, primitives, stage, stage1, stage2, stage_common
from .assigners import *  # noqa: F401,F403
from .common import *  # noqa: F401,F403
from .distillation import *  # noqa: F401,F403
from .primitives import *  # noqa: F401,F403
from .stage import *  # noqa: F401,F403
from . import losses
from .losses import build_loss, compute_loss, list_loss_names

__all__ = [
    "assigners",
    "bundles",
    "common",
    "distillation",
    "metrics",
    "primitives",
    "stage",
    "stage1",
    "stage2",
    "stage_common",
    "losses",
    "build_loss",
    "compute_loss",
    "list_loss_names",
    *assigners.__all__,
    *common.__all__,
    *primitives.__all__,
    *stage.__all__,
    *distillation.__all__,
]
