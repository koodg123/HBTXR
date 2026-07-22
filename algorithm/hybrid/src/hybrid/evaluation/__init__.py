"""Dependency-free evaluation contracts and coordinate conversion."""

from .contracts import (
    BBox,
    Center,
    Domain,
    Ellipse,
    EvaluationContractError,
    EvaluationPolicy,
    EvaluationTask,
    Size,
)
from .handover_bridge import convert_record

__all__ = [
    "BBox", "Center", "Domain", "Ellipse", "EvaluationContractError",
    "EvaluationPolicy", "EvaluationTask", "Size", "convert_record",
]
