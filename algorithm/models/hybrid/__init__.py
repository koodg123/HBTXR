"""models.hybrid — Hybrid HBTXR model (paper full system).

Shared ViT backbone with a frame/search branch (full depth, box head, anchor
refresh) and an event/track branch (early-exit, ellipse-residual head), selected
at runtime by the reliability-driven scheduler (models.hybrid.scheduler).
"""
from models.hybrid.model import HybridModel, HybridModelConfig, build_hybrid_model

__all__ = ["HybridModel", "HybridModelConfig", "build_hybrid_model"]
