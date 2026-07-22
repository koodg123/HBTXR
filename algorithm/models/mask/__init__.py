"""models.mask — standalone pupil-mask segmentation model (UNet).

A separate task from the HBTXR ellipse-regression models: dense per-pixel pupil
segmentation, independently trainable.
"""
from models.mask.model import MaskModel, MaskModelConfig, build_mask_model

__all__ = ["MaskModel", "MaskModelConfig", "build_mask_model"]
