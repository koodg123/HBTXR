"""models.heads — shared, type-based pupil heads + a config-driven factory.

Paper-canonical: PupilBoxHead (search, Eq 2) and PupilEllipseHead (track, Eq 7).
Auxiliary: PupilMaskHead (Eq stage-1), ReliabilityHead (Eq 17-18), EyeRegionHead
(g_eye, Eq 15). Alternative: CenterHeatmapHead. Any modality selects its head by
name through build_head(), so heads are swapped by config, not per-model code.
"""
from models.heads.bbox import PupilBoxHead
from models.heads.ellipse import PupilEllipseHead
from models.heads.factory import HEAD_REGISTRY, build_head, list_head_names
from models.heads.heatmap import CenterHeatmapHead
from models.heads.mask import PupilMaskHead
from models.heads.reliability import ReliabilityHead
from models.heads.roi_guidance import EyeRegionHead

__all__ = [
    "PupilBoxHead",
    "PupilEllipseHead",
    "PupilMaskHead",
    "ReliabilityHead",
    "EyeRegionHead",
    "CenterHeatmapHead",
    "build_head",
    "list_head_names",
    "HEAD_REGISTRY",
]
