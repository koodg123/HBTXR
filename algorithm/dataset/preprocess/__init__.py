"""dataset.preprocess — offline dataset preparation (canonicalize, manifests, relocate).

Only the preparation drivers are re-exported. The GroundedSAM annotation types used to
be re-exported here too, from a ``.annotation_groundedsam`` that no longer exists under
this package: since the flat rewrite they live in ``dataset.annotation``, and the one
in-repo consumer (``dataset.annotation.groundedsam_pipeline``) imports them from there
directly. Re-pointing the re-export at ``dataset.annotation`` would only pull a heavy
annotation backend into a preparation package, so it is dropped rather than repaired.
"""
from .build_manifests import build_manifests
from .canonicalize import canonicalize_dataset
from .relocate_dataset import relocate_dataset

__all__ = [
    "build_manifests",
    "canonicalize_dataset",
    "relocate_dataset",
]
