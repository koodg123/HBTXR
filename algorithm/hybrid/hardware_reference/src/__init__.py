__all__ = [
    "HGTXRDataset",
    "HGTXRTracker",
    "RuntimeHGTXRTracker",
    "TrackSearchSchedulerFSM",
    "make_synthetic_batch",
]


def __getattr__(name):
    if name in {"HGTXRDataset", "make_synthetic_batch"}:
        from .dataset import HGTXRDataset, make_synthetic_batch

        return {"HGTXRDataset": HGTXRDataset, "make_synthetic_batch": make_synthetic_batch}[name]
    if name == "HGTXRTracker":
        from .model import HGTXRTracker

        return HGTXRTracker
    if name == "RuntimeHGTXRTracker":
        from .runtime import RuntimeHGTXRTracker

        return RuntimeHGTXRTracker
    if name == "TrackSearchSchedulerFSM":
        from .scheduler import TrackSearchSchedulerFSM

        return TrackSearchSchedulerFSM
    raise AttributeError(name)
