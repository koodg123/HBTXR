from __future__ import annotations

from utils.component_registry import resolve_component
from .codec import TrackStateCodec
from .encoder import TrackerTokenEncoder
from .event_branch import EventStateBranch
from .head_factory import TrackerHeadFactory
from .runtime_policy import RuntimeStepPolicy
from .search_branch import SearchBranch, SearchMaskGuidanceRefiner
from .track_branch import TrackStateBranch


TRACKER_COMPONENT_VARIANTS = {
    "encoder": {
        "default": TrackerTokenEncoder,
        "split_v1": TrackerTokenEncoder,
    },
    "head_factory": {
        "default": TrackerHeadFactory,
        "split_v1": TrackerHeadFactory,
    },
    "search_refiner": {
        "default": SearchMaskGuidanceRefiner,
        "split_v1": SearchMaskGuidanceRefiner,
    },
    "search_branch": {
        "default": SearchBranch,
        "split_v1": SearchBranch,
    },
    "event_branch": {
        "default": EventStateBranch,
        "split_v1": EventStateBranch,
    },
    "track_branch": {
        "default": TrackStateBranch,
        "split_v1": TrackStateBranch,
    },
    "runtime_policy": {
        "default": RuntimeStepPolicy,
        "split_v1": RuntimeStepPolicy,
    },
    "track_codec": {
        "default": TrackStateCodec,
        "split_v1": TrackStateCodec,
    },
}


def resolve_tracker_component(component_cfg: dict[str, Any], key: str):
    return resolve_component(
        component_cfg,
        key=key,
        variants=TRACKER_COMPONENT_VARIANTS,
        group_name="tracker",
    )
