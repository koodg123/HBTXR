from __future__ import annotations

import torch

from fecet_hbtxr.model import HBTXRTracker
from fecet_hbtxr.runtime import RuntimeFECETHBTXRTracker
from fecet_hbtxr.scheduler import TrackSearchSchedulerFSM


def test_scheduler_fsm_transitions():
    fsm = TrackSearchSchedulerFSM()
    ready = fsm.step(
        search_conf=0.9,
        track_conf=0.9,
        track_quality=0.9,
        similarity=0.9,
        event_density=0.1,
        closed_eye_flag=False,
    )
    assert ready.reason == "track_ready"
    keep = fsm.step(
        search_conf=0.2,
        track_conf=0.9,
        track_quality=0.9,
        similarity=0.9,
        event_density=0.1,
        closed_eye_flag=False,
    )
    assert keep.reason == "track_keep"
    degraded = fsm.step(
        search_conf=0.2,
        track_conf=0.1,
        track_quality=0.1,
        similarity=0.1,
        event_density=0.0,
        closed_eye_flag=False,
    )
    assert degraded.reason == "track_degraded"
    closed = fsm.step(
        search_conf=0.9,
        track_conf=0.9,
        track_quality=0.9,
        similarity=0.9,
        event_density=0.1,
        closed_eye_flag=True,
    )
    assert closed.reason == "closed_eye"


def test_runtime_tracker_bootstrap_and_reason():
    model = HBTXRTracker(embed_dim=24, depth=1, num_heads=3, patch_size=16, input_size=(256, 256))
    tracker = RuntimeFECETHBTXRTracker(model)
    batch = {
        "frame": torch.randn(1, 1, 256, 256),
        "event": torch.randn(1, 2, 256, 256),
        "event_density": torch.ones(1),
        "closed_eye_flag": torch.zeros(1),
    }
    out1 = tracker.step(**batch, session_key="s1")
    assert out1["runtime/reason"] == "bootstrap"
    out2 = tracker.step(**batch, session_key="s1")
    assert str(out2["runtime/reason"]) in {"track_ready", "search_keep", "search_cooldown", "track_keep", "track_degraded"}
