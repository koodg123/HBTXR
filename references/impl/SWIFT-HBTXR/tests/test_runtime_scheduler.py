from __future__ import annotations

import torch

from swift_hbtxr.model import HBTXRTracker
from swift_hbtxr.runtime import RuntimeSwiftHBTXRTracker
from swift_hbtxr.scheduler import TrackSearchSchedulerFSM


class _AlwaysClosedDetector:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, frame, ellipse_xywht):
        batch = frame.shape[0]
        return {
            "mask_logits": torch.zeros(batch, 2, frame.shape[-2], frame.shape[-1], device=frame.device),
            "mask_probability": torch.zeros(batch, frame.shape[-2], frame.shape[-1], device=frame.device),
            "mask_binary": torch.zeros(batch, frame.shape[-2], frame.shape[-1], dtype=torch.uint8, device=frame.device),
            "open_extent": torch.zeros(batch, device=frame.device),
            "closed_eye_flag": torch.ones(batch, device=frame.device),
            "should_hold": torch.ones(batch, dtype=torch.bool, device=frame.device),
        }


class _DeterministicTrackerModel:
    def __init__(self):
        self.scheduler = TrackSearchSchedulerFSM()
        self._call_idx = 0

    def forward_train(self, batch):
        device = batch["frame"].device
        if self._call_idx == 0:
            search_state = torch.tensor([[16.0, 16.0, 8.0, 8.0, 0.0, 1.0]], device=device)
            track_state = torch.tensor([[16.0, 16.0, 8.0, 8.0, 0.0, 1.0]], device=device)
            search_logit = 0.0
            track_logit = 0.0
            track_quality = 0.0
        elif self._call_idx == 1:
            search_state = torch.tensor([[32.0, 32.0, 20.0, 16.0, 0.0, 1.0]], device=device)
            track_state = torch.tensor([[32.0, 32.0, 20.0, 16.0, 0.0, 1.0]], device=device)
            search_logit = 4.0
            track_logit = 4.0
            track_quality = 4.0
        else:
            search_state = torch.tensor([[33.0, 32.0, 20.0, 16.0, 0.0, 1.0]], device=device)
            track_state = torch.tensor([[33.0, 32.0, 20.0, 16.0, 0.0, 1.0]], device=device)
            search_logit = 4.0
            track_logit = 4.0
            track_quality = 4.0
        self._call_idx += 1
        search_pupil = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, search_logit, 0.0]], device=device)
        track_pupil = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, track_logit, track_quality]], device=device)
        return {
            "search/state": search_state,
            "track/state": track_state,
            "search/pupil": search_pupil,
            "track/pupil": track_pupil,
        }


class _TinyEllipseTrackerModel(_DeterministicTrackerModel):
    def forward_train(self, batch):
        outputs = super().forward_train(batch)
        tiny_state = torch.tensor([[1.5, 2.0, 1.4, 1.3, 0.0, 1.0]], device=batch["frame"].device)
        outputs["search/state"] = tiny_state
        outputs["track/state"] = tiny_state
        outputs["search/pupil"][:, 6] = 4.0
        outputs["track/pupil"][:, 6] = 4.0
        outputs["track/pupil"][:, 7] = 4.0
        return outputs


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
    tracker = RuntimeSwiftHBTXRTracker(model)
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


def test_runtime_tracker_hold_last_on_blink():
    tracker = RuntimeSwiftHBTXRTracker(_DeterministicTrackerModel())
    frame = torch.randn(1, 1, 256, 256)
    event = torch.randn(1, 2, 256, 256)
    tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    assert tracker.state.last_valid_state is not None
    last_valid = tracker.state.last_valid_state.clone()
    held = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.ones(1), session_key="s1")
    assert held["runtime/output_mode"] == "hold"
    assert held["runtime/reason"] == "hold_last"
    assert torch.allclose(held["runtime/ellipse_state"], last_valid)


def test_runtime_tracker_arms_antiblink_only_after_track():
    tracker = RuntimeSwiftHBTXRTracker(_DeterministicTrackerModel(), antiblink_detector=_AlwaysClosedDetector())
    frame = torch.randn(1, 1, 256, 256)
    event = torch.randn(1, 2, 256, 256)

    bootstrap = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    assert bootstrap["runtime/output_mode"] == "search"
    assert tracker.state.antiblink_armed is False
    assert tracker.state.last_valid_state is None

    track_ready = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    assert track_ready["runtime/output_mode"] == "track"
    assert track_ready["runtime/reason"] == "track_ready"
    assert tracker.state.antiblink_armed is True
    assert tracker.state.last_valid_state is not None

    held = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    assert held["runtime/output_mode"] == "hold"
    assert held["runtime/reason"] == "hold_last"


def test_runtime_tracker_ignores_antiblink_for_implausible_ellipse():
    tracker = RuntimeSwiftHBTXRTracker(_TinyEllipseTrackerModel(), antiblink_detector=_AlwaysClosedDetector())
    frame = torch.randn(1, 1, 256, 256)
    event = torch.randn(1, 2, 256, 256)

    tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    step2 = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")
    step3 = tracker.step(frame=frame, event=event, event_density=torch.ones(1), closed_eye_flag=torch.zeros(1), session_key="s1")

    assert step2["runtime/output_mode"] == "track"
    assert step3["runtime/output_mode"] in {"track", "search"}
    assert step3["runtime/reason"] != "hold_last"
