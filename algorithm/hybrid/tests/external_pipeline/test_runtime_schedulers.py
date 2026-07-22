from __future__ import annotations

import pytest

from hybrid.models.controller import TrackSearchSchedulerFSM
from hybrid.runtime.runtime_schedulers import (
    build_runtime_scheduler,
    get_runtime_scheduler_class,
    list_runtime_scheduler_names,
)


def test_alias_list_is_sorted_and_complete() -> None:
    assert list_runtime_scheduler_names() == ["fsm", "track_search", "track_search_fsm"]


def test_every_alias_resolves_to_the_same_class() -> None:
    resolved = {get_runtime_scheduler_class(name) for name in list_runtime_scheduler_names()}
    assert resolved == {TrackSearchSchedulerFSM}


def test_unknown_alias_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_runtime_scheduler_class("nope")


def test_build_returns_configured_instance() -> None:
    scheduler = build_runtime_scheduler("fsm", relocalize_cooldown=3)
    assert isinstance(scheduler, TrackSearchSchedulerFSM)
    assert scheduler.relocalize_cooldown == 3
    assert scheduler.state == "search"


def test_representative_transition_track_to_search_on_closed_eye() -> None:
    scheduler = build_runtime_scheduler()
    scheduler.state = "track"
    decision = scheduler.step(
        search_conf=0.9,
        track_conf=0.9,
        track_quality=0.9,
        event_density=1.0,
        similarity=0.9,
        closed_eye_flag=True,
    )
    assert decision.state == "search"
    assert decision.changed is True
    assert decision.reason == "closed_eye"
    assert scheduler.state == "search"
