"""Reliability-driven runtime scheduler FSM (paper Sec III-C.3, Eq 21).

CPU-side mode selection (the FPGA only enforces the chosen mode). States are
``{search, track}``. Bookkeeping per the paper: a valid-anchor flag ``v``, the
consecutive-track count ``d``, and the decoded anchor state ``z``.

Transition rule (Eq 21), applied after scoring the current step:
    next = search  if v == 0                                  (no valid anchor)
    next = track   if mode == search and rho_s >= tau_search  (anchor accepted)
    next = track   if mode == track  and rho_e >= tau_event and d < d_max
    next = search  otherwise

Acceptance also updates memory: an accepted search anchor is written (d<-0, v<-1);
an accepted track residual advances the state and increments d; otherwise the
stored reference is left unchanged and the next mode falls back to search.

Operates on scalar reliability at runtime (one stream window at a time); training
does not use the scheduler (each branch trains independently).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.hybrid.scheduler.policy import SchedulerConfig, combine_reliability


@dataclass
class SchedulerState:
    mode: str = "search"          # current operating mode
    valid: bool = False           # v_t: a valid reference anchor/state exists
    track_count: int = 0          # d_t: consecutive event-track updates
    anchor_state: Any = None      # z_t: decoded reference state


class TrackSearchScheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.config = config or SchedulerConfig()

    def initial_state(self) -> SchedulerState:
        return SchedulerState(mode="search", valid=False, track_count=0, anchor_state=None)

    def after_search(self, state: SchedulerState, reliability, accepted_anchor_state) -> SchedulerState:
        """Score a completed search step and return the next state (Eq 21)."""
        cfg = self.config
        confidence, quality = reliability
        rho = combine_reliability(confidence, quality, cfg.w_search_c, cfg.w_search_q)
        if rho >= cfg.tau_search:
            # accept the refreshed anchor: write it, reset d, mark valid, go to track
            return SchedulerState(mode="track", valid=True, track_count=0,
                                  anchor_state=accepted_anchor_state)
        # not accepted: keep prior reference, stay in search
        return SchedulerState(mode="search", valid=state.valid,
                              track_count=state.track_count, anchor_state=state.anchor_state)

    def after_track(self, state: SchedulerState, reliability, updated_state) -> SchedulerState:
        """Score a completed track step and return the next state (Eq 21)."""
        cfg = self.config
        confidence, quality = reliability
        rho = combine_reliability(confidence, quality, cfg.w_event_c, cfg.w_event_q)
        if rho >= cfg.tau_event and state.track_count < cfg.d_max:
            # accept the residual update: advance state, increment d, stay in track
            return SchedulerState(mode="track", valid=True,
                                  track_count=state.track_count + 1, anchor_state=updated_state)
        # not retained: keep prior reference, fall back to search
        return SchedulerState(mode="search", valid=state.valid,
                              track_count=state.track_count, anchor_state=state.anchor_state)

    @staticmethod
    def select_mode(state: SchedulerState) -> str:
        """Mode to run next: without a valid anchor the system must search."""
        return "search" if not state.valid else state.mode
