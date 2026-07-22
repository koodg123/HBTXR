"""Scheduler policy: reliability combination and thresholds (paper Eq 20).

Each branch emits (confidence c, IoU-quality q); the policy combines them into a
scalar score ``rho = w_c*c + w_q*q`` with fixed branch-specific weights, and
holds the acceptance/retention thresholds used by the FSM.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    tau_search: float = 0.5   # tau^+_s : accept a refreshed search anchor
    tau_event: float = 0.5    # tau^-_e : retain the event-track mode
    d_max: int = 16           # D_max   : max consecutive track updates
    w_search_c: float = 0.5   # w^f_c
    w_search_q: float = 0.5   # w^f_q
    w_event_c: float = 0.5    # w^e_c
    w_event_q: float = 0.5    # w^e_q


def combine_reliability(confidence: float, quality: float, w_c: float, w_q: float) -> float:
    """rho = w_c * c + w_q * q (Eq 20)."""
    return w_c * float(confidence) + w_q * float(quality)
