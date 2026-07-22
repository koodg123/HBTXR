from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SchedulerDecision:
    state: str
    changed: bool
    reason: str


class TrackSearchSchedulerFSM:
    def __init__(
        self,
        *,
        search_conf_threshold: float = 0.35,
        track_conf_threshold: float = 0.45,
        track_quality_threshold: float = 0.45,
        similarity_threshold: float = 0.5,
        density_threshold: float = 0.002,
        relocalize_cooldown: int = 2,
        max_track_updates: int | None = None,
    ) -> None:
        self.search_conf_threshold = float(search_conf_threshold)
        self.track_conf_threshold = float(track_conf_threshold)
        self.track_quality_threshold = float(track_quality_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.density_threshold = float(density_threshold)
        self.relocalize_cooldown = int(relocalize_cooldown)
        self.max_track_updates = None if max_track_updates is None else int(max_track_updates)
        self.state = "search"
        self.cooldown = 0
        self.track_updates = 0

    def reset(self) -> None:
        self.state = "search"
        self.cooldown = 0
        self.track_updates = 0

    def step(
        self,
        *,
        search_conf: float,
        track_conf: float,
        track_quality: float,
        event_density: float,
        similarity: float,
        closed_eye_flag: bool,
    ) -> SchedulerDecision:
        if self.cooldown > 0:
            self.cooldown -= 1

        if closed_eye_flag:
            changed = self.state != "search"
            self.state = "search"
            self.cooldown = self.relocalize_cooldown
            self.track_updates = 0
            return SchedulerDecision(state=self.state, changed=changed, reason="closed_eye")

        if self.state == "track":
            if (
                track_conf < self.track_conf_threshold
                or track_quality < self.track_quality_threshold
                or similarity < self.similarity_threshold
                or event_density < self.density_threshold
            ):
                self.state = "search"
                self.cooldown = self.relocalize_cooldown
                self.track_updates = 0
                return SchedulerDecision(state=self.state, changed=True, reason="track_degraded")
            if self.max_track_updates is not None and self.track_updates >= self.max_track_updates:
                self.state = "search"
                self.cooldown = self.relocalize_cooldown
                self.track_updates = 0
                return SchedulerDecision(state=self.state, changed=True, reason="track_duration_limit")
            self.track_updates += 1
            return SchedulerDecision(state=self.state, changed=False, reason="track_keep")

        if self.cooldown > 0:
            return SchedulerDecision(state=self.state, changed=False, reason="search_cooldown")

        if (
            search_conf >= self.search_conf_threshold
            and track_conf >= self.track_conf_threshold
            and track_quality >= self.track_quality_threshold
            and similarity >= self.similarity_threshold
            and event_density >= self.density_threshold
        ):
            self.state = "track"
            self.track_updates = 0
            return SchedulerDecision(state=self.state, changed=True, reason="track_ready")
        return SchedulerDecision(state=self.state, changed=False, reason="search_keep")
