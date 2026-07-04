from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerDecision:
    state: str
    reason: str


class TrackSearchSchedulerFSM:
    def __init__(
        self,
        search_conf_threshold: float = 0.35,
        search_accept_threshold: float = 0.50,
        track_conf_threshold: float = 0.45,
        track_quality_threshold: float = 0.45,
        reliability_threshold: float = 0.50,
        similarity_threshold: float = 0.5,
        density_threshold: float = 0.002,
        relocalize_cooldown: int = 2,
        beta_conf: float = 0.5,
        beta_quality: float = 0.5,
        **_: object,
    ) -> None:
        self.search_conf_threshold = float(search_conf_threshold)
        self.search_accept_threshold = float(search_accept_threshold)
        self.track_conf_threshold = float(track_conf_threshold)
        self.track_quality_threshold = float(track_quality_threshold)
        self.reliability_threshold = float(reliability_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.density_threshold = float(density_threshold)
        self.relocalize_cooldown = int(relocalize_cooldown)
        self.beta_conf = float(beta_conf)
        self.beta_quality = float(beta_quality)
        self.state = "search"
        self.cooldown = 0

    def reset(self) -> None:
        self.state = "search"
        self.cooldown = 0

    def reliability(self, confidence: float, quality: float) -> float:
        return self.beta_conf * float(confidence) + self.beta_quality * float(quality)

    def step(
        self,
        *,
        search_conf: float,
        track_conf: float,
        track_quality: float,
        similarity: float,
        event_density: float,
        closed_eye_flag: bool,
    ) -> SchedulerDecision:
        if closed_eye_flag:
            self.state = "search"
            self.cooldown = self.relocalize_cooldown
            return SchedulerDecision("search", "closed_eye")
        if event_density < self.density_threshold:
            self.state = "search"
            return SchedulerDecision("search", "low_event_density")
        track_rel = self.reliability(track_conf, track_quality)
        search_rel = self.reliability(search_conf, search_conf)
        track_ok = (
            track_conf >= self.track_conf_threshold
            and track_quality >= self.track_quality_threshold
            and track_rel >= self.reliability_threshold
            and similarity >= self.similarity_threshold
        )
        if track_ok and self.cooldown <= 0:
            self.state = "track"
            return SchedulerDecision("track", "track_reliable")
        self.cooldown = max(0, self.cooldown - 1)
        if search_conf >= self.search_conf_threshold or search_rel >= self.search_accept_threshold:
            self.state = "search"
            return SchedulerDecision("search", "anchor_refresh")
        self.state = "search"
        return SchedulerDecision("search", "relocalize")
