"""Immutable fail-closed evaluation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class EvaluationContractError(ValueError):
    pass


class Domain(str, Enum):
    SENSOR = "SENSOR"
    ROI = "ROI"
    POST_TRANSFORM = "POST_TRANSFORM"


class EvaluationTask(str, Enum):
    CENTER = "CENTER"
    BBOX = "BBOX"
    ELLIPSE = "ELLIPSE"


_DOMAIN_SIZE = {
    Domain.ROI: (80, 60),
    Domain.SENSOR: (640, 480),
    Domain.POST_TRANSFORM: (64, 64),
}


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
            or (self.width, self.height) not in _DOMAIN_SIZE.values()
        ):
            raise EvaluationContractError("unsupported evaluation size")


def validate_domain_size(domain: Domain, size: Size) -> None:
    if not isinstance(domain, Domain) or not isinstance(size, Size):
        raise EvaluationContractError(f"invalid domain or size: {domain!r}, {size!r}")
    if _DOMAIN_SIZE[domain] != (size.width, size.height):
        raise EvaluationContractError(f"domain and size mismatch: {domain!r}, {size!r}")


def _validate_numbers(*values: float) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationContractError("geometry values must be finite numbers")
        try:
            finite = isfinite(value)
        except (TypeError, OverflowError):
            raise EvaluationContractError(
                "geometry values must be finite numbers"
            ) from None
        if not finite:
            raise EvaluationContractError("geometry values must be finite numbers")


@dataclass(frozen=True)
class Center:
    domain: Domain
    size: Size
    x: float
    y: float

    def __post_init__(self) -> None:
        validate_domain_size(self.domain, self.size)
        _validate_numbers(self.x, self.y)


@dataclass(frozen=True)
class BBox:
    domain: Domain
    size: Size
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        validate_domain_size(self.domain, self.size)
        _validate_numbers(self.x, self.y, self.width, self.height)
        if self.width <= 0 or self.height <= 0:
            raise EvaluationContractError("bounding-box dimensions must be positive")


@dataclass(frozen=True)
class Ellipse:
    domain: Domain
    size: Size
    cx: float
    cy: float
    axis_x: float
    axis_y: float
    theta: float

    def __post_init__(self) -> None:
        validate_domain_size(self.domain, self.size)
        _validate_numbers(self.cx, self.cy, self.axis_x, self.axis_y, self.theta)
        if self.axis_x <= 0 or self.axis_y <= 0:
            raise EvaluationContractError("ellipse axes must be positive")


@dataclass(frozen=True)
class EvaluationPolicy:
    invalid_sample: str = "reject"
    subject_weighting: str = "macro"
    aggregation: str = "macro"
    pixel_tolerances: tuple[int, ...] = (1, 5, 10)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.invalid_sample, str)
            or self.invalid_sample not in {"reject", "exclude"}
        ):
            raise EvaluationContractError("invalid invalid_sample policy")
        if (
            not isinstance(self.subject_weighting, str)
            or self.subject_weighting not in {"macro", "micro"}
        ):
            raise EvaluationContractError("invalid subject_weighting policy")
        if (
            not isinstance(self.aggregation, str)
            or self.aggregation not in {"macro", "micro"}
        ):
            raise EvaluationContractError("invalid aggregation policy")
        if (
            not isinstance(self.pixel_tolerances, tuple)
            or any(type(value) is not int for value in self.pixel_tolerances)
            or self.pixel_tolerances != (1, 5, 10)
        ):
            raise EvaluationContractError("pixel tolerances must be exactly (1, 5, 10)")
