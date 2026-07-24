"""Fully-specifiable quantization schema (Part B).

A ``TensorQuantSpec`` captures *everything* about how one tensor (a weight or an
activation) is quantized: bit-width, granularity, symmetric/asymmetric, scale
representation, and the calibration method. A ``QuantScheme`` bundles a weight spec
+ an activation spec + per-layer ``overrides`` (mixed precision) + a ``skip`` list,
and ``resolve(name)`` returns the effective ``(weight, activation)`` specs for a
given module path.

Separation of concerns (so each axis is independent):

- **granularity** — how many scale slots and how they tile the tensor (grouping.py).
- **symmetric/asymmetric** — whether ``zero_point`` is forced to 0 (observer.py).
- **calibration.method** — how the clip range per slot is chosen (observer.py).
- **scale_type** — how the chosen float scale is *represented*: plain ``float``, a
  power-of-two ``2^k``, or a ``dyadic`` ``multiplier / 2^shift`` (this module's
  :func:`apply_scale_type`; the observer applies it as the final step).

Only ``granularity`` touches the fake-quant forward (via slot reshape); the other
three are resolved entirely at calibration time into the stored scale/zero_point.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from quantization.scheme import QuantDtype

_GRANULARITIES = {"per-tensor", "per-channel", "per-group", "per-block"}
_SCALE_TYPES = {"float", "dyadic", "pot"}
_METHODS = {"minmax", "percentile", "kl", "mse"}


@dataclass(frozen=True)
class CalibrationSpec:
    """How each slot's clip range is chosen from observed data."""

    method: str = "minmax"        # minmax | percentile | kl | mse
    percentile: float = 99.9      # percentile / kl clip percentile
    num_bins: int = 2048          # kl histogram bins
    mse_grid: int = 80            # mse clip-ratio search points

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"calibration method must be one of {sorted(_METHODS)}, got {self.method!r}")

    @classmethod
    def from_config(cls, cfg: Any) -> "CalibrationSpec":
        if cfg is None:
            return cls()
        if isinstance(cfg, str):
            return cls(method=cfg)
        return cls(
            method=str(cfg.get("method", "minmax")),
            percentile=float(cfg.get("percentile", 99.9)),
            num_bins=int(cfg.get("num_bins", 2048)),
            mse_grid=int(cfg.get("mse_grid", 80)),
        )


@dataclass(frozen=True)
class TensorQuantSpec:
    """Full quantization spec for one tensor role (weight or activation)."""

    bits: int = 8
    signed: bool = True
    granularity: str = "per-tensor"          # per-tensor | per-channel | per-group | per-block
    group_size: int | None = None            # per-group: elements per group along in-features
    block_size: tuple[int, int] | None = None  # per-block: (out_block, in_block)
    symmetric: bool = True
    scale_type: str = "float"                # float | dyadic | pot
    ch_axis: int = 0                         # per-channel axis (weights 0, activations -1)
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")
        if self.granularity not in _GRANULARITIES:
            raise ValueError(f"granularity must be one of {sorted(_GRANULARITIES)}, got {self.granularity!r}")
        if self.scale_type not in _SCALE_TYPES:
            raise ValueError(f"scale_type must be one of {sorted(_SCALE_TYPES)}, got {self.scale_type!r}")

    @property
    def dtype(self) -> QuantDtype:
        return QuantDtype(self.bits, signed=self.signed)

    @classmethod
    def from_config(cls, cfg: Any, *, ch_axis: int = 0) -> "TensorQuantSpec":
        if cfg is None:
            return cls(ch_axis=ch_axis)
        block = cfg.get("block_size")
        return cls(
            bits=int(cfg.get("bits", 8)),
            signed=bool(cfg.get("signed", True)),
            granularity=str(cfg.get("granularity", "per-tensor")),
            group_size=(int(cfg["group_size"]) if cfg.get("group_size") is not None else None),
            block_size=(tuple(int(v) for v in block) if block is not None else None),  # type: ignore[arg-type]
            symmetric=bool(cfg.get("symmetric", True)),
            scale_type=str(cfg.get("scale_type", "float")),
            ch_axis=int(cfg.get("ch_axis", ch_axis)),
            calibration=CalibrationSpec.from_config(cfg.get("calibration")),
        )


def _default_weight() -> TensorQuantSpec:
    return TensorQuantSpec(ch_axis=0)


def _default_activation() -> TensorQuantSpec:
    return TensorQuantSpec(ch_axis=-1)


@dataclass(frozen=True)
class QuantScheme:
    """Weight + activation specs, a skip list, and per-layer overrides."""

    weight: TensorQuantSpec = field(default_factory=_default_weight)
    activation: TensorQuantSpec = field(default_factory=_default_activation)
    skip: tuple[str, ...] = ()
    # (name_substring, {"weight": {...patch}, "activation": {...patch}})
    overrides: tuple[tuple[str, dict[str, Any]], ...] = ()

    def resolve(self, name: str) -> tuple[TensorQuantSpec, TensorQuantSpec]:
        """Effective (weight, activation) specs for module ``name`` after overrides."""
        weight, activation = self.weight, self.activation
        for match, patch in self.overrides:
            if match in name:
                if patch.get("weight"):
                    weight = _patch(weight, patch["weight"])
                if patch.get("activation"):
                    activation = _patch(activation, patch["activation"])
        return weight, activation

    @classmethod
    def from_config(cls, cfg: Any) -> "QuantScheme":
        """Build from the ``quantization`` config block (see configs/experiment/*_quant.yaml)."""
        cfg = cfg or {}
        weight = TensorQuantSpec.from_config(cfg.get("weight"), ch_axis=0)
        activation = TensorQuantSpec.from_config(cfg.get("activation"), ch_axis=-1)
        # legacy flat keys (weight_bits / act_bits) still honored when weight/activation absent
        if cfg.get("weight") is None and cfg.get("weight_bits") is not None:
            weight = replace(weight, bits=int(cfg["weight_bits"]))
        if cfg.get("activation") is None and cfg.get("act_bits") is not None:
            activation = replace(activation, bits=int(cfg["act_bits"]))
        overrides = tuple(
            (str(o["match"]), {k: v for k, v in o.items() if k in ("weight", "activation")})
            for o in (cfg.get("overrides") or [])
        )
        return cls(weight=weight, activation=activation, skip=tuple(cfg.get("skip", []) or []), overrides=overrides)


def _patch(spec: TensorQuantSpec, patch: dict[str, Any]) -> TensorQuantSpec:
    fields = {k: v for k, v in patch.items() if k != "calibration"}
    if "block_size" in fields and fields["block_size"] is not None:
        fields["block_size"] = tuple(int(v) for v in fields["block_size"])
    if "calibration" in patch:
        fields["calibration"] = CalibrationSpec.from_config(patch["calibration"])
    return replace(spec, **fields)


# --- scale representation (scale_type) ---------------------------------------

def _dyadic_scale(scale: torch.Tensor, *, shift_min: int = 1, shift_max: int = 24) -> torch.Tensor:
    """Per-element dyadic approximation: effective ``round(scale*2^s) / 2^s`` (best s).

    Vectorized port of the HG-PIPE ``_dyadic_approx`` (references/) — picks, per
    element, the shift in ``[shift_min, shift_max]`` whose ``multiplier / 2^shift``
    is closest to ``scale``, and returns that effective float scale.
    """
    shifts = torch.arange(shift_min, shift_max + 1, device=scale.device, dtype=torch.float64)
    s = scale.reshape(-1, 1).to(torch.float64)                  # [N, 1]
    two_pow = torch.pow(2.0, shifts).reshape(1, -1)             # [1, S]
    mult = torch.clamp(torch.round(s * two_pow), min=1.0)       # [N, S]
    effective = mult / two_pow                                  # [N, S]
    err = (effective - s).abs()
    best = torch.argmin(err, dim=1)                             # [N]
    chosen = effective.gather(1, best.reshape(-1, 1)).reshape(scale.shape)
    return chosen.to(scale.dtype)


def apply_scale_type(scale: torch.Tensor, scale_type: str) -> torch.Tensor:
    """Represent ``scale`` as float / power-of-two / dyadic (returns effective float)."""
    if scale_type == "float":
        return scale
    if scale_type == "pot":
        exp = torch.round(torch.log2(scale.clamp_min(1e-12)))
        return torch.pow(2.0, exp).to(scale.dtype)
    if scale_type == "dyadic":
        return _dyadic_scale(scale)
    raise ValueError(f"unknown scale_type {scale_type!r}")


__all__ = [
    "CalibrationSpec",
    "TensorQuantSpec",
    "QuantScheme",
    "apply_scale_type",
]
