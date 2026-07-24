"""Integer quantization scheme primitives (dtype, range, clamp).

Adapted from the HG-PIPE quantization reference (references/) for the HBTXR ViT
path. Pure integer semantics, no torch dependency, so the dtype/range helpers can
be used from both the fake-quant (training) and integer-inference paths.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantDtype:
    """A signed/unsigned integer quantization dtype with derived qmin/qmax."""

    bits: int
    signed: bool = True

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("bits must be positive")

    @property
    def qmin(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def qmax(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1


INT8 = QuantDtype(8, signed=True)
UINT8 = QuantDtype(8, signed=False)
INT4 = QuantDtype(4, signed=True)
UINT4 = QuantDtype(4, signed=False)


def qrange(bits: int, signed: bool = True) -> tuple[int, int]:
    """(qmin, qmax) for a bit width — HG-PIPE ``_qrange`` semantics."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def clamp_int(value: int, lo: int, hi: int) -> int:
    """Bit-exact HG-PIPE integer clamp."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def quantize_clamp(value: int, bits: int, signed: bool) -> int:
    """Clamp an integer into the signed/unsigned range for ``bits`` (HG-PIPE)."""
    lo, hi = qrange(bits, signed=signed)
    return clamp_int(value, lo, hi)


__all__ = [
    "QuantDtype",
    "INT8",
    "UINT8",
    "INT4",
    "UINT4",
    "qrange",
    "clamp_int",
    "quantize_clamp",
]
