"""Higher-level quantization scheme helpers for HG-PIPE experiments."""

from __future__ import annotations

import numpy as np

from .lut_calibration import calibrate_gelu_requant, calibrate_rsqrt, calibrate_softmax


def _qrange(bits: int, signed: bool = True) -> tuple[int, int]:
    if bits <= 0:
        raise ValueError("bits must be positive")
    if signed:
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def _safe_scale(max_abs: np.ndarray, qmax: int) -> np.ndarray:
    scale = np.asarray(max_abs, dtype=np.float64) / float(max(qmax, 1))
    return np.where(scale > 0.0, scale, 1.0)


def quantize_group_vector(values, *, tensor_role: str, bits: int = 8, group_size: int | None = None, signed: bool = True) -> dict[str, object]:
    """Quantize activations per token and weights per output channel.

    Activations use group-vector scales over the last dimension for each token.
    Weights use one scale per output channel, matching the common linear-layer
    layout ``[out_channel, in_channel]``.
    """

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("group-vector quantization expects a 2D tensor")
    qmin, qmax = _qrange(bits, signed=signed)
    role = tensor_role.lower()

    if role in {"activation", "x"}:
        tokens, channels = arr.shape
        group = int(group_size or channels)
        if group <= 0 or channels % group != 0:
            raise ValueError("group_size must divide the last dimension")
        grouped = arr.reshape(tokens, channels // group, group)
        scales = _safe_scale(np.max(np.abs(grouped), axis=2), qmax)
        quantized = np.clip(np.rint(grouped / scales[..., None]), qmin, qmax).astype(np.int64).reshape(arr.shape)
        return {
            "granularity": "per-token",
            "tensor_role": "activation",
            "group_size": group,
            "bits": bits,
            "signed": signed,
            "quantized": quantized,
            "scales": scales,
        }

    if role in {"weight", "w"}:
        scales = _safe_scale(np.max(np.abs(arr), axis=1), qmax)
        quantized = np.clip(np.rint(arr / scales[:, None]), qmin, qmax).astype(np.int64)
        return {
            "granularity": "per-channel",
            "tensor_role": "weight",
            "group_size": arr.shape[1],
            "bits": bits,
            "signed": signed,
            "quantized": quantized,
            "scales": scales,
        }

    raise ValueError("tensor_role must be activation/x or weight/w")


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray) -> float:
    eps = 1e-12
    ref = reference.astype(np.float64) + eps
    cand = candidate.astype(np.float64) + eps
    ref /= np.sum(ref)
    cand /= np.sum(cand)
    return float(np.sum(ref * np.log(ref / cand)))


def _dyadic_approx(scale: float, shift_min: int, shift_max: int) -> tuple[int, int, float]:
    best = None
    for shift in range(shift_min, shift_max + 1):
        multiplier = max(1, int(round(scale * (1 << shift))))
        effective = multiplier / float(1 << shift)
        error = abs(effective - scale)
        if best is None or error < best[0]:
            best = (error, multiplier, shift, effective)
    assert best is not None
    return int(best[1]), int(best[2]), float(best[3])


def calibrate_dyadic_scale_kl(
    values,
    *,
    bits: int = 8,
    signed: bool = True,
    histogram_bins: int = 2048,
    percentiles: tuple[float, ...] = (99.0, 99.5, 99.9, 100.0),
    shift_min: int = 4,
    shift_max: int = 24,
) -> dict[str, object]:
    """Choose a dyadic dequant scale with a KL-divergence calibration loop."""

    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("calibration values must not be empty")
    qmin, qmax = _qrange(bits, signed=signed)
    max_q = max(abs(qmin), abs(qmax)) if signed else qmax
    best = None

    for percentile in percentiles:
        clip = float(np.percentile(np.abs(arr), percentile)) if signed else float(np.percentile(arr, percentile))
        clip = max(clip, 1e-12)
        raw_scale = clip / float(max(max_q, 1))
        multiplier, shift, scale = _dyadic_approx(raw_scale, shift_min, shift_max)
        clipped = np.clip(arr, -clip if signed else 0.0, clip)
        q = np.clip(np.rint(clipped / scale), qmin, qmax)
        dequant = q * scale
        hist_range = (-clip, clip) if signed else (0.0, clip)
        ref_hist, _ = np.histogram(clipped, bins=histogram_bins, range=hist_range)
        deq_hist, _ = np.histogram(np.clip(dequant, hist_range[0], hist_range[1]), bins=histogram_bins, range=hist_range)
        kl = _kl_divergence(ref_hist, deq_hist)
        candidate = (kl, percentile, clip, multiplier, shift, scale)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    kl, percentile, clip, multiplier, shift, scale = best
    return {
        "method": "kl_divergence",
        "scale_type": "dyadic",
        "linear_units": ["convolution", "SMU", "RMU"],
        "bits": bits,
        "signed": signed,
        "clip": float(clip),
        "percentile": float(percentile),
        "multiplier": int(multiplier),
        "shift": int(shift),
        "effective_scale": float(scale),
        "kl_divergence": float(kl),
    }


def quantize_nonlinear_lut(values, *, op: str, entries: int = 64, bits: int = 3, percentile: float = 99.9, **kwargs) -> dict[str, object]:
    """Build the requested non-linear LUT with percentile-based clipping."""

    normalized = op.lower()
    if normalized in {"gelu", "gelu-requant"}:
        return calibrate_gelu_requant(values, entries=entries, bits=bits, percentile=percentile, **kwargs)
    if normalized in {"layernorm", "rsqrt"}:
        return calibrate_rsqrt(values, entries=entries, bits=kwargs.pop("rsqrt_bits", 12), percentile=percentile, **kwargs)
    if normalized == "softmax":
        return calibrate_softmax(values, exp_entries=entries, recip_entries=kwargs.pop("recip_entries", entries), output_bits=bits, percentile=percentile, **kwargs)
    raise ValueError("op must be gelu, layernorm/rsqrt, or softmax")


def hardware_lut_index(values, *, offset: int, shift: int, entries: int):
    """Map integer input values to bounded LUT addresses."""

    if entries <= 0:
        raise ValueError("entries must be positive")
    if shift < 0:
        raise ValueError("shift must be non-negative")
    x = np.asarray(values, dtype=np.int64)
    return np.clip((x + int(offset)) >> int(shift), 0, int(entries) - 1).astype(np.int64)
