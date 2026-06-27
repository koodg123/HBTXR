"""Image tensor to HG-PIPE patch input bridge.

The public ICCAD24-HG-PIPE checkout exposes the integer patch embedding
input contract, but not the original image preprocessing or calibration script.
These helpers provide an explicit, experimental bridge from normalized image
tensors to the signed int8 patch input shape consumed by the artifact graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_PATCH_TOKENS = 196
DEFAULT_PATCH_CHANNELS = 768
DEFAULT_INPUT_RANGE = (-102, 127)
SIGNED_INT8_RANGE = (-128, 127)


@dataclass(frozen=True)
class PatchInputBridgeConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    tokens: int = DEFAULT_PATCH_TOKENS
    channels: int = DEFAULT_PATCH_CHANNELS
    signed_min: int = SIGNED_INT8_RANGE[0]
    signed_max: int = SIGNED_INT8_RANGE[1]
    observed_min: int = DEFAULT_INPUT_RANGE[0]
    observed_max: int = DEFAULT_INPUT_RANGE[1]
    cls_slot: bool = True


def _as_chw_array(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D CHW or HWC image tensor, got shape {arr.shape}")
    if arr.shape[0] in {1, 3}:
        return arr
    if arr.shape[-1] in {1, 3}:
        return arr.transpose(2, 0, 1)
    raise ValueError(f"cannot infer channel dimension from shape {arr.shape}")


def extract_patch_vectors(image: np.ndarray, config: PatchInputBridgeConfig = PatchInputBridgeConfig()) -> np.ndarray:
    chw = _as_chw_array(image)
    channels, height, width = chw.shape
    if height != config.image_size or width != config.image_size:
        raise ValueError(f"expected {config.image_size}x{config.image_size}, got {height}x{width}")
    if channels != 3:
        raise ValueError(f"expected 3 channels, got {channels}")

    patch = config.patch_size
    grid_h = height // patch
    grid_w = width // patch
    if grid_h * grid_w != config.tokens:
        raise ValueError(f"expected {config.tokens} patches, got {grid_h * grid_w}")

    patches = (
        chw.reshape(channels, grid_h, patch, grid_w, patch)
        .transpose(1, 3, 0, 2, 4)
        .reshape(config.tokens, channels * patch * patch)
    )
    if patches.shape[1] != config.channels:
        raise ValueError(f"expected patch channel width {config.channels}, got {patches.shape[1]}")
    return patches


def quantize_patches_symmetric(patches: np.ndarray, scale: float, config: PatchInputBridgeConfig = PatchInputBridgeConfig()) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    quantized = np.rint(np.asarray(patches, dtype=np.float32) / float(scale))
    quantized = np.clip(quantized, config.signed_min, config.signed_max)
    return quantized.astype(np.int64)


def estimate_symmetric_scale(images: Iterable[np.ndarray], config: PatchInputBridgeConfig = PatchInputBridgeConfig()) -> float:
    max_abs = 0.0
    for image in images:
        patches = extract_patch_vectors(image, config)
        if patches.size:
            max_abs = max(max_abs, float(np.max(np.abs(patches))))
    if max_abs == 0.0:
        raise ValueError("cannot estimate scale from all-zero images")
    return max_abs / float(config.signed_max)


def to_hgpipe_patch_input(
    image: np.ndarray,
    *,
    scale: float,
    config: PatchInputBridgeConfig = PatchInputBridgeConfig(),
) -> np.ndarray:
    patches = quantize_patches_symmetric(extract_patch_vectors(image, config), scale, config)
    if not config.cls_slot:
        return patches.reshape(-1)

    output = np.zeros((config.tokens, config.channels), dtype=np.int64)
    output[1:, :] = patches[: config.tokens - 1, :]
    return output.reshape(-1)


def patch_input_contract(config: PatchInputBridgeConfig = PatchInputBridgeConfig()) -> dict[str, int | bool]:
    return {
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "tokens": config.tokens,
        "channels": config.channels,
        "signed_min": config.signed_min,
        "signed_max": config.signed_max,
        "observed_min": config.observed_min,
        "observed_max": config.observed_max,
        "cls_slot": config.cls_slot,
    }


def iter_images_from_npy(array: np.ndarray):
    """Yield CHW/HWC image tensors from a single-image or batch .npy array."""

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        yield arr
        return
    if arr.ndim != 4:
        raise ValueError(f"expected 3D image or 4D image batch, got shape {arr.shape}")
    if arr.shape[1] in {1, 3} or arr.shape[-1] in {1, 3}:
        for image in arr:
            yield image
        return
    raise ValueError(f"cannot infer channel dimension from batch shape {arr.shape}")


def estimate_scale_from_npy_array(array: np.ndarray, config: PatchInputBridgeConfig = PatchInputBridgeConfig()) -> dict[str, object]:
    images = list(iter_images_from_npy(array))
    scale = estimate_symmetric_scale(images, config)
    return {
        "scale": scale,
        "images": len(images),
        "contract": patch_input_contract(config),
        "paper_equivalent": False,
        "method": "symmetric_max_abs_over_patch_vectors",
    }
