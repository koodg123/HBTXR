from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset.preprocess.io_utils import load_events_from_txt


DEFAULT_INTERPOLATION_BACKEND = "linear_blend"
SUPPORTED_INTERPOLATION_BACKENDS = ("linear_blend", "timelens", "timelens_xl")
DEFAULT_INTERPOLATION_COUNT_POLICY = "round"
DEFAULT_TIMELENS_BINS = 5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_third_party_checkpoint_root() -> Path:
    return (_project_root() / "workspace" / "third_party_checkpoints").resolve()


@dataclass
class _TimeLensImports:
    attention_average_network: Any
    event_module: Any
    transformers_module: Any
    size_adapter_module: Any


class _TimeLensRuntime:
    def __init__(
        self,
        *,
        import_root: Path,
        checkpoint_path: Path,
        device: str,
        number_of_bins_in_voxel_grid: int,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TimeLens backend requires torch to be installed") from exc

        self.torch = torch
        self.imports = _import_timelens_modules(import_root)
        self.device = torch.device(device)
        self.transform_list = self.imports.transformers_module.initialize_transformers(
            number_of_bins_in_voxel_grid=number_of_bins_in_voxel_grid
        )
        self.size_adapter = self.imports.size_adapter_module.SizeAdapter(minimum_size=16)
        self.network = self.imports.attention_average_network.AttentionAverage().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = unwrap_timelens_checkpoint_state_dict(checkpoint)
        self.network.load_state_dict(state_dict, strict=True)
        self.network.eval()

    def _move_tensors_to_device(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {key: self._move_tensors_to_device(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._move_tensors_to_device(value) for value in payload]
        if isinstance(payload, self.torch.Tensor):
            return payload.to(self.device)
        return payload

    def _pack_event_sequence(
        self,
        *,
        events_path: Path,
        image_height: int,
        image_width: int,
        timestamp0_us: int,
        timestamp1_us: int,
    ):
        suffix = str(events_path.suffix).strip().lower()
        if suffix == ".txt":
            eye = None
            for part in events_path.parts:
                lowered = str(part).strip().lower()
                if lowered in {"left", "right"}:
                    eye = lowered
                    break
            raw = load_events_from_txt(events_path, eye=eye)
        else:
            raw = np.load(events_path)
        features = np.stack(
            [
                np.asarray(raw["x"], dtype=np.float64),
                np.asarray(raw["y"], dtype=np.float64),
                np.asarray(raw["t"], dtype=np.float64),
                np.asarray(raw["p"], dtype=np.float64),
            ],
            axis=-1,
        )
        return self.imports.event_module.EventSequence(
            features=features,
            image_height=image_height,
            image_width=image_width,
            start_time=float(timestamp0_us),
            end_time=float(timestamp1_us),
        )

    def interpolate(
        self,
        *,
        frame0: np.ndarray,
        frame1: np.ndarray,
        alpha: float,
        timestamp0_us: int,
        timestamp1_us: int,
        events_path: Path,
    ) -> np.ndarray:
        left_image = Image.fromarray(np.asarray(frame0, dtype=np.uint8)).convert("RGB")
        right_image = Image.fromarray(np.asarray(frame1, dtype=np.uint8)).convert("RGB")
        split_timestamp = float(synth_timestamp_us(timestamp0_us, timestamp1_us, alpha))
        interframe_events = self._pack_event_sequence(
            events_path=events_path,
            image_height=int(frame0.shape[0]),
            image_width=int(frame0.shape[1]),
            timestamp0_us=timestamp0_us,
            timestamp1_us=timestamp1_us,
        )
        left_events, right_events = interframe_events.split_in_two(split_timestamp)
        example = {
            "before": {"rgb_image": left_image, "events": left_events},
            "middle": {"weight": float(alpha)},
            "after": {"rgb_image": right_image, "events": right_events},
        }
        example = self.imports.transformers_module.apply_transforms(example, self.transform_list)
        example = self.imports.transformers_module.collate([example])

        for packet_name in ("before", "after"):
            for field_name in ("rgb_image_tensor", "voxel_grid", "reversed_voxel_grid"):
                if field_name in example.get(packet_name, {}):
                    example[packet_name][field_name] = self.size_adapter.pad(example[packet_name][field_name])

        example = self._move_tensors_to_device(example)
        with self.torch.no_grad():
            frame_tensor, _ = self.network.run_attention_averaging(example)
        frame_tensor = self.size_adapter.unpad(frame_tensor)
        frame_tensor = frame_tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
        if frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3:
            frame_tensor = frame_tensor.mean(dim=0)
        elif frame_tensor.ndim == 3:
            frame_tensor = frame_tensor.squeeze(0)
        return (frame_tensor.numpy() * 255.0).round().clip(0.0, 255.0).astype(np.uint8)


_TIMELENS_RUNTIME_CACHE: dict[tuple[str, str, str, int], _TimeLensRuntime] = {}


def unwrap_timelens_checkpoint_state_dict(checkpoint: Any) -> Any:
    state_dict = checkpoint.get("networks", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    if isinstance(state_dict, dict) and state_dict and all(str(key).startswith("net.") for key in state_dict.keys()):
        state_dict = {str(key)[4:]: value for key, value in state_dict.items()}
    return state_dict


def resolve_interpolation_backend(
    *,
    interpolation_backend: str | None = None,
    interpolation_model: str | None = None,
) -> str:
    backend = interpolation_backend if interpolation_backend is not None else interpolation_model
    text = str(backend or DEFAULT_INTERPOLATION_BACKEND).strip().lower()
    if text in SUPPORTED_INTERPOLATION_BACKENDS:
        return text
    raise ValueError(f"Unsupported interpolation backend: {backend}")


def _import_timelens_modules(timelens_root: Path) -> _TimeLensImports:
    root = Path(timelens_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"TimeLens root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        attention_average_network = importlib.import_module("timelens.attention_average_network")
        event_module = importlib.import_module("timelens.common.event")
        transformers_module = importlib.import_module("timelens.common.transformers")
        size_adapter_module = importlib.import_module("timelens.common.size_adapter")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import TimeLens modules from {root}. "
            "Check the reference repo path and its Python dependencies."
        ) from exc
    return _TimeLensImports(
        attention_average_network=attention_average_network,
        event_module=event_module,
        transformers_module=transformers_module,
        size_adapter_module=size_adapter_module,
    )


def _resolve_timelens_import_root(*, backend: str, repo_root: Path) -> Path:
    resolved_backend = resolve_interpolation_backend(interpolation_backend=backend)
    root = Path(repo_root).resolve()
    if resolved_backend == "timelens":
        return root
    candidates = [root / "models" / "timelens", root]
    for candidate in candidates:
        if (candidate / "timelens").exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_timelens_checkpoint(
    *,
    backend: str,
    timelens_root: Path | None,
    checkpoint_path: Path | None,
) -> Path:
    resolved_backend = resolve_interpolation_backend(interpolation_backend=backend)
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"TimeLens checkpoint does not exist: {resolved}")
        return resolved
    if resolved_backend == "timelens_xl":
        shared_root = _default_third_party_checkpoint_root() / "FI" / "TimeLens-XL"
        if shared_root.exists():
            shared_candidates = sorted(
                [
                    *shared_root.glob("*.pt"),
                    *shared_root.glob("*.pth"),
                    *shared_root.glob("*.bin"),
                    *shared_root.rglob("TimeLens_*.pt"),
                    *shared_root.rglob("*.pth"),
                    *shared_root.rglob("*.bin"),
                ],
                key=lambda path: (path.stat().st_mtime, str(path)),
                reverse=True,
            )
            for candidate in shared_candidates:
                if candidate.is_file():
                    return candidate.resolve()
    if timelens_root is not None:
        repo_root = Path(timelens_root).resolve()
        candidates = (
            [
                repo_root / "refined_model" / "attention.bin",
                repo_root / "checkpoint.bin",
            ]
            if resolved_backend == "timelens"
            else [
                repo_root / "models" / "timelens" / "checkpoint.bin",
                repo_root / "checkpoint.bin",
                repo_root / "models" / "timelens" / "refined_model" / "attention.bin",
                repo_root / "refined_model" / "attention.bin",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        f"{resolved_backend} backend requires a checkpoint. "
        "Provide an explicit interpolation checkpoint, or place the expected checkpoint under the package root."
    )


def _get_timelens_runtime(
    *,
    backend: str,
    timelens_root: Path,
    checkpoint_path: Path,
    device: str,
    number_of_bins_in_voxel_grid: int,
) -> _TimeLensRuntime:
    import_root = _resolve_timelens_import_root(backend=backend, repo_root=Path(timelens_root))
    cache_key = (
        f"{resolve_interpolation_backend(interpolation_backend=backend)}::{Path(import_root).resolve()}",
        str(Path(checkpoint_path).resolve()),
        str(device),
        int(number_of_bins_in_voxel_grid),
    )
    runtime = _TIMELENS_RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = _TimeLensRuntime(
            import_root=import_root,
            checkpoint_path=Path(checkpoint_path),
            device=str(device),
            number_of_bins_in_voxel_grid=int(number_of_bins_in_voxel_grid),
        )
        _TIMELENS_RUNTIME_CACHE[cache_key] = runtime
    return runtime


def _apply_count_policy(value: float, policy: str) -> int:
    normalized = str(policy or DEFAULT_INTERPOLATION_COUNT_POLICY).strip().lower()
    if normalized == "round":
        return int(round(value))
    if normalized == "floor":
        return int(math.floor(value))
    if normalized == "ceil":
        return int(math.ceil(value))
    raise ValueError(f"Unsupported interpolation count policy: {policy}")


def compute_insert_count(
    *,
    timestamp0_us: int,
    timestamp1_us: int,
    target_fps: float | None = None,
    fixed_insert: int | None = None,
    count_policy: str = DEFAULT_INTERPOLATION_COUNT_POLICY,
    max_insert: int | None = None,
    legacy_default_insert: int = 1,
) -> int:
    gap_us = max(0, int(timestamp1_us) - int(timestamp0_us))
    if fixed_insert is not None:
        insert_count = max(int(fixed_insert), 0)
    elif target_fps is not None:
        fps = float(target_fps)
        if fps <= 0.0:
            raise ValueError("target_fps must be > 0 when provided")
        target_step_us = 1_000_000.0 / fps
        segment_count = max(1, _apply_count_policy(float(gap_us) / target_step_us, count_policy))
        insert_count = max(0, segment_count - 1)
    else:
        insert_count = max(int(legacy_default_insert), 0)
    if max_insert is not None:
        insert_count = min(insert_count, max(int(max_insert), 0))
    return int(insert_count)


def build_alpha_schedule(insert_count: int, *, legacy_alpha: float = 0.5) -> list[float]:
    count = max(int(insert_count), 0)
    if count <= 0:
        return []
    if count == 1:
        return [float(np.clip(legacy_alpha, 0.0, 1.0))]
    denom = float(count + 1)
    return [float((idx + 1) / denom) for idx in range(count)]


def synth_timestamp_us(timestamp0_us: int, timestamp1_us: int, alpha: float) -> int:
    t = float(np.clip(alpha, 0.0, 1.0))
    return int(round((1.0 - t) * int(timestamp0_us) + t * int(timestamp1_us)))


def interpolate_linear_blend(
    frame0: np.ndarray,
    frame1: np.ndarray,
    *,
    alpha: float,
    device: str = "cpu",
) -> np.ndarray:
    t = float(np.clip(alpha, 0.0, 1.0))
    requested_device = str(device or "cpu").strip().lower()
    if requested_device not in {"", "cpu"}:
        try:
            import torch

            resolved_device = requested_device
            if requested_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved_device != "cpu":
                frame0_t = torch.as_tensor(np.asarray(frame0, dtype=np.float32), device=resolved_device)
                frame1_t = torch.as_tensor(np.asarray(frame1, dtype=np.float32), device=resolved_device)
                blended = ((1.0 - t) * frame0_t + t * frame1_t).round().clamp(0.0, 255.0).to(dtype=torch.uint8)
                return blended.detach().cpu().numpy()
        except Exception:
            pass
    return (
        (1.0 - t) * np.asarray(frame0, dtype=np.float32)
        + t * np.asarray(frame1, dtype=np.float32)
    ).round().clip(0.0, 255.0).astype(np.uint8)


def interpolate_timelens(
    *,
    backend: str,
    frame0: np.ndarray,
    frame1: np.ndarray,
    alpha: float,
    timestamp0_us: int,
    timestamp1_us: int,
    events_path: Path | None = None,
    timelens_root: Path | None = None,
    timelens_checkpoint: Path | None = None,
    timelens_device: str = "cpu",
    timelens_xl_root: Path | None = None,
    timelens_xl_checkpoint: Path | None = None,
    timelens_xl_device: str = "cpu",
    number_of_bins_in_voxel_grid: int = DEFAULT_TIMELENS_BINS,
) -> np.ndarray:
    resolved_backend = resolve_interpolation_backend(interpolation_backend=backend)
    if events_path is None:
        raise ValueError(f"{resolved_backend} interpolation requires events_path")
    repo_root = timelens_root if resolved_backend == "timelens" else timelens_xl_root
    checkpoint = timelens_checkpoint if resolved_backend == "timelens" else timelens_xl_checkpoint
    device = timelens_device if resolved_backend == "timelens" else timelens_xl_device
    if repo_root is None:
        raise ValueError(f"{resolved_backend} interpolation requires the matching package root")
    resolved_checkpoint = resolve_timelens_checkpoint(
        backend=resolved_backend,
        timelens_root=Path(repo_root),
        checkpoint_path=None if checkpoint is None else Path(checkpoint),
    )
    runtime = _get_timelens_runtime(
        backend=resolved_backend,
        timelens_root=Path(repo_root),
        checkpoint_path=resolved_checkpoint,
        device=str(device),
        number_of_bins_in_voxel_grid=int(number_of_bins_in_voxel_grid),
    )
    return runtime.interpolate(
        frame0=frame0,
        frame1=frame1,
        alpha=alpha,
        timestamp0_us=timestamp0_us,
        timestamp1_us=timestamp1_us,
        events_path=Path(events_path),
    )


def interpolate_pair(
    *,
    backend: str,
    frame0: np.ndarray,
    frame1: np.ndarray,
    alpha: float,
    timestamp0_us: int,
    timestamp1_us: int,
    events_path: Path | None = None,
    timelens_root: Path | None = None,
    timelens_checkpoint: Path | None = None,
    timelens_device: str = "cpu",
    timelens_xl_root: Path | None = None,
    timelens_xl_checkpoint: Path | None = None,
    timelens_xl_device: str = "cpu",
    number_of_bins_in_voxel_grid: int = DEFAULT_TIMELENS_BINS,
) -> np.ndarray:
    resolved_backend = resolve_interpolation_backend(interpolation_backend=backend)
    if resolved_backend == "linear_blend":
        linear_blend_device = str(timelens_xl_device if str(timelens_xl_device).strip().lower() not in {"", "cpu"} else timelens_device)
        return interpolate_linear_blend(frame0, frame1, alpha=alpha, device=linear_blend_device)
    if resolved_backend in {"timelens", "timelens_xl"}:
        return interpolate_timelens(
            backend=resolved_backend,
            frame0=frame0,
            frame1=frame1,
            alpha=alpha,
            timestamp0_us=timestamp0_us,
            timestamp1_us=timestamp1_us,
            events_path=events_path,
            timelens_root=timelens_root,
            timelens_checkpoint=timelens_checkpoint,
            timelens_device=timelens_device,
            timelens_xl_root=timelens_xl_root,
            timelens_xl_checkpoint=timelens_xl_checkpoint,
            timelens_xl_device=timelens_xl_device,
            number_of_bins_in_voxel_grid=number_of_bins_in_voxel_grid,
        )
    raise ValueError(f"Unsupported interpolation backend: {backend}")
