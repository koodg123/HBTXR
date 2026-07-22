from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from dataset.preprocess.path_utils import ResolvedPaths
from dataset.preprocess.target_fps_build import normalize_target_fps_tag, resolve_target_fps_root
from utils.io import read_json, write_json


DEFAULT_TIMELENS_XL_MODEL_NAME = "TimeLens"
DEFAULT_TIMELENS_XL_PARAM_NAME = "hbtxr_timelens_xl_tuning"
DEFAULT_TIMELENS_XL_DATALOADER = "loader_timelens"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_timelens_xl_shared_checkpoint_dir(project_root: Path | None = None) -> Path:
    root = _project_root() if project_root is None else Path(project_root).resolve()
    return (root / "workspace" / "third_party_checkpoints" / "FI" / "TimeLens-XL").resolve()


def _resolve_timelens_xl_pretrained_path(
    *,
    model_pretrained: str | Path | None,
    repo_root: Path,
) -> str | None:
    if model_pretrained not in {None, ""}:
        resolved = Path(model_pretrained).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"TimeLens-XL pretrained checkpoint not found: {resolved}")
        return str(resolved)

    shared_root = _default_timelens_xl_shared_checkpoint_dir()
    if shared_root.exists():
        candidates = sorted(
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
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

    default_candidates = [
        repo_root / "models" / "timelens" / "checkpoint.bin",
        repo_root / "checkpoint.bin",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


@dataclass(frozen=True)
class TimeLensXLExportedSession:
    session_key: str
    split: str
    sequence_name: str
    images_dir: Path
    events_dir: Path
    n_images: int
    n_event_packets: int
    frame_shape_hw: tuple[int, int]


def _import_h5py(required: bool = True):
    try:
        import h5py  # type: ignore
    except ImportError:
        if required:
            raise RuntimeError("h5py is required to read HDF5 TimeLens-XL fine-tune session stores") from None
        return None
    return h5py


def _normalize_split_name(text: str | None) -> str:
    value = str(text or "train").strip().lower()
    if value not in {"train", "val"}:
        raise ValueError(f"Unsupported split name: {text!r}")
    return value


def _safe_sequence_name(session_key: str) -> str:
    text = str(session_key).strip().replace("\\", "/")
    return "_".join(part for part in text.split("/") if part)


def _list_materialized_session_meta_paths(target_fps_root: Path) -> list[Path]:
    return sorted((target_fps_root / "sessions").glob("user*/*/session_*/session_meta.json"))


def _load_npz_session_store(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path)
    return {key: np.asarray(raw[key]) for key in raw.files}


def _load_h5_session_store(path: Path) -> dict[str, np.ndarray]:
    h5py = _import_h5py(required=True)
    with h5py.File(path, "r") as handle:
        payload = {
            "frame_timestamps_us": np.asarray(handle["frames/timestamps_us"]),
            "frame_event_ranges": np.asarray(handle["frames/event_ranges"]),
            "event_t": np.asarray(handle["events/t"]),
            "event_x": np.asarray(handle["events/x"]),
            "event_y": np.asarray(handle["events/y"]),
            "event_p": np.asarray(handle["events/p"]),
        }
        if "images" in handle["frames"]:
            payload["frame_images"] = np.asarray(handle["frames/images"])
        if "source_frames" in handle and "images" in handle["source_frames"]:
            payload["source_frame_images"] = np.asarray(handle["source_frames/images"])
        return payload


def _load_materialized_session_store(path: Path) -> dict[str, np.ndarray]:
    suffix = str(path.suffix).strip().lower()
    if suffix == ".npz":
        return _load_npz_session_store(path)
    if suffix == ".h5":
        return _load_h5_session_store(path)
    raise ValueError(f"Unsupported session store suffix: {path}")


def _ensure_materialized_frame_images(store: dict[str, np.ndarray], *, session_store_path: Path) -> np.ndarray:
    frames = store.get("frame_images")
    if frames is None:
        raise ValueError(
            f"TimeLens-XL fine-tune export currently requires frame_storage_mode='materialized_target_frames'. "
            f"Missing frame_images in {session_store_path}"
        )
    frames_np = np.asarray(frames, dtype=np.uint8)
    if frames_np.ndim != 3:
        raise ValueError(f"Expected [N, H, W] frame stack in {session_store_path}, got {frames_np.shape}")
    return frames_np


def _normalize_polarity(values: np.ndarray) -> np.ndarray:
    signs = np.asarray(values, dtype=np.int8)
    return np.where(signs > 0, 1.0, -1.0).astype(np.float32)


def _events_to_signed_count_map(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    ps: np.ndarray,
    height: int,
    width: int,
    device: str = "cpu",
) -> np.ndarray:
    requested_device = str(device or "cpu").strip().lower()
    if requested_device not in {"", "cpu"}:
        try:
            import torch

            resolved_device = requested_device
            if requested_device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved_device != "cpu":
                xs_t = torch.as_tensor(xs, dtype=torch.long, device=resolved_device)
                ys_t = torch.as_tensor(ys, dtype=torch.long, device=resolved_device)
                ps_t = torch.as_tensor(np.where(np.asarray(ps) > 0, 1.0, -1.0), dtype=torch.float32, device=resolved_device)
                valid = (xs_t >= 0) & (xs_t < int(width)) & (ys_t >= 0) & (ys_t < int(height))
                if bool(valid.any()):
                    data = torch.zeros((int(height) * int(width),), dtype=torch.float32, device=resolved_device)
                    indices = ys_t[valid] * int(width) + xs_t[valid]
                    data.scatter_add_(0, indices, ps_t[valid])
                    return data.view(int(height), int(width)).detach().cpu().numpy()
                return np.zeros((height, width), dtype=np.float32)
        except Exception:
            pass

    data = np.zeros((height, width), dtype=np.float32)
    if xs.size == 0:
        return data
    xs_i = np.asarray(xs, dtype=np.int64)
    ys_i = np.asarray(ys, dtype=np.int64)
    valid = (xs_i >= 0) & (xs_i < width) & (ys_i >= 0) & (ys_i < height)
    if not np.any(valid):
        return data
    np.add.at(data, (ys_i[valid], xs_i[valid]), _normalize_polarity(ps[valid]))
    return data


def _event_range_for_interval(frame_event_ranges: np.ndarray, interval_index: int) -> tuple[int, int]:
    if interval_index < 0 or interval_index + 1 >= int(frame_event_ranges.shape[0]):
        raise IndexError(f"Interval index {interval_index} out of range for frame_event_ranges")
    row = np.asarray(frame_event_ranges[interval_index + 1], dtype=np.int64)
    return int(row[0]), int(row[1])


def _write_rgb_png(frame: np.ndarray, path: Path) -> None:
    gray = np.asarray(frame, dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    Image.fromarray(rgb, mode="RGB").save(path)


def _split_assignment_for_session(
    *,
    session_key: str,
    sorted_index: int,
    split_map: dict[str, set[str]] | None,
    val_every_nth_sequence: int,
) -> str:
    if split_map is not None:
        if session_key in split_map["val"]:
            return "val"
        return "train"
    if int(val_every_nth_sequence) > 0 and (sorted_index + 1) % int(val_every_nth_sequence) == 0:
        return "val"
    return "train"


def _load_split_map(path: str | Path | None) -> dict[str, set[str]] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid split map: {path}")
    return {
        "train": {str(item).strip() for item in payload.get("train", []) if str(item).strip()},
        "val": {str(item).strip() for item in payload.get("val", []) if str(item).strip()},
    }


def export_timelens_xl_finetune_dataset(
    *,
    target_root: str | Path,
    target_fps: float,
    output_root: str | Path,
    overwrite: bool = False,
    split_map_path: str | Path | None = None,
    val_every_nth_sequence: int = 5,
    export_device: str = "cpu",
) -> dict[str, Any]:
    target_fps_root = resolve_target_fps_root(target_root, target_fps)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_map = _load_split_map(split_map_path)
    meta_paths = _list_materialized_session_meta_paths(target_fps_root)
    exported: list[TimeLensXLExportedSession] = []
    skipped: list[dict[str, Any]] = []

    for sorted_index, meta_path in enumerate(meta_paths):
        meta = read_json(meta_path)
        session_key = str(meta.get("session_key", ""))
        if not session_key:
            skipped.append({"session_meta_path": str(meta_path), "reason": "missing_session_key"})
            continue
        session_store_path = Path(meta["session_store_path"]).resolve()
        store = _load_materialized_session_store(session_store_path)
        try:
            frames = _ensure_materialized_frame_images(store, session_store_path=session_store_path)
        except ValueError as exc:
            skipped.append({"session_key": session_key, "session_meta_path": str(meta_path), "reason": str(exc)})
            continue

        frame_event_ranges = np.asarray(store["frame_event_ranges"], dtype=np.int64)
        event_t = np.asarray(store["event_t"], dtype=np.int64)
        event_x = np.asarray(store["event_x"], dtype=np.int16)
        event_y = np.asarray(store["event_y"], dtype=np.int16)
        event_p = np.asarray(store["event_p"], dtype=np.int8)
        if frames.shape[0] < 3:
            skipped.append({"session_key": session_key, "session_meta_path": str(meta_path), "reason": "insufficient_target_frames"})
            continue
        if frame_event_ranges.shape[0] != frames.shape[0]:
            skipped.append(
                {
                    "session_key": session_key,
                    "session_meta_path": str(meta_path),
                    "reason": f"frame_event_ranges length {frame_event_ranges.shape[0]} does not match frame count {frames.shape[0]}",
                }
            )
            continue

        split = _split_assignment_for_session(
            session_key=session_key,
            sorted_index=sorted_index,
            split_map=split_map,
            val_every_nth_sequence=val_every_nth_sequence,
        )
        sequence_name = _safe_sequence_name(session_key)
        images_dir = output_root / split / sequence_name / "images"
        events_dir = output_root / split / sequence_name / "events"
        images_dir.mkdir(parents=True, exist_ok=True)
        events_dir.mkdir(parents=True, exist_ok=True)

        height = int(frames.shape[1])
        width = int(frames.shape[2])
        for frame_index, frame in enumerate(frames):
            _write_rgb_png(np.asarray(frame, dtype=np.uint8), images_dir / f"{frame_index:06d}.png")
        for interval_index in range(int(frames.shape[0]) - 1):
            start_idx, end_idx = _event_range_for_interval(frame_event_ranges, interval_index)
            packet = _events_to_signed_count_map(
                xs=event_x[start_idx:end_idx],
                ys=event_y[start_idx:end_idx],
                ps=event_p[start_idx:end_idx],
                height=height,
                width=width,
                device=str(export_device),
            )
            np.savez_compressed(
                events_dir / f"{interval_index:06d}.npz",
                data=np.asarray(packet, dtype=np.float32),
                timestamp_start_us=np.asarray([int(event_t[start_idx])] if start_idx < end_idx else [0], dtype=np.int64),
                timestamp_end_us=np.asarray([int(event_t[end_idx - 1])] if start_idx < end_idx else [0], dtype=np.int64),
            )

        sequence_meta = {
            "session_key": session_key,
            "split": split,
            "sequence_name": sequence_name,
            "session_meta_path": str(meta_path),
            "session_store_path": str(session_store_path),
            "frame_shape_hw": [height, width],
            "n_images": int(frames.shape[0]),
            "n_event_packets": int(frames.shape[0] - 1),
            "interpolation_backend": meta.get("interpolation_backend"),
            "event_generation_backend": meta.get("event_generation_backend"),
            "target_fps": meta.get("target_fps"),
        }
        write_json(sequence_meta, output_root / split / sequence_name / "sequence_meta.json")
        exported.append(
            TimeLensXLExportedSession(
                session_key=session_key,
                split=split,
                sequence_name=sequence_name,
                images_dir=images_dir,
                events_dir=events_dir,
                n_images=int(frames.shape[0]),
                n_event_packets=int(frames.shape[0] - 1),
                frame_shape_hw=(height, width),
            )
        )

    summary = {
        "stage": "timelens_xl_finetune_export",
        "target_root": str(Path(target_root).resolve()),
        "target_fps_root": str(target_fps_root),
        "target_fps": float(target_fps),
        "output_root": str(output_root),
        "split_map_path": None if split_map_path is None else str(Path(split_map_path).resolve()),
        "val_every_nth_sequence": int(val_every_nth_sequence),
        "export_device": str(export_device),
        "n_sequences_exported": int(len(exported)),
        "n_sequences_train": int(sum(1 for item in exported if item.split == "train")),
        "n_sequences_val": int(sum(1 for item in exported if item.split == "val")),
        "n_sequences_skipped": int(len(skipped)),
        "exported_sequences": [
            {
                "session_key": item.session_key,
                "split": item.split,
                "sequence_name": item.sequence_name,
                "images_dir": str(item.images_dir),
                "events_dir": str(item.events_dir),
                "n_images": item.n_images,
                "n_event_packets": item.n_event_packets,
                "frame_shape_hw": [int(item.frame_shape_hw[0]), int(item.frame_shape_hw[1])],
            }
            for item in exported
        ],
        "skipped_sequences": skipped,
    }
    write_json(summary, output_root / "export_summary.json")
    return summary


def _ensure_timelens_xl_repo_importable(repo_root: Path) -> None:
    root = repo_root.resolve()
    candidates = [
        root,
        root / "models" / "timelens",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def build_timelens_xl_run_argv(
    *,
    param_name: str,
    model_name: str,
    model_pretrained: str | None = None,
    init_step: int | None = None,
    skip_training: bool = False,
    clear_previous: bool = False,
    extension: str = "",
) -> list[str]:
    argv = [
        "run_network.py",
        "--param_name",
        str(param_name),
        "--model_name",
        str(model_name),
    ]
    if model_pretrained:
        argv.extend(["--model_pretrained", str(model_pretrained)])
    if init_step is not None:
        argv.extend(["--init_step", str(int(init_step))])
    if skip_training:
        argv.append("--skip_training")
    if clear_previous:
        argv.append("--clear_previous")
    if extension:
        argv.extend(["--extension", str(extension)])
    return argv


def register_hbtxr_timelens_xl_param(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    param_name: str = DEFAULT_TIMELENS_XL_PARAM_NAME,
    model_name: str = DEFAULT_TIMELENS_XL_MODEL_NAME,
    dataloader: str = DEFAULT_TIMELENS_XL_DATALOADER,
    crop_size: int = 128,
    batch_size: int = 2,
    num_workers: int = 1,
    interp_ratio: int = 2,
    rgb_sampling_ratio: int = 1,
    random_t: bool = True,
    learning_rate: float = 1.0e-4,
    milestones: Sequence[int] = (12, 24),
    gamma: float = 0.1,
    max_epoch: int = 27,
    training_stage: str = "tuning",
    enable_perceptual_metrics: bool = False,
) -> str:
    repo_root = Path(repo_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    _ensure_timelens_xl_repo_importable(repo_root)

    from easydict import EasyDict as ED  # type: ignore
    from params.models import model_arch_config  # type: ignore
    from tools.file_path_index import parse_path_common  # type: ignore
    from tools.registery import PARAM_REGISTRY  # type: ignore

    if param_name in set(PARAM_REGISTRY.keys()):
        return str(param_name)

    def _builder(args) -> Any:
        train_root = dataset_root / "train"
        val_root = dataset_root / "val"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing TimeLens-XL train split under {dataset_root}")
        if not val_root.exists():
            val_root = train_root

        paths = ED()
        paths.train_rgb = str(train_root)
        paths.train_evs = str(train_root)
        paths.test_rgb = str(val_root)
        paths.test_evs = str(val_root)

        paths.save = ED()
        paths.save.save_path = str(output_root)
        paths.save.exp_path = str(output_root / f"{args.model_name}{args.extension}")
        paths.save.record_txt = str(Path(paths.save.exp_path) / "training_record.txt")
        paths.save.train_im_path = str(Path(paths.save.exp_path) / "training_visual_examples")
        paths.save.val_im_path = str(Path(paths.save.exp_path) / "validation_visual_examples")
        paths.save.weights = str(Path(paths.save.exp_path) / "weights")

        if args.clear_previous and Path(paths.save.exp_path).exists():
            shutil.rmtree(paths.save.exp_path)
        for key in ("exp_path", "train_im_path", "val_im_path", "weights"):
            Path(paths.save[key]).mkdir(parents=True, exist_ok=True)

        model_config = ED()
        model_config.name = str(model_name)
        model_config.model_pretrained = args.model_pretrained
        for key, value in model_arch_config[str(model_name)].items():
            model_config[key] = value

        training_config = ED()
        training_config.dataloader = str(dataloader)
        training_config.crop_size = None if int(crop_size) <= 0 else int(crop_size)
        training_config.num_workers = int(num_workers)
        training_config.batch_size = int(batch_size)
        if not args.calc_flops and not args.skip_training:
            training_config.data_paths = parse_path_common(paths.train_rgb, paths.train_evs, bsergb=True)
        training_config.data_index_offset = 1
        training_config.rgb_sampling_ratio = int(rgb_sampling_ratio)
        training_config.interp_ratio = int(interp_ratio)
        training_config.random_t = bool(random_t)
        training_config.color = "RGB"
        training_config.max_epoch = int(max_epoch)
        training_config.optim = ED()
        training_config.optim.name = "Adam"
        training_config.optim.optim_params = ED()
        training_config.optim.optim_params.lr = float(learning_rate)
        training_config.optim.scheduler = "multilr"
        training_config.optim.scheduler_params = ED()
        training_config.optim.scheduler_params.milestones = [int(item) for item in milestones]
        training_config.optim.scheduler_params.gamma = float(gamma)
        training_config.losses = ED()
        training_config.losses.Charbonier = ED(weight=1.0, as_loss=True)
        training_config.losses.psnr = ED(weight=1.0, as_loss=False, test_y_channel=False)
        if enable_perceptual_metrics:
            training_config.losses.lpips = ED(weight=0.1, as_loss=True)
        training_config.train_stats = ED(print_freq=200, save_im_ep=max(1, min(int(max_epoch), 10)))

        validation_config = ED()
        validation_config.dataloader = str(dataloader)
        validation_config.val_epochs = int(max_epoch)
        validation_config.val_imsave_epochs = int(max_epoch)
        validation_config.weights_save_freq = max(1, int(max_epoch) // 3)
        validation_config.crop_size = None
        if not args.calc_flops:
            validation_config.data_paths = parse_path_common(paths.test_rgb, paths.test_evs, bsergb=True)
        validation_config.data_index_offset = 1
        validation_config.rgb_sampling_ratio = int(rgb_sampling_ratio)
        validation_config.interp_ratio = int(interp_ratio)
        validation_config.random_t = False
        validation_config.color = "RGB"
        validation_config.losses = ED()
        validation_config.losses.l1_loss = ED(weight=1.0, as_loss=False)
        validation_config.losses.psnr = ED(weight=1.0, as_loss=False, test_y_channel=False)
        validation_config.losses.ssim = ED(weight=1.0, as_loss=False, test_y_channel=False)
        if enable_perceptual_metrics:
            validation_config.losses.lpips = ED(weight=1.0, as_loss=False)
            validation_config.losses.dists = ED(weight=1.0, as_loss=False)

        params = ED()
        params.paths = paths
        params.training_config = training_config
        params.validation_config = validation_config
        params.model_config = model_config
        params.training_stage = str(training_stage)
        return params

    _builder.__name__ = str(param_name)
    PARAM_REGISTRY.register(_builder, suffix=None)
    return str(param_name)


def launch_timelens_xl_finetune(
    *,
    repo_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    param_name: str = DEFAULT_TIMELENS_XL_PARAM_NAME,
    model_name: str = DEFAULT_TIMELENS_XL_MODEL_NAME,
    dataloader: str = DEFAULT_TIMELENS_XL_DATALOADER,
    crop_size: int = 128,
    batch_size: int = 2,
    num_workers: int = 1,
    interp_ratio: int = 2,
    rgb_sampling_ratio: int = 1,
    random_t: bool = True,
    learning_rate: float = 1.0e-4,
    milestones: Sequence[int] = (12, 24),
    gamma: float = 0.1,
    max_epoch: int = 27,
    training_stage: str = "tuning",
    enable_perceptual_metrics: bool = False,
    model_pretrained: str | None = None,
    init_step: int | None = None,
    skip_training: bool = False,
    clear_previous: bool = False,
    extension: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    resolved_model_pretrained = _resolve_timelens_xl_pretrained_path(
        model_pretrained=model_pretrained,
        repo_root=repo_root,
    )
    argv = build_timelens_xl_run_argv(
        param_name=param_name,
        model_name=model_name,
        model_pretrained=resolved_model_pretrained,
        init_step=init_step,
        skip_training=skip_training,
        clear_previous=clear_previous,
        extension=extension,
    )
    summary = {
        "stage": "timelens_xl_finetune_launch",
        "repo_root": str(repo_root),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "param_name": str(param_name),
        "model_name": str(model_name),
        "dataloader": str(dataloader),
        "crop_size": int(crop_size),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "interp_ratio": int(interp_ratio),
        "rgb_sampling_ratio": int(rgb_sampling_ratio),
        "random_t": bool(random_t),
        "learning_rate": float(learning_rate),
        "milestones": [int(item) for item in milestones],
        "gamma": float(gamma),
        "max_epoch": int(max_epoch),
        "training_stage": str(training_stage),
        "enable_perceptual_metrics": bool(enable_perceptual_metrics),
        "model_pretrained": None if resolved_model_pretrained is None else str(resolved_model_pretrained),
        "skip_training": bool(skip_training),
        "clear_previous": bool(clear_previous),
        "extension": str(extension),
        "argv": argv,
    }
    if dry_run:
        return summary

    child_argv = build_timelens_xl_run_argv(
        param_name=param_name,
        model_name=model_name,
        model_pretrained=resolved_model_pretrained,
        init_step=init_step,
        skip_training=skip_training,
        clear_previous=clear_previous,
        extension=extension,
    )
    summary["param_name"] = str(param_name)
    summary["argv"] = child_argv
    project_root = _project_root()
    summary["launcher"] = {"python": sys.executable, "cwd": str(repo_root), "entrypoint": "src.preprocess.timelens_xl_native_entry"}
    env = dict(os.environ)
    src_root = project_root / "src"
    existing_pythonpath = str(env.get("PYTHONPATH", "")).strip()
    env["PYTHONPATH"] = (
        str(src_root)
        if not existing_pythonpath
        else f"{str(src_root)}{os.pathsep}{existing_pythonpath}"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.preprocess.timelens_xl_native_entry",
            "--repo-root",
            str(repo_root),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--param-name",
            str(param_name),
            "--model-name",
            str(model_name),
            "--dataloader",
            str(dataloader),
            "--crop-size",
            str(int(crop_size)),
            "--batch-size",
            str(int(batch_size)),
            "--num-workers",
            str(int(num_workers)),
            "--interp-ratio",
            str(int(interp_ratio)),
            "--rgb-sampling-ratio",
            str(int(rgb_sampling_ratio)),
            "--random-t",
            str(int(bool(random_t))),
            "--learning-rate",
            str(float(learning_rate)),
            "--milestones",
            ",".join(str(int(item)) for item in milestones),
            "--gamma",
            str(float(gamma)),
            "--max-epoch",
            str(int(max_epoch)),
            "--training-stage",
            str(training_stage),
            "--enable-perceptual-metrics",
            str(int(bool(enable_perceptual_metrics))),
            "--skip-training",
            str(int(bool(skip_training))),
            "--clear-previous",
            str(int(bool(clear_previous))),
            "--extension",
            str(extension),
            "--child-argv-json",
            json.dumps(child_argv),
            "--model-pretrained",
            "" if resolved_model_pretrained is None else str(resolved_model_pretrained),
            "--init-step",
            "" if init_step is None else str(int(init_step)),
        ],
        check=True,
        cwd=str(repo_root),
        env=env,
    )
    return summary


def default_timelens_xl_dataset_root(paths: ResolvedPaths) -> Path:
    return (paths.project_root / "workspace" / "timelens_xl_finetune_dataset").resolve()


def default_timelens_xl_runs_root(paths: ResolvedPaths) -> Path:
    return (paths.project_root / "workspace" / "timelens_xl_finetune_runs").resolve()


__all__ = [
    "DEFAULT_TIMELENS_XL_DATALOADER",
    "DEFAULT_TIMELENS_XL_MODEL_NAME",
    "DEFAULT_TIMELENS_XL_PARAM_NAME",
    "TimeLensXLExportedSession",
    "build_timelens_xl_run_argv",
    "default_timelens_xl_dataset_root",
    "default_timelens_xl_runs_root",
    "export_timelens_xl_finetune_dataset",
    "launch_timelens_xl_finetune",
    "register_hbtxr_timelens_xl_param",
]
