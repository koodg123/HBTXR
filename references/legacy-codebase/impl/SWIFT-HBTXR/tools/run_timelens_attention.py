from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch as th
from torchvision import transforms


def _install_tqdm_shim() -> None:
    try:
        import tqdm  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *args, **kwargs):
        return iterable

    module.tqdm = _tqdm
    sys.modules["tqdm"] = module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TimeLens interpolation with SWIFT-HBTXR-compatible CLI arguments")
    parser.add_argument("--timelens-root", required=True)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--root-event-folder", required=True)
    parser.add_argument("--root-image-folder", required=True)
    parser.add_argument("--root-output-folder", required=True)
    parser.add_argument("--number-of-frames-to-skip", type=int, default=0)
    parser.add_argument("--number-of-frames-to-insert", type=int, default=199)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _resolve_device(raw: str) -> torch.device:
    requested = str(raw).strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _move_tensors_to_device(device: torch.device, payload):
    if isinstance(payload, dict):
        return {key: _move_tensors_to_device(device, value) for key, value in payload.items()}
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    return payload


def _pack_to_example(left_image, right_image, left_events, right_events, right_weight):
    return {
        "before": {"rgb_image": left_image, "events": left_events},
        "middle": {"weight": right_weight},
        "after": {"rgb_image": right_image, "events": right_events},
    }


def _load_network(device: torch.device, checkpoint_file: Path, timelens_root: Path):
    _install_tqdm_shim()
    if str(timelens_root) not in sys.path:
        sys.path.insert(0, str(timelens_root))
    from timelens import attention_average_network

    network = attention_average_network.AttentionAverage().to(device)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    network.load_state_dict(checkpoint["networks"])
    network.eval()
    return network


def _import_timelens_modules(timelens_root: Path):
    _install_tqdm_shim()
    if str(timelens_root) not in sys.path:
        sys.path.insert(0, str(timelens_root))
    from timelens.common import hybrid_storage, image_sequence, os_tools, pytorch_tools, transformers

    return hybrid_storage, image_sequence, os_tools, pytorch_tools, transformers


def _interpolate(
    *,
    network,
    transform_list,
    interframe_events_iterator,
    boundary_frames_iterator,
    number_of_frames_to_interpolate: int,
    output_folder: Path,
    device: torch.device,
):
    output_frames, output_timestamps = [], []
    counter = 0
    timings: list[float] = []
    last_end_timestamp: float | None = None

    combined_iterator = zip(boundary_frames_iterator, interframe_events_iterator)
    for (left_frame, right_frame), event_sequence in combined_iterator:
        print(f"Counter: {counter:04d}")
        last_end_timestamp = float(event_sequence.end_time())
        output_timestamps += list(
            np.linspace(
                event_sequence.start_time(),
                event_sequence.end_time(),
                2 + number_of_frames_to_interpolate,
            )
        )[:-1]
        iterator_over_splits = event_sequence.make_iterator_over_splits(number_of_frames_to_interpolate)
        output_frames.append(left_frame)
        output_frames[-1].save(output_folder / f"{counter:06d}.png")
        counter += 1

        start_time = time.time()
        for split_index, (left_events, right_events) in enumerate(iterator_over_splits):
            example = _pack_to_example(
                left_frame,
                right_frame,
                left_events,
                right_events,
                float(split_index + 1.0) / float(number_of_frames_to_interpolate + 1.0),
            )
            example = transform_list.apply_transforms(example, transform_list.initialize_transformers())
            example = transform_list.collate([example])
            example = _move_tensors_to_device(device, example)
            with torch.no_grad():
                frame, _ = network.train2_run_attention_averaging(example, device)
            interpolated = th.clamp(frame.squeeze().detach().cpu(), 0, 1)
            output_frames.append(transforms.ToPILImage()(interpolated))
            output_frames[-1].save(output_folder / f"{counter:06d}.png")
            counter += 1

        elapsed = time.time() - start_time
        timings.append(elapsed)
        print(f"time: {elapsed}")
        print(f"average time: {float(np.mean(timings))}")
        print(f"lately time: {float(np.mean(timings[-10:]))}")

    output_frames.append(right_frame)
    output_frames[-1].save(output_folder / f"{counter:06d}.png")
    if last_end_timestamp is not None:
        output_timestamps.append(last_end_timestamp)
    return output_frames, output_timestamps


def run() -> None:
    args = _parse_args()
    timelens_root = Path(args.timelens_root).resolve()
    checkpoint_file = Path(args.checkpoint_file).resolve()
    root_event_folder = Path(args.root_event_folder).resolve()
    root_image_folder = Path(args.root_image_folder).resolve()
    root_output_folder = Path(args.root_output_folder).resolve()
    root_output_folder.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    hybrid_storage, image_sequence, os_tools, pytorch_tools, transformers_mod = _import_timelens_modules(timelens_root)
    if device.type == "cuda":
        pytorch_tools.set_fastest_cuda_mode()
    network = _load_network(device, checkpoint_file, timelens_root)
    leaf_image_folders = os_tools.find_leaf_folders(str(root_image_folder))
    for leaf_image_folder in leaf_image_folders:
        relative_path = os.path.relpath(leaf_image_folder, str(root_image_folder))
        leaf_event_folder = root_event_folder / relative_path
        leaf_output_folder = root_output_folder / relative_path
        leaf_output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Processing {relative_path}")
        storage = hybrid_storage.HybridStorage.from_folders(
            str(leaf_event_folder),
            str(leaf_image_folder),
            "*.npz",
            "*.png",
        )
        interframe_events_iterator = storage.make_interframe_events_iterator(int(args.number_of_frames_to_skip))
        boundary_frames_iterator = storage.make_boundary_frames_iterator(int(args.number_of_frames_to_skip))
        output_frames, output_timestamps = _interpolate(
            network=network,
            transform_list=transformers_mod,
            interframe_events_iterator=interframe_events_iterator,
            boundary_frames_iterator=boundary_frames_iterator,
            number_of_frames_to_interpolate=int(args.number_of_frames_to_insert),
            output_folder=leaf_output_folder,
            device=device,
        )
        output_image_sequence = image_sequence.ImageSequence(output_frames, output_timestamps)
        output_image_sequence.to_folder(str(leaf_output_folder), file_template="frame_{:06d}.png")


if __name__ == "__main__":
    run()
