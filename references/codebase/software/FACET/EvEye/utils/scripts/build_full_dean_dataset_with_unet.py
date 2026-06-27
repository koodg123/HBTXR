import argparse
import gc
from datetime import datetime, timezone
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from EvEye.model.DavisEyeEllipse.UNet.UNet import UNet
from EvEye.utils.scripts.build_dean_dataset_from_ev_eye import (
    ELLIPSE_DTYPE,
    EVENT_DTYPE,
    create_memmap,
    ellipse_from_mask,
    merge_structured,
    natural_key,
    parse_frame_timestamp,
)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_torch_backend():
    if env_flag("FACET_DISABLE_CUDNN"):
        torch.backends.cudnn.enabled = False


def discover_sessions(data_root: Path) -> list[Path]:
    sessions = []
    for user_dir in sorted(data_root.glob("user*"), key=natural_key):
        for side in ("left", "right"):
            side_dir = user_dir / side
            if not side_dir.exists():
                continue
            for session_dir in sorted(side_dir.glob("session_*"), key=natural_key):
                if (session_dir / "frames").exists() and (
                    session_dir / "events" / "events.txt"
                ).exists():
                    sessions.append(session_dir)
    return sessions


def load_events(path: Path) -> np.ndarray:
    events = np.loadtxt(path, dtype=EVENT_DTYPE)
    return np.atleast_1d(events)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def preprocess_frame(frame_path: Path, input_size: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(frame_path).convert("L")
    original_size = image.size
    image = image.resize(input_size, Image.BILINEAR)
    array = np.asarray(image).astype(np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor, original_size


def infer_masks(
    model: torch.nn.Module,
    frame_paths: list[Path],
    device: torch.device,
    input_size: tuple[int, int],
    threshold: float,
) -> list[np.ndarray]:
    tensors = []
    original_sizes = []
    for frame_path in frame_paths:
        tensor, original_size = preprocess_frame(frame_path, input_size)
        tensors.append(tensor)
        original_sizes.append(original_size)

    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        masks = (probs >= threshold).detach().cpu().numpy().astype(np.uint8)

    resized_masks = []
    for mask, original_size in zip(masks, original_sizes):
        resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        resized_masks.append(resized)
    return resized_masks


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def save_sample(
    output_root: Path,
    sample_index: int,
    frame_path: Path,
    mask: np.ndarray,
    ellipse: tuple,
    split: str,
    session_dir: Path,
    event_count: int,
):
    sample_root = output_root / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_index:03d}_{split}_{safe_name(session_dir.parent.parent.name)}_{safe_name(session_dir.parent.name)}_{safe_name(session_dir.name)}"

    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        frame = np.asarray(Image.open(frame_path).convert("L"))
    mask_u8 = (mask.astype(np.uint8) * 255)
    overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(overlay)
    color_mask[:, :, 1] = mask_u8
    overlay = cv2.addWeighted(overlay, 0.75, color_mask, 0.25, 0)

    _, x, y, a, b, angle = ellipse
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(a / 2.0))), max(1, int(round(b / 2.0))))
    cv2.ellipse(overlay, center, axes, float(angle), 0, 360, (0, 0, 255), 1)

    frame_out = sample_root / f"{stem}_frame.png"
    mask_out = sample_root / f"{stem}_mask.png"
    overlay_out = sample_root / f"{stem}_overlay.png"
    label_out = sample_root / f"{stem}_label.txt"
    cv2.imwrite(str(frame_out), frame)
    cv2.imwrite(str(mask_out), mask_u8)
    cv2.imwrite(str(overlay_out), overlay)
    label_out.write_text(
        "\n".join(
            [
                f"frame={frame_path}",
                f"session={session_dir}",
                f"split={split}",
                f"event_count={event_count}",
                f"t={ellipse[0]}",
                f"x={ellipse[1]:.6f}",
                f"y={ellipse[2]:.6f}",
                f"a={ellipse[3]:.6f}",
                f"b={ellipse[4]:.6f}",
                f"ang={ellipse[5]:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "frame": str(frame_out),
        "mask": str(mask_out),
        "overlay": str(overlay_out),
        "label": str(label_out),
        "source_frame": str(frame_path),
        "split": split,
        "ellipse": {
            "t": int(ellipse[0]),
            "x": float(ellipse[1]),
            "y": float(ellipse[2]),
            "a": float(ellipse[3]),
            "b": float(ellipse[4]),
            "ang": float(ellipse[5]),
        },
        "event_count": int(event_count),
    }


class SplitWriter:
    def __init__(self, split: str, output_root: Path, batch_size: int, resume: bool = False):
        self.split = split
        self.batch_size = batch_size
        self.split_root = output_root / split
        self.data_root = self.split_root / "cached_data"
        self.ellipse_root = self.split_root / "cached_ellipse"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.ellipse_root.mkdir(parents=True, exist_ok=True)
        self.event_batch = []
        self.ellipses = []
        self.batch_id = 0
        self.ellipse_records_path = self.ellipse_root / "ellipse_records.npy"
        if resume:
            self._load_resume_state()

    def _load_resume_state(self):
        batch_ids = []
        for path in self.data_root.glob("events_batch_*.memmap"):
            try:
                batch_ids.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        self.batch_id = max(batch_ids) + 1 if batch_ids else 0
        if self.ellipse_records_path.exists():
            ellipses = np.load(self.ellipse_records_path)
            self.ellipses = [np.array(row, dtype=ELLIPSE_DTYPE) for row in ellipses]

    def add(self, events: np.ndarray, ellipse: tuple):
        self.event_batch.append(events)
        self.ellipses.append(np.array(ellipse, dtype=ELLIPSE_DTYPE))
        if len(self.event_batch) >= self.batch_size:
            self.flush_events()

    def flush_events(self):
        if not self.event_batch:
            return
        events_merged, event_indices = merge_structured(self.event_batch)
        create_memmap(
            events_merged,
            self.data_root / f"events_batch_{self.batch_id}.memmap",
            self.data_root / f"events_batch_info_{self.batch_id}.txt",
        )
        np.save(self.data_root / f"events_indices_{self.batch_id}.npy", event_indices)
        self.event_batch = []
        self.batch_id += 1

    def save_resume_state(self):
        ellipses = np.zeros(len(self.ellipses), dtype=ELLIPSE_DTYPE)
        if self.ellipses:
            ellipses[:] = self.ellipses
        tmp_path = self.ellipse_records_path.with_suffix(".npy.tmp")
        with tmp_path.open("wb") as out:
            np.save(out, ellipses)
        os.replace(tmp_path, self.ellipse_records_path)

    def close(self):
        self.flush_events()
        ellipses = np.zeros(len(self.ellipses), dtype=ELLIPSE_DTYPE)
        if self.ellipses:
            ellipses[:] = self.ellipses
        ellipse_indices = np.array(
            [[index, index + 1] for index in range(len(self.ellipses))],
            dtype=np.int32,
        )
        create_memmap(
            ellipses,
            self.ellipse_root / "ellipses_batch_0.memmap",
            self.ellipse_root / "ellipses_batch_info_0.txt",
        )
        np.save(self.ellipse_root / "ellipses_indices_0.npy", ellipse_indices)

    @property
    def count(self):
        return len(self.ellipses)


def split_for_session(index: int, total: int, train_ratio: float) -> str:
    return "train" if index < int(total * train_ratio) else "val"


def process_session(
    session_dir: Path,
    split: str,
    writer: SplitWriter,
    model: torch.nn.Module,
    device: torch.device,
    args,
    sample_records: list[dict],
):
    frames_dir = session_dir / "frames"
    events_path = session_dir / "events" / "events.txt"
    frame_paths = sorted(frames_dir.glob("*.png"), key=natural_key)
    if args.max_frames_per_session:
        frame_paths = frame_paths[: args.max_frames_per_session]
    if not frame_paths:
        return {"frames": 0, "valid": 0, "skipped": 0, "skip_no_ellipse": 0, "skip_no_events": 0}

    events = load_events(events_path)
    valid = 0
    skipped = 0
    skip_no_ellipse = 0
    skip_no_events = 0
    for start in range(0, len(frame_paths), args.inference_batch_size):
        batch_paths = frame_paths[start : start + args.inference_batch_size]
        masks = infer_masks(
            model=model,
            frame_paths=batch_paths,
            device=device,
            input_size=(args.input_width, args.input_height),
            threshold=args.mask_threshold,
        )
        for frame_path, mask in zip(batch_paths, masks):
            timestamp = parse_frame_timestamp(frame_path)
            ellipse = ellipse_from_mask(mask, timestamp, args.min_mask_pixels)
            if ellipse is None:
                skipped += 1
                skip_no_ellipse += 1
                continue
            end = np.searchsorted(events["t"], timestamp)
            start_event = max(0, end - args.events_per_sample)
            event_segment = events[start_event:end]
            if event_segment.shape[0] == 0:
                skipped += 1
                skip_no_events += 1
                continue
            writer.add(event_segment, ellipse)
            if len(sample_records) < args.sample_count:
                sample_records.append(
                    save_sample(
                        output_root=args.output_root,
                        sample_index=len(sample_records),
                        frame_path=frame_path,
                        mask=mask,
                        ellipse=ellipse,
                        split=split,
                        session_dir=session_dir,
                        event_count=event_segment.shape[0],
                    )
                )
            valid += 1
        del masks
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "frames": len(frame_paths),
        "valid": valid,
        "skipped": skipped,
        "skip_no_ellipse": skip_no_ellipse,
        "skip_no_events": skip_no_events,
    }


def write_json_atomic(path: Path, payload: dict):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2)
        out.write("\n")
    os.replace(tmp_path, path)


def save_progress(
    output_root: Path,
    completed_session_count: int,
    session_summaries: list[dict],
    sample_records: list[dict],
    totals: dict,
    writers: dict[str, SplitWriter],
):
    for writer in writers.values():
        writer.flush_events()
        writer.save_resume_state()
    payload = {
        "completed_session_count": completed_session_count,
        "session_summaries": session_summaries,
        "sample_records": sample_records,
        "totals": totals,
        "writer_counts": {split: writer.count for split, writer in writers.items()},
        "writer_next_batch_ids": {
            split: writer.batch_id for split, writer in writers.items()
        },
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(output_root / "progress_state.json", payload)


def main():
    configure_torch_backend()
    parser = argparse.ArgumentParser(
        description=(
            "Use a trained U-Net checkpoint to expand full Data_davis frames into "
            "a DavisEyeEllipseDataset-compatible DeanDataset."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Lightning checkpoint for EvEye.model.DavisEyeEllipse.UNet.UNet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--events-per-sample", type=int, default=5000)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--max-frames-per-session", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_root = args.raw_root / "Data_davis"
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    sessions = discover_sessions(data_root)
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]
    if not sessions:
        raise FileNotFoundError(f"No Data_davis sessions found under {data_root}")

    dry_manifest = {
        "dry_run": args.dry_run,
        "raw_root": str(args.raw_root),
        "data_root": str(data_root),
        "output_root": str(args.output_root),
        "num_sessions": len(sessions),
        "train_ratio": args.train_ratio,
        "events_per_sample": args.events_per_sample,
        "max_sessions": args.max_sessions,
        "max_frames_per_session": args.max_frames_per_session,
        "preview_sessions": [str(path) for path in sessions[:10]],
    }
    if args.dry_run:
        print(json.dumps(dry_manifest, indent=2))
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --dry-run is set")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    progress_path = args.output_root / "progress_state.json"
    resume_state = None
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if args.overwrite:
            shutil.rmtree(args.output_root)
        elif args.resume:
            if not progress_path.exists():
                raise SystemExit(
                    f"{progress_path} is missing. Cannot resume this partial output; "
                    "rerun with --overwrite."
                )
            with progress_path.open("r", encoding="utf-8") as src:
                resume_state = json.load(src)
        else:
            raise SystemExit(
                f"{args.output_root} is not empty. Re-run with --overwrite to replace "
                "it or --resume to continue a resumable run."
            )
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    writers = {
        "train": SplitWriter(
            "train", args.output_root, args.batch_size, resume=bool(resume_state)
        ),
        "val": SplitWriter(
            "val", args.output_root, args.batch_size, resume=bool(resume_state)
        ),
    }
    if resume_state:
        start_index = int(resume_state["completed_session_count"])
        session_summaries = list(resume_state.get("session_summaries", []))
        sample_records = list(resume_state.get("sample_records", []))
        totals = dict(resume_state.get("totals", {}))
        total_frames = int(totals.get("frames", 0))
        total_valid = int(totals.get("valid", 0))
        total_skipped = int(totals.get("skipped", 0))
        total_skip_no_ellipse = int(totals.get("skip_no_ellipse", 0))
        total_skip_no_events = int(totals.get("skip_no_events", 0))
    else:
        start_index = 0
        session_summaries = []
        sample_records = []
        total_frames = 0
        total_valid = 0
        total_skipped = 0
        total_skip_no_ellipse = 0
        total_skip_no_events = 0
    iterator = enumerate(sessions[start_index:], start=start_index)
    for index, session_dir in tqdm(
        iterator,
        total=len(sessions) - start_index,
        initial=0,
        desc="Expanding Data_davis",
    ):
        split = split_for_session(index, len(sessions), args.train_ratio)
        summary = process_session(
            session_dir, split, writers[split], model, device, args, sample_records
        )
        summary.update({"session": str(session_dir), "split": split})
        session_summaries.append(summary)
        total_frames += summary["frames"]
        total_valid += summary["valid"]
        total_skipped += summary["skipped"]
        total_skip_no_ellipse += summary["skip_no_ellipse"]
        total_skip_no_events += summary["skip_no_events"]
        save_progress(
            output_root=args.output_root,
            completed_session_count=index + 1,
            session_summaries=session_summaries,
            sample_records=sample_records,
            totals={
                "frames": total_frames,
                "valid": total_valid,
                "skipped": total_skipped,
                "skip_no_ellipse": total_skip_no_ellipse,
                "skip_no_events": total_skip_no_events,
            },
            writers=writers,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for writer in writers.values():
        writer.close()

    manifest = {
        "source": "Data_davis frames + U-Net predicted masks + Data_davis events",
        "raw_root": str(args.raw_root),
        "checkpoint": str(args.checkpoint),
        "output_root": str(args.output_root),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "generation_command": " ".join(sys.argv),
        "device": str(device),
        "torch_version": torch.__version__,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "facet_disable_cudnn": env_flag("FACET_DISABLE_CUDNN"),
        "resumed_from_progress": bool(resume_state),
        "train_ratio": args.train_ratio,
        "split_rule": "session-order split",
        "events_per_sample": args.events_per_sample,
        "input_size": [args.input_height, args.input_width],
        "mask_threshold": args.mask_threshold,
        "min_mask_pixels": args.min_mask_pixels,
        "num_sessions": len(sessions),
        "total_frames_scanned": total_frames,
        "valid_ellipse_count": total_valid,
        "skipped_frame_count": total_skipped,
        "skip_no_ellipse_count": total_skip_no_ellipse,
        "skip_no_events_count": total_skip_no_events,
        "num_train": writers["train"].count,
        "num_val": writers["val"].count,
        "sample_count": len(sample_records),
        "samples": sample_records,
        "session_summaries_preview": session_summaries[:20],
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
