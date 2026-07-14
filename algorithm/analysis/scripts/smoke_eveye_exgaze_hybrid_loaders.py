from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.cache.MemmapCacheStructedEvents import load_event_segment


DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")
DEFAULT_SOURCE_ROOT = Path(
    "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/"
    "DeanDataset_full_unet_subject_independent"
)
SENSOR_W = 346.0
SENSOR_H = 260.0
FRAME_SIZE = 128
EVENT_SIZE = 64
PATCH_SIZE = 16
PATCH_NUM = 8


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


class HybridEVEyeSmokeDataset(Dataset):
    """EV-Eye adaptation smoke dataset.

    This is not the final model dataset. It verifies the agreed input contract:
    frame tensor Bx1x128x128, mask tensor Bx128x128, and event tensor Bx2x64x64.
    """

    def __init__(self, root: Path, split: str):
        self.root = root
        self.split = split
        self.rows = read_csv_rows(root / split / "labels.csv")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict]:
        row = self.rows[index]
        frame = cv2.imread(str(self.root / row["frame128_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / row["mask128_path"]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {row['frame128_path']}")
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {row['mask128_path']}")
        event = np.load(self.root / row["event64_path"])

        frame_t = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy((mask > 0).astype(np.int64))
        event_t = torch.from_numpy(event.astype(np.float32))
        label_t = torch.tensor(
            [
                float(row["gt_x_128"]),
                float(row["gt_y_128"]),
                float(row["gt_a_128"]),
                float(row["gt_b_128"]),
                float(row["gt_ang"]),
            ],
            dtype=torch.float32,
        )
        return {
            "frame": frame_t,
            "mask": mask_t,
            "event": event_t,
            "ellipse128": label_t,
            "meta": {
                "sample_idx": int(row["sample_idx"]),
                "subject": int(row["subject"]),
                "eye": row["eye"],
                "session_code": int(row["session_code"]),
                "frame_idx": int(row["frame_idx"]),
            },
        }


class HybridEXGazeSmokeDataset(Dataset):
    """EX-Gaze adaptation smoke dataset for exported annotation JSON."""

    def __init__(self, root: Path, split: str):
        self.root = root
        self.split = split
        payload = json.loads((root / split / "annotations.json").read_text())
        self.metainfo = payload["metainfo"]
        self.rows = payload["data_list"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict]:
        row = self.rows[index]
        frame = cv2.imread(str(self.root / row["img_filename"]), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {row['img_filename']}")
        event = np.load(self.root / row["input_volume"])
        patches = np.load(self.root / row["event_patches"])

        frame_t = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        event_t = torch.from_numpy(event.astype(np.float32))
        patches_t = torch.from_numpy(patches.astype(np.float32))
        pre_state_t = torch.tensor(row["pre_state"], dtype=torch.float32)
        pupil_t = torch.tensor(row["pupil"], dtype=torch.float32)
        regions_t = torch.tensor(row["sample_regions"], dtype=torch.int64)
        return {
            "frame": frame_t,
            "input_volume": event_t,
            "event_patches": patches_t,
            "pre_state": pre_state_t,
            "pupil": pupil_t,
            "sample_regions": regions_t,
            "meta": {
                "img_id": int(row["img_id"]),
                "subject": int(row["subject"]),
                "eye": row["eye"],
                "session": int(row["session"]),
                "frame_idx": int(row["frame_idx"]),
            },
        }


def make_event64(events: np.ndarray) -> np.ndarray:
    volume = np.zeros((2, EVENT_SIZE, EVENT_SIZE), dtype=np.uint16)
    if len(events) == 0:
        return volume
    xs = np.clip((events["x"].astype(np.float32) * EVENT_SIZE / SENSOR_W).astype(np.int64), 0, EVENT_SIZE - 1)
    ys = np.clip((events["y"].astype(np.float32) * EVENT_SIZE / SENSOR_H).astype(np.int64), 0, EVENT_SIZE - 1)
    ps = (events["p"].astype(np.int64) > 0).astype(np.int64)
    np.add.at(volume, (ps, ys, xs), 1)
    return volume


def make_mask128(ellipse128: list[float]) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    x, y, a, b, angle = ellipse128
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(a / 2.0))), max(1, int(round(b / 2.0))))
    cv2.ellipse(mask, center, axes, float(angle), 0, 360, 1, -1)
    return mask


def sample_regions_from_ellipse64(ellipse64: list[float]) -> list[list[int]]:
    cx, cy, a, b, angle = ellipse64
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    regions = []
    for i in range(PATCH_NUM):
        theta = 2.0 * np.pi * i / PATCH_NUM
        px = cx + (a / 2.0) * np.cos(theta) * cos_a - (b / 2.0) * np.sin(theta) * sin_a
        py = cy + (a / 2.0) * np.cos(theta) * sin_a + (b / 2.0) * np.sin(theta) * cos_a
        x1 = int(np.ceil(px)) - PATCH_SIZE // 2
        y1 = int(np.ceil(py)) - PATCH_SIZE // 2
        x1 = int(np.clip(x1, 0, EVENT_SIZE - PATCH_SIZE))
        y1 = int(np.clip(y1, 0, EVENT_SIZE - PATCH_SIZE))
        regions.append([x1, y1, x1 + PATCH_SIZE, y1 + PATCH_SIZE])
    return regions


def event_patches(volume: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    return np.stack([volume[:, y1:y2, x1:x2] for x1, y1, x2, y2 in regions]).astype(np.uint16)


class HybridLiveManifestDataset(Dataset):
    """Live training-ready dataset backed by compact manifest rows and FACET cache."""

    def __init__(self, root: Path, source_root: Path, split: str):
        self.root = root
        self.source_root = source_root
        self.split = split
        self.rows = read_csv_rows(root / split / "labels.csv")
        self.data_root = source_root / split / "cached_data"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict]:
        row = self.rows[index]
        frame = cv2.imread(row["source_frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {row['source_frame']}")
        frame128 = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        ellipse128 = [
            float(row["gt_x_128"]),
            float(row["gt_y_128"]),
            float(row["gt_a_128"]),
            float(row["gt_b_128"]),
            float(row["gt_ang"]),
        ]
        ellipse64 = [
            float(row["gt_x_64"]),
            float(row["gt_y_64"]),
            float(row["gt_a_64"]),
            float(row["gt_b_64"]),
            float(row["gt_ang"]),
        ]
        event64 = make_event64(load_event_segment(int(row["source_global_idx"]), self.data_root))
        mask128 = make_mask128(ellipse128)
        regions = sample_regions_from_ellipse64(ellipse64)
        patches = event_patches(event64, regions)
        return {
            "frame": torch.from_numpy(frame128.astype(np.float32) / 255.0).unsqueeze(0),
            "mask": torch.from_numpy(mask128.astype(np.int64)),
            "event": torch.from_numpy(event64.astype(np.float32)),
            "ellipse128": torch.tensor(ellipse128, dtype=torch.float32),
            "input_volume": torch.from_numpy(event64.astype(np.float32)),
            "event_patches": torch.from_numpy(patches.astype(np.float32)),
            "pre_state": torch.tensor(ellipse64, dtype=torch.float32),
            "pupil": torch.tensor(ellipse128, dtype=torch.float32),
            "sample_regions": torch.tensor(regions, dtype=torch.int64),
            "meta": {
                "sample_idx": int(row["sample_idx"]),
                "subject": int(row["subject"]),
                "eye": row["eye"],
                "session_code": int(row["session_code"]),
                "frame_idx": int(row["frame_idx"]),
            },
        }


def summarize_tensor(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def run_loader_smoke(tag: str, batch_size: int, target_base: Path, source_root: Path, live: bool) -> dict:
    ev_root = target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent" / tag
    ex_root = target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent" / tag
    result = {
        "tag": tag,
        "batch_size": batch_size,
        "live": live,
        "ev_eye_root": str(ev_root),
        "ex_gaze_root": str(ex_root),
        "source_root": str(source_root),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        if live:
            ev_ds = HybridLiveManifestDataset(ev_root, source_root, split)
            ex_ds = HybridLiveManifestDataset(ex_root, source_root, split)
        else:
            ev_ds = HybridEVEyeSmokeDataset(ev_root, split)
            ex_ds = HybridEXGazeSmokeDataset(ex_root, split)
        ev_batch = next(iter(DataLoader(ev_ds, batch_size=batch_size, shuffle=False, num_workers=0)))
        ex_batch = next(iter(DataLoader(ex_ds, batch_size=batch_size, shuffle=False, num_workers=0)))

        checks = {
            "ev_frame": summarize_tensor(ev_batch["frame"]),
            "ev_mask": summarize_tensor(ev_batch["mask"]),
            "ev_event": summarize_tensor(ev_batch["event"]),
            "ev_ellipse128": summarize_tensor(ev_batch["ellipse128"]),
            "ex_frame": summarize_tensor(ex_batch["frame"]),
            "ex_input_volume": summarize_tensor(ex_batch["input_volume"]),
            "ex_event_patches": summarize_tensor(ex_batch["event_patches"]),
            "ex_pre_state": summarize_tensor(ex_batch["pre_state"]),
            "ex_pupil": summarize_tensor(ex_batch["pupil"]),
            "ex_sample_regions": summarize_tensor(ex_batch["sample_regions"]),
            "ev_len": len(ev_ds),
            "ex_len": len(ex_ds),
        }

        assert checks["ev_frame"]["shape"] == [batch_size, 1, 128, 128]
        assert checks["ev_mask"]["shape"] == [batch_size, 128, 128]
        assert checks["ev_event"]["shape"] == [batch_size, 2, 64, 64]
        assert checks["ev_ellipse128"]["shape"] == [batch_size, 5]
        assert checks["ex_frame"]["shape"] == [batch_size, 1, 128, 128]
        assert checks["ex_input_volume"]["shape"] == [batch_size, 2, 64, 64]
        assert checks["ex_event_patches"]["shape"] == [batch_size, 8, 2, 16, 16]
        assert checks["ex_pre_state"]["shape"] == [batch_size, 5]
        assert checks["ex_pupil"]["shape"] == [batch_size, 5]
        assert checks["ex_sample_regions"]["shape"] == [batch_size, 8, 4]
        result["splits"][split] = checks
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# EV-Eye and EX-Gaze Hybrid Loader Smoke",
        "",
        "Date: 2026-07-04",
        "",
        f"- Tag: `{result['tag']}`",
        f"- Batch size: `{result['batch_size']}`",
        f"- Live cache mode: `{result['live']}`",
        f"- EV-Eye root: `{result['ev_eye_root']}`",
        f"- EX-Gaze root: `{result['ex_gaze_root']}`",
        f"- Source root: `{result['source_root']}`",
        "",
        "| Split | EV frame | EV mask | EV event | EX frame | EX volume | EX patches | EX regions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, data in result["splits"].items():
        lines.append(
            f"| {split} | `{data['ev_frame']['shape']}` | `{data['ev_mask']['shape']}` | "
            f"`{data['ev_event']['shape']}` | `{data['ex_frame']['shape']}` | "
            f"`{data['ex_input_volume']['shape']}` | `{data['ex_event_patches']['shape']}` | "
            f"`{data['ex_sample_regions']['shape']}` |"
        )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            "- EV-Eye loader smoke passed for `frame`, `mask`, `event`, and `ellipse128` tensors.",
            "- EX-Gaze loader smoke passed for `frame`, `input_volume`, `event_patches`, `pre_state`, `pupil`, and `sample_regions` tensors.",
            "- This validates the package as a PyTorch-loadable contract. It does not yet validate the final model training loops.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="smoke_32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--live", action="store_true", help="Use compact manifests and source FACET cache instead of exported tensors.")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_loader_smoke_2026-07-04.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_loader_smoke_2026-07-04.md"),
    )
    parser.add_argument(
        "--mirror-markdown-output",
        type=Path,
        default=Path("references/report/EX-Gaze/EV_Eye_EX_Gaze_hybrid_loader_smoke_2026-07-04.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_loader_smoke(args.tag, args.batch_size, args.target_base, args.source_root, args.live)
    markdown = render_markdown(result)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown + "\n")
    if args.mirror_markdown_output:
        args.mirror_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.mirror_markdown_output.write_text(markdown + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
