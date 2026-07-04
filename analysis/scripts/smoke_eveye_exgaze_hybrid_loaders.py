from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")


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


def summarize_tensor(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def run_loader_smoke(tag: str, batch_size: int, target_base: Path) -> dict:
    ev_root = target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent" / tag
    ex_root = target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent" / tag
    result = {
        "tag": tag,
        "batch_size": batch_size,
        "ev_eye_root": str(ev_root),
        "ex_gaze_root": str(ex_root),
        "splits": {},
    }

    for split in ("train", "val", "test"):
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
        f"- EV-Eye root: `{result['ev_eye_root']}`",
        f"- EX-Gaze root: `{result['ex_gaze_root']}`",
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
            "- This validates the exported smoke package as a PyTorch-loadable contract. It does not yet validate the final model training loops.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="smoke_32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
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
    result = run_loader_smoke(args.tag, args.batch_size, args.target_base)
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
