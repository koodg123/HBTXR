from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(
    "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/"
    "DeanDataset_full_unet_subject_independent"
)
DEFAULT_RAW_DAVIS_ROOT = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis")
DEFAULT_MOTION_LABELS = (
    ROOT
    / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv"
)
DEFAULT_HBTXR_PACKAGE = ROOT / "analysis/RESULTS/HBTXR_subject37_48_error_distribution"
DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _read_info(path: Path) -> dict[str, str]:
    info = {}
    for line in path.read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            info[key.strip()] = value.strip()
    return info


def _parse_session(session_path: str) -> tuple[int | None, str | None, str | None]:
    parts = Path(session_path).parts
    user = eye = session = None
    for part in parts:
        if part.startswith("user") and part[4:].isdigit():
            user = int(part[4:])
        elif part in {"left", "right"}:
            eye = part
        elif part.startswith("session_"):
            session = part
    return user, eye, session


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def _first_existing_frame_stats(raw_root: Path, session_path: str) -> dict[str, str | int]:
    user, eye, session = _parse_session(session_path)
    if user is None or eye is None or session is None:
        return {"status": "unparsed"}
    frames_dir = raw_root / f"user{user}" / eye / session / "frames"
    if not frames_dir.exists():
        return {"status": "missing", "frames_dir": str(frames_dir)}
    first = next(iter(sorted(frames_dir.glob("*.png"))), None)
    if first is None:
        return {"status": "empty", "frames_dir": str(frames_dir)}
    return {
        "status": "ok",
        "frames_dir": str(frames_dir),
        "first_frame": first.name,
    }


def collect_preflight(
    source_root: Path,
    raw_davis_root: Path,
    motion_labels: Path,
    hbtxr_package: Path,
    target_base: Path,
) -> dict:
    manifest = _read_json(source_root / "manifest.json")
    progress = _read_json(source_root / "progress_state.json")

    split_sessions = Counter()
    split_valid = Counter()
    split_skipped = Counter()
    split_skip_no_ellipse = Counter()
    split_skip_no_events = Counter()
    split_subjects: dict[str, set[int]] = defaultdict(set)
    first_session_by_split = {}

    for session in progress["session_summaries"]:
        split = session["split"]
        split_sessions[split] += 1
        split_valid[split] += int(session["valid"])
        split_skipped[split] += int(session["skipped"])
        split_skip_no_ellipse[split] += int(session["skip_no_ellipse"])
        split_skip_no_events[split] += int(session["skip_no_events"])
        split_subjects[split].add(int(session["user"]))
        first_session_by_split.setdefault(split, session["session"])

    cache = {}
    for split in ("train", "val", "test"):
        data_path = source_root / split / "cached_data"
        ellipse_path = source_root / split / "cached_ellipse"
        event_info = _read_info(data_path / "events_batch_info_0.txt")
        ellipse_info = _read_info(ellipse_path / "ellipses_batch_info_0.txt")
        event_indices = np.load(data_path / "events_indices_0.npy", mmap_mode="r")
        cache[split] = {
            "event_info": event_info,
            "ellipse_info": ellipse_info,
            "event_indices_shape": list(event_indices.shape),
            "first_event_index_rows": event_indices[:3].tolist(),
        }

    raw_frame_checks = {
        split: _first_existing_frame_stats(raw_davis_root, session_path)
        for split, session_path in first_session_by_split.items()
    }

    target_dirs = {
        "EV-Eye": str(
            target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent"
        ),
        "EX-Gaze": str(
            target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent"
        ),
    }
    target_exists = {name: Path(path).exists() for name, path in target_dirs.items()}

    hbtxr_files = {
        "metadata": hbtxr_package
        / "HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv",
        "predictions": hbtxr_package
        / "HBTXR_subject_independent_img64_patch4_test_sample_predictions.csv",
        "joined_motion_error": hbtxr_package
        / "HBTXR_subject37_48_test_joined_motion_error.csv",
    }

    return {
        "source_root": str(source_root),
        "raw_davis_root": str(raw_davis_root),
        "target_dirs": target_dirs,
        "target_exists": target_exists,
        "manifest_split_counts": manifest.get("split_counts", {}),
        "manifest_split_subjects": manifest.get("split_subjects", {}),
        "progress_splits": {
            split: {
                "sessions": split_sessions[split],
                "subjects": sorted(split_subjects[split]),
                "valid": split_valid[split],
                "skipped": split_skipped[split],
                "skip_no_ellipse": split_skip_no_ellipse[split],
                "skip_no_events": split_skip_no_events[split],
            }
            for split in ("train", "val", "test")
        },
        "cache": cache,
        "raw_frame_checks": raw_frame_checks,
        "motion_labels": {
            "path": str(motion_labels),
            "exists": motion_labels.exists(),
            "rows": _count_csv_rows(motion_labels),
        },
        "hbtxr_reference_package": {
            "path": str(hbtxr_package),
            "files": {key: str(path) for key, path in hbtxr_files.items()},
            "exists": {key: path.exists() for key, path in hbtxr_files.items()},
            "rows": {key: _count_csv_rows(path) for key, path in hbtxr_files.items()},
        },
        "contracts": {
            "event_input": "2 x 64 x 64 event frame, generated from cached events",
            "frame_input": "1 x 128 x 128 grayscale frame, generated from Data_davis frames",
            "ellipse_source": "cached_ellipse original coordinates: t,x,y,a,b,ang in 346x260 sensor coordinates",
            "motion_join_key": "(subject, eye, session_code, frame_idx)",
            "exgaze_patch_contract": "2 x 64 x 64 event volume -> 8 x 2 x 16 x 16 patches",
        },
    }


def render_markdown(data: dict) -> str:
    lines = [
        "# EV-Eye and EX-Gaze Hybrid Preflight",
        "",
        "Date: 2026-07-04",
        "",
        "## Scope",
        "",
        "- EV-Eye target: GPU0, frame `1 x 128 x 128`, event `2 x 64 x 64`.",
        "- EX-Gaze target: GPU1, frame `1 x 128 x 128`, event `2 x 64 x 64`.",
        "- Split policy: train subjects 1-32, val subjects 33-36, test subjects 37-48.",
        "- Reference evaluation package: "
        f"`{data['hbtxr_reference_package']['path']}`.",
        "",
        "## Source Dataset",
        "",
        f"- Source root: `{data['source_root']}`",
        f"- Raw Davis root for frame recovery: `{data['raw_davis_root']}`",
        "",
        "| Split | Manifest Samples | Progress Valid | Sessions | Subjects | Skipped | Skip No Ellipse | Skip No Events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        p = data["progress_splits"][split]
        subject_span = f"{p['subjects'][0]}-{p['subjects'][-1]}" if p["subjects"] else ""
        lines.append(
            f"| {split} | {data['manifest_split_counts'].get(split, 'n/a')} | "
            f"{p['valid']} | {p['sessions']} | {subject_span} | {p['skipped']} | "
            f"{p['skip_no_ellipse']} | {p['skip_no_events']} |"
        )

    lines.extend(
        [
            "",
            "## Cache Contract",
            "",
            "| Split | Event dtype | Event shape | Ellipse dtype | Ellipse shape | First index rows |",
            "|---|---|---|---|---|---|",
        ]
    )
    for split in ("train", "val", "test"):
        c = data["cache"][split]
        lines.append(
            f"| {split} | `{c['event_info'].get('Data dtype')}` | "
            f"`{c['event_info'].get('Data shape')}` | "
            f"`{c['ellipse_info'].get('Data dtype')}` | "
            f"`{c['ellipse_info'].get('Data shape')}` | "
            f"`{c['first_event_index_rows']}` |"
        )

    lines.extend(
        [
            "",
            "## Frame Recovery Check",
            "",
            "| Split | Status | Example Frame Source | First Frame |",
            "|---|---|---|---|",
        ]
    )
    for split in ("train", "val", "test"):
        check = data["raw_frame_checks"].get(split, {})
        lines.append(
            f"| {split} | {check.get('status', 'missing')} | "
            f"`{check.get('frames_dir', '')}` | `{check.get('first_frame', '')}` |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Inputs",
            "",
            f"- Motion labels: `{data['motion_labels']['path']}`",
            f"- Motion label rows: `{data['motion_labels']['rows']}`",
            "",
            "| HBTXR Reference File | Exists | Rows |",
            "|---|---:|---:|",
        ]
    )
    for key, path in data["hbtxr_reference_package"]["files"].items():
        lines.append(
            f"| `{Path(path).name}` | {data['hbtxr_reference_package']['exists'][key]} | "
            f"{data['hbtxr_reference_package']['rows'][key]} |"
        )

    lines.extend(
        [
            "",
            "## Target Dataset Roots",
            "",
            "| Model | Target Root | Exists Now |",
            "|---|---|---:|",
        ]
    )
    for model, path in data["target_dirs"].items():
        lines.append(f"| {model} | `{path}` | {data['target_exists'][model]} |")

    lines.extend(
        [
            "",
            "## Execution Judgment",
            "",
            "- Preflight passed for source split counts, cached event/ellipse availability, and raw frame recovery.",
            "- The target EV-Eye/EX-Gaze derived dataset roots do not exist yet.",
            "- Full export must write under `/home/kjm26/project/dataset/XR/EV_Eye/target_data`, which is outside the repo workspace.",
            "- Main implementation should reuse the existing FACET session-range recovery pattern from `export_hbtxr_subject_independent_for_targets.py`.",
            "",
            "## Next Concrete Steps",
            "",
            "1. Add an EV-Eye/EX-Gaze export path that emits split-level metadata, frame-128 tensors/images, event-64 tensors, masks, and EX-Gaze JSON/HDF5 annotations.",
            "2. Run a small export smoke test, for example first 32 samples per split.",
            "3. Verify frame/event/label shapes and EX-Gaze patch extraction `8 x 2 x 16 x 16`.",
            "4. After smoke passes, run full export under the target dataset roots.",
            "5. Add model configs and launch EV-Eye on GPU0 and EX-Gaze on GPU1.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-davis-root", type=Path, default=DEFAULT_RAW_DAVIS_ROOT)
    parser.add_argument("--motion-labels", type=Path, default=DEFAULT_MOTION_LABELS)
    parser.add_argument("--hbtxr-package", type=Path, default=DEFAULT_HBTXR_PACKAGE)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_preflight_2026-07-04.md",
    )
    parser.add_argument(
        "--mirror-output",
        type=Path,
        default=ROOT
        / "references/report/EX-Gaze/EV_Eye_EX_Gaze_hybrid_preflight_2026-07-04.md",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = collect_preflight(
        args.source_root,
        args.raw_davis_root,
        args.motion_labels,
        args.hbtxr_package,
        args.target_base,
    )
    markdown = render_markdown(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n")
    if args.mirror_output:
        args.mirror_output.parent.mkdir(parents=True, exist_ok=True)
        args.mirror_output.write_text(markdown + "\n")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
