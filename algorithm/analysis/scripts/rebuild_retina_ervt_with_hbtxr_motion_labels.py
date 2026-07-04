#!/usr/bin/env python3
"""Rebuild Retina/ERVT result tables using the HBTXR joined motion label map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_ROOT = REPO_ROOT / "analysis" / "results"
HBTXR_PACKAGE = RESULTS_ROOT / "HBTXR"

sys.path.insert(0, str(SCRIPT_DIR))
from generate_retina_ervt_error_distribution import read_motion_labels, write_distribution_package  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("Retina", "ERVT"), default=["Retina", "ERVT"])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=HBTXR_PACKAGE / "HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv",
    )
    parser.add_argument(
        "--motion-label-map",
        type=Path,
        default=HBTXR_PACKAGE / "HBTXR_subject37_48_test_joined_motion_error.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    motion_labels = read_motion_labels(args.motion_label_map)

    for model in args.models:
        out_dir = args.results_root / model
        pred_path = out_dir / f"{model}_subject37_48_test_sample_predictions.csv"
        pred = pd.read_csv(pred_path)
        write_distribution_package(
            model,
            out_dir,
            metadata,
            pred,
            motion_labels,
            args.motion_label_map,
            overwrite=True,
        )
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
