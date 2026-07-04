from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from software.dataset import make_synthetic_batch
from software.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="hardware/refs")
    args = parser.parse_args()
    root = Path(args.output_root)
    batch = make_synthetic_batch(1)
    ensure_dir(root / "inputs")
    ensure_dir(root / "outputs")
    np.save(root / "inputs" / "frame.npy", batch["frame"].numpy())
    np.save(root / "inputs" / "event.npy", batch["event"].numpy())
    np.save(root / "inputs" / "prev_state.npy", batch["prev_state"].numpy())
    np.save(root / "outputs" / "target_state.npy", batch["target_state"].numpy())
    write_json({"status": "ok", "format": "npy", "root": str(root)}, root / "manifest.json")
    print({"output_root": str(root)})


if __name__ == "__main__":
    main()

