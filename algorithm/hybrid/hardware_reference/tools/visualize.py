from __future__ import annotations

import argparse

import numpy as np

import _bootstrap  # noqa: F401
from software.visualization import save_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/visualize/sample.png")
    args = parser.parse_args()
    frame = np.zeros((256, 256), dtype=np.float32)
    out = save_overlay(frame, [128, 128, 42, 26, 0], args.output)
    print({"output": str(out)})


if __name__ == "__main__":
    main()

