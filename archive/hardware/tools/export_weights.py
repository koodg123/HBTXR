from __future__ import annotations

import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from software.config import load_config
from software.io import ensure_dir
from software.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="software/configs/base.yaml")
    parser.add_argument("--output-root", default="hardware/refs/weights")
    args = parser.parse_args()
    model = build_model(load_config(args.config))
    root = ensure_dir(args.output_root)
    torch.save(model.state_dict(), Path(root) / "software_initial_weights.pt")
    print({"output": str(Path(root) / "software_initial_weights.pt")})


if __name__ == "__main__":
    main()

