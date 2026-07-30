from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from software.config import load_config
from software.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hardware/configs/vitis_hls.yaml")
    parser.add_argument("--output", default="hardware/refs/hls_config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    write_json(cfg, Path(args.output))
    print({"output": args.output})


if __name__ == "__main__":
    main()

