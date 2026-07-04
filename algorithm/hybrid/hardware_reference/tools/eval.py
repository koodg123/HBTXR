from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from software.config import apply_overrides, load_config
from software.evaluator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="software/configs/eval.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    print(evaluate(cfg, checkpoint=args.checkpoint))


if __name__ == "__main__":
    main()

