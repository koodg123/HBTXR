from __future__ import annotations

import argparse

import torch

import _bootstrap  # noqa: F401
from software.modules.antiblink import AntiBlinkDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-checkpoint", default="runs/import_antiblink/antiblink_detector.pt")
    args = parser.parse_args()
    detector = AntiBlinkDetector()
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        detector.load_state_dict(state, strict=False)
    torch.save({"model": detector.state_dict()}, args.output_checkpoint)
    print({"output_checkpoint": args.output_checkpoint})


if __name__ == "__main__":
    main()

