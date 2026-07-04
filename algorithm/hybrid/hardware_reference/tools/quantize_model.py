from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--output", default="hardware/refs/weights/quantization.json")
    args = parser.parse_args()
    payload = {"activation_bits": args.activation_bits, "weight_bits": args.weight_bits, "status": "calibration_required"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(payload)


if __name__ == "__main__":
    main()

