from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from software.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal manifest from a canonical root.")
    parser.add_argument("--canonical-root", default="data/_internal/canonical")
    parser.add_argument("--output", default="data/_internal/manifests/train_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.canonical_root)
    frames = sorted(root.rglob("*.png"))
    if args.limit:
        frames = frames[: args.limit]
    rows = [{"sample_id": f.stem, "frame_path": str(f), "ellipse_xywht": [128, 128, 40, 24, 0]} for f in frames]
    write_jsonl(rows, args.output)
    print({"rows": len(rows), "output": args.output})


if __name__ == "__main__":
    main()

