from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from software.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-key", default="synthetic/session")
    parser.add_argument("--prepared-root", default="data/_internal/timelens_ready")
    args = parser.parse_args()
    root = ensure_dir(Path(args.prepared_root))
    ensure_dir(root / "images")
    ensure_dir(root / "events")
    summary = {"session_key": args.session_key, "prepared_root": str(root), "status": "prepared_empty_structure"}
    write_json(summary, root / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()

