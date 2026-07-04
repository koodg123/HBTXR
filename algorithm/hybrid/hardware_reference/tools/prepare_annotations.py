from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Placeholder for paper annotation backend integration.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", default="data/_internal/canonical")
    args = parser.parse_args()
    print({"status": "annotation_backend_not_bundled", "input_root": args.input_root, "output_root": args.output_root})


if __name__ == "__main__":
    main()

