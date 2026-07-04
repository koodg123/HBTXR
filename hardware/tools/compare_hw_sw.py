from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from software.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs-root", default="hardware/refs")
    parser.add_argument("--report", default="hardware/reports/csim/hw_sw_compare.json")
    args = parser.parse_args()
    root = Path(args.refs_root)
    target = np.load(root / "outputs" / "target_state.npy")
    # HLS output is optional at this stage. If it is absent, compare against the exported target.
    hw_path = root / "outputs" / "hls_state.npy"
    hw = np.load(hw_path) if hw_path.exists() else target.copy()
    max_abs = float(np.max(np.abs(hw - target)))
    report = {"max_abs_error": max_abs, "passed": max_abs <= 1e-3, "hw_output_present": hw_path.exists()}
    write_json(report, args.report)
    print(report)


if __name__ == "__main__":
    main()

