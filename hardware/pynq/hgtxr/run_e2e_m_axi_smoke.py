from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hgtxr.e2e_m_axi_overlay import DEFAULT_BIT, DEFAULT_HWH, E2EMaxiRuntimeConfig, HgtxrE2EMaxiOverlay
    from hgtxr.e2e_m_axi_weights import EXPECTED_RAW, build_active196_b6_ff768_weights
else:
    from .e2e_m_axi_overlay import DEFAULT_BIT, DEFAULT_HWH, E2EMaxiRuntimeConfig, HgtxrE2EMaxiOverlay
    from .e2e_m_axi_weights import EXPECTED_RAW, build_active196_b6_ff768_weights

FRAME_SHAPE = (256, 256)


def make_frame(pattern: str) -> np.ndarray:
    if pattern == "zero":
        return np.zeros(FRAME_SHAPE, dtype=np.float32)
    if pattern == "ramp":
        y, x = np.indices(FRAME_SHAPE, dtype=np.int32)
        return (((x + y) & 0xFF).astype(np.float32) / 128.0).astype(np.float32)
    raise ValueError(f"unsupported frame pattern: {pattern}")


def make_weights(mode: str, weights_bin: Path | None = None) -> np.ndarray | None:
    if mode == "zero":
        return None
    if mode == "live":
        return np.array([1], dtype=np.uint32)
    if mode == "golden":
        return build_active196_b6_ff768_weights()
    if mode == "file":
        if weights_bin is None:
            raise ValueError("--weights-bin is required when --weights-mode file")
        return np.fromfile(weights_bin, dtype=np.uint32)
    raise ValueError(f"unsupported weights mode: {mode}")


def default_expected_runtime_state(mode: str) -> int:
    return 1 if mode == "zero" else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGTXR A2 E2E m_axi PYNQ smoke.")
    parser.add_argument("--bitfile", type=Path, default=DEFAULT_BIT)
    parser.add_argument("--hwhfile", type=Path, default=DEFAULT_HWH)
    parser.add_argument("--ip-name", default=None)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-s", type=float, default=0.001)
    parser.add_argument("--frame-pattern", choices=["ramp", "zero"], default="ramp")
    parser.add_argument("--weights-mode", choices=["live", "zero", "golden", "file"], default="live")
    parser.add_argument("--weights-bin", type=Path, default=None)
    parser.add_argument("--expect-runtime-state", type=int, default=None)
    parser.add_argument("--expect-out-raw", type=int, nargs=6, default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    expected_runtime_state = (
        default_expected_runtime_state(args.weights_mode)
        if args.expect_runtime_state is None
        else int(args.expect_runtime_state)
    )
    config = E2EMaxiRuntimeConfig(
        bitfile=args.bitfile,
        hwhfile=args.hwhfile,
        ip_name=args.ip_name,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
    )
    frame = make_frame(args.frame_pattern)
    weights = make_weights(args.weights_mode, args.weights_bin)
    with HgtxrE2EMaxiOverlay(config) as runtime:
        out_state, runtime_state_arr = runtime.run(frame, weights_u32=weights)

    runtime_values = np.asarray(runtime_state_arr, dtype=np.int32).reshape(-1)
    runtime_state = int(runtime_values[0]) if runtime_values.size else None
    out_raw = (np.asarray(out_state, dtype=np.float32).reshape(-1) * 16.0).astype(np.int32).tolist()
    expected_out_raw = list(EXPECTED_RAW) if args.weights_mode == "golden" else args.expect_out_raw
    runtime_match = runtime_state == expected_runtime_state
    output_match = True if expected_out_raw is None else out_raw == list(expected_out_raw)
    passed = args.no_check or (runtime_match and output_match)
    return {
        "status": "pass" if passed else "fail",
        "bitfile": str(args.bitfile),
        "hwhfile": str(args.hwhfile),
        "frame_pattern": args.frame_pattern,
        "weights_mode": args.weights_mode,
        "expected_runtime_state": expected_runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": runtime_match,
        "out_state": np.asarray(out_state, dtype=np.float32).reshape(-1).tolist(),
        "out_raw": out_raw,
        "expected_out_raw": expected_out_raw,
        "output_match": output_match,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        write_json(args.json_out, result)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
