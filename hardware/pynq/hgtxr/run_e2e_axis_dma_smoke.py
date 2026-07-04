from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hgtxr.e2e_axis_dma_overlay import DEFAULT_BIT, DEFAULT_HWH, E2EAxisDmaRuntimeConfig, HgtxrE2EAxisDmaOverlay
    from hgtxr.e2e_m_axi_weights import EXPECTED_RAW, build_active196_b6_ff768_weights
else:
    from .e2e_axis_dma_overlay import DEFAULT_BIT, DEFAULT_HWH, E2EAxisDmaRuntimeConfig, HgtxrE2EAxisDmaOverlay
    from .e2e_m_axi_weights import EXPECTED_RAW, build_active196_b6_ff768_weights

FRAME_SHAPE = (256, 256)
SEARCH_FRAME_SHAPE = (128, 128)
TRACK_FRAME_SHAPE = (64, 64)
MODE_PROFILE_SHAPES = {
    "full": FRAME_SHAPE,
    "search": SEARCH_FRAME_SHAPE,
    "track": TRACK_FRAME_SHAPE,
}
MODE_PROFILE_CONTROL = {
    "full": None,
    "search": SEARCH_FRAME_SHAPE[0] * SEARCH_FRAME_SHAPE[1],
    "track": 1,
}
MODE_PROFILE_RUNTIME_STATE = {
    "search": 0,
    "track": 1,
}
ARTIFACT_PREFIXES = {
    "a1": "hgtxr_e2e_axis_dma",
    "c1-par16": "hgtxr_e2e_axis_dma_par16",
    "c3b-mem16": "hgtxr_e2e_axis_dma_c3b_mem16",
    "runtime-mode-par32": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
    "par32-rom-compute-300": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_300_mem16"
    ),
    "par32-prefetchall4-300": (
        "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
        "corefabric_normuram_compute_prefetchall4_300_mem16"
    ),
    "par32-patch32-dtok4-300": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
    "vref-p0-softmax-input-x2-dsp-mixed-stream": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
    "vref-p0-softmax-input-x2-qkv-uram": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
}


def make_frame(pattern: str, shape: tuple[int, int] = FRAME_SHAPE) -> np.ndarray:
    if pattern == "zero":
        return np.zeros(shape, dtype=np.float32)
    if pattern == "ramp":
        y, x = np.indices(shape, dtype=np.int32)
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


def artifact_paths(variant: str) -> tuple[Path, Path]:
    prefix = ARTIFACT_PREFIXES[variant]
    base = DEFAULT_BIT.parent
    return base / f"{prefix}.bit", base / f"{prefix}.hwh"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGTXR A1 E2E AXIS/DMA PYNQ smoke.")
    parser.add_argument("--variant", choices=sorted(ARTIFACT_PREFIXES), default="a1")
    parser.add_argument("--bitfile", type=Path, default=None)
    parser.add_argument("--hwhfile", type=Path, default=None)
    parser.add_argument("--ip-name", default=None)
    parser.add_argument("--dma-in-name", default="axi_dma_in")
    parser.add_argument("--dma-out-name", default="axi_dma_out")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-s", type=float, default=0.001)
    parser.add_argument("--frame-pattern", choices=["ramp", "zero"], default="ramp")
    parser.add_argument("--weights-mode", choices=["live", "zero", "golden", "file"], default="live")
    parser.add_argument("--weights-bin", type=Path, default=None)
    parser.add_argument("--expect-runtime-state", type=int, default=None)
    parser.add_argument("--expect-out-raw", type=int, nargs=6, default=None)
    parser.add_argument("--mode-profile", choices=sorted(MODE_PROFILE_SHAPES), default="full")
    parser.add_argument("--repeat", type=int, default=1, help="number of measured board invocations")
    parser.add_argument("--warmup", type=int, default=0, help="number of unmeasured warmup invocations")
    parser.add_argument("--mode-label", choices=["search", "track", "unknown"], default="unknown")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    return args


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct, method="linear"))


def summarize_samples(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "p99": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
    }


def summarize_optional_samples(values: list[float | None]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    summary: dict[str, Any] = summarize_samples(present)
    summary["missing_count"] = len(values) - len(present)
    return summary


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    expected_runtime_state = (
        MODE_PROFILE_RUNTIME_STATE.get(args.mode_profile, default_expected_runtime_state(args.weights_mode))
        if args.expect_runtime_state is None
        else int(args.expect_runtime_state)
    )
    default_bitfile, default_hwhfile = artifact_paths(args.variant)
    bitfile = default_bitfile if args.bitfile is None else args.bitfile
    hwhfile = default_hwhfile if args.hwhfile is None else args.hwhfile
    config = E2EAxisDmaRuntimeConfig(
        bitfile=bitfile,
        hwhfile=hwhfile,
        ip_name=args.ip_name,
        dma_in_name=args.dma_in_name,
        dma_out_name=args.dma_out_name,
        timeout_s=args.timeout_s,
        poll_s=args.poll_s,
    )
    frame = make_frame(args.frame_pattern, MODE_PROFILE_SHAPES[args.mode_profile])
    weights = make_weights(args.weights_mode, args.weights_bin)
    num_pixels_control = MODE_PROFILE_CONTROL[args.mode_profile]
    mode_label = args.mode_profile if args.mode_label == "unknown" and args.mode_profile in {"search", "track"} else args.mode_label
    measured_runs: list[dict[str, Any]] = []
    with HgtxrE2EAxisDmaOverlay(config) as runtime:
        for _ in range(args.warmup):
            runtime.run(frame, weights_u32=weights, num_pixels_control=num_pixels_control)
        for index in range(args.repeat):
            out_state, runtime_state_arr, metrics = runtime.run_profiled(
                frame, weights_u32=weights, num_pixels_control=num_pixels_control
            )
            measured_runs.append(
                {
                    "index": index,
                    "mode_label": mode_label,
                    "out_state": np.asarray(out_state, dtype=np.float32).reshape(-1).tolist(),
                    "runtime_state": int(np.asarray(runtime_state_arr, dtype=np.int32).reshape(-1)[0]),
                    "metrics": metrics,
                }
            )

    final_run = measured_runs[-1]
    out_state = np.asarray(final_run["out_state"], dtype=np.float32)
    runtime_state_arr = np.asarray([final_run["runtime_state"]], dtype=np.int32)
    runtime_values = np.asarray(runtime_state_arr, dtype=np.int32).reshape(-1)
    runtime_state = int(runtime_values[0]) if runtime_values.size else None
    out_raw = (np.asarray(out_state, dtype=np.float32).reshape(-1) * 16.0).astype(np.int32).tolist()
    expected_out_raw = list(EXPECTED_RAW) if args.weights_mode == "golden" else args.expect_out_raw
    runtime_matches = [run["runtime_state"] == expected_runtime_state for run in measured_runs]
    output_matches = []
    for run in measured_runs:
        run_raw = (np.asarray(run["out_state"], dtype=np.float32).reshape(-1) * 16.0).astype(np.int32).tolist()
        output_matches.append(True if expected_out_raw is None else run_raw == list(expected_out_raw))
    runtime_match = all(runtime_matches)
    output_match = all(output_matches)
    passed = args.no_check or (runtime_match and output_match)

    latency_ms_samples = [float(run["metrics"]["latency_s"]) * 1000.0 for run in measured_runs]
    kernel_observed_ms_samples = [float(run["metrics"]["kernel_observed_s"]) * 1000.0 for run in measured_runs]
    output_wait_plus_compute_ms_samples = [
        float(run["metrics"]["output_dma_wait_plus_compute_s"]) * 1000.0 for run in measured_runs
    ]
    input_dma_bandwidth_samples = [run["metrics"]["input_dma_measured_bandwidth_Bps"] for run in measured_runs]
    aggregate_dma_bandwidth_samples = [run["metrics"]["aggregate_dma_effective_bandwidth_Bps"] for run in measured_runs]
    mode_counts = {"search": 0, "track": 0, "unknown": 0}
    mode_counts[mode_label] = len(measured_runs)
    return {
        "status": "pass" if passed else "fail",
        "variant": args.variant,
        "bitfile": str(bitfile),
        "hwhfile": str(hwhfile),
        "dma_in_name": args.dma_in_name,
        "dma_out_name": args.dma_out_name,
        "frame_pattern": args.frame_pattern,
        "weights_mode": args.weights_mode,
        "mode_profile": args.mode_profile,
        "num_pixels_control": num_pixels_control,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "mode_label": mode_label,
        "expected_runtime_state": expected_runtime_state,
        "runtime_state": runtime_state,
        "runtime_match": runtime_match,
        "out_state": np.asarray(out_state, dtype=np.float32).reshape(-1).tolist(),
        "out_raw": out_raw,
        "expected_out_raw": expected_out_raw,
        "output_match": output_match,
        "board_latency_ms": latency_ms_samples[-1],
        "latency_ms_samples": latency_ms_samples,
        "latency_ms_summary": summarize_samples(latency_ms_samples),
        "accelerator_latency_ms": kernel_observed_ms_samples[-1],
        "accelerator_latency_ms_samples": kernel_observed_ms_samples,
        "accelerator_latency_ms_summary": summarize_samples(kernel_observed_ms_samples),
        "output_wait_plus_compute_ms_samples": output_wait_plus_compute_ms_samples,
        "output_wait_plus_compute_ms_summary": summarize_samples(output_wait_plus_compute_ms_samples),
        "throughput_fps_summary": summarize_optional_samples(
            [(1000.0 / value) if value > 0.0 else None for value in latency_ms_samples]
        ),
        "dma_bandwidth_Bps_samples": input_dma_bandwidth_samples,
        "dma_bandwidth_Bps_summary": summarize_optional_samples(input_dma_bandwidth_samples),
        "aggregate_dma_effective_bandwidth_Bps_samples": aggregate_dma_bandwidth_samples,
        "aggregate_dma_effective_bandwidth_Bps_summary": summarize_optional_samples(aggregate_dma_bandwidth_samples),
        "search_track_invocation_distribution": mode_counts,
        "search_track_distribution_note": (
            "C3b AXIS/DMA smoke does not expose hardware-internal search/track counters; "
            "this distribution reflects the runner --mode-label assigned to measured invocations."
        ),
        "measurement_runs": measured_runs,
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
