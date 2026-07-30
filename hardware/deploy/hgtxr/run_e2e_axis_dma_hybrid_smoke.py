from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hgtxr.e2e_axis_dma_overlay import E2EAxisDmaRuntimeConfig, HgtxrE2EAxisDmaOverlay
    from hgtxr.run_e2e_axis_dma_smoke import (
        ARTIFACT_PREFIXES,
        MODE_PROFILE_CONTROL,
        MODE_PROFILE_SHAPES,
        artifact_paths,
        make_frame,
        summarize_optional_samples,
        summarize_samples,
    )
else:
    from .e2e_axis_dma_overlay import E2EAxisDmaRuntimeConfig, HgtxrE2EAxisDmaOverlay
    from .run_e2e_axis_dma_smoke import (
        ARTIFACT_PREFIXES,
        MODE_PROFILE_CONTROL,
        MODE_PROFILE_SHAPES,
        artifact_paths,
        make_frame,
        summarize_optional_samples,
        summarize_samples,
    )


EXPECTED_BY_VARIANT = {
    "par32-prefetchall4-300": {
        "search": {
            "runtime_state": 0,
            "out_raw": [-1169, -1169, -1169, -1169, -1169, -1125],
            "max_accelerator_latency_ms": 4.0,
        },
        "track": {
            "runtime_state": 1,
            "out_raw": [-235, -235, -235, -235, -235, -239],
            "max_accelerator_latency_ms": 1.0,
        },
    },
    "par32-patch32-dtok4-300": {
        "search": {
            "runtime_state": 0,
            "out_raw": [3, 3, 3, 3, 3, 3],
            "max_accelerator_latency_ms": 4.0,
        },
        "track": {
            "runtime_state": 1,
            "out_raw": [217, 217, 217, 217, 217, 217],
            "max_accelerator_latency_ms": 1.0,
        },
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGTXR E2E AXIS/DMA interleaved Search/Track smoke.")
    parser.add_argument("--variant", choices=sorted(ARTIFACT_PREFIXES), default="par32-prefetchall4-300")
    parser.add_argument("--bitfile", type=Path, default=None)
    parser.add_argument("--hwhfile", type=Path, default=None)
    parser.add_argument("--ip-name", default=None)
    parser.add_argument("--dma-in-name", default="axi_dma_in")
    parser.add_argument("--dma-out-name", default="axi_dma_out")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-s", type=float, default=0.001)
    parser.add_argument("--frame-pattern", choices=["ramp", "zero"], default="ramp")
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--repeat-cycles", type=int, default=5)
    parser.add_argument("--sequence", choices=["search1-track9"], default="search1-track9")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.variant not in {"par32-prefetchall4-300", "par32-patch32-dtok4-300"}:
        parser.error("hybrid smoke currently supports --variant par32-prefetchall4-300 or par32-patch32-dtok4-300")
    if args.warmup_cycles < 0:
        parser.error("--warmup-cycles must be >= 0")
    if args.repeat_cycles < 1:
        parser.error("--repeat-cycles must be >= 1")
    return args


def mode_sequence(sequence: str) -> list[str]:
    if sequence == "search1-track9":
        return ["search"] + ["track"] * 9
    raise ValueError(f"unsupported sequence: {sequence}")


def run_one(
    runtime: HgtxrE2EAxisDmaOverlay,
    mode: str,
    frame_pattern: str,
    index: int,
    measured: bool,
    expected_by_mode: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    frame = make_frame(frame_pattern, MODE_PROFILE_SHAPES[mode])
    out_state, runtime_state_arr, metrics = runtime.run_profiled(
        frame,
        weights_u32=None,
        num_pixels_control=MODE_PROFILE_CONTROL[mode],
    )
    runtime_state = int(np.asarray(runtime_state_arr, dtype=np.int32).reshape(-1)[0])
    out_raw = (np.asarray(out_state, dtype=np.float32).reshape(-1) * 16.0).astype(np.int32).tolist()
    expected = expected_by_mode[mode]
    return {
        "index": index,
        "measured": measured,
        "mode_label": mode,
        "runtime_state": runtime_state,
        "expected_runtime_state": expected["runtime_state"],
        "runtime_match": runtime_state == expected["runtime_state"],
        "out_state": np.asarray(out_state, dtype=np.float32).reshape(-1).tolist(),
        "out_raw": out_raw,
        "expected_out_raw": expected["out_raw"],
        "output_match": out_raw == expected["out_raw"],
        "metrics": metrics,
    }


def samples_for(runs: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        value = run["metrics"].get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def ms_samples_for(runs: list[dict[str, Any]], field: str) -> list[float]:
    return [value * 1000.0 for value in samples_for(runs, field)]


def mode_summary(runs: list[dict[str, Any]], mode: str, expected_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mode_runs = [run for run in runs if run["mode_label"] == mode]
    latency_ms = ms_samples_for(mode_runs, "latency_s")
    accelerator_ms = ms_samples_for(mode_runs, "kernel_observed_s")
    dma_bps = samples_for(mode_runs, "input_dma_measured_bandwidth_Bps")
    aggregate_dma_bps = samples_for(mode_runs, "aggregate_dma_effective_bandwidth_Bps")
    max_target = float(expected_by_mode[mode]["max_accelerator_latency_ms"])
    accelerator_summary = summarize_samples(accelerator_ms)
    max_latency = accelerator_summary.get("max")
    target_pass = isinstance(max_latency, (int, float)) and float(max_latency) <= max_target
    return {
        "count": len(mode_runs),
        "expected_runtime_state": expected_by_mode[mode]["runtime_state"],
        "expected_out_raw": expected_by_mode[mode]["out_raw"],
        "runtime_match": all(run["runtime_match"] for run in mode_runs),
        "output_match": all(run["output_match"] for run in mode_runs),
        "latency_ms_samples": latency_ms,
        "latency_ms_summary": summarize_samples(latency_ms),
        "accelerator_latency_ms_samples": accelerator_ms,
        "accelerator_latency_ms_summary": accelerator_summary,
        "dma_bandwidth_Bps_samples": dma_bps,
        "dma_bandwidth_Bps_summary": summarize_optional_samples(dma_bps),
        "aggregate_dma_effective_bandwidth_Bps_samples": aggregate_dma_bps,
        "aggregate_dma_effective_bandwidth_Bps_summary": summarize_optional_samples(aggregate_dma_bps),
        "max_accelerator_latency_ms": max_target,
        "latency_target_match": target_pass,
    }


def run_hybrid_smoke(args: argparse.Namespace) -> dict[str, Any]:
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
    sequence = mode_sequence(args.sequence)
    expected_by_mode = EXPECTED_BY_VARIANT[args.variant]
    measured_runs: list[dict[str, Any]] = []
    warmup_runs: list[dict[str, Any]] = []
    with HgtxrE2EAxisDmaOverlay(config) as runtime:
        index = 0
        for _cycle in range(args.warmup_cycles):
            for mode in sequence:
                warmup_runs.append(run_one(runtime, mode, args.frame_pattern, index, measured=False, expected_by_mode=expected_by_mode))
                index += 1
        for _cycle in range(args.repeat_cycles):
            for mode in sequence:
                measured_runs.append(run_one(runtime, mode, args.frame_pattern, index, measured=True, expected_by_mode=expected_by_mode))
                index += 1

    latency_ms = ms_samples_for(measured_runs, "latency_s")
    accelerator_ms = ms_samples_for(measured_runs, "kernel_observed_s")
    dma_bps = samples_for(measured_runs, "input_dma_measured_bandwidth_Bps")
    aggregate_dma_bps = samples_for(measured_runs, "aggregate_dma_effective_bandwidth_Bps")
    distribution = {
        "search": sum(1 for run in measured_runs if run["mode_label"] == "search"),
        "track": sum(1 for run in measured_runs if run["mode_label"] == "track"),
        "unknown": 0,
    }
    search_summary = mode_summary(measured_runs, "search", expected_by_mode)
    track_summary = mode_summary(measured_runs, "track", expected_by_mode)
    checks_pass = (
        all(run["runtime_match"] and run["output_match"] for run in measured_runs)
        and search_summary["latency_target_match"]
        and track_summary["latency_target_match"]
    )
    status_pass = args.no_check or checks_pass
    return {
        "status": "pass" if status_pass else "fail",
        "variant": args.variant,
        "hybrid_profile": "search10-track90",
        "bitfile": str(bitfile),
        "hwhfile": str(hwhfile),
        "dma_in_name": args.dma_in_name,
        "dma_out_name": args.dma_out_name,
        "frame_pattern": args.frame_pattern,
        "weights_mode": "zero",
        "sequence": sequence,
        "warmup_cycles": args.warmup_cycles,
        "repeat_cycles": args.repeat_cycles,
        "repeat": len(measured_runs),
        "batch_size": 1,
        "search_track_invocation_distribution": distribution,
        "search_track_distribution_note": "Measured interleaved runner sequence: one Search invocation followed by nine Track invocations per cycle.",
        "mode_summaries": {
            "search": search_summary,
            "track": track_summary,
        },
        "runtime_match": all(run["runtime_match"] for run in measured_runs),
        "output_match": all(run["output_match"] for run in measured_runs),
        "latency_target_match": bool(search_summary["latency_target_match"] and track_summary["latency_target_match"]),
        "latency_ms_samples": latency_ms,
        "latency_ms_summary": summarize_samples(latency_ms),
        "accelerator_latency_ms_samples": accelerator_ms,
        "accelerator_latency_ms_summary": summarize_samples(accelerator_ms),
        "throughput_fps_summary": summarize_optional_samples(
            [(1000.0 / value) if value > 0.0 else None for value in accelerator_ms]
        ),
        "board_throughput_fps_summary": summarize_optional_samples(
            [(1000.0 / value) if value > 0.0 else None for value in latency_ms]
        ),
        "dma_bandwidth_Bps_samples": dma_bps,
        "dma_bandwidth_Bps_summary": summarize_optional_samples(dma_bps),
        "aggregate_dma_effective_bandwidth_Bps_samples": aggregate_dma_bps,
        "aggregate_dma_effective_bandwidth_Bps_summary": summarize_optional_samples(aggregate_dma_bps),
        "measurement_runs": measured_runs,
        "warmup_runs": warmup_runs,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_hybrid_smoke(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        write_json(args.json_out, result)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
