#!/usr/bin/env python3
"""Validate HGTXR PYNQ smoke result JSON captured from a physical board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

EXPECTED_RAW = [32, -13, 26, -6, 14, -11]
VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW = [58, -51, 42, -28, 36, -41]
PAR32_ROM_COMPUTE_300_SEARCH_EXPECTED_RAW = [-1169, -1169, -1169, -1169, -1169, -1133]
PAR32_ROM_COMPUTE_300_TRACK_EXPECTED_RAW = [-235, -235, -235, -235, -235, -226]
PAR32_ROM_COMPUTE_300_PREFIX = (
    "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
    "corefabric_normuram_compute_300_mem16"
)
PAR32_PREFETCHALL4_300_SEARCH_EXPECTED_RAW = [-1169, -1169, -1169, -1169, -1169, -1125]
PAR32_PREFETCHALL4_300_TRACK_EXPECTED_RAW = [-235, -235, -235, -235, -235, -239]
PAR32_PREFETCHALL4_300_PREFIX = (
    "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
    "corefabric_normuram_compute_prefetchall4_300_mem16"
)
PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW = [3, 3, 3, 3, 3, 3]
PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW = [217, 217, 217, 217, 217, 217]
PAR32_PATCH32_DTOK4_300_PREFIX = "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16"

VARIANT_PRESETS: dict[str, dict[str, Any]] = {
    "axis-a1": {
        "kind": "axis",
        "variant": "a1",
        "artifact_prefix": "hgtxr_e2e_axis_dma",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": EXPECTED_RAW,
    },
    "axis-c1-par16": {
        "kind": "axis",
        "variant": "c1-par16",
        "artifact_prefix": "hgtxr_e2e_axis_dma_par16",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": EXPECTED_RAW,
    },
    "axis-c3b-mem16": {
        "kind": "axis",
        "variant": "c3b-mem16",
        "artifact_prefix": "hgtxr_e2e_axis_dma_c3b_mem16",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": EXPECTED_RAW,
    },
    "axis-runtime-mode-par32-search": {
        "kind": "axis",
        "variant": "runtime-mode-par32",
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
        "weights_mode": "zero",
        "mode_profile": "search",
        "expected_runtime_state": 0,
        "expected_out_raw": None,
        "max_accelerator_latency_ms": 4.0,
    },
    "axis-runtime-mode-par32-track": {
        "kind": "axis",
        "variant": "runtime-mode-par32",
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
        "weights_mode": "zero",
        "mode_profile": "track",
        "expected_runtime_state": 1,
        "expected_out_raw": None,
        "max_accelerator_latency_ms": 1.0,
    },
    "axis-par32-rom-compute-300-search": {
        "kind": "axis",
        "variant": "par32-rom-compute-300",
        "artifact_prefix": PAR32_ROM_COMPUTE_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "search",
        "expected_runtime_state": 0,
        "expected_out_raw": PAR32_ROM_COMPUTE_300_SEARCH_EXPECTED_RAW,
        "max_accelerator_latency_ms": 4.0,
    },
    "axis-par32-rom-compute-300-track": {
        "kind": "axis",
        "variant": "par32-rom-compute-300",
        "artifact_prefix": PAR32_ROM_COMPUTE_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "track",
        "expected_runtime_state": 1,
        "expected_out_raw": PAR32_ROM_COMPUTE_300_TRACK_EXPECTED_RAW,
        "max_accelerator_latency_ms": 1.0,
    },
    "axis-par32-prefetchall4-300-search": {
        "kind": "axis",
        "variant": "par32-prefetchall4-300",
        "artifact_prefix": PAR32_PREFETCHALL4_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "search",
        "expected_runtime_state": 0,
        "expected_out_raw": PAR32_PREFETCHALL4_300_SEARCH_EXPECTED_RAW,
        "max_accelerator_latency_ms": 4.0,
    },
    "axis-par32-prefetchall4-300-track": {
        "kind": "axis",
        "variant": "par32-prefetchall4-300",
        "artifact_prefix": PAR32_PREFETCHALL4_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "track",
        "expected_runtime_state": 1,
        "expected_out_raw": PAR32_PREFETCHALL4_300_TRACK_EXPECTED_RAW,
        "max_accelerator_latency_ms": 1.0,
    },
    "axis-par32-prefetchall4-300-hybrid-10-90": {
        "kind": "axis_hybrid",
        "variant": "par32-prefetchall4-300",
        "artifact_prefix": PAR32_PREFETCHALL4_300_PREFIX,
        "weights_mode": "zero",
        "hybrid_profile": "search10-track90",
        "search": {
            "expected_runtime_state": 0,
            "expected_out_raw": PAR32_PREFETCHALL4_300_SEARCH_EXPECTED_RAW,
            "max_accelerator_latency_ms": 4.0,
        },
        "track": {
            "expected_runtime_state": 1,
            "expected_out_raw": PAR32_PREFETCHALL4_300_TRACK_EXPECTED_RAW,
            "max_accelerator_latency_ms": 1.0,
        },
    },
    "axis-par32-patch32-dtok4-300-search": {
        "kind": "axis",
        "variant": "par32-patch32-dtok4-300",
        "artifact_prefix": PAR32_PATCH32_DTOK4_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "search",
        "expected_runtime_state": 0,
        "expected_out_raw": PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW,
        "max_accelerator_latency_ms": 4.0,
    },
    "axis-par32-patch32-dtok4-300-track": {
        "kind": "axis",
        "variant": "par32-patch32-dtok4-300",
        "artifact_prefix": PAR32_PATCH32_DTOK4_300_PREFIX,
        "weights_mode": "zero",
        "mode_profile": "track",
        "expected_runtime_state": 1,
        "expected_out_raw": PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW,
        "max_accelerator_latency_ms": 1.0,
    },
    "axis-par32-patch32-dtok4-300-hybrid-10-90": {
        "kind": "axis_hybrid",
        "variant": "par32-patch32-dtok4-300",
        "artifact_prefix": PAR32_PATCH32_DTOK4_300_PREFIX,
        "weights_mode": "zero",
        "hybrid_profile": "search10-track90",
        "search": {
            "expected_runtime_state": 0,
            "expected_out_raw": PAR32_PATCH32_DTOK4_300_SEARCH_EXPECTED_RAW,
            "max_accelerator_latency_ms": 4.0,
        },
        "track": {
            "expected_runtime_state": 1,
            "expected_out_raw": PAR32_PATCH32_DTOK4_300_TRACK_EXPECTED_RAW,
            "max_accelerator_latency_ms": 1.0,
        },
    },
    "axis-vref-p0-softmax-input-x2-dsp-mixed-stream": {
        "kind": "axis",
        "variant": "vref-p0-softmax-input-x2-dsp-mixed-stream",
        "artifact_prefix": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
    },
    "axis-vref-p0-softmax-input-x2-qkv-uram": {
        "kind": "axis",
        "variant": "vref-p0-softmax-input-x2-qkv-uram",
        "artifact_prefix": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": VREF_P0_SOFTMAX_INPUT_X2_EXPECTED_RAW,
    },
    "m_axi-a2": {
        "kind": "m_axi",
        "artifact_prefix": "hgtxr_e2e_m_axi",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "expected_out_raw": EXPECTED_RAW,
    },
}


def add_error(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    errors.append(f"{label}: got {actual!r}, expected {expected!r}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_measurement_fields(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    repeat = payload.get("repeat")
    samples = payload.get("latency_ms_samples")
    summary = payload.get("latency_ms_summary")

    if repeat is not None:
        if not isinstance(repeat, int) or repeat < 1:
            errors.append(f"repeat: got {repeat!r}, expected positive integer")
        if not isinstance(samples, list) or not samples:
            errors.append("latency_ms_samples: missing or empty despite repeat being present")
        elif isinstance(repeat, int) and len(samples) != repeat:
            errors.append(f"latency_ms_samples length: got {len(samples)}, expected repeat {repeat}")
        if isinstance(samples, list):
            for index, value in enumerate(samples):
                if not is_number(value) or float(value) <= 0.0:
                    errors.append(f"latency_ms_samples[{index}]: got {value!r}, expected positive number")
        if not isinstance(summary, dict):
            errors.append("latency_ms_summary: missing or non-object despite repeat being present")
        elif isinstance(repeat, int) and summary.get("count") != repeat:
            errors.append(f"latency_ms_summary.count: got {summary.get('count')!r}, expected repeat {repeat}")

    bandwidth_summary = payload.get("dma_bandwidth_Bps_summary")
    if bandwidth_summary is not None:
        if not isinstance(bandwidth_summary, dict):
            errors.append("dma_bandwidth_Bps_summary: expected object")
        else:
            count = bandwidth_summary.get("count")
            if not isinstance(count, int) or count < 0:
                errors.append(f"dma_bandwidth_Bps_summary.count: got {count!r}, expected non-negative integer")

    distribution = payload.get("search_track_invocation_distribution")
    if distribution is not None:
        if not isinstance(distribution, dict):
            errors.append("search_track_invocation_distribution: expected object")
        else:
            for key in ["search", "track", "unknown"]:
                value = distribution.get(key)
                if not isinstance(value, int) or value < 0:
                    errors.append(f"search_track_invocation_distribution.{key}: got {value!r}, expected non-negative integer")
            if isinstance(repeat, int):
                total = sum(int(distribution.get(key, 0)) for key in ["search", "track", "unknown"])
                if total != repeat:
                    errors.append(f"search_track_invocation_distribution total: got {total}, expected repeat {repeat}")

    return errors


def validate_latency_target(payload: dict[str, Any], preset: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    threshold = preset.get("max_accelerator_latency_ms")
    if threshold is None:
        return errors
    if not is_number(threshold) or float(threshold) <= 0.0:
        errors.append(f"max_accelerator_latency_ms preset invalid: {threshold!r}")
        return errors

    summary = payload.get("accelerator_latency_ms_summary")
    samples = payload.get("accelerator_latency_ms_samples")
    if not isinstance(summary, dict):
        errors.append("accelerator_latency_ms_summary: missing or non-object for latency-target preset")
        return errors
    max_latency = summary.get("max")
    if not is_number(max_latency):
        errors.append(f"accelerator_latency_ms_summary.max: got {max_latency!r}, expected number")
    elif float(max_latency) > float(threshold):
        errors.append(f"accelerator_latency_ms_summary.max: got {max_latency!r}, expected <= {threshold!r}")
    if not isinstance(samples, list) or not samples:
        errors.append("accelerator_latency_ms_samples: missing or empty for latency-target preset")
    else:
        for index, value in enumerate(samples):
            if not is_number(value) or float(value) <= 0.0:
                errors.append(f"accelerator_latency_ms_samples[{index}]: got {value!r}, expected positive number")
            elif float(value) > float(threshold):
                errors.append(f"accelerator_latency_ms_samples[{index}]: got {value!r}, expected <= {threshold!r}")
    return errors


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("result JSON must be an object")
    return payload


def expected_artifact_name(preset: dict[str, Any], field: str) -> str:
    suffix_by_field = {"bitfile": ".bit", "hwhfile": ".hwh"}
    return f"{preset['artifact_prefix']}{suffix_by_field[field]}"


def validate_result(payload: dict[str, Any], preset_name: str, require_paths: bool = True) -> list[str]:
    if preset_name not in VARIANT_PRESETS:
        raise ValueError(f"unsupported preset: {preset_name}")
    preset = VARIANT_PRESETS[preset_name]
    if preset["kind"] == "axis_hybrid":
        return validate_hybrid_result(payload, preset, require_paths=require_paths)
    errors: list[str] = []

    if payload.get("status") != "pass":
        add_error(errors, "status", payload.get("status"), "pass")
    if payload.get("weights_mode") != preset["weights_mode"]:
        add_error(errors, "weights_mode", payload.get("weights_mode"), preset["weights_mode"])
    if payload.get("expected_runtime_state") != preset["expected_runtime_state"]:
        add_error(errors, "expected_runtime_state", payload.get("expected_runtime_state"), preset["expected_runtime_state"])
    if payload.get("runtime_state") != preset["expected_runtime_state"]:
        add_error(errors, "runtime_state", payload.get("runtime_state"), preset["expected_runtime_state"])
    if payload.get("runtime_match") is not True:
        add_error(errors, "runtime_match", payload.get("runtime_match"), True)
    if preset["expected_out_raw"] is None:
        if payload.get("expected_out_raw") is not None:
            add_error(errors, "expected_out_raw", payload.get("expected_out_raw"), None)
    else:
        if payload.get("expected_out_raw") != preset["expected_out_raw"]:
            add_error(errors, "expected_out_raw", payload.get("expected_out_raw"), preset["expected_out_raw"])
        if payload.get("out_raw") != preset["expected_out_raw"]:
            add_error(errors, "out_raw", payload.get("out_raw"), preset["expected_out_raw"])
    if payload.get("output_match") is not True:
        add_error(errors, "output_match", payload.get("output_match"), True)

    if preset["kind"] == "axis":
        if payload.get("variant") != preset["variant"]:
            add_error(errors, "variant", payload.get("variant"), preset["variant"])
        for field in ["dma_in_name", "dma_out_name"]:
            if not payload.get(field):
                errors.append(f"{field}: missing or empty")
        if "mode_profile" in preset and payload.get("mode_profile") != preset["mode_profile"]:
            add_error(errors, "mode_profile", payload.get("mode_profile"), preset["mode_profile"])

    if require_paths:
        for field in ["bitfile", "hwhfile"]:
            value = payload.get(field)
            if not value:
                errors.append(f"{field}: missing or empty")
                continue
            expected_name = expected_artifact_name(preset, field)
            actual_name = Path(str(value)).name
            if actual_name != expected_name:
                add_error(errors, field, actual_name, expected_name)

    out_state = payload.get("out_state")
    if not isinstance(out_state, list) or len(out_state) != len(EXPECTED_RAW):
        errors.append(f"out_state: got length {len(out_state) if isinstance(out_state, list) else 'non-list'}, expected {len(EXPECTED_RAW)}")

    errors.extend(validate_measurement_fields(payload))
    errors.extend(validate_latency_target(payload, preset))

    return errors


def validate_hybrid_result(payload: dict[str, Any], preset: dict[str, Any], require_paths: bool = True) -> list[str]:
    errors: list[str] = []

    if payload.get("status") != "pass":
        add_error(errors, "status", payload.get("status"), "pass")
    if payload.get("variant") != preset["variant"]:
        add_error(errors, "variant", payload.get("variant"), preset["variant"])
    if payload.get("weights_mode") != preset["weights_mode"]:
        add_error(errors, "weights_mode", payload.get("weights_mode"), preset["weights_mode"])
    if payload.get("hybrid_profile") != preset["hybrid_profile"]:
        add_error(errors, "hybrid_profile", payload.get("hybrid_profile"), preset["hybrid_profile"])
    if payload.get("runtime_match") is not True:
        add_error(errors, "runtime_match", payload.get("runtime_match"), True)
    if payload.get("output_match") is not True:
        add_error(errors, "output_match", payload.get("output_match"), True)
    if payload.get("latency_target_match") is not True:
        add_error(errors, "latency_target_match", payload.get("latency_target_match"), True)

    if require_paths:
        for field in ["bitfile", "hwhfile"]:
            value = payload.get(field)
            if not value:
                errors.append(f"{field}: missing or empty")
                continue
            expected_name = expected_artifact_name(preset, field)
            actual_name = Path(str(value)).name
            if actual_name != expected_name:
                add_error(errors, field, actual_name, expected_name)

    repeat = payload.get("repeat")
    if not isinstance(repeat, int) or repeat < 10 or repeat % 10 != 0:
        errors.append(f"repeat: got {repeat!r}, expected positive multiple of 10")
    distribution = payload.get("search_track_invocation_distribution")
    if not isinstance(distribution, dict):
        errors.append("search_track_invocation_distribution: expected object")
    else:
        search_count = distribution.get("search")
        track_count = distribution.get("track")
        unknown_count = distribution.get("unknown")
        if not isinstance(search_count, int) or search_count < 1:
            errors.append(f"search_track_invocation_distribution.search: got {search_count!r}, expected positive integer")
        if not isinstance(track_count, int) or track_count < 1:
            errors.append(f"search_track_invocation_distribution.track: got {track_count!r}, expected positive integer")
        if unknown_count != 0:
            errors.append(f"search_track_invocation_distribution.unknown: got {unknown_count!r}, expected 0")
        if isinstance(search_count, int) and isinstance(track_count, int):
            if track_count != search_count * 9:
                errors.append(
                    f"search_track_invocation_distribution ratio: got search={search_count}, track={track_count}, expected 1:9"
                )
            if isinstance(repeat, int) and search_count + track_count + int(unknown_count or 0) != repeat:
                errors.append(
                    "search_track_invocation_distribution total: "
                    f"got {search_count + track_count + int(unknown_count or 0)}, expected repeat {repeat}"
                )

    summaries = payload.get("mode_summaries")
    if not isinstance(summaries, dict):
        errors.append("mode_summaries: expected object")
        summaries = {}
    for mode in ["search", "track"]:
        expected = preset[mode]
        summary = summaries.get(mode)
        if not isinstance(summary, dict):
            errors.append(f"mode_summaries.{mode}: expected object")
            continue
        if summary.get("expected_runtime_state") != expected["expected_runtime_state"]:
            add_error(
                errors,
                f"mode_summaries.{mode}.expected_runtime_state",
                summary.get("expected_runtime_state"),
                expected["expected_runtime_state"],
            )
        if summary.get("expected_out_raw") != expected["expected_out_raw"]:
            add_error(errors, f"mode_summaries.{mode}.expected_out_raw", summary.get("expected_out_raw"), expected["expected_out_raw"])
        if summary.get("runtime_match") is not True:
            add_error(errors, f"mode_summaries.{mode}.runtime_match", summary.get("runtime_match"), True)
        if summary.get("output_match") is not True:
            add_error(errors, f"mode_summaries.{mode}.output_match", summary.get("output_match"), True)
        if summary.get("latency_target_match") is not True:
            add_error(errors, f"mode_summaries.{mode}.latency_target_match", summary.get("latency_target_match"), True)
        accelerator_summary = summary.get("accelerator_latency_ms_summary")
        if not isinstance(accelerator_summary, dict):
            errors.append(f"mode_summaries.{mode}.accelerator_latency_ms_summary: expected object")
            continue
        max_latency = accelerator_summary.get("max")
        threshold = expected["max_accelerator_latency_ms"]
        if not is_number(max_latency):
            errors.append(f"mode_summaries.{mode}.accelerator_latency_ms_summary.max: got {max_latency!r}, expected number")
        elif float(max_latency) > float(threshold):
            errors.append(
                f"mode_summaries.{mode}.accelerator_latency_ms_summary.max: got {max_latency!r}, expected <= {threshold!r}"
            )

    errors.extend(validate_measurement_fields(payload))
    errors.extend(validate_latency_target(payload, {"max_accelerator_latency_ms": 4.0}))
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HGTXR physical PYNQ smoke result JSON.")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--preset", choices=sorted(VARIANT_PRESETS), default="axis-c3b-mem16")
    parser.add_argument("--no-require-paths", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_json(args.result_json)
    errors = validate_result(payload, args.preset, require_paths=not args.no_require_paths)
    result = {
        "status": "pass" if not errors else "fail",
        "preset": args.preset,
        "result_json": str(args.result_json),
        "errors": errors,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
