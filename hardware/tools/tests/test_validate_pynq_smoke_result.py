#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_pynq_smoke_result as validator  # noqa: E402


def c3b_result(**overrides):
    payload = {
        "status": "pass",
        "variant": "c3b-mem16",
        "bitfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
        "hwhfile": "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
        "dma_in_name": "axi_dma_in",
        "dma_out_name": "axi_dma_out",
        "frame_pattern": "ramp",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "runtime_state": 2,
        "runtime_match": True,
        "out_state": [2.0, -0.8125, 1.625, -0.375, 0.875, -0.6875],
        "out_raw": [32, -13, 26, -6, 14, -11],
        "expected_out_raw": [32, -13, 26, -6, 14, -11],
        "output_match": True,
    }
    payload.update(overrides)
    return payload


def successor_result(**overrides):
    payload = c3b_result(
        variant="vref-p0-softmax-input-x2-dsp-mixed-stream",
        bitfile="hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
        hwhfile="hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
        out_state=[3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
        out_raw=[58, -51, 42, -28, 36, -41],
        expected_out_raw=[58, -51, 42, -28, 36, -41],
    )
    payload.update(overrides)
    return payload


def qkv_uram_result(**overrides):
    payload = c3b_result(
        variant="vref-p0-softmax-input-x2-qkv-uram",
        bitfile="hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit",
        hwhfile="hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh",
        out_state=[3.625, -3.1875, 2.625, -1.75, 2.25, -2.5625],
        out_raw=[58, -51, 42, -28, 36, -41],
        expected_out_raw=[58, -51, 42, -28, 36, -41],
    )
    payload.update(overrides)
    return payload


def runtime_mode_result(profile: str, **overrides):
    runtime_state = 0 if profile == "search" else 1
    accelerator_latency_ms = 2.5 if profile == "search" else 0.7
    payload = c3b_result(
        variant="runtime-mode-par32",
        bitfile="hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit",
        hwhfile="hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh",
        weights_mode="zero",
        mode_profile=profile,
        expected_runtime_state=runtime_state,
        runtime_state=runtime_state,
        expected_out_raw=None,
        output_match=True,
        accelerator_latency_ms=accelerator_latency_ms,
        accelerator_latency_ms_samples=[accelerator_latency_ms],
        accelerator_latency_ms_summary={
            "count": 1,
            "mean": accelerator_latency_ms,
            "median": accelerator_latency_ms,
            "min": accelerator_latency_ms,
            "max": accelerator_latency_ms,
            "p95": accelerator_latency_ms,
            "p99": accelerator_latency_ms,
        },
    )
    payload.update(overrides)
    return payload


def par32_rom_compute_result(profile: str, **overrides):
    runtime_state = 0 if profile == "search" else 1
    accelerator_latency_ms = 3.5 if profile == "search" else 0.8
    expected = (
        validator.PAR32_ROM_COMPUTE_300_SEARCH_EXPECTED_RAW
        if profile == "search"
        else validator.PAR32_ROM_COMPUTE_300_TRACK_EXPECTED_RAW
    )
    payload = c3b_result(
        variant="par32-rom-compute-300",
        bitfile=f"hgtxr/{validator.PAR32_ROM_COMPUTE_300_PREFIX}.bit",
        hwhfile=f"hgtxr/{validator.PAR32_ROM_COMPUTE_300_PREFIX}.hwh",
        weights_mode="zero",
        mode_profile=profile,
        expected_runtime_state=runtime_state,
        runtime_state=runtime_state,
        out_state=[float(value) / 16.0 for value in expected],
        out_raw=list(expected),
        expected_out_raw=list(expected),
        output_match=True,
        accelerator_latency_ms=accelerator_latency_ms,
        accelerator_latency_ms_samples=[accelerator_latency_ms],
        accelerator_latency_ms_summary={
            "count": 1,
            "mean": accelerator_latency_ms,
            "median": accelerator_latency_ms,
            "min": accelerator_latency_ms,
            "max": accelerator_latency_ms,
            "p95": accelerator_latency_ms,
            "p99": accelerator_latency_ms,
        },
    )
    payload.update(overrides)
    return payload


def par32_prefetchall4_result(profile: str, **overrides):
    runtime_state = 0 if profile == "search" else 1
    accelerator_latency_ms = 3.5 if profile == "search" else 0.8
    expected = (
        validator.PAR32_PREFETCHALL4_300_SEARCH_EXPECTED_RAW
        if profile == "search"
        else validator.PAR32_PREFETCHALL4_300_TRACK_EXPECTED_RAW
    )
    payload = c3b_result(
        variant="par32-prefetchall4-300",
        bitfile=f"hgtxr/{validator.PAR32_PREFETCHALL4_300_PREFIX}.bit",
        hwhfile=f"hgtxr/{validator.PAR32_PREFETCHALL4_300_PREFIX}.hwh",
        weights_mode="zero",
        mode_profile=profile,
        expected_runtime_state=runtime_state,
        runtime_state=runtime_state,
        out_state=[float(value) / 16.0 for value in expected],
        out_raw=list(expected),
        expected_out_raw=list(expected),
        output_match=True,
        accelerator_latency_ms=accelerator_latency_ms,
        accelerator_latency_ms_samples=[accelerator_latency_ms],
        accelerator_latency_ms_summary={
            "count": 1,
            "mean": accelerator_latency_ms,
            "median": accelerator_latency_ms,
            "min": accelerator_latency_ms,
            "max": accelerator_latency_ms,
            "p95": accelerator_latency_ms,
            "p99": accelerator_latency_ms,
        },
    )
    payload.update(overrides)
    return payload


def par32_prefetchall4_hybrid_result(**overrides):
    search_expected = validator.PAR32_PREFETCHALL4_300_SEARCH_EXPECTED_RAW
    track_expected = validator.PAR32_PREFETCHALL4_300_TRACK_EXPECTED_RAW
    payload = c3b_result(
        variant="par32-prefetchall4-300",
        bitfile=f"hgtxr/{validator.PAR32_PREFETCHALL4_300_PREFIX}.bit",
        hwhfile=f"hgtxr/{validator.PAR32_PREFETCHALL4_300_PREFIX}.hwh",
        weights_mode="zero",
        hybrid_profile="search10-track90",
        runtime_match=True,
        output_match=True,
        latency_target_match=True,
        expected_runtime_state=None,
        runtime_state=None,
        expected_out_raw=None,
        out_raw=track_expected,
        out_state=[float(value) / 16.0 for value in track_expected],
        repeat=10,
        latency_ms_samples=[4.0] + [1.0] * 9,
        latency_ms_summary={"count": 10, "mean": 1.3, "median": 1.0, "min": 1.0, "max": 4.0, "p95": 2.65, "p99": 3.73},
        accelerator_latency_ms_samples=[3.5] + [0.8] * 9,
        accelerator_latency_ms_summary={
            "count": 10,
            "mean": 1.07,
            "median": 0.8,
            "min": 0.8,
            "max": 3.5,
            "p95": 2.285,
            "p99": 3.257,
        },
        search_track_invocation_distribution={"search": 1, "track": 9, "unknown": 0},
        mode_summaries={
            "search": {
                "count": 1,
                "expected_runtime_state": 0,
                "expected_out_raw": search_expected,
                "runtime_match": True,
                "output_match": True,
                "latency_target_match": True,
                "accelerator_latency_ms_summary": {"count": 1, "max": 3.5},
            },
            "track": {
                "count": 9,
                "expected_runtime_state": 1,
                "expected_out_raw": track_expected,
                "runtime_match": True,
                "output_match": True,
                "latency_target_match": True,
                "accelerator_latency_ms_summary": {"count": 9, "max": 0.8},
            },
        },
    )
    payload.update(overrides)
    return payload


class ValidatePynqSmokeResultTests(unittest.TestCase):
    def test_validate_accepts_c3b_axis_result(self) -> None:
        errors = validator.validate_result(c3b_result(), "axis-c3b-mem16")
        self.assertEqual(errors, [])

    def test_validate_accepts_vref_successor_axis_result(self) -> None:
        errors = validator.validate_result(successor_result(), "axis-vref-p0-softmax-input-x2-dsp-mixed-stream")
        self.assertEqual(errors, [])

    def test_validate_accepts_qkv_uram_axis_result(self) -> None:
        errors = validator.validate_result(qkv_uram_result(), "axis-vref-p0-softmax-input-x2-qkv-uram")
        self.assertEqual(errors, [])

    def test_validate_accepts_runtime_mode_search_result_without_golden_output(self) -> None:
        errors = validator.validate_result(runtime_mode_result("search"), "axis-runtime-mode-par32-search")
        self.assertEqual(errors, [])

    def test_validate_accepts_runtime_mode_track_result_without_golden_output(self) -> None:
        errors = validator.validate_result(runtime_mode_result("track"), "axis-runtime-mode-par32-track")
        self.assertEqual(errors, [])

    def test_validate_accepts_par32_rom_compute_search_result_with_expected_output(self) -> None:
        errors = validator.validate_result(par32_rom_compute_result("search"), "axis-par32-rom-compute-300-search")
        self.assertEqual(errors, [])

    def test_validate_accepts_par32_rom_compute_track_result_with_expected_output(self) -> None:
        errors = validator.validate_result(par32_rom_compute_result("track"), "axis-par32-rom-compute-300-track")
        self.assertEqual(errors, [])

    def test_validate_accepts_par32_prefetchall4_search_result_with_expected_output(self) -> None:
        errors = validator.validate_result(par32_prefetchall4_result("search"), "axis-par32-prefetchall4-300-search")
        self.assertEqual(errors, [])

    def test_validate_accepts_par32_prefetchall4_track_result_with_expected_output(self) -> None:
        errors = validator.validate_result(par32_prefetchall4_result("track"), "axis-par32-prefetchall4-300-track")
        self.assertEqual(errors, [])

    def test_validate_accepts_par32_prefetchall4_hybrid_10_90_result(self) -> None:
        errors = validator.validate_result(
            par32_prefetchall4_hybrid_result(),
            "axis-par32-prefetchall4-300-hybrid-10-90",
        )
        self.assertEqual(errors, [])

    def test_validate_rejects_par32_rom_compute_wrong_output(self) -> None:
        payload = par32_rom_compute_result("search", out_raw=[0, 0, 0, 0, 0, 0])
        errors = validator.validate_result(payload, "axis-par32-rom-compute-300-search")
        self.assertTrue(any("out_raw" in error for error in errors))

    def test_validate_rejects_runtime_mode_search_latency_over_target(self) -> None:
        payload = runtime_mode_result(
            "search",
            accelerator_latency_ms_samples=[4.2],
            accelerator_latency_ms_summary={"count": 1, "max": 4.2},
        )
        errors = validator.validate_result(payload, "axis-runtime-mode-par32-search")
        self.assertTrue(any("expected <= 4.0" in error for error in errors))

    def test_validate_rejects_runtime_mode_track_latency_over_target(self) -> None:
        payload = runtime_mode_result(
            "track",
            accelerator_latency_ms_samples=[1.2],
            accelerator_latency_ms_summary={"count": 1, "max": 1.2},
        )
        errors = validator.validate_result(payload, "axis-runtime-mode-par32-track")
        self.assertTrue(any("expected <= 1.0" in error for error in errors))

    def test_validate_rejects_par32_prefetchall4_search_latency_over_target(self) -> None:
        payload = par32_prefetchall4_result(
            "search",
            accelerator_latency_ms_samples=[4.2],
            accelerator_latency_ms_summary={"count": 1, "max": 4.2},
        )
        errors = validator.validate_result(payload, "axis-par32-prefetchall4-300-search")
        self.assertTrue(any("expected <= 4.0" in error for error in errors))

    def test_validate_rejects_par32_prefetchall4_track_latency_over_target(self) -> None:
        payload = par32_prefetchall4_result(
            "track",
            accelerator_latency_ms_samples=[1.2],
            accelerator_latency_ms_summary={"count": 1, "max": 1.2},
        )
        errors = validator.validate_result(payload, "axis-par32-prefetchall4-300-track")
        self.assertTrue(any("expected <= 1.0" in error for error in errors))

    def test_validate_rejects_par32_prefetchall4_hybrid_wrong_distribution(self) -> None:
        payload = par32_prefetchall4_hybrid_result(
            search_track_invocation_distribution={"search": 2, "track": 8, "unknown": 0}
        )
        errors = validator.validate_result(payload, "axis-par32-prefetchall4-300-hybrid-10-90")
        self.assertTrue(any("expected 1:9" in error for error in errors))

    def test_validate_rejects_par32_prefetchall4_hybrid_track_latency_over_target(self) -> None:
        payload = par32_prefetchall4_hybrid_result()
        payload["mode_summaries"]["track"]["accelerator_latency_ms_summary"] = {"count": 9, "max": 1.2}
        errors = validator.validate_result(payload, "axis-par32-prefetchall4-300-hybrid-10-90")
        self.assertTrue(any("mode_summaries.track" in error and "expected <= 1.0" in error for error in errors))

    def test_validate_rejects_wrong_variant_and_output(self) -> None:
        errors = validator.validate_result(c3b_result(variant="a1", out_raw=[0, 0, 0, 0, 0, 0]), "axis-c3b-mem16")
        self.assertTrue(any("variant" in error for error in errors))
        self.assertTrue(any("out_raw" in error for error in errors))

    def test_validate_rejects_wrong_c3b_bitfile_basename(self) -> None:
        errors = validator.validate_result(c3b_result(bitfile="hgtxr/hgtxr_e2e_axis_dma.bit"), "axis-c3b-mem16")
        self.assertTrue(any("bitfile" in error and "hgtxr_e2e_axis_dma_c3b_mem16.bit" in error for error in errors))

    def test_validate_rejects_wrong_c3b_hwhfile_basename(self) -> None:
        errors = validator.validate_result(c3b_result(hwhfile="hgtxr/hgtxr_e2e_axis_dma.hwh"), "axis-c3b-mem16")
        self.assertTrue(any("hwhfile" in error and "hgtxr_e2e_axis_dma_c3b_mem16.hwh" in error for error in errors))

    def test_validate_no_require_paths_skips_overlay_basename_check(self) -> None:
        errors = validator.validate_result(
            c3b_result(bitfile="hgtxr/hgtxr_e2e_axis_dma.bit", hwhfile="hgtxr/hgtxr_e2e_axis_dma.hwh"),
            "axis-c3b-mem16",
            require_paths=False,
        )
        self.assertEqual(errors, [])

    def test_cli_writes_json_out_and_returns_zero_on_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "c3b.json"
            out_path = Path(tmp) / "validation.json"
            result_path.write_text(json.dumps(c3b_result()))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = validator.main([str(result_path), "--preset", "axis-c3b-mem16", "--json-out", str(out_path)])
            self.assertEqual(code, 0, stream.getvalue())
            payload = json.loads(out_path.read_text())
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["preset"], "axis-c3b-mem16")

    def test_cli_returns_one_on_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "bad.json"
            result_path.write_text(json.dumps(c3b_result(status="fail", runtime_state=0)))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = validator.main([str(result_path), "--preset", "axis-c3b-mem16"])
            self.assertEqual(code, 1)
            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["status"], "fail")
            self.assertTrue(payload["errors"])


if __name__ == "__main__":
    unittest.main()
