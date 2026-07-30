#!/usr/bin/env python3
"""Transfer, run, fetch, and import HGTXR ZCU104 PYNQ smoke bundles over SSH."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from import_pynq_smoke_result import import_result
from validate_pynq_bundle_package import sha256_file


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DATE_TAG = "2026_06_10"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
DEFAULT_SHA256 = HARDWARE_ROOT / "generated" / "signoff" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
DEFAULT_LOCAL_RESULT = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_file_smoke.remote.json"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / f"zcu104_c3b_smoke_remote_run_{DATE_TAG}.md"
DEFAULT_REMOTE_DIR = "/home/xilinx/hgtxr_c3b_smoke"
RESULT_JSON = "e2e_axis_dma_c3b_mem16_file_smoke.json"
VALIDATION_JSON = "e2e_axis_dma_c3b_mem16_file_smoke_validation.json"
BUNDLE_ROOT = "e2e_axis_dma_c3b_mem16_smoke_bundle"
DEFAULT_PROFILE = "c3b-mem16"
PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "c3b-mem16": {
        "title": "C3b",
        "tar": DEFAULT_TAR,
        "sha256": DEFAULT_SHA256,
        "local_result": DEFAULT_LOCAL_RESULT,
        "json": DEFAULT_JSON,
        "markdown": DEFAULT_MARKDOWN,
        "remote_dir": DEFAULT_REMOTE_DIR,
        "result_json": RESULT_JSON,
        "validation_json": VALIDATION_JSON,
        "bundle_root": BUNDLE_ROOT,
        "run_script": "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "preset": "axis-c3b-mem16",
    },
    "runtime-mode-par32-search": {
        "title": "Runtime-mode PAR32 Search",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_runtime_mode_search_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_runtime_mode_search_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_runtime_mode_par32_search_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_runtime_mode_par32_search_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_runtime_mode_par32_search_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_runtime_mode_search_smoke",
        "result_json": "e2e_axis_dma_runtime_mode_par32_search_file_smoke.json",
        "validation_json": "e2e_axis_dma_runtime_mode_par32_search_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_runtime_mode_search_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
        "preset": "axis-runtime-mode-par32-search",
    },
    "runtime-mode-par32-track": {
        "title": "Runtime-mode PAR32 Track",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_runtime_mode_track_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_runtime_mode_track_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_runtime_mode_par32_track_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_runtime_mode_par32_track_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_runtime_mode_par32_track_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_runtime_mode_track_smoke",
        "result_json": "e2e_axis_dma_runtime_mode_par32_track_file_smoke.json",
        "validation_json": "e2e_axis_dma_runtime_mode_par32_track_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_runtime_mode_track_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
        "preset": "axis-runtime-mode-par32-track",
    },
    "par32-rom-compute-300-search": {
        "title": "PAR32 ROM compute 300MHz Search",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_rom_compute_300_search_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_rom_compute_300_search_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_rom_compute_300_search_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_rom_compute_300_search_smoke",
        "result_json": "e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_rom_compute_300_search_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_rom_compute_300_search_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_rom_compute_300_search_file_smoke.sh",
        "preset": "axis-par32-rom-compute-300-search",
    },
    "par32-rom-compute-300-track": {
        "title": "PAR32 ROM compute 300MHz Track",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_rom_compute_300_track_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_rom_compute_300_track_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_rom_compute_300_track_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_rom_compute_300_track_smoke",
        "result_json": "e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_rom_compute_300_track_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_rom_compute_300_track_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_rom_compute_300_track_file_smoke.sh",
        "preset": "axis-par32-rom-compute-300-track",
    },
    "par32-prefetchall4-300-search": {
        "title": "PAR32 prefetch-all4 300MHz Search",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_search_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_search_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_prefetchall4_300_search_smoke",
        "result_json": "e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_prefetchall4_300_search_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.sh",
        "preset": "axis-par32-prefetchall4-300-search",
    },
    "par32-prefetchall4-300-track": {
        "title": "PAR32 prefetch-all4 300MHz Track",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_track_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_track_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_prefetchall4_300_track_smoke",
        "result_json": "e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_prefetchall4_300_track_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.sh",
        "preset": "axis-par32-prefetchall4-300-track",
    },
    "par32-prefetchall4-300-hybrid-10-90": {
        "title": "PAR32 prefetch-all4 300MHz Hybrid 10/90",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_hybrid_10_90_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_prefetchall4_300_hybrid_10_90_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_prefetchall4_300_hybrid_10_90_smoke",
        "result_json": "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
        "preset": "axis-par32-prefetchall4-300-hybrid-10-90",
    },
    "par32-patch32-dtok4-300-search": {
        "title": "PAR32 patch32 dtok4 300MHz Search",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_patch32_dtok4_300_search_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_search_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_search_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_search_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_patch32_dtok4_300_search_smoke",
        "result_json": "e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_patch32_dtok4_300_search_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.sh",
        "preset": "axis-par32-patch32-dtok4-300-search",
    },
    "par32-patch32-dtok4-300-track": {
        "title": "PAR32 patch32 dtok4 300MHz Track",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_patch32_dtok4_300_track_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_track_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_track_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_track_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_patch32_dtok4_300_track_smoke",
        "result_json": "e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_patch32_dtok4_300_track_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.sh",
        "preset": "axis-par32-patch32-dtok4-300-track",
    },
    "par32-patch32-dtok4-300-hybrid-10-90": {
        "title": "PAR32 patch32 dtok4 300MHz Hybrid 10/90",
        "tar": HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_hybrid_10_90_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_par32_patch32_dtok4_300_hybrid_10_90_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_par32_patch32_dtok4_300_hybrid_10_90_smoke",
        "result_json": "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.json",
        "validation_json": "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.sh",
        "preset": "axis-par32-patch32-dtok4-300-hybrid-10-90",
    },
    "vref-p0-softmax-input-x2-dsp-mixed-stream": {
        "title": "VREF-P0 softmax_input_x2 dsp_mixed_stream",
        "tar": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_vref_p0_softmax_input_x2_smoke",
        "result_json": "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json",
        "validation_json": "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
        "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
    },
    "vref-p0-softmax-input-x2-qkv-uram": {
        "title": "VREF-P0 softmax_input_x2 qkv_uram",
        "tar": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz",
        "sha256": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz.sha256",
        "local_result": HARDWARE_ROOT
        / "generated"
        / "pynq"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.remote.json",
        "json": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.json",
        "markdown": HARDWARE_ROOT
        / "generated"
        / "signoff"
        / f"zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_{DATE_TAG}.md",
        "remote_dir": "/home/xilinx/hgtxr_vref_p0_qkv_uram_smoke",
        "result_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
        "validation_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json",
        "bundle_root": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle",
        "run_script": "./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
        "validate_script": "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
        "preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
    },
}

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def remote_target(host: str, user: str | None) -> str:
    return f"{user}@{host}" if user else host


def ssh_base(identity_file: Path | None, port: int | None) -> list[str]:
    cmd = ["ssh"]
    if identity_file is not None:
        cmd.extend(["-i", str(identity_file)])
    if port is not None:
        cmd.extend(["-p", str(port)])
    return cmd


def scp_base(identity_file: Path | None, port: int | None) -> list[str]:
    cmd = ["scp"]
    if identity_file is not None:
        cmd.extend(["-i", str(identity_file)])
    if port is not None:
        cmd.extend(["-P", str(port)])
    return cmd


def shell_join(parts: Sequence[str]) -> str:
    return " && ".join(parts)


def quote_remote_shell_path(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)


def validate_local_inputs(tar_path: Path, sha256_path: Path) -> list[str]:
    errors: list[str] = []
    if not tar_path.exists():
        errors.append(f"tar missing: {tar_path}")
    if not sha256_path.exists():
        errors.append(f"sha256 file missing: {sha256_path}")
    if not errors:
        expected = f"{sha256_file(tar_path)}  {tar_path.name}"
        actual = sha256_path.read_text().strip()
        if actual != expected:
            errors.append(f"sha256 line mismatch: got {actual!r}, expected {expected!r}")
    return errors


def build_commands(
    *,
    host: str,
    user: str | None,
    remote_dir: str,
    tar_path: Path,
    sha256_path: Path,
    local_result: Path,
    identity_file: Path | None,
    port: int | None,
    bundle_root: str = BUNDLE_ROOT,
    result_json: str = RESULT_JSON,
    validation_json: str = VALIDATION_JSON,
    run_script: str = "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
    validate_script: str = "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
) -> dict[str, list[str]]:
    target = remote_target(host, user)
    remote_dir_q = quote_remote_shell_path(remote_dir)
    tar_name_q = shlex.quote(tar_path.name)
    sha_name_q = shlex.quote(sha256_path.name)
    bundle_q = shlex.quote(bundle_root)
    result_q = shlex.quote(result_json)
    run_script_q = shlex.quote(run_script)
    validate_script_q = shlex.quote(validate_script)

    remote_run = shell_join(
        [
            f"cd {remote_dir_q}",
            f"sha256sum -c {sha_name_q}",
            f"tar -xzf {tar_name_q}",
            f"cd {bundle_q}",
            run_script_q,
            validate_script_q,
        ]
    )
    remote_result = f"{target}:{remote_dir.rstrip('/')}/{bundle_root}/{result_json}"

    return {
        "mkdir": ssh_base(identity_file, port) + [target, f"mkdir -p {remote_dir_q}"],
        "copy_tar": scp_base(identity_file, port) + [str(tar_path), str(sha256_path), f"{target}:{remote_dir.rstrip('/')}/"],
        "run": ssh_base(identity_file, port) + [target, remote_run],
        "fetch_result": scp_base(identity_file, port) + [remote_result, str(local_result)],
        "fetch_validation": scp_base(identity_file, port) + [
            f"{target}:{remote_dir.rstrip('/')}/{bundle_root}/{validation_json}",
            str(local_result.with_name(validation_json)),
        ],
    }


def run_command(cmd: list[str], runner: Runner) -> dict[str, Any]:
    completed = runner(cmd)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)


def build_summary(
    *,
    root: Path,
    execute: bool,
    host: str,
    user: str | None,
    remote_dir: str,
    tar_path: Path,
    sha256_path: Path,
    local_result: Path,
    identity_file: Path | None,
    port: int | None,
    profile: str = DEFAULT_PROFILE,
    bundle_root: str = BUNDLE_ROOT,
    result_json: str = RESULT_JSON,
    validation_json: str = VALIDATION_JSON,
    run_script: str = "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
    validate_script: str = "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
    preset: str = "axis-c3b-mem16",
    runner: Runner = default_runner,
) -> dict[str, Any]:
    errors = validate_local_inputs(tar_path, sha256_path)
    commands = build_commands(
        host=host,
        user=user,
        remote_dir=remote_dir,
        tar_path=tar_path,
        sha256_path=sha256_path,
        local_result=local_result,
        identity_file=identity_file,
        port=port,
        bundle_root=bundle_root,
        result_json=result_json,
        validation_json=validation_json,
        run_script=run_script,
        validate_script=validate_script,
    )
    summary: dict[str, Any] = {
        "status": "dry-run" if not execute else "pending",
        "profile": profile,
        "preset": preset,
        "execute": execute,
        "target": "ZCU104 PYNQ",
        "host": host,
        "user": user,
        "remote_dir": remote_dir,
        "tar": str(tar_path),
        "sha256_file": str(sha256_path),
        "local_result": str(local_result),
        "canonical_result": str(root / "hardware" / "pynq" / "hgtxr" / result_json),
        "bundle_root": bundle_root,
        "result_json": result_json,
        "validation_json": validation_json,
        "commands": commands,
        "command_results": [],
        "import_result": None,
        "errors": errors,
    }
    if errors:
        summary["status"] = "blocked-local-inputs"
        return summary
    if not execute:
        return summary

    local_result.parent.mkdir(parents=True, exist_ok=True)
    for name in ["mkdir", "copy_tar", "run", "fetch_result", "fetch_validation"]:
        result = run_command(commands[name], runner)
        result["name"] = name
        summary["command_results"].append(result)
        if result["returncode"] != 0:
            summary["status"] = "remote-command-failed"
            summary["errors"].append(f"{name} failed with returncode {result['returncode']}")
            return summary

    imported = import_result(local_result, preset=preset, hgtxr_root=root)
    summary["import_result"] = imported
    if imported["status"] == "pass":
        summary["status"] = "pass"
    else:
        summary["status"] = "import-failed"
        summary["errors"].extend(imported["errors"])
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# HGTXR ZCU104 Remote Smoke Runner",
        "",
        f"- status: `{summary['status']}`",
        f"- profile: `{summary['profile']}`",
        f"- preset: `{summary['preset']}`",
        f"- execute: `{summary['execute']}`",
        f"- host: `{summary['host']}`",
        f"- user: `{summary['user']}`",
        f"- remote_dir: `{summary['remote_dir']}`",
        f"- local_result: `{summary['local_result']}`",
        f"- canonical_result: `{summary['canonical_result']}`",
        "",
        "## Commands",
        "",
    ]
    for name, cmd in summary["commands"].items():
        lines.append(f"### {name}")
        lines.append("```sh")
        lines.append(" ".join(shlex.quote(part) for part in cmd))
        lines.append("```")
        lines.append("")
    lines.append("## Errors")
    lines.append("")
    if summary["errors"]:
        lines.extend(f"- `{error}`" for error in summary["errors"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(summary))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HGTXR ZCU104 smoke bundle over SSH/SCP.")
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default=DEFAULT_PROFILE)
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--host", default="zcu104.local")
    parser.add_argument("--user", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--remote-dir", default=None)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=None)
    parser.add_argument("--sha256-file", type=Path, default=None)
    parser.add_argument("--local-result", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="run SSH/SCP commands; otherwise emit dry-run plan only")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args(argv)
    profile = PROFILE_CONFIGS[args.profile]
    args.remote_dir = args.remote_dir if args.remote_dir is not None else profile["remote_dir"]
    args.tar_path = args.tar_path if args.tar_path is not None else profile["tar"]
    args.sha256_file = args.sha256_file if args.sha256_file is not None else profile["sha256"]
    args.local_result = args.local_result if args.local_result is not None else profile["local_result"]
    args.json_out = args.json_out if args.json_out is not None else profile["json"]
    args.markdown_out = args.markdown_out if args.markdown_out is not None else profile["markdown"]
    args.profile_config = profile
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        root=args.root.resolve(),
        execute=args.execute,
        host=args.host,
        user=args.user,
        remote_dir=args.remote_dir,
        tar_path=args.tar_path,
        sha256_path=args.sha256_file,
        local_result=args.local_result,
        identity_file=args.identity_file,
        port=args.port,
        profile=args.profile,
        bundle_root=args.profile_config["bundle_root"],
        result_json=args.profile_config["result_json"],
        validation_json=args.profile_config["validation_json"],
        run_script=args.profile_config["run_script"],
        validate_script=args.profile_config["validate_script"],
        preset=args.profile_config["preset"],
    )
    write_outputs(summary, args.json_out, args.markdown_out)
    print(
        f"[zcu104-smoke-remote] profile={summary['profile']} "
        f"status={summary['status']} execute={summary['execute']} errors={len(summary['errors'])}"
    )
    return 0 if summary["status"] in {"dry-run", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
