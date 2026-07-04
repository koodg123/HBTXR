#!/usr/bin/env python3
"""Validate a generated HGTXR PYNQ transfer bundle directory and tarball."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Sequence


REQUIRED_BY_VARIANT = {
    "c3b-mem16": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_c3b_mem16",
        "command_tokens": ["--variant c3b-mem16", "--weights-mode file"],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-c3b-mem16"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        },
    },
    "runtime-mode-par32-search": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
        "command_tokens": [
            "--variant runtime-mode-par32",
            "--weights-mode zero",
            "--mode-profile search",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-runtime-mode-par32-search"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
            "validate_e2e_axis_dma_runtime_mode_par32_search_file_smoke.sh",
        },
    },
    "runtime-mode-par32-track": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_runtime_mode_mem16",
        "command_tokens": [
            "--variant runtime-mode-par32",
            "--weights-mode zero",
            "--mode-profile track",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-runtime-mode-par32-track"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
            "validate_e2e_axis_dma_runtime_mode_par32_track_file_smoke.sh",
        },
    },
    "par32-rom-compute-300-search": {
        "artifact_prefix": (
            "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
            "corefabric_normuram_compute_300_mem16"
        ),
        "command_tokens": [
            "--variant par32-rom-compute-300",
            "--weights-mode zero",
            "--mode-profile search",
            "--expect-out-raw -1169 -1169 -1169 -1169 -1169 -1133",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-rom-compute-300-search"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_rom_compute_300_search_file_smoke.sh",
            "validate_e2e_axis_dma_par32_rom_compute_300_search_file_smoke.sh",
        },
    },
    "par32-rom-compute-300-track": {
        "artifact_prefix": (
            "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
            "corefabric_normuram_compute_300_mem16"
        ),
        "command_tokens": [
            "--variant par32-rom-compute-300",
            "--weights-mode zero",
            "--mode-profile track",
            "--expect-out-raw -235 -235 -235 -235 -235 -226",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-rom-compute-300-track"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_rom_compute_300_track_file_smoke.sh",
            "validate_e2e_axis_dma_par32_rom_compute_300_track_file_smoke.sh",
        },
    },
    "par32-prefetchall4-300-search": {
        "artifact_prefix": (
            "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
            "corefabric_normuram_compute_prefetchall4_300_mem16"
        ),
        "command_tokens": [
            "--variant par32-prefetchall4-300",
            "--weights-mode zero",
            "--mode-profile search",
            "--expect-out-raw -1169 -1169 -1169 -1169 -1169 -1125",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-prefetchall4-300-search"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.sh",
            "validate_e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.sh",
        },
    },
    "par32-prefetchall4-300-track": {
        "artifact_prefix": (
            "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
            "corefabric_normuram_compute_prefetchall4_300_mem16"
        ),
        "command_tokens": [
            "--variant par32-prefetchall4-300",
            "--weights-mode zero",
            "--mode-profile track",
            "--expect-out-raw -235 -235 -235 -235 -235 -239",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-prefetchall4-300-track"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.sh",
            "validate_e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.sh",
        },
    },
    "par32-prefetchall4-300-hybrid-10-90": {
        "artifact_prefix": (
            "hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_"
            "corefabric_normuram_compute_prefetchall4_300_mem16"
        ),
        "command_tokens": [
            "hgtxr.run_e2e_axis_dma_hybrid_smoke",
            "--variant par32-prefetchall4-300",
            "--warmup-cycles 1",
            "--repeat-cycles 5",
        ],
        "validation_tokens": [
            "tools/validate_pynq_smoke_result.py",
            "--preset axis-par32-prefetchall4-300-hybrid-10-90",
        ],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "hgtxr/run_e2e_axis_dma_hybrid_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
            "validate_e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.sh",
        },
    },
    "par32-patch32-dtok4-300-search": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
        "command_tokens": [
            "--variant par32-patch32-dtok4-300",
            "--weights-mode zero",
            "--mode-profile search",
            "--expect-out-raw 3 3 3 3 3 3",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-patch32-dtok4-300-search"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.sh",
            "validate_e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.sh",
        },
    },
    "par32-patch32-dtok4-300-track": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
        "command_tokens": [
            "--variant par32-patch32-dtok4-300",
            "--weights-mode zero",
            "--mode-profile track",
            "--expect-out-raw 217 217 217 217 217 217",
            "--warmup 1",
            "--repeat 5",
        ],
        "validation_tokens": ["tools/validate_pynq_smoke_result.py", "--preset axis-par32-patch32-dtok4-300-track"],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.sh",
            "validate_e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.sh",
        },
    },
    "par32-patch32-dtok4-300-hybrid-10-90": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16",
        "command_tokens": [
            "hgtxr.run_e2e_axis_dma_hybrid_smoke",
            "--variant par32-patch32-dtok4-300",
            "--warmup-cycles 1",
            "--repeat-cycles 5",
        ],
        "validation_tokens": [
            "tools/validate_pynq_smoke_result.py",
            "--preset axis-par32-patch32-dtok4-300-hybrid-10-90",
        ],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.bit",
            "hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "hgtxr/run_e2e_axis_dma_hybrid_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.sh",
            "validate_e2e_axis_dma_par32_patch32_dtok4_300_hybrid_10_90_file_smoke.sh",
        },
    },
    "vref-p0-softmax-input-x2-dsp-mixed-stream": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
        "command_tokens": ["--variant vref-p0-softmax-input-x2-dsp-mixed-stream", "--weights-mode file"],
        "validation_tokens": [
            "tools/validate_pynq_smoke_result.py",
            "--preset axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
        ],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit",
            "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
            "validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh",
        },
    },
    "vref-p0-softmax-input-x2-qkv-uram": {
        "artifact_prefix": "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
        "command_tokens": ["--variant vref-p0-softmax-input-x2-qkv-uram", "--weights-mode file"],
        "validation_tokens": [
            "tools/validate_pynq_smoke_result.py",
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram",
        ],
        "files": {
            "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit",
            "hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.hwh",
            "hgtxr/run_e2e_axis_dma_smoke.py",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
            "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
            "tools/validate_pynq_smoke_result.py",
            "run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            "validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tar_member(tar: tarfile.TarFile, member_name: str) -> str:
    extracted = tar.extractfile(member_name)
    if extracted is None:
        return ""
    digest = hashlib.sha256()
    for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    payload = json.loads((bundle_dir / "BUNDLE_MANIFEST.json").read_text())
    if not isinstance(payload, dict):
        raise ValueError("BUNDLE_MANIFEST.json must be an object")
    return payload


def validate_bundle(bundle_dir: Path, tar_path: Path | None, variant: str) -> list[str]:
    if variant not in REQUIRED_BY_VARIANT:
        raise ValueError(f"unsupported variant: {variant}")
    rule = REQUIRED_BY_VARIANT[variant]
    errors: list[str] = []

    if not bundle_dir.exists():
        return [f"bundle_dir missing: {bundle_dir}"]
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        return [f"manifest missing: {manifest_path}"]

    try:
        manifest = load_manifest(bundle_dir)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [f"manifest invalid: {exc}"]

    if manifest.get("variant") != variant:
        errors.append(f"variant got {manifest.get('variant')!r}, expected {variant!r}")
    if manifest.get("artifact_prefix") != rule["artifact_prefix"]:
        errors.append(f"artifact_prefix got {manifest.get('artifact_prefix')!r}, expected {rule['artifact_prefix']!r}")
    command = manifest.get("command")
    if not isinstance(command, str) or any(token not in command for token in rule["command_tokens"]):
        errors.append(f"command missing required tokens: {rule['command_tokens']}")
    validation_command = manifest.get("validation_command")
    if not isinstance(validation_command, str) or any(token not in validation_command for token in rule["validation_tokens"]):
        errors.append(f"validation_command missing required tokens: {rule['validation_tokens']}")

    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    entries = [entry for entry in files if isinstance(entry, dict)]
    bundle_names = {entry.get("bundle") for entry in entries}
    missing = sorted(rule["files"] - bundle_names)
    if missing:
        errors.append(f"manifest missing bundle files: {missing}")

    for entry in entries:
        bundle_name = entry.get("bundle")
        if not isinstance(bundle_name, str):
            errors.append(f"manifest file entry has invalid bundle name: {entry}")
            continue
        path = bundle_dir / bundle_name
        if not path.exists():
            errors.append(f"bundle file missing: {bundle_name}")
            continue
        expected_sha = entry.get("sha256")
        if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
            errors.append(f"bundle file sha256 mismatch: {bundle_name}")

    if tar_path is None:
        tar_info = manifest.get("tar") if isinstance(manifest.get("tar"), dict) else {}
        tar_path_raw = tar_info.get("path")
        if isinstance(tar_path_raw, str):
            tar_path = Path(tar_path_raw)
    if tar_path is None:
        errors.append("tar path not provided and manifest tar.path missing")
        return errors
    if not tar_path.exists():
        errors.append(f"tar missing: {tar_path}")
        return errors

    manifest_tar = manifest.get("tar") if isinstance(manifest.get("tar"), dict) else {}
    expected_tar_sha = manifest_tar.get("sha256")
    if isinstance(expected_tar_sha, str) and sha256_file(tar_path) != expected_tar_sha:
        errors.append("tar sha256 mismatch")

    root_name = bundle_dir.name
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar_names = set(tar.getnames())
            required_in_tar = {f"{root_name}/BUNDLE_MANIFEST.json"} | {f"{root_name}/{name}" for name in rule["files"]}
            missing_in_tar = sorted(required_in_tar - tar_names)
            if missing_in_tar:
                errors.append(f"tar missing files: {missing_in_tar}")
            for entry in entries:
                bundle_name = entry.get("bundle")
                expected_sha = entry.get("sha256")
                if not isinstance(bundle_name, str) or not isinstance(expected_sha, str):
                    continue
                member_name = f"{root_name}/{bundle_name}"
                if member_name in tar_names and sha256_tar_member(tar, member_name) != expected_sha:
                    errors.append(f"tar member sha256 mismatch: {bundle_name}")
    except tarfile.TarError as exc:
        errors.append(f"tar invalid: {exc}")

    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HGTXR PYNQ transfer bundle package.")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=None)
    parser.add_argument("--variant", choices=sorted(REQUIRED_BY_VARIANT), default="c3b-mem16")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate_bundle(args.bundle_dir, args.tar_path, args.variant)
    result = {
        "status": "pass" if not errors else "fail",
        "variant": args.variant,
        "bundle_dir": str(args.bundle_dir),
        "tar": str(args.tar_path) if args.tar_path is not None else None,
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
