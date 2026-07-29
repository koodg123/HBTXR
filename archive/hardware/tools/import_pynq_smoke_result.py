#!/usr/bin/env python3
"""Validate and import a board-produced HGTXR PYNQ smoke result JSON."""

from __future__ import annotations

import argparse
import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_smoke_result import VARIANT_PRESETS, load_json, validate_result


DEFAULT_DEST_BY_PRESET = {
    "axis-a1": "hardware/pynq/hgtxr/e2e_axis_dma_a1_file_smoke.json",
    "axis-c1-par16": "hardware/pynq/hgtxr/e2e_axis_dma_c1_par16_file_smoke.json",
    "axis-c3b-mem16": "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json",
    "axis-runtime-mode-par32-search": (
        "hardware/pynq/hgtxr/e2e_axis_dma_runtime_mode_par32_search_file_smoke.json"
    ),
    "axis-runtime-mode-par32-track": (
        "hardware/pynq/hgtxr/e2e_axis_dma_runtime_mode_par32_track_file_smoke.json"
    ),
    "axis-par32-rom-compute-300-search": (
        "hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json"
    ),
    "axis-par32-rom-compute-300-track": (
        "hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json"
    ),
    "axis-par32-prefetchall4-300-search": (
        "hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json"
    ),
    "axis-par32-prefetchall4-300-track": (
        "hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json"
    ),
    "axis-par32-prefetchall4-300-hybrid-10-90": (
        "hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json"
    ),
    "axis-vref-p0-softmax-input-x2-dsp-mixed-stream": (
        "hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
    ),
    "axis-vref-p0-softmax-input-x2-qkv-uram": (
        "hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
    ),
    "m_axi-a2": "hardware/pynq/hgtxr/e2e_m_axi_file_smoke.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "variant": payload.get("variant"),
        "runtime_state": payload.get("runtime_state"),
        "out_raw": payload.get("out_raw"),
        "hybrid_profile": payload.get("hybrid_profile"),
        "search_track_invocation_distribution": payload.get("search_track_invocation_distribution"),
    }


def destination_validation(path: Path, *, preset: str, require_paths: bool, source_sha256: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "dest_sha256": "",
            "dest_matches_source": False,
            "payload_summary": {},
            "errors": [f"destination missing: {path}"],
        }
    dest_sha256 = sha256_file(path)
    try:
        payload = load_json(path)
        errors = validate_result(payload, preset, require_paths=require_paths)
        summary = payload_summary(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors = [f"destination JSON invalid: {exc}"]
        summary = {}
    return {
        "status": "pass" if not errors else "fail",
        "path": str(path),
        "dest_sha256": dest_sha256,
        "dest_matches_source": dest_sha256 == source_sha256,
        "payload_summary": summary,
        "errors": errors,
    }


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def import_result(
    source: Path,
    *,
    preset: str,
    hgtxr_root: Path,
    dest: Path | None = None,
    require_paths: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    if preset not in DEFAULT_DEST_BY_PRESET:
        raise ValueError(f"unsupported preset: {preset}")
    payload = load_json(source)
    errors = validate_result(payload, preset, require_paths=require_paths)
    default_target = hgtxr_root / DEFAULT_DEST_BY_PRESET[preset]
    target = dest if dest is not None else default_target
    dest_exists_before = target.exists()
    source_sha256 = sha256_file(source)
    result = {
        "status": "pass" if not errors else "fail",
        "preset": preset,
        "source": str(source),
        "source_name": source.name,
        "source_sha256": source_sha256,
        "dest": str(target),
        "default_dest": str(default_target),
        "dest_is_default": target.resolve() == default_target.resolve(),
        "dest_exists_before": dest_exists_before,
        "dest_exists_after": dest_exists_before,
        "dest_validation": destination_validation(
            target,
            preset=preset,
            require_paths=require_paths,
            source_sha256=source_sha256,
        ) if dest_exists_before else {
            "status": "missing",
            "path": str(target),
            "dest_sha256": "",
            "dest_matches_source": False,
            "payload_summary": {},
            "errors": [f"destination missing before import: {target}"],
        },
        "would_clear_current_gate": False,
        "payload_summary": payload_summary(payload),
        "copied": False,
        "dry_run": dry_run,
        "require_paths": require_paths,
        "errors": errors,
    }
    if errors:
        return result

    if dry_run:
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    result["copied"] = True
    result["dest_exists_after"] = target.exists()
    result["dest_validation"] = destination_validation(
        target,
        preset=preset,
        require_paths=require_paths,
        source_sha256=source_sha256,
    )
    result["would_clear_current_gate"] = (
        result["dest_is_default"]
        and result["dest_exists_after"]
        and result["dest_validation"]["status"] == "pass"
        and result["dest_validation"]["dest_matches_source"] is True
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and import HGTXR PYNQ smoke result JSON.")
    parser.add_argument("source_json", type=Path)
    parser.add_argument("--preset", choices=sorted(VARIANT_PRESETS), default="axis-c3b-mem16")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--no-require-paths", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate only; do not copy into the canonical smoke-result path")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--validation-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = import_result(
        args.source_json,
        preset=args.preset,
        hgtxr_root=args.root.resolve(),
        dest=args.dest,
        require_paths=not args.no_require_paths,
        dry_run=args.dry_run,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    if args.validation_out is not None:
        validation_payload = {
            "status": result["status"],
            "preset": result["preset"],
            "result_json": str(args.source_json),
            "dry_run": result["dry_run"],
            "source_sha256": result["source_sha256"],
            "dest": result["dest"],
            "dest_is_default": result["dest_is_default"],
            "dest_validation": result["dest_validation"],
            "would_clear_current_gate": result["would_clear_current_gate"],
            "copied": result["copied"],
            "payload_summary": result["payload_summary"],
            "errors": result["errors"],
        }
        args.validation_out.parent.mkdir(parents=True, exist_ok=True)
        args.validation_out.write_text(json.dumps(validation_payload, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
