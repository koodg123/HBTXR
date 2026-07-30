#!/usr/bin/env python3
"""Prepare a reproducible ZCU104 PYNQ smoke-session manifest and runbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from import_pynq_smoke_result import DEFAULT_DEST_BY_PRESET
from validate_pynq_bundle_package import load_manifest, sha256_file, validate_bundle


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DEFAULT_VARIANT = "c3b-mem16"
DEFAULT_PRESET = "axis-c3b-mem16"
DEFAULT_BUNDLE_DIR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.md"


def result_name(variant: str) -> str:
    return f"e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.json"


def validation_name(variant: str) -> str:
    return f"e2e_axis_dma_{variant.replace('-', '_')}_file_smoke_validation.json"


def run_script_name(variant: str) -> str:
    return f"run_e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.sh"


def validate_script_name(variant: str) -> str:
    return f"validate_e2e_axis_dma_{variant.replace('-', '_')}_file_smoke.sh"


def host_copyback_steps(*, result_json: str, preset: str, hgtxr_root: Path) -> list[str]:
    root = hgtxr_root.as_posix()
    canonical = DEFAULT_DEST_BY_PRESET[preset]
    if preset == "axis-c3b-mem16":
        import_arg = "--import-c3b-smoke-json"
        dry_run_arg = "--dry-run-import-c3b-smoke"
    elif preset == "axis-vref-p0-softmax-input-x2-dsp-mixed-stream":
        import_arg = "--import-vref-successor-smoke-json"
        dry_run_arg = "--dry-run-import-vref-successor-smoke"
    else:
        variant_tag = preset.replace("axis-", "").replace("-", "_")
        return [
            f"python3 tools/import_pynq_smoke_result.py /path/to/{result_json} --preset {preset} --dry-run --json-out /tmp/hgtxr_{variant_tag}_smoke_import_dry_run.json --validation-out /tmp/hgtxr_{variant_tag}_smoke_validation_dry_run.json",
            f"python3 tools/import_pynq_smoke_result.py /path/to/{result_json} --preset {preset} --json-out /tmp/hgtxr_{variant_tag}_smoke_import.json --validation-out /tmp/hgtxr_{variant_tag}_smoke_validation.json",
            f"python3 tools/validate_pynq_smoke_result.py {canonical} --preset {preset}",
            f"python3 tools/check_third_goal_preflight.py --root {root} --mode final-signoff",
        ]

    return [
        f"python3 tools/run_third_goal_final_signoff.py --root {root} {import_arg} /path/to/{result_json} {dry_run_arg} --allow-blocked",
        f"python3 tools/run_third_goal_final_signoff.py --root {root} {import_arg} /path/to/{result_json} --allow-blocked",
        f"python3 tools/validate_pynq_smoke_result.py {canonical} --preset {preset}",
        f"python3 tools/check_third_goal_preflight.py --root {root} --mode final-signoff",
    ]


def build_session(
    *,
    bundle_dir: Path,
    tar_path: Path,
    variant: str,
    preset: str,
    hgtxr_root: Path,
) -> dict[str, Any]:
    errors = validate_bundle(bundle_dir, tar_path, variant)
    manifest = load_manifest(bundle_dir) if (bundle_dir / "BUNDLE_MANIFEST.json").exists() else {}
    result_json = result_name(variant)
    validation_json = validation_name(variant)
    canonical_dest = hgtxr_root / DEFAULT_DEST_BY_PRESET[preset]
    tar_basename = tar_path.name
    bundle_root = bundle_dir.name
    tar_info = {
        "path": str(tar_path),
        "name": tar_basename,
        "bytes": tar_path.stat().st_size if tar_path.exists() else None,
        "sha256": sha256_file(tar_path) if tar_path.exists() else None,
    }
    return {
        "status": "pass" if not errors else "fail",
        "target": "ZCU104 PYNQ",
        "variant": variant,
        "preset": preset,
        "bundle_dir": str(bundle_dir),
        "bundle_root": bundle_root,
        "tar": tar_info,
        "expected_runtime_state": manifest.get("expected_runtime_state"),
        "expected_out_raw": manifest.get("expected_out_raw"),
        "result_json": result_json,
        "validation_json": validation_json,
        "canonical_result_path": str(canonical_dest),
        "bundle_validation_errors": errors,
        "board_steps": [
            f"tar -xzf {tar_basename}",
            f"cd {bundle_root}",
            f"./{run_script_name(variant)}",
            f"./{validate_script_name(variant)}",
        ],
        "host_steps": host_copyback_steps(result_json=result_json, preset=preset, hgtxr_root=hgtxr_root),
    }


def render_markdown(session: dict[str, Any]) -> str:
    board_steps = "\n".join(f"{idx}. `{step}`" for idx, step in enumerate(session["board_steps"], start=1))
    host_steps = "\n".join(f"{idx}. `{step}`" for idx, step in enumerate(session["host_steps"], start=1))
    errors = session.get("bundle_validation_errors") or []
    error_text = "None" if not errors else "\n".join(f"- `{error}`" for error in errors)
    return (
        "# HGTXR ZCU104 Smoke Session\n\n"
        f"- Status: `{session['status']}`\n"
        f"- Target: `{session['target']}`\n"
        f"- Variant: `{session['variant']}`\n"
        f"- Preset: `{session['preset']}`\n"
        f"- Tarball: `{session['tar']['path']}`\n"
        f"- Tarball SHA256: `{session['tar']['sha256']}`\n"
        f"- Expected runtime_state: `{session['expected_runtime_state']}`\n"
        f"- Expected out_raw: `{session['expected_out_raw']}`\n"
        f"- Board result JSON: `{session['result_json']}`\n"
        f"- Canonical host result: `{session['canonical_result_path']}`\n\n"
        "## Board Steps\n\n"
        f"{board_steps}\n\n"
        "## Host Copy-Back Steps\n\n"
        f"{host_steps}\n\n"
        "## Bundle Validation Errors\n\n"
        f"{error_text}\n"
    )


def write_outputs(session: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(session) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HGTXR ZCU104 PYNQ smoke-session runbook.")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session = build_session(
        bundle_dir=args.bundle_dir,
        tar_path=args.tar_path,
        variant=args.variant,
        preset=args.preset,
        hgtxr_root=args.root.resolve(),
    )
    write_outputs(session, args.json_out, args.markdown_out)
    print(json.dumps(session, indent=2, sort_keys=True))
    return 0 if session["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
