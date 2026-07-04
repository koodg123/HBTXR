#!/usr/bin/env python3
"""Write transfer manifest for the C3b ZCU104 physical smoke bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_bundle_package import sha256_file, validate_bundle


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DEFAULT_SESSION = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
DEFAULT_BUNDLE_DIR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "c3b_smoke_transfer_manifest_2026_06_10.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "c3b_smoke_transfer_manifest_2026_06_10.md"
DEFAULT_SHA256 = HARDWARE_ROOT / "generated" / "signoff" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_manifest(*, session_path: Path, bundle_dir: Path, tar_path: Path, hgtxr_root: Path) -> dict[str, Any]:
    session = load_json(session_path)
    errors = validate_bundle(bundle_dir, tar_path, "c3b-mem16")
    tar_sha = sha256_file(tar_path) if tar_path.exists() else ""
    tar_name = tar_path.name
    expected_sha = session.get("tar", {}).get("sha256") if isinstance(session.get("tar"), dict) else None
    if expected_sha != tar_sha:
        errors.append(f"session tar sha256 mismatch: {expected_sha!r} != {tar_sha!r}")

    result_json = str(session.get("result_json", "e2e_axis_dma_c3b_mem16_file_smoke.json"))
    validation_json = str(session.get("validation_json", "e2e_axis_dma_c3b_mem16_file_smoke_validation.json"))
    canonical_result = hgtxr_root / "hardware" / "pynq" / "hgtxr" / result_json
    sha256_file_name = f"{tar_name}.sha256"
    return {
        "status": "pass" if not errors else "fail",
        "target": "ZCU104 PYNQ",
        "variant": "c3b-mem16",
        "preset": "axis-c3b-mem16",
        "tar": {
            "path": str(tar_path),
            "name": tar_name,
            "bytes": tar_path.stat().st_size if tar_path.exists() else None,
            "sha256": tar_sha,
            "sha256_file": sha256_file_name,
            "sha256_line": f"{tar_sha}  {tar_name}",
        },
        "transfer_files": [
            str(tar_path),
            str(DEFAULT_SHA256),
        ],
        "board_verify_commands": [
            f"sha256sum -c {sha256_file_name}",
        ],
        "board_run_commands": list(session.get("board_steps", [])),
        "board_expected_outputs": {
            "result_json": result_json,
            "validation_json": validation_json,
            "expected_runtime_state": session.get("expected_runtime_state"),
            "expected_out_raw": session.get("expected_out_raw"),
        },
        "host_copyback": {
            "source_on_board": result_json,
            "canonical_result_path": str(canonical_result),
            "import_command": f"python3 tools/import_pynq_smoke_result.py /path/to/{result_json} --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json",
            "validate_command": f"python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/{result_json} --preset axis-c3b-mem16",
            "final_signoff_command": "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
        },
        "bundle_validation_errors": errors,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# HGTXR C3b Smoke Transfer Manifest",
        "",
        f"- status: `{manifest['status']}`",
        f"- target: `{manifest['target']}`",
        f"- variant: `{manifest['variant']}`",
        f"- tar: `{manifest['tar']['path']}`",
        f"- tar_sha256: `{manifest['tar']['sha256']}`",
        "",
        "## Transfer Files",
        "",
    ]
    for path in manifest["transfer_files"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Board Verify", "", "```sh"])
    lines.extend(manifest["board_verify_commands"])
    lines.extend(["```", "", "## Board Run", "", "```sh"])
    lines.extend(manifest["board_run_commands"])
    lines.extend(["```", "", "## Host Copy-Back", "", "```sh"])
    host = manifest["host_copyback"]
    lines.extend([host["import_command"], host["validate_command"], host["final_signoff_command"]])
    lines.extend(["```", "", "## Expected Output", ""])
    expected = manifest["board_expected_outputs"]
    lines.append(f"- result_json: `{expected['result_json']}`")
    lines.append(f"- validation_json: `{expected['validation_json']}`")
    lines.append(f"- expected_runtime_state: `{expected['expected_runtime_state']}`")
    lines.append(f"- expected_out_raw: `{expected['expected_out_raw']}`")
    lines.extend(["", "## Bundle Validation Errors", ""])
    if manifest["bundle_validation_errors"]:
        for error in manifest["bundle_validation_errors"]:
            lines.append(f"- `{error}`")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_outputs(manifest: dict[str, Any], json_out: Path, markdown_out: Path, sha256_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    sha256_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(manifest))
    sha256_out.write_text(manifest["tar"]["sha256_line"] + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write C3b smoke transfer manifest.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--session-json", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--sha256-out", type=Path, default=DEFAULT_SHA256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        session_path=args.session_json,
        bundle_dir=args.bundle_dir,
        tar_path=args.tar_path,
        hgtxr_root=args.root.resolve(),
    )
    write_outputs(manifest, args.json_out, args.markdown_out, args.sha256_out)
    print(
        f"[c3b-smoke-transfer-manifest] status={manifest['status']} "
        f"sha256={manifest['tar']['sha256']} errors={len(manifest['bundle_validation_errors'])}"
    )
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
