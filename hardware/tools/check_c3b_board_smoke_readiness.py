#!/usr/bin/env python3
"""Check whether the C3b ZCU104 smoke package is ready for board execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_bundle_package import sha256_file, validate_bundle
from validate_pynq_smoke_result import EXPECTED_RAW, validate_result


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DEFAULT_TRANSFER_MANIFEST = HGTXR_ROOT / "docs" / "resources" / "c3b_smoke_transfer_manifest_2026_06_10.json"
DEFAULT_SESSION = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
DEFAULT_BUNDLE_DIR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
DEFAULT_TAR = HARDWARE_ROOT / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
DEFAULT_SHA256_FILE = HGTXR_ROOT / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
DEFAULT_RESULT = HARDWARE_ROOT / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "c3b_board_smoke_readiness_2026_06_10.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "c3b_board_smoke_readiness_2026_06_10.md"

EXPECTED_RUNTIME_STATE = 2
EXPECTED_RESULT_NAME = "e2e_axis_dma_c3b_mem16_file_smoke.json"
EXPECTED_VALIDATION_NAME = "e2e_axis_dma_c3b_mem16_file_smoke_validation.json"
EXPECTED_TAR_NAME = "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require_token(errors: list[str], values: Sequence[str], token: str, label: str) -> None:
    if not any(token in value for value in values):
        errors.append(f"{label} missing token: {token}")


def check_expected_outputs(errors: list[str], payload: dict[str, Any], label: str) -> None:
    runtime = payload.get("expected_runtime_state")
    out_raw = payload.get("expected_out_raw")
    if runtime != EXPECTED_RUNTIME_STATE:
        errors.append(f"{label} expected_runtime_state: got {runtime!r}, expected {EXPECTED_RUNTIME_STATE!r}")
    if out_raw != EXPECTED_RAW:
        errors.append(f"{label} expected_out_raw: got {out_raw!r}, expected {EXPECTED_RAW!r}")


def build_readiness(
    *,
    transfer_manifest_path: Path,
    session_path: Path,
    bundle_dir: Path,
    tar_path: Path,
    sha256_file_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []
    transfer: dict[str, Any] = {}
    session: dict[str, Any] = {}

    if not transfer_manifest_path.exists():
        errors.append(f"transfer manifest missing: {transfer_manifest_path}")
    else:
        try:
            transfer = load_json(transfer_manifest_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"transfer manifest invalid: {exc}")

    if not session_path.exists():
        errors.append(f"session json missing: {session_path}")
    else:
        try:
            session = load_json(session_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"session json invalid: {exc}")

    if transfer and transfer.get("status") != "pass":
        errors.append(f"transfer manifest status: got {transfer.get('status')!r}, expected 'pass'")
    if session and session.get("status") != "pass":
        errors.append(f"session status: got {session.get('status')!r}, expected 'pass'")

    if bundle_dir.exists():
        errors.extend(validate_bundle(bundle_dir, tar_path, "c3b-mem16"))
    else:
        errors.append(f"bundle dir missing: {bundle_dir}")

    tar_sha = ""
    if not tar_path.exists():
        errors.append(f"tar missing: {tar_path}")
    else:
        tar_sha = sha256_file(tar_path)
        if tar_path.name != EXPECTED_TAR_NAME:
            errors.append(f"tar name: got {tar_path.name!r}, expected {EXPECTED_TAR_NAME!r}")

    sha256_line = ""
    if not sha256_file_path.exists():
        errors.append(f"sha256 file missing: {sha256_file_path}")
    else:
        sha256_line = sha256_file_path.read_text().strip()
        expected_line = f"{tar_sha}  {tar_path.name}"
        if sha256_line != expected_line:
            errors.append(f"sha256 file line: got {sha256_line!r}, expected {expected_line!r}")

    transfer_tar = transfer.get("tar") if isinstance(transfer.get("tar"), dict) else {}
    session_tar = session.get("tar") if isinstance(session.get("tar"), dict) else {}
    for label, tar_info in [("transfer", transfer_tar), ("session", session_tar)]:
        if tar_info:
            if tar_info.get("sha256") != tar_sha:
                errors.append(f"{label} tar sha256: got {tar_info.get('sha256')!r}, expected {tar_sha!r}")
            if tar_info.get("name") not in (None, tar_path.name):
                errors.append(f"{label} tar name: got {tar_info.get('name')!r}, expected {tar_path.name!r}")

    if transfer:
        check_expected_outputs(errors, transfer.get("board_expected_outputs", {}), "transfer")
        require_token(errors, list(transfer.get("board_verify_commands", [])), "sha256sum -c", "board verify commands")
        board_commands = list(transfer.get("board_run_commands", []))
        require_token(errors, board_commands, "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh", "board run commands")
        require_token(errors, board_commands, "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh", "board run commands")
        host = transfer.get("host_copyback") if isinstance(transfer.get("host_copyback"), dict) else {}
        import_command = str(host.get("import_command", ""))
        if "tools/import_pynq_smoke_result.py" not in import_command or "--preset axis-c3b-mem16" not in import_command:
            errors.append("host import command missing import tool or axis-c3b-mem16 preset")
        if host.get("canonical_result_path") and Path(str(host["canonical_result_path"])) != result_path:
            errors.append(f"canonical result path: got {host['canonical_result_path']!r}, expected {str(result_path)!r}")

    if session:
        check_expected_outputs(errors, session, "session")
        if session.get("result_json") != EXPECTED_RESULT_NAME:
            errors.append(f"session result_json: got {session.get('result_json')!r}, expected {EXPECTED_RESULT_NAME!r}")
        if session.get("validation_json") != EXPECTED_VALIDATION_NAME:
            errors.append(f"session validation_json: got {session.get('validation_json')!r}, expected {EXPECTED_VALIDATION_NAME!r}")
        if session.get("canonical_result_path") and Path(str(session["canonical_result_path"])) != result_path:
            errors.append(f"session canonical_result_path: got {session['canonical_result_path']!r}, expected {str(result_path)!r}")

    physical_result: dict[str, Any] = {
        "path": str(result_path),
        "status": "missing",
        "validation_errors": [],
    }
    if result_path.exists():
        try:
            result_payload = load_json(result_path)
            result_errors = validate_result(result_payload, "axis-c3b-mem16")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            result_errors = [f"physical smoke result invalid: {exc}"]
        physical_result["status"] = "pass" if not result_errors else "fail"
        physical_result["validation_errors"] = result_errors
        errors.extend(result_errors)
    else:
        next_steps.append("Run C3b smoke on ZCU104 and copy back e2e_axis_dma_c3b_mem16_file_smoke.json.")
        next_steps.append("Import result with tools/import_pynq_smoke_result.py --preset axis-c3b-mem16.")
        warnings.append("Physical C3b board smoke result is not present yet.")

    if errors:
        status = "blocked"
        ready = False
    elif physical_result["status"] == "pass":
        status = "complete"
        ready = True
    else:
        status = "ready-for-board"
        ready = True

    return {
        "status": status,
        "ready": ready,
        "target": "ZCU104 PYNQ",
        "variant": "c3b-mem16",
        "preset": "axis-c3b-mem16",
        "transfer_manifest": str(transfer_manifest_path),
        "session_json": str(session_path),
        "bundle_dir": str(bundle_dir),
        "tar": {
            "path": str(tar_path),
            "name": tar_path.name,
            "sha256": tar_sha,
            "sha256_file": str(sha256_file_path),
            "sha256_line": sha256_line,
        },
        "physical_result": physical_result,
        "expected_runtime_state": EXPECTED_RUNTIME_STATE,
        "expected_out_raw": EXPECTED_RAW,
        "errors": errors,
        "warnings": warnings,
        "next_steps": next_steps,
    }


def render_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# HGTXR C3b Board Smoke Readiness",
        "",
        f"- status: `{readiness['status']}`",
        f"- ready: `{readiness['ready']}`",
        f"- target: `{readiness['target']}`",
        f"- variant: `{readiness['variant']}`",
        f"- tar_sha256: `{readiness['tar']['sha256']}`",
        f"- physical_result: `{readiness['physical_result']['status']}`",
        "",
        "## Inputs",
        "",
        f"- transfer_manifest: `{readiness['transfer_manifest']}`",
        f"- session_json: `{readiness['session_json']}`",
        f"- bundle_dir: `{readiness['bundle_dir']}`",
        f"- sha256_file: `{readiness['tar']['sha256_file']}`",
        "",
        "## Board Commands",
        "",
        "```sh",
        "sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
        "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
        "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
        "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "```",
        "",
        "## Host Import",
        "",
        "```sh",
        "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json",
        "```",
        "",
        "## Errors",
        "",
    ]
    if readiness["errors"]:
        lines.extend(f"- `{error}`" for error in readiness["errors"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if readiness["warnings"]:
        lines.extend(f"- `{warning}`" for warning in readiness["warnings"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Steps", ""])
    if readiness["next_steps"]:
        lines.extend(f"- {step}" for step in readiness["next_steps"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_outputs(readiness: dict[str, Any], json_out: Path | None, markdown_out: Path | None) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(readiness))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check C3b ZCU104 board smoke readiness.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--transfer-manifest", type=Path, default=DEFAULT_TRANSFER_MANIFEST)
    parser.add_argument("--session-json", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--tar", dest="tar_path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--sha256-file", type=Path, default=DEFAULT_SHA256_FILE)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    hgtxr_root = args.root.resolve()
    result_path = args.result_json
    if not result_path.is_absolute():
        result_path = hgtxr_root / result_path
    readiness = build_readiness(
        transfer_manifest_path=args.transfer_manifest,
        session_path=args.session_json,
        bundle_dir=args.bundle_dir,
        tar_path=args.tar_path,
        sha256_file_path=args.sha256_file,
        result_path=(
            hgtxr_root / "hardware" / "pynq" / "hgtxr" / EXPECTED_RESULT_NAME
            if args.result_json == DEFAULT_RESULT and hgtxr_root != HGTXR_ROOT.resolve()
            else result_path
        ),
    )
    write_outputs(readiness, args.json_out, args.markdown_out)
    print(
        f"[c3b-board-smoke-readiness] status={readiness['status']} "
        f"ready={readiness['ready']} errors={len(readiness['errors'])} "
        f"warnings={len(readiness['warnings'])}"
    )
    return 0 if readiness["status"] in {"ready-for-board", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
