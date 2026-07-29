#!/usr/bin/env python3
"""Audit Req9 DeiT-Tiny C-Syn image reference evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
IMAGE_NAME = "DeiT-Tiny C-Syn Results.png"
REQUESTED_USER_SUFFIX = f"PAPER_PRJXR\\05_RESOURCES\\{IMAGE_NAME}"
PNG_MAGIC = bytes.fromhex("89504e470d0a1a0a")


def normalize_roots(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        hgtxr = root.parent
        hardware = root
    else:
        hgtxr = root
        hardware = root / "hardware"
    xr_vit = hgtxr.parent
    return hgtxr, hardware, xr_vit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    info: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": sha256_file(path) if exists and path.is_file() else "",
        "png_magic": path.read_bytes()[:8].hex() if exists and path.is_file() else "",
    }
    return info


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware, xr_vit = normalize_roots(root)
    requested = xr_vit / "PAPER_PRJXR" / "05_RESOURCES" / IMAGE_NAME
    hgpipe = xr_vit / "HGPIPE" / IMAGE_NAME
    docs_copy = hardware / "docs" / IMAGE_NAME
    normalized_requested_suffix = REQUESTED_USER_SUFFIX.replace("\\", "/")
    requested_info = file_info(requested)
    hgpipe_info = file_info(hgpipe)
    docs_info = file_info(docs_copy)
    checks: list[dict[str, Any]] = []

    add_check(
        checks,
        "requested_path_normalizes_to_paper_prjxr",
        requested.as_posix().endswith(normalized_requested_suffix),
        {"requested_user_suffix": REQUESTED_USER_SUFFIX, "normalized": requested.as_posix()},
    )
    add_check(checks, "requested_image_exists", requested.exists(), str(requested))
    add_check(checks, "requested_image_nonempty", requested_info["size_bytes"] > 0, requested_info["size_bytes"])
    add_check(checks, "requested_image_png_magic", requested_info["png_magic"] == PNG_MAGIC.hex(), requested_info["png_magic"])
    add_check(checks, "requested_image_sha256_valid", len(requested_info["sha256"]) == 64, requested_info["sha256"])
    add_check(checks, "hgpipe_substitute_exists", hgpipe.exists(), str(hgpipe))
    add_check(checks, "hgpipe_substitute_png_magic", hgpipe_info["png_magic"] == PNG_MAGIC.hex(), hgpipe_info["png_magic"])
    add_check(
        checks,
        "hgpipe_substitute_matches_requested_sha256",
        requested_info["sha256"] != "" and requested_info["sha256"] == hgpipe_info["sha256"],
        {"requested": requested_info["sha256"], "hgpipe": hgpipe_info["sha256"]},
    )
    add_check(checks, "hardware_docs_copy_exists", docs_copy.exists(), str(docs_copy))
    add_check(checks, "hardware_docs_copy_png_magic", docs_info["png_magic"] == PNG_MAGIC.hex(), docs_info["png_magic"])
    add_check(
        checks,
        "hardware_docs_copy_matches_requested_sha256",
        requested_info["sha256"] != "" and requested_info["sha256"] == docs_info["sha256"],
        {"requested": requested_info["sha256"], "docs_copy": docs_info["sha256"]},
    )

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "xr_vit_root": str(xr_vit),
        "requirement": "Req9 requested PAPER_PRJXR DeiT-Tiny C-Syn image reference",
        "requested_user_path": f"{xr_vit}\\{REQUESTED_USER_SUFFIX}",
        "normalized_requested_path": str(requested),
        "requested_image": requested_info,
        "hgpipe_substitute": hgpipe_info,
        "hardware_docs_copy": docs_info,
        "checks": checks,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "safety": {
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Req9 DeiT Image Reference Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- requested_user_path: `{audit['requested_user_path']}`",
        f"- normalized_requested_path: `{audit['normalized_requested_path']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        f"- fail_count: `{audit['fail_count']}`",
        "",
        "## Images",
        "",
        "| Role | Exists | Size | SHA256 | Path |",
        "|---|---:|---:|---|---|",
    ]
    for role in ["requested_image", "hgpipe_substitute", "hardware_docs_copy"]:
        info = audit[role]
        lines.append(
            f"| {role} | `{info['exists']}` | `{info['size_bytes']}` | `{info['sha256']}` | {info['path']} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Req9 DeiT image reference audit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(f"[req9-deit-image-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
