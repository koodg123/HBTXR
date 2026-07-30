#!/usr/bin/env python3
"""Audit requested third-goal reference inputs and nearby fallback candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_candidate(path: Path, role: str) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "role": role, "exists": path.exists()}
    if path.exists() and path.is_file():
        row["bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    return row


def dir_candidate(path: Path, role: str) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "role": role, "exists": path.exists()}
    if path.exists() and path.is_dir():
        markers = []
        for marker in ["hardware", "hls", "vivado", "src", "README.md", "AGENTS.md"]:
            if (path / marker).exists():
                markers.append(marker)
        row["markers"] = markers
    return row


def build_audit(xr_vit_root: Path) -> dict[str, Any]:
    requested_image = xr_vit_root / "PAPER_PRJXR" / "05_RESOURCES" / "DeiT-Tiny C-Syn Results.png"
    requested_xr_vits = xr_vit_root.parent / "XR-VITs"

    image_candidates = [
        file_candidate(requested_image, "requested"),
        file_candidate(xr_vit_root / "HGPIPE" / "DeiT-Tiny C-Syn Results.png", "candidate-hgpipe"),
    ]
    hls_candidates = [
        dir_candidate(requested_xr_vits, "requested"),
        dir_candidate(xr_vit_root / "XR_Accel", "candidate-xr-accel"),
        dir_candidate(xr_vit_root / "analysis" / "XR_Accel", "candidate-analysis-xr-accel"),
        dir_candidate(xr_vit_root / "ViT_Accel", "candidate-vit-accel"),
    ]

    blockers = []
    if not requested_image.exists():
        blockers.append(
            {
                "name": "requested PAPER_PRJXR DeiT image",
                "path": str(requested_image),
                "reason": "requested file missing",
                "candidate_count": sum(1 for candidate in image_candidates if candidate["role"] != "requested" and candidate["exists"]),
                "required_action": "restore requested file or get explicit approval for a replacement source",
            }
        )
    if not requested_xr_vits.exists():
        blockers.append(
            {
                "name": "requested XR-VITs sibling",
                "path": str(requested_xr_vits),
                "reason": "requested directory missing",
                "candidate_count": sum(1 for candidate in hls_candidates if candidate["role"] != "requested" and candidate["exists"]),
                "required_action": "restore requested checkout or get explicit approval for a replacement source",
            }
        )

    return {
        "status": "pass" if not blockers else "needs-reference-input",
        "approved_replacements": False,
        "xr_vit_root": str(xr_vit_root),
        "blocker_count": len(blockers),
        "blockers": blockers,
        "image_candidates": image_candidates,
        "hls_code_candidates": hls_candidates,
        "restore_commands": [
            "mkdir -p /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES",
            "cp -a /home/kjm26/project/PRJXR/XR-VIT/HGPIPE/DeiT-Tiny\\ C-Syn\\ Results.png /home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny\\ C-Syn\\ Results.png",
            "# restore or clone the requested /home/kjm26/project/PRJXR/XR-VITs checkout",
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Reference Input Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- approved_replacements: `{audit['approved_replacements']}`",
        f"- blockers: `{audit['blocker_count']}`",
        "",
        "## Blockers",
        "",
    ]
    if audit["blockers"]:
        for blocker in audit["blockers"]:
            lines.append(f"- `{blocker['name']}`: {blocker['reason']}")
            lines.append(f"  - path: `{blocker['path']}`")
            lines.append(f"  - candidates_found: `{blocker['candidate_count']}`")
            lines.append(f"  - action: {blocker['required_action']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Image Candidates", ""])
    for candidate in audit["image_candidates"]:
        suffix = f", sha256=`{candidate['sha256']}`" if candidate.get("sha256") else ""
        lines.append(f"- `{candidate['role']}` exists=`{candidate['exists']}` path=`{candidate['path']}`{suffix}")

    lines.extend(["", "## HLS Code Candidates", ""])
    for candidate in audit["hls_code_candidates"]:
        markers = ",".join(candidate.get("markers", [])) or "none"
        lines.append(f"- `{candidate['role']}` exists=`{candidate['exists']}` path=`{candidate['path']}` markers=`{markers}`")

    lines.extend(["", "## Restore Commands", "", "```sh"])
    lines.extend(audit["restore_commands"])
    lines.append("```")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xr-vit-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args(argv)

    audit = build_audit(Path(args.xr_vit_root).resolve())
    json_out = Path(args.json_out)
    markdown_out = Path(args.markdown_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2) + "\n")
    markdown_out.write_text(render_markdown(audit))
    print(f"[reference-input-audit] status={audit['status']} blockers={audit['blocker_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
