#!/usr/bin/env python3
"""Rank local candidates that could replace the missing XR-VITs reference tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HLS_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hpp", ".tcl", ".v", ".sv"}
PRIMITIVE_TOKENS = ["layernorm", "gelu", "softmax", "quant"]


def list_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def candidate_report(root: Path, role: str) -> dict[str, Any]:
    files = list_files(root)
    file_names = [rel(path, root) for path in files]
    lower_names = [name.lower() for name in file_names]
    hls_files = [name for name in file_names if Path(name).suffix.lower() in HLS_EXTENSIONS]

    zcu104_configs = [name for name in file_names if "zcu104" in name.lower() and name.endswith((".json", ".tcl"))]
    cyclic_deit_configs = [
        name
        for name in file_names
        if name.endswith(".json") and ("deit" in name.lower() or "cyclic" in name.lower()) and "zcu104" in name.lower()
    ]
    primitive_hits = {
        token: sorted(name for name in file_names if token in name.lower())[:8]
        for token in PRIMITIVE_TOKENS
    }
    vivado_zcu104 = [name for name in file_names if "vivado" in name.lower() and "zcu104" in name.lower()]
    cyclic_top = [name for name in file_names if "cyclic" in name.lower() and ("top" in name.lower() or "vit" in name.lower())]

    score = 0
    score += 20 if zcu104_configs else 0
    score += 20 if cyclic_deit_configs else 0
    score += min(20, len(hls_files) // 5)
    score += 5 * sum(1 for hits in primitive_hits.values() if hits)
    score += 10 if vivado_zcu104 else 0
    score += 10 if cyclic_top else 0

    return {
        "path": str(root),
        "role": role,
        "exists": root.exists(),
        "score": score,
        "file_count": len(files),
        "hls_file_count": len(hls_files),
        "zcu104_configs": sorted(zcu104_configs)[:12],
        "cyclic_deit_configs": sorted(cyclic_deit_configs)[:12],
        "primitive_hits": primitive_hits,
        "vivado_zcu104": sorted(vivado_zcu104)[:12],
        "cyclic_top": sorted(cyclic_top)[:12],
        "top_hls_samples": sorted(hls_files)[:20],
        "readme_exists": "readme.md" in lower_names,
    }


def build_audit(xr_vit_root: Path) -> dict[str, Any]:
    candidates = [
        candidate_report(xr_vit_root / "XR_Accel", "candidate-xr-accel"),
        candidate_report(xr_vit_root / "analysis" / "XR_Accel", "candidate-analysis-xr-accel"),
        candidate_report(xr_vit_root / "ViT_Accel", "candidate-vit-accel"),
    ]
    ranked = sorted(candidates, key=lambda item: (item["score"], item["hls_file_count"], item["file_count"]), reverse=True)
    recommendation = ranked[0] if ranked and ranked[0]["exists"] else None
    return {
        "status": "candidate-found" if recommendation else "no-candidate",
        "requested_path": str(xr_vit_root.parent / "XR-VITs"),
        "approved_replacement": False,
        "recommendation": {
            "role": recommendation["role"],
            "path": recommendation["path"],
            "score": recommendation["score"],
            "reason": "highest score for ZCU104/cyclic/DeiT/HLS evidence",
        }
        if recommendation
        else None,
        "candidates": ranked,
        "policy": "Do not treat any candidate as XR-VITs until the user approves the replacement or the exact requested path is restored.",
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# XR-VITs Candidate Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- requested_path: `{audit['requested_path']}`",
        f"- approved_replacement: `{audit['approved_replacement']}`",
        f"- policy: {audit['policy']}",
        "",
        "## Recommendation",
        "",
    ]
    recommendation = audit["recommendation"]
    if recommendation:
        lines.append(f"- `{recommendation['role']}` score=`{recommendation['score']}` path=`{recommendation['path']}`")
        lines.append(f"- reason: {recommendation['reason']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Candidates", ""])
    for candidate in audit["candidates"]:
        lines.append(f"### {candidate['role']}")
        lines.append(f"- path: `{candidate['path']}`")
        lines.append(f"- exists: `{candidate['exists']}`")
        lines.append(f"- score: `{candidate['score']}`")
        lines.append(f"- files: `{candidate['file_count']}`")
        lines.append(f"- hls_files: `{candidate['hls_file_count']}`")
        lines.append(f"- zcu104_configs: `{len(candidate['zcu104_configs'])}`")
        lines.append(f"- cyclic_deit_configs: `{len(candidate['cyclic_deit_configs'])}`")
        primitive_summary = ", ".join(f"{name}:{len(hits)}" for name, hits in candidate["primitive_hits"].items())
        lines.append(f"- primitives: `{primitive_summary}`")
        if candidate["zcu104_configs"]:
            lines.append("- zcu104 samples:")
            for name in candidate["zcu104_configs"][:5]:
                lines.append(f"  - `{name}`")
        lines.append("")
    return "\n".join(lines)


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
    rec = audit["recommendation"]["role"] if audit["recommendation"] else "none"
    print(f"[xr-vits-candidate-audit] status={audit['status']} recommendation={rec}")
    return 0 if audit["status"] == "candidate-found" else 1


if __name__ == "__main__":
    raise SystemExit(main())
