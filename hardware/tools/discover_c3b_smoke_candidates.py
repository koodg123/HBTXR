#!/usr/bin/env python3
"""Discover side-effect-free C3b PYNQ smoke result candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_smoke_result import load_json, validate_result


DATE_TAG = "2026_06_10"
DEFAULT_PATTERNS = ["*c3b*smoke*.json", "*e2e_axis_dma_c3b_mem16*.json"]
SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache"}
SKIP_PATH_PARTS = {
    "generated/signoff",
    "docs/resources/final_",
    "docs/resources/third_goal_",
    "docs/resources/zcu104_",
    "docs/resources/c3b_axis_dma_smoke_bundle_",
    "docs/resources/c3b_board_smoke_readiness_",
    "docs/resources/c3b_physical_smoke_gate_audit_",
    "docs/resources/c3b_smoke_candidate_discovery_",
    "docs/resources/c3b_smoke_result_contract_",
    "docs/resources/c3b_smoke_transfer_manifest_",
    "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json",
}


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_scan_roots(root: Path) -> list[Path]:
    return [
        root / "hardware" / "pynq" / "hgtxr",
        root / "hardware" / "generated" / "pynq",
        root / "docs" / "resources",
    ]


def should_skip(path: Path) -> bool:
    text = path.as_posix()
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    return any(marker in text for marker in SKIP_PATH_PARTS)


def iter_candidate_paths(scan_roots: Sequence[Path], patterns: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for root in scan_roots:
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = []
            for pattern in patterns:
                paths.extend(root.rglob(pattern))
        for path in paths:
            resolved = path.resolve()
            if resolved in seen or should_skip(path):
                continue
            seen.add(resolved)
            out.append(path)
    return sorted(out, key=lambda item: item.as_posix())


def inspect_candidate(path: Path, *, require_paths: bool) -> dict[str, Any]:
    try:
        payload = load_json(path)
        errors = validate_result(payload, "axis-c3b-mem16", require_paths=require_paths)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors = [f"unreadable or invalid JSON: {exc}"]
        payload = {}
    return {
        "path": str(path),
        "status": "pass" if not errors else "fail",
        "would_clear": not errors,
        "errors": errors,
        "runtime_state": payload.get("runtime_state"),
        "out_raw": payload.get("out_raw"),
        "variant": payload.get("variant"),
        "weights_mode": payload.get("weights_mode"),
    }


def build_discovery(root: Path, scan_roots: Sequence[Path], *, require_paths: bool) -> dict[str, Any]:
    root = root.resolve()
    candidates = [
        inspect_candidate(path, require_paths=require_paths)
        for path in iter_candidate_paths(scan_roots, DEFAULT_PATTERNS)
    ]
    pass_candidates = [item for item in candidates if item["status"] == "pass"]
    recommended = pass_candidates[0] if pass_candidates else None
    return {
        "status": "found" if recommended else "missing",
        "root": str(root),
        "scan_roots": [str(path) for path in scan_roots],
        "candidate_count": len(candidates),
        "pass_count": len(pass_candidates),
        "recommended_candidate": recommended,
        "candidates": candidates,
        "dry_run_import_command": (
            f"python3 tools/run_third_goal_final_signoff.py --root {root} "
            f"--import-c3b-smoke-json {recommended['path']} --dry-run-import-c3b-smoke --allow-blocked"
            if recommended
            else ""
        ),
        "import_command": (
            f"python3 tools/run_third_goal_final_signoff.py --root {root} "
            f"--import-c3b-smoke-json {recommended['path']} --allow-blocked"
            if recommended
            else ""
        ),
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(discovery: dict[str, Any]) -> str:
    lines = [
        "# HGTXR C3b Smoke Candidate Discovery",
        "",
        f"- status: `{discovery['status']}`",
        f"- root: `{discovery['root']}`",
        f"- candidates: `{discovery['candidate_count']}`",
        f"- pass: `{discovery['pass_count']}`",
        "",
        "## Recommended Candidate",
        "",
    ]
    recommended = discovery.get("recommended_candidate")
    if recommended:
        lines.append(f"- `{recommended['path']}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Dry-run Import Command", "", "```sh"])
    lines.append(discovery["dry_run_import_command"] or "# No valid C3b smoke candidate found.")
    lines.extend(["```", "", "## Import Command", "", "```sh"])
    lines.append(discovery["import_command"] or "# No valid C3b smoke candidate found.")
    lines.extend(["```", "", "## Candidates", "", "| Status | Path | Errors |", "|---|---|---|"])
    for candidate in discovery["candidates"]:
        errors = "; ".join(str(error) for error in candidate["errors"]).replace("|", "\\|")
        lines.append(f"| `{candidate['status']}` | `{candidate['path']}` | {errors} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Does not execute commands.",
            "- Does not create board smoke results.",
            "- Does not create XR-VITs replacement policy.",
            "- Does not write canonical unblock inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover C3b PYNQ smoke result candidates.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--scan-root", type=Path, action="append", default=None)
    parser.add_argument("--no-require-paths", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    scan_roots = [path.resolve() for path in args.scan_root] if args.scan_root else default_scan_roots(root)
    discovery = build_discovery(root, scan_roots, require_paths=not args.no_require_paths)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(discovery))
    print(
        f"[c3b-smoke-candidate-discovery] status={discovery['status']} "
        f"pass={discovery['pass_count']} candidates={discovery['candidate_count']}"
    )
    return 0 if discovery["status"] == "found" else 1


if __name__ == "__main__":
    raise SystemExit(main())
