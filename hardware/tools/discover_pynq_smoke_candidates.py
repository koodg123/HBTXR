#!/usr/bin/env python3
"""Discover side-effect-free HGTXR PYNQ smoke result candidates by preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_smoke_result import VARIANT_PRESETS, load_json, validate_result


PRESET_PATTERNS: dict[str, list[str]] = {
    "axis-c3b-mem16": ["*c3b*smoke*.json", "*e2e_axis_dma_c3b_mem16*.json"],
    "axis-vref-p0-softmax-input-x2-dsp-mixed-stream": [
        "*vref*p0*softmax*input*x2*smoke*.json",
        "*e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream*.json",
    ],
    "axis-vref-p0-softmax-input-x2-qkv-uram": [
        "*vref*p0*softmax*input*x2*qkv*uram*smoke*.json",
        "*e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram*.json",
    ],
    "axis-a1": ["*a1*smoke*.json", "*e2e_axis_dma_a1*.json"],
    "axis-c1-par16": ["*c1*par16*smoke*.json", "*e2e_axis_dma_c1_par16*.json"],
    "m_axi-a2": ["*m_axi*smoke*.json", "*e2e_m_axi*.json"],
}

SKIP_DIR_NAMES = {".git", "__pycache__", ".pytest_cache"}
SKIP_NAME_PARTS = {
    "BUNDLE_MANIFEST",
    "_bundle_validation",
    "_candidate_discovery",
    "_gate_audit_",
    "_import_",
    "_readiness_",
    "_remote_run_",
    "_session",
    "_validation",
    "final_",
    "third_goal_",
}
SKIP_PATH_PARTS = {
    "generated/signoff",
    "docs/resources/final_",
    "docs/resources/third_goal_",
    "docs/resources/zcu104_",
    "docs/resources/c3b_axis_dma_smoke_bundle_",
    "docs/resources/c3b_board_smoke_readiness_",
    "docs/resources/c3b_smoke_candidate_discovery_",
    "docs/resources/c3b_smoke_result_contract_",
    "docs/resources/c3b_smoke_transfer_manifest_",
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
    name = path.name
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if any(marker in name for marker in SKIP_NAME_PARTS):
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


def inspect_candidate(path: Path, *, preset: str, require_paths: bool) -> dict[str, Any]:
    try:
        payload = load_json(path)
        errors = validate_result(payload, preset, require_paths=require_paths)
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


def runner_import_command(root: Path, preset: str, source: str, *, dry_run: bool) -> str:
    if preset == "axis-c3b-mem16":
        args = ["--import-c3b-smoke-json", source]
        if dry_run:
            args.append("--dry-run-import-c3b-smoke")
    elif preset == "axis-vref-p0-softmax-input-x2-dsp-mixed-stream":
        args = ["--import-vref-successor-smoke-json", source]
        if dry_run:
            args.append("--dry-run-import-vref-successor-smoke")
    elif preset == "axis-vref-p0-softmax-input-x2-qkv-uram":
        args = ["--import-qkv-uram-smoke-json", source]
        if dry_run:
            args.append("--dry-run-import-qkv-uram-smoke")
    else:
        base = (
            f"python3 tools/import_pynq_smoke_result.py {source} --preset {preset} "
            f"--root {root}"
        )
        return f"{base} --dry-run" if dry_run else base
    args_text = " ".join(args)
    return f"python3 tools/run_third_goal_final_signoff.py --root {root} {args_text} --allow-blocked"


def build_discovery(
    root: Path,
    scan_roots: Sequence[Path],
    *,
    preset: str,
    require_paths: bool,
    patterns: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if preset not in VARIANT_PRESETS:
        raise ValueError(f"unsupported preset: {preset}")
    selected_patterns = list(patterns) if patterns else PRESET_PATTERNS[preset]
    candidates = [
        inspect_candidate(path, preset=preset, require_paths=require_paths)
        for path in iter_candidate_paths(scan_roots, selected_patterns)
    ]
    pass_candidates = [item for item in candidates if item["status"] == "pass"]
    recommended = pass_candidates[0] if pass_candidates else None
    recommended_path = recommended["path"] if recommended else ""
    return {
        "status": "found" if recommended else "missing",
        "root": str(root),
        "preset": preset,
        "patterns": selected_patterns,
        "scan_roots": [str(path) for path in scan_roots],
        "candidate_count": len(candidates),
        "pass_count": len(pass_candidates),
        "recommended_candidate": recommended,
        "candidates": candidates,
        "dry_run_import_command": runner_import_command(root, preset, recommended_path, dry_run=True) if recommended else "",
        "import_command": runner_import_command(root, preset, recommended_path, dry_run=False) if recommended else "",
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(discovery: dict[str, Any]) -> str:
    lines = [
        "# HGTXR PYNQ Smoke Candidate Discovery",
        "",
        f"- status: `{discovery['status']}`",
        f"- preset: `{discovery['preset']}`",
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
    lines.append(discovery["dry_run_import_command"] or "# No valid smoke candidate found.")
    lines.extend(["```", "", "## Import Command", "", "```sh"])
    lines.append(discovery["import_command"] or "# No valid smoke candidate found.")
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
    parser = argparse.ArgumentParser(description="Discover HGTXR PYNQ smoke result candidates.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--preset", choices=sorted(VARIANT_PRESETS), default="axis-c3b-mem16")
    parser.add_argument("--scan-root", type=Path, action="append", default=None)
    parser.add_argument("--pattern", action="append", default=None)
    parser.add_argument("--no-require-paths", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    scan_roots = [path.resolve() for path in args.scan_root] if args.scan_root else default_scan_roots(root)
    discovery = build_discovery(
        root,
        scan_roots,
        preset=args.preset,
        require_paths=not args.no_require_paths,
        patterns=args.pattern,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(discovery))
    print(
        f"[pynq-smoke-candidate-discovery] preset={discovery['preset']} "
        f"status={discovery['status']} pass={discovery['pass_count']} "
        f"candidates={discovery['candidate_count']}"
    )
    return 0 if discovery["status"] == "found" else 1


if __name__ == "__main__":
    raise SystemExit(main())
