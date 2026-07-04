#!/usr/bin/env python3
"""Preview whether supplied evidence would clear the final unblock gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy
from validate_pynq_smoke_result import load_json, validate_result


DATE_TAG = "2026_06_10"


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_c3b_candidate(path: Path | None, *, require_paths: bool) -> dict[str, Any]:
    if path is None:
        return {
            "status": "missing",
            "path": "",
            "would_clear": False,
            "errors": ["--c3b-smoke-json is required to preview C3b blocker clearance"],
        }
    try:
        payload = load_json(path)
        errors = validate_result(payload, "axis-c3b-mem16", require_paths=require_paths)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors = [f"candidate C3b smoke JSON invalid: {exc}"]
    return {
        "status": "pass" if not errors else "fail",
        "path": str(path),
        "would_clear": not errors,
        "errors": errors,
    }


def check_xr_vits_candidate(
    *,
    root: Path,
    mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
) -> dict[str, Any]:
    requested_path = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    if mode == "exact":
        errors = [] if requested_path.is_dir() else [f"requested XR-VITs path missing: {requested_path}"]
        return {
            "status": "pass" if not errors else "fail",
            "mode": mode,
            "would_clear": not errors,
            "requested_path": str(requested_path),
            "replacement_path": "",
            "errors": errors,
        }

    selected_replacement = (replacement_path or create_xr_vits_replacement_policy.default_replacement_path(root)).resolve()
    candidate_audit_rel = create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
    candidate_audit_path = (root / candidate_audit_rel).resolve()
    errors, _audit = create_xr_vits_replacement_policy.validate_inputs(
        approve=True,
        dry_run=True,
        requested_path=requested_path,
        replacement_path=selected_replacement,
        approved_by=approved_by,
        approved_at=create_xr_vits_replacement_policy.default_approved_at(),
        reason=reason,
        reason_code=create_xr_vits_replacement_policy.DEFAULT_REASON_CODE,
        candidate_audit_path=candidate_audit_path,
        candidate_audit_rel=candidate_audit_rel,
    )
    return {
        "status": "pass" if not errors else "fail",
        "mode": mode,
        "would_clear": not errors,
        "requested_path": str(requested_path),
        "replacement_path": str(selected_replacement),
        "errors": errors,
    }


def build_audit(
    *,
    root: Path,
    c3b_smoke_json: Path | None,
    c3b_require_paths: bool,
    xr_vits_mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
) -> dict[str, Any]:
    c3b = check_c3b_candidate(c3b_smoke_json, require_paths=c3b_require_paths)
    xr_vits = check_xr_vits_candidate(
        root=root,
        mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
    )
    would_clear_all = bool(c3b["would_clear"] and xr_vits["would_clear"])
    return {
        "status": "would-clear" if would_clear_all else "blocked",
        "root": str(root),
        "would_clear_all": would_clear_all,
        "c3b_smoke": c3b,
        "xr_vits": xr_vits,
        "remaining_blockers": [
            name
            for name, check in [
                ("C3b AXIS/DMA physical smoke result", c3b),
                ("requested XR-VITs sibling", xr_vits),
            ]
            if not check["would_clear"]
        ],
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Unblock Candidate Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- would_clear_all: `{audit['would_clear_all']}`",
        "",
        "## C3b Smoke Candidate",
        "",
        f"- status: `{audit['c3b_smoke']['status']}`",
        f"- path: `{audit['c3b_smoke']['path']}`",
        f"- would_clear: `{audit['c3b_smoke']['would_clear']}`",
        "",
        "## XR-VITs Candidate",
        "",
        f"- status: `{audit['xr_vits']['status']}`",
        f"- mode: `{audit['xr_vits']['mode']}`",
        f"- requested_path: `{audit['xr_vits']['requested_path']}`",
        f"- replacement_path: `{audit['xr_vits']['replacement_path']}`",
        f"- would_clear: `{audit['xr_vits']['would_clear']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    if audit["remaining_blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in audit["remaining_blockers"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Safety", ""])
    lines.append("- Does not create board smoke results.")
    lines.append("- Does not create XR-VITs replacement policy.")
    lines.append("- Does not execute network commands.")
    lines.append("- Does not write canonical unblock inputs.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview final unblock candidates without side effects.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--c3b-smoke-json", type=Path, default=None)
    parser.add_argument("--c3b-no-require-paths", action="store_true")
    parser.add_argument("--xr-vits-mode", choices=["exact", "replacement"], default="replacement")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--replacement-path", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(
        root=args.root.resolve(),
        c3b_smoke_json=args.c3b_smoke_json,
        c3b_require_paths=not args.c3b_no_require_paths,
        xr_vits_mode=args.xr_vits_mode,
        approved_by=args.approved_by,
        reason=args.reason,
        replacement_path=args.replacement_path,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(
        f"[final-unblock-candidate-audit] status={audit['status']} "
        f"remaining={len(audit['remaining_blockers'])}"
    )
    return 0 if audit["status"] == "would-clear" else 1


if __name__ == "__main__":
    raise SystemExit(main())
