#!/usr/bin/env python3
"""Check whether HGTXR final blockers can be closed now or with supplied candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import audit_final_unblock_candidates
import create_xr_vits_replacement_policy
import discover_pynq_smoke_candidates
from validate_pynq_smoke_result import load_json, validate_result


DATE_TAG = "2026_06_10"


def default_hgtxr_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_c3b_path(root: Path) -> Path:
    return root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"


def active_policy_path(root: Path) -> Path:
    return root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL


def check_c3b_smoke(path: Path, *, require_paths: bool) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "would_clear": False,
            "errors": [f"C3b smoke JSON missing: {path}"],
        }
    try:
        errors = validate_result(load_json(path), "axis-c3b-mem16", require_paths=require_paths)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors = [f"C3b smoke JSON invalid: {exc}"]
    return {
        "status": "pass" if not errors else "fail",
        "path": str(path),
        "would_clear": not errors,
        "errors": errors,
    }


def check_active_policy(root: Path) -> dict[str, Any]:
    path = active_policy_path(root)
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "would_clear": False,
            "errors": [f"active replacement policy missing: {path}"],
        }
    try:
        policy = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "fail",
            "path": str(path),
            "would_clear": False,
            "errors": [f"active replacement policy invalid: {exc}"],
        }
    requested = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    replacement = Path(str(policy.get("replacement_path", ""))).resolve()
    errors: list[str] = []
    if policy.get("approved_replacement") is not True:
        errors.append("approved_replacement is not true")
    if str(requested) != str(policy.get("requested_path", "")):
        errors.append(f"requested_path mismatch: {policy.get('requested_path')}")
    if not replacement.is_dir():
        errors.append(f"replacement_path missing or not directory: {replacement}")
    if not str(policy.get("approved_by", "")).strip():
        errors.append("approved_by missing")
    if not str(policy.get("reason", "")).strip():
        errors.append("reason missing")
    integrity = create_xr_vits_replacement_policy.validate_policy_integrity(root, policy)
    if integrity["status"] != "pass":
        errors.extend(integrity.get("errors", []))
        errors.extend(integrity.get("warnings", []))
    return {
        "status": "pass" if not errors else "fail",
        "path": str(path),
        "would_clear": not errors,
        "errors": errors,
        "integrity": integrity,
    }


def check_current_xr_vits(root: Path) -> dict[str, Any]:
    requested = create_xr_vits_replacement_policy.default_requested_path(root).resolve()
    exact = {
        "status": "pass" if requested.is_dir() else "missing",
        "path": str(requested),
        "would_clear": requested.is_dir(),
        "errors": [] if requested.is_dir() else [f"requested XR-VITs path missing: {requested}"],
    }
    policy = check_active_policy(root)
    would_clear = bool(exact["would_clear"] or policy["would_clear"])
    return {
        "status": "pass" if would_clear else "missing",
        "would_clear": would_clear,
        "mode": "exact" if exact["would_clear"] else "replacement-policy" if policy["would_clear"] else "missing",
        "exact": exact,
        "replacement_policy": policy,
        "errors": [] if would_clear else exact["errors"] + policy["errors"],
    }


def summarize_pynq_discovery(discovery: dict[str, Any]) -> dict[str, Any]:
    recommended = discovery.get("recommended_candidate")
    return {
        "status": discovery.get("status"),
        "preset": discovery.get("preset"),
        "candidate_count": discovery.get("candidate_count"),
        "pass_count": discovery.get("pass_count"),
        "recommended_candidate_path": recommended.get("path", "") if isinstance(recommended, dict) else "",
        "dry_run_import_command": discovery.get("dry_run_import_command", ""),
        "import_command": discovery.get("import_command", ""),
        "safety": discovery.get("safety", {}),
    }


def build_pynq_discovery_summary(root: Path) -> dict[str, Any]:
    scan_roots = discover_pynq_smoke_candidates.default_scan_roots(root)
    presets = {
        "c3b": "axis-c3b-mem16",
        "vref_p0": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
        "qkv_uram": "axis-vref-p0-softmax-input-x2-qkv-uram",
    }
    return {
        label: summarize_pynq_discovery(
            discover_pynq_smoke_candidates.build_discovery(
                root,
                scan_roots,
                preset=preset,
                require_paths=True,
            )
        )
        for label, preset in presets.items()
    }


def build_final_command(
    root: Path,
    *,
    candidate_c3b: Path | None,
    xr_vits_mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
    current_ready: bool,
    candidate_ready: bool,
    dry_run: bool = False,
) -> str:
    base = f"python3 tools/run_third_goal_final_signoff.py --root {root}"
    if current_ready:
        return base
    if not candidate_ready or candidate_c3b is None:
        return ""
    command = f"{base} --import-c3b-smoke-json {candidate_c3b}"
    if dry_run:
        command += " --dry-run-import-c3b-smoke"
    if xr_vits_mode == "replacement":
        command += (
            " --approve-xr-vits-replacement"
            f" --xr-vits-approved-by {approved_by or '<approved-by>'}"
            f' --xr-vits-replacement-reason "{reason or "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"}"'
        )
        if replacement_path is not None:
            command += f" --xr-vits-replacement-path {replacement_path}"
        if dry_run:
            command += " --dry-run-xr-vits-replacement"
    if dry_run:
        command += " --allow-blocked"
    return command


def build_c3b_import_template(root: Path, *, dry_run: bool) -> str:
    command = (
        f"python3 tools/run_third_goal_final_signoff.py --root {root} "
        "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json"
    )
    if dry_run:
        command += " --dry-run-import-c3b-smoke --allow-blocked"
    return command


def build_xr_vits_policy_template(root: Path, *, dry_run: bool) -> str:
    command = (
        f"python3 tools/run_third_goal_final_signoff.py --root {root} "
        "--approve-xr-vits-replacement "
        "--xr-vits-approved-by USER "
        '--xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'
    )
    if dry_run:
        command += " --dry-run-xr-vits-replacement --allow-blocked"
    return command


def build_operator_unblock_plan(
    *,
    root: Path,
    current_c3b: dict[str, Any],
    current_xr: dict[str, Any],
    candidate_c3b: dict[str, Any],
    candidate_xr: dict[str, Any],
    current_ready: bool,
    candidate_ready: bool,
    final_runner_command: str,
    dry_run_final_runner_command: str,
) -> dict[str, Any]:
    c3b_ready = bool(current_c3b.get("would_clear") or candidate_c3b.get("would_clear"))
    xr_ready = bool(current_xr.get("would_clear") or candidate_xr.get("would_clear"))
    exact = current_xr.get("exact", {}) if isinstance(current_xr.get("exact"), dict) else {}
    policy = (
        current_xr.get("replacement_policy", {})
        if isinstance(current_xr.get("replacement_policy"), dict)
        else {}
    )
    status = (
        "ready-to-run-final"
        if current_ready
        else "ready-with-candidates"
        if candidate_ready
        else "needs-external-inputs"
    )
    return {
        "status": status,
        "required_inputs": [
            {
                "blocker": "C3b AXIS/DMA physical smoke result",
                "kind": "board-produced-json",
                "ready": c3b_ready,
                "current_status": str(current_c3b.get("status", "unknown")),
                "candidate_status": str(candidate_c3b.get("status", "unknown")),
                "canonical_path": str(current_c3b.get("path", canonical_c3b_path(root))),
                "candidate_path": str(candidate_c3b.get("path", "")),
                "acceptance": [
                    "preset axis-c3b-mem16 validates",
                    "runtime_state is 2",
                    "out_raw matches expected C3b vector",
                    "bitfile/hwhfile basenames match hgtxr_e2e_axis_dma_c3b_mem16",
                ],
            },
            {
                "blocker": "requested XR-VITs sibling",
                "kind": "exact-source-or-approved-policy",
                "ready": xr_ready,
                "current_status": str(current_xr.get("status", "unknown")),
                "candidate_status": str(candidate_xr.get("status", "unknown")),
                "exact_path": str(exact.get("path", "")),
                "policy_path": str(policy.get("path", active_policy_path(root))),
                "replacement_path": str(candidate_xr.get("replacement_path", "")),
                "acceptance": [
                    "exact XR-VITs checkout exists, or",
                    "replacement policy is generated by create_xr_vits_replacement_policy.py",
                    "policy has candidate_audit_fingerprint, recommendation snapshot, and policy_fingerprint",
                    "approved_by is a real non-placeholder identifier",
                ],
            },
        ],
        "dry_run_sequence": [
            {
                "step": "c3b-import-dry-run",
                "ready_to_execute": bool(candidate_c3b.get("would_clear")),
                "command": dry_run_final_runner_command if candidate_ready else build_c3b_import_template(root, dry_run=True),
                "side_effects": False,
            },
            {
                "step": "xr-vits-policy-preview",
                "ready_to_execute": bool(candidate_xr.get("would_clear")),
                "command": dry_run_final_runner_command if candidate_ready else build_xr_vits_policy_template(root, dry_run=True),
                "side_effects": False,
            },
            {
                "step": "combined-final-runner-dry-run",
                "ready_to_execute": bool(dry_run_final_runner_command),
                "command": dry_run_final_runner_command,
                "side_effects": False,
            },
        ],
        "active_sequence": [
            {
                "step": "combined-final-runner-active",
                "ready_to_execute": bool(final_runner_command),
                "command": final_runner_command,
                "side_effects": not current_ready,
            }
        ],
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
            "templates_only_when_inputs_missing": not (current_ready or candidate_ready),
        },
    }


def build_readiness(
    *,
    root: Path,
    c3b_smoke_json: Path | None,
    c3b_require_paths: bool,
    xr_vits_mode: str,
    approved_by: str,
    reason: str,
    replacement_path: Path | None,
) -> dict[str, Any]:
    root = root.resolve()
    current_c3b = check_c3b_smoke(canonical_c3b_path(root), require_paths=c3b_require_paths)
    current_xr = check_current_xr_vits(root)
    pynq_discovery = build_pynq_discovery_summary(root)
    current_ready = bool(current_c3b["would_clear"] and current_xr["would_clear"])

    candidate_c3b = (
        audit_final_unblock_candidates.check_c3b_candidate(c3b_smoke_json, require_paths=c3b_require_paths)
        if c3b_smoke_json is not None
        else {"status": "not-supplied", "path": "", "would_clear": False, "errors": []}
    )
    candidate_xr = audit_final_unblock_candidates.check_xr_vits_candidate(
        root=root,
        mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
    )
    candidate_ready = bool(candidate_c3b["would_clear"] and candidate_xr["would_clear"])

    if current_ready:
        status = "ready-to-run-final"
    elif candidate_ready:
        status = "would-clear-with-candidates"
    else:
        status = "blocked"

    missing = []
    if not current_c3b["would_clear"]:
        missing.append("C3b AXIS/DMA physical smoke result")
    if not current_xr["would_clear"]:
        missing.append("requested XR-VITs sibling")

    final_runner_command = build_final_command(
        root,
        candidate_c3b=c3b_smoke_json,
        xr_vits_mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
        current_ready=current_ready,
        candidate_ready=candidate_ready,
    )
    dry_run_final_runner_command = build_final_command(
        root,
        candidate_c3b=c3b_smoke_json,
        xr_vits_mode=xr_vits_mode,
        approved_by=approved_by,
        reason=reason,
        replacement_path=replacement_path,
        current_ready=current_ready,
        candidate_ready=candidate_ready,
        dry_run=True,
    )

    return {
        "status": status,
        "root": str(root),
        "current_ready": current_ready,
        "candidate_ready": candidate_ready,
        "current": {
            "c3b_smoke": current_c3b,
            "xr_vits": current_xr,
        },
        "candidates": {
            "c3b_smoke": candidate_c3b,
            "xr_vits": candidate_xr,
        },
        "pynq_discovery": pynq_discovery,
        "missing_current_blockers": missing,
        "operator_unblock_plan": build_operator_unblock_plan(
            root=root,
            current_c3b=current_c3b,
            current_xr=current_xr,
            candidate_c3b=candidate_c3b,
            candidate_xr=candidate_xr,
            current_ready=current_ready,
            candidate_ready=candidate_ready,
            final_runner_command=final_runner_command,
            dry_run_final_runner_command=dry_run_final_runner_command,
        ),
        "final_runner_command": final_runner_command,
        "dry_run_final_runner_command": dry_run_final_runner_command,
        "safety": {
            "executes_commands": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
            "discovers_pynq_candidates_only": True,
        },
    }


def render_markdown(readiness: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Blocker Closure Readiness",
        "",
        f"- status: `{readiness['status']}`",
        f"- root: `{readiness['root']}`",
        f"- current_ready: `{readiness['current_ready']}`",
        f"- candidate_ready: `{readiness['candidate_ready']}`",
        "",
        "## Current Evidence",
        "",
        f"- C3b smoke: `{readiness['current']['c3b_smoke']['status']}`",
        f"- XR-VITs: `{readiness['current']['xr_vits']['status']}`",
        "",
        "## Candidate Evidence",
        "",
        f"- C3b smoke: `{readiness['candidates']['c3b_smoke']['status']}`",
        f"- XR-VITs: `{readiness['candidates']['xr_vits']['status']}`",
        "",
        "## PYNQ Discovery",
        "",
    ]
    for label, discovery in readiness["pynq_discovery"].items():
        lines.append(
            f"- {label}: status `{discovery['status']}`, pass `{discovery['pass_count']}`, "
            f"candidates `{discovery['candidate_count']}`"
        )
    lines.extend(
        [
            "",
            "## Missing Current Blockers",
            "",
        ]
    )
    if readiness["missing_current_blockers"]:
        lines.extend(f"- `{item}`" for item in readiness["missing_current_blockers"])
    else:
        lines.append("- None.")
    plan = readiness.get("operator_unblock_plan", {})
    lines.extend(["", "## Operator Unblock Plan", ""])
    lines.append(f"- status: `{plan.get('status', 'missing')}`")
    lines.extend(["", "### Required Inputs", ""])
    for item in plan.get("required_inputs", []) if isinstance(plan, dict) else []:
        lines.append(
            f"- `{item['blocker']}`: ready `{item['ready']}`, kind `{item['kind']}`"
        )
    lines.extend(["", "### Dry-Run Sequence", "", "```sh"])
    for step in plan.get("dry_run_sequence", []) if isinstance(plan, dict) else []:
        lines.append(f"# {step['step']} ready={step['ready_to_execute']}")
        lines.append(step["command"] or "# Not ready.")
    lines.extend(["```", "", "### Active Sequence", "", "```sh"])
    for step in plan.get("active_sequence", []) if isinstance(plan, dict) else []:
        lines.append(f"# {step['step']} ready={step['ready_to_execute']}")
        lines.append(step["command"] or "# Not ready.")
    lines.extend(["```"])
    lines.extend(["", "## Final Runner Command", "", "```sh"])
    lines.append(readiness["final_runner_command"] or "# Not ready.")
    lines.extend(["```", "", "## Dry-Run Final Runner Command", "", "```sh"])
    lines.append(readiness["dry_run_final_runner_command"] or "# Not ready.")
    lines.extend(
        [
            "```",
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
    parser = argparse.ArgumentParser(description="Check final blocker closure readiness without side effects.")
    parser.add_argument("--root", type=Path, default=default_hgtxr_root())
    parser.add_argument("--c3b-smoke-json", type=Path, default=None)
    parser.add_argument("--c3b-no-require-paths", action="store_true")
    parser.add_argument("--xr-vits-mode", choices=["exact", "replacement"], default="exact")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--replacement-path", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    readiness = build_readiness(
        root=args.root,
        c3b_smoke_json=args.c3b_smoke_json,
        c3b_require_paths=not args.c3b_no_require_paths,
        xr_vits_mode=args.xr_vits_mode,
        approved_by=args.approved_by,
        reason=args.reason,
        replacement_path=args.replacement_path,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(readiness))
    print(
        f"[final-blocker-closure-readiness] status={readiness['status']} "
        f"current_ready={readiness['current_ready']} candidate_ready={readiness['candidate_ready']}"
    )
    return 0 if readiness["status"] in {"ready-to-run-final", "would-clear-with-candidates"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
