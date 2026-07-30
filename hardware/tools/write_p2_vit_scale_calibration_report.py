#!/usr/bin/env python3
"""Write SW-first P2-ViT scale calibration report for Q4/Q8 E2E evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def int_at_least(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and value >= minimum


def build_report(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    signoff = hardware / "generated" / "signoff"
    req5 = load_json_object(signoff / f"req5_q4q8_swhw_match_audit_{DATE_TAG}.json")
    pot_audit = load_json_object(signoff / f"vref_p0_pot_scale_audit_{DATE_TAG}.json")
    pot_sweep = load_json_object(signoff / f"vref_p0_pot_scale_sweep_{DATE_TAG}.json")

    req5_precision = req5.get("precision", {}) if isinstance(req5.get("precision"), dict) else {}
    sweep_summary = pot_sweep.get("summary", {}) if isinstance(pot_sweep.get("summary"), dict) else {}
    sweep_specs = pot_sweep.get("specs", []) if isinstance(pot_sweep.get("specs"), list) else []
    sweep_safety = pot_sweep.get("safety", {}) if isinstance(pot_sweep.get("safety"), dict) else {}
    audit_safety = pot_audit.get("safety", {}) if isinstance(pot_audit.get("safety"), dict) else {}

    checks: list[dict[str, Any]] = []
    add_check(checks, "req5_q4q8_pass", req5.get("status") == "pass" and req5.get("fail_count") == 0, str(req5.get("status")))
    add_check(
        checks,
        "req5_precision_q4w8a",
        req5_precision.get("weight_bits") == 4 and req5_precision.get("activation_bits") == 8,
        str(req5_precision),
    )
    add_check(
        checks,
        "req5_csim_coverage",
        int_at_least(req5.get("pass_count"), 6) and len(req5.get("csim_logs", [])) >= 3,
        f"pass={req5.get('pass_count')} csim_logs={len(req5.get('csim_logs', []))}",
    )
    add_check(
        checks,
        "pot_scale_readiness_pass",
        pot_audit.get("status") == "pass" and pot_audit.get("fail_count") == 0,
        f"status={pot_audit.get('status')} fail={pot_audit.get('fail_count')}",
    )
    add_check(
        checks,
        "pot_scale_readiness_checks",
        int_at_least(pot_audit.get("check_count"), 14),
        str(pot_audit.get("check_count")),
    )
    add_check(
        checks,
        "scale_sweep_pass",
        pot_sweep.get("status") == "pass" and sweep_summary.get("total_fail_count") == 0,
        f"status={pot_sweep.get('status')} fail={sweep_summary.get('total_fail_count')}",
    )
    add_check(
        checks,
        "scale_sweep_candidate_coverage",
        int_at_least(pot_sweep.get("spec_count"), 3) and int_at_least(sweep_summary.get("total_candidate_count"), 45),
        f"specs={pot_sweep.get('spec_count')} candidates={sweep_summary.get('total_candidate_count')}",
    )
    add_check(
        checks,
        "scale_decision_current_for_c3b",
        sweep_summary.get("specs_recommending_current") == pot_sweep.get("spec_count")
        and "keep current PoT scales" in str(sweep_summary.get("next_action", "")),
        str(sweep_summary),
    )
    add_check(
        checks,
        "successor_only_promotion_policy",
        "successor experiments" in str(sweep_summary.get("next_action", ""))
        and "header" in str(pot_sweep.get("policy", {}).get("promotion_gate", "")).lower()
        and "csim" in str(pot_sweep.get("policy", {}).get("promotion_gate", "")).lower(),
        str(pot_sweep.get("policy", {})),
    )
    add_check(
        checks,
        "no_hardware_side_effects",
        all(value is False for value in [
            sweep_safety.get("executes_hls"),
            sweep_safety.get("executes_vivado"),
            sweep_safety.get("writes_hls_source"),
            sweep_safety.get("overwrites_c3b_artifacts"),
            audit_safety.get("executes_hls"),
            audit_safety.get("executes_vivado"),
            audit_safety.get("writes_hls_source"),
            audit_safety.get("overwrites_c3b_artifacts"),
        ]),
        f"sweep={sweep_safety} audit={audit_safety}",
    )

    failed = [check["name"] for check in checks if check["status"] != "pass"]
    per_spec = [
        {
            "spec": item.get("spec"),
            "recommended_candidate": item.get("recommended_candidate"),
            "recommendation": item.get("recommendation"),
            "candidate_count": item.get("candidate_count"),
            "pass_count": item.get("pass_count"),
            "fail_count": item.get("fail_count"),
            "exact_match_count": item.get("exact_match_count"),
            "baseline_expected_raw": item.get("baseline_expected_raw"),
        }
        for item in sweep_specs
        if isinstance(item, dict)
    ]
    return {
        "status": "pass" if not failed else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "policy": {
            "source": "P2-ViT",
            "scope": "SW-first Q4/Q8 scale calibration closure for current C3b baseline",
            "decision": "keep_current_pot_scales_for_c3b",
            "non_current_candidates": "successor_only_until_header_regen_csim_hls_and_board_evidence",
        },
        "inputs": {
            "req5_q4q8_swhw": str(signoff / f"req5_q4q8_swhw_match_audit_{DATE_TAG}.json"),
            "pot_scale_audit": str(signoff / f"vref_p0_pot_scale_audit_{DATE_TAG}.json"),
            "pot_scale_sweep": str(signoff / f"vref_p0_pot_scale_sweep_{DATE_TAG}.json"),
        },
        "precision": req5_precision,
        "sweep_summary": sweep_summary,
        "per_spec": per_spec,
        "checks": checks,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "failed_checks": failed,
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P2-ViT Scale Calibration Report",
        "",
        f"- status: `{report['status']}`",
        f"- decision: `{report['policy']['decision']}`",
        f"- checks: `{report['pass_count']}/{report['check_count']}`",
        f"- fail_count: `{report['fail_count']}`",
        f"- weight_bits: `{report['precision'].get('weight_bits')}`",
        f"- activation_bits: `{report['precision'].get('activation_bits')}`",
        f"- total_candidates: `{report['sweep_summary'].get('total_candidate_count')}`",
        f"- next_action: `{report['sweep_summary'].get('next_action')}`",
        "",
        "## Per Spec",
        "",
        "| Spec | Recommended | Candidates | Exact Matches |",
        "|---|---|---:|---:|",
    ]
    for item in report["per_spec"]:
        lines.append(
            f"| {Path(str(item.get('spec'))).name} | `{item.get('recommended_candidate')}` | "
            f"{item.get('pass_count')}/{item.get('candidate_count')} | {item.get('exact_match_count')} |"
        )
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in report["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in report["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write P2-ViT SW-first scale calibration report.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _, hardware = normalize_roots(args.root)
    json_out = args.json_out or hardware / "generated" / "signoff" / f"p2_vit_scale_calibration_report_{DATE_TAG}.json"
    markdown_out = args.markdown_out or hardware / "generated" / "signoff" / f"p2_vit_scale_calibration_report_{DATE_TAG}.md"
    report = build_report(args.root)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(report))
    print(f"[p2-vit-scale-calibration] status={report['status']} checks={report['pass_count']}/{report['check_count']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
