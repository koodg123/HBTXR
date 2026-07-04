#!/usr/bin/env python3
"""Audit plan, progress, HANDOVER, and signoff source documents for req0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
MARKERS = (
    "HGTXR",
    "ZCU104",
    "C3b",
    "VREF",
    "E2E",
    "third",
    "3차",
    "XR-VIT",
    "HANDOVER",
    "HG-PIPE",
    "ViT",
    "accelerator",
)


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped
    return ""


def source_entry(base: Path, rel: str, group: str, required: bool) -> dict[str, Any]:
    path = base / rel
    entry: dict[str, Any] = {
        "group": group,
        "relative_path": rel,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
    }
    if not path.exists():
        entry["status"] = "missing" if required else "absent-optional"
        return entry
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    markers = [marker for marker in MARKERS if marker.lower() in text.lower()]
    entry.update(
        {
            "status": "pass",
            "bytes": len(data),
            "sha256": sha256_file(path),
            "first_heading": first_heading(text),
            "matched_markers": markers,
            "marker_count": len(markers),
        }
    )
    return entry


def discover_optional(base: Path, roots: list[str], known: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for root_rel in roots:
        root = base / root_rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".png"}:
                continue
            rel = path.relative_to(base).as_posix()
            if rel in known:
                continue
            name = path.name.lower()
            if any(
                token in name
                for token in [
                    "handover",
                    "progress",
                    "plan",
                    "spec",
                    "validation",
                    "execution",
                    "audit",
                    "signoff",
                    "discovery",
                    "requirements",
                    "closeout",
                    "unblock",
                    "choice",
                    "todo",
                    "adr",
                    "log",
                ]
            ):
                group = "optional-discovered"
                entries.append(source_entry(base, rel, group, required=False))
    return entries


def required_sources() -> list[tuple[str, str, str]]:
    return [
        ("hardware-plan", "hardware", "docs/Master-Plan.md"),
        ("hardware-plan", "hardware", "docs/Sub-Plan.md"),
        ("hardware-plan", "hardware", "docs/Spec.md"),
        ("hardware-plan", "hardware", "docs/Execution.md"),
        ("hardware-plan", "hardware", "docs/CHOICE.md"),
        ("hardware-validation", "hardware", "docs/Validation.md"),
        ("hardware-progress", "hardware", "docs/track/PROGRESS.md"),
        ("hardware-progress", "hardware", "docs/track/THIRD_GOAL_REQUIREMENTS_2026_06_16.md"),
        ("hardware-progress", "hardware", "docs/track/log.md"),
        ("handover", "hardware", "docs/track/HANDOVER.md"),
        ("handover", "hgtxr", "docs/track/HANDOVER.md"),
        ("handover", "hgtxr", "docs/track/HANDOVER-2026-06-10-E2E.md"),
        ("parent-plan", "hgtxr", "docs/Master-Plan.md"),
        ("parent-plan", "hgtxr", "docs/Sub-Plan.md"),
        ("parent-plan", "hgtxr", "docs/Spec.md"),
        ("parent-plan", "hgtxr", "docs/Execution.md"),
        ("parent-validation", "hgtxr", "docs/Validation.md"),
        ("parent-progress", "hgtxr", "docs/track/PROGRESS.md"),
        ("parent-progress", "hgtxr", "docs/track/CHOICE.md"),
        ("reference", "hardware", "docs/SRC_CASE_MODULE_GUIDE.md"),
        ("reference", "hardware", "docs/legacy/legacy_experiment_analysis_2026_06_12.md"),
        ("reference", "hardware", "analysis/vit-accel/README.md"),
        ("reference", "hardware", "analysis/vit-accel/experiment_extensions_2026_06_15.md"),
        ("reference", "hardware", "analysis/vit-accel/third_goal_integration_2026_06_15.md"),
        ("signoff", "hardware", "generated/signoff/third_goal_current_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/third_goal_current_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_validation_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_validation_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_vref_required_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_closeout_packet_vref_required_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/final_blocker_closure_readiness_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_blocker_closure_readiness_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/final_unblock_intake_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_intake_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/final_unblock_commands_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_commands_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/final_unblock_candidate_audit_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_unblock_candidate_audit_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/final_operator_handoff_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_operator_handoff_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/final_operator_handoff_validation_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/final_operator_handoff_validation_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/xr_vits_unblock_packet_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/xr_vits_unblock_packet_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/xr_vits_replacement_policy_preview_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/xr_vits_replacement_policy_preview_2026_06_10.json"),
        (
            "signoff",
            "hardware",
            "generated/signoff/final_unblock_closeout_packet_vref_required_validation_2026_06_16.md",
        ),
        (
            "signoff",
            "hardware",
            "generated/signoff/final_unblock_closeout_packet_vref_required_validation_2026_06_16.json",
        ),
        ("signoff", "hardware", "generated/signoff/vref_p0_pot_scale_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/vref_p0_pot_scale_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/hgpipe_operator_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/hgpipe_operator_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/xr_vits_reference_resolution_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/xr_vits_reference_resolution_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/xr_vits_gate_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/xr_vits_gate_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.md"),
        ("signoff", "hardware", "generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.json"),
        ("signoff", "hardware", "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/p2_vit_scale_calibration_report_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/p2_vit_scale_calibration_report_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/req6_parameterization_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/req6_parameterization_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/req1_environment_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/req1_environment_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/req9_deit_image_reference_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/req9_deit_image_reference_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.md"),
        ("signoff", "hardware", "generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.json"),
        ("signoff", "hardware", "generated/signoff/c3b_protection_checklist_2026_06_16.md"),
    ]


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    entries: list[dict[str, Any]] = []
    known_by_base: dict[str, set[str]] = {"hgtxr": set(), "hardware": set()}
    bases = {"hgtxr": hgtxr, "hardware": hardware}
    for group, base_name, rel in required_sources():
        known_by_base[base_name].add(rel)
        entries.append(source_entry(bases[base_name], rel, group, required=True))

    entries.extend(discover_optional(hardware, ["docs", "generated/signoff"], known_by_base["hardware"]))
    entries.extend(discover_optional(hgtxr, ["docs"], known_by_base["hgtxr"]))

    missing_required = [
        entry["path"]
        for entry in entries
        if entry.get("required") and entry.get("status") != "pass"
    ]
    markerless_required = [
        entry["path"]
        for entry in entries
        if entry.get("required") and entry.get("status") == "pass" and entry.get("marker_count", 0) == 0
    ]
    group_summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        group = str(entry["group"])
        summary = group_summary.setdefault(group, {"total": 0, "pass": 0, "missing": 0, "optional": 0})
        summary["total"] += 1
        if not entry.get("required"):
            summary["optional"] += 1
        if entry.get("status") == "pass":
            summary["pass"] += 1
        elif entry.get("status") == "missing":
            summary["missing"] += 1

    status = "pass" if not missing_required else "partial"
    return {
        "status": status,
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "required_count": len(required_sources()),
        "source_count": len(entries),
        "missing_required": missing_required,
        "markerless_required": markerless_required,
        "group_summary": group_summary,
        "sources": entries,
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
        "# Third Goal Source Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- date_tag: `{audit['date_tag']}`",
        f"- required_count: `{audit['required_count']}`",
        f"- source_count: `{audit['source_count']}`",
        "",
        "## Group Summary",
        "",
        "| Group | Total | Pass | Missing | Optional |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, summary in sorted(audit["group_summary"].items()):
        lines.append(
            f"| {group} | {summary['total']} | {summary['pass']} | {summary['missing']} | {summary['optional']} |"
        )
    lines.extend(["", "## Required Sources", "", "| Group | Status | Path | Heading | Markers |"])
    lines.append("|---|---|---|---|---|")
    for entry in audit["sources"]:
        if not entry.get("required"):
            continue
        heading = str(entry.get("first_heading", "")).replace("|", "\\|")
        markers = ", ".join(entry.get("matched_markers", [])) if entry.get("matched_markers") else "-"
        lines.append(
            f"| {entry['group']} | `{entry['status']}` | `{entry['path']}` | {heading or '-'} | {markers} |"
        )
    if audit["missing_required"]:
        lines.extend(["", "## Missing Required"])
        for path in audit["missing_required"]:
            lines.append(f"- `{path}`")
    if audit["markerless_required"]:
        lines.extend(["", "## Markerless Required"])
        for path in audit["markerless_required"]:
            lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/third_goal_source_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/third_goal_source_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "required_count": audit["required_count"],
                "source_count": audit["source_count"],
                "missing_required": len(audit["missing_required"]),
                "markerless_required": len(audit["markerless_required"]),
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
