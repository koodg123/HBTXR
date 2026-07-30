#!/usr/bin/env python3
"""Write one operator packet for closing the remaining final-signoff blockers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import create_xr_vits_replacement_policy as xr_policy


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def artifact_entry(root: Path, rel: str, *, required: bool = True) -> dict[str, Any]:
    path = root / rel
    entry: dict[str, Any] = {
        "relative_path": rel,
        "path": str(path),
        "required": required,
        "exists": path.exists(),
    }
    if path.exists():
        entry.update(
            {
                "status": "pass",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    else:
        entry["status"] = "missing" if required else "absent-optional"
    return entry


def optional_json(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    if not path.exists():
        return {}
    return load_json(path)


def build_vref_successor_gate(root: Path, required_for_final_signoff: bool = False) -> dict[str, Any]:
    rel = "hardware/generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json"
    package = optional_json(root, rel)
    if not package:
        return {
            "status": "not_available",
            "artifact": str(root / rel),
            "required_for_final_signoff": required_for_final_signoff,
        }

    physical = package.get("physical_smoke_result", {}) if isinstance(package.get("physical_smoke_result"), dict) else {}
    projection = package.get("recommended_c3b_projection", {}) if isinstance(package.get("recommended_c3b_projection"), dict) else {}
    physical_status = physical.get("status")
    if physical_status == "pass":
        gate_status = "pass"
    elif required_for_final_signoff:
        gate_status = "blocked"
    elif physical_status == "not_captured":
        gate_status = "ready-for-physical-smoke"
    else:
        gate_status = physical_status or "unknown"
    return {
        "status": gate_status,
        "artifact": str(root / rel),
        "required_for_final_signoff": required_for_final_signoff,
        "candidate": package.get("candidate"),
        "recommended_resource_variant": package.get("recommended_resource_variant"),
        "projection_status": projection.get("status"),
        "projection_failures": projection.get("failures", []),
        "projection_pending": projection.get("pending", []),
        "physical_smoke_status": physical.get("status"),
        "physical_smoke_preset": physical.get("preset"),
        "physical_smoke_result_json": physical.get("result_json"),
        "expected_result_json": str(
            root
            / "hardware"
            / "pynq"
            / "hgtxr"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
        ),
    }


def build_qkv_uram_successor_gate(root: Path) -> dict[str, Any]:
    rel = "hardware/generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json"
    package = optional_json(root, rel)
    expected_result_json = str(
        root
        / "hardware"
        / "pynq"
        / "hgtxr"
        / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
    )
    if not package:
        return {
            "status": "not_available",
            "artifact": str(root / rel),
            "required_for_final_signoff": False,
            "expected_result_json": expected_result_json,
        }

    physical = package.get("physical_smoke_result", {}) if isinstance(package.get("physical_smoke_result"), dict) else {}
    csynth = package.get("csynth", {}) if isinstance(package.get("csynth"), dict) else {}
    resources = csynth.get("resources", {}) if isinstance(csynth.get("resources"), dict) else {}
    overlay = package.get("overlay", {}) if isinstance(package.get("overlay"), dict) else {}
    timing = overlay.get("timing", {}) if isinstance(overlay.get("timing"), dict) else {}
    route_status = overlay.get("route_status", {}) if isinstance(overlay.get("route_status"), dict) else {}
    physical_status = str(physical.get("status", "not_captured"))
    return {
        "status": "pass" if physical_status == "pass" else "ready-for-physical-smoke",
        "artifact": str(root / rel),
        "required_for_final_signoff": False,
        "candidate": package.get("candidate"),
        "macro": package.get("macro"),
        "csynth_status": package.get("status"),
        "latency_cycles": csynth.get("latency_cycles"),
        "resources": resources,
        "routed_wns_ns": timing.get("wns_ns"),
        "route_errors": route_status.get("routing_error_nets"),
        "physical_smoke_status": physical_status,
        "physical_smoke_preset": physical.get("preset") or "axis-vref-p0-softmax-input-x2-qkv-uram",
        "physical_smoke_result_json": physical.get("result_json"),
        "expected_result_json": expected_result_json,
    }


def section_by_id(card: dict[str, Any], section_id: str) -> dict[str, Any]:
    sections = card.get("sections", [])
    if not isinstance(sections, list):
        return {}
    for section in sections:
        if isinstance(section, dict) and section.get("id") == section_id:
            return section
    return {}


def commands_from(section: dict[str, Any]) -> list[str]:
    commands = section.get("commands", [])
    return [str(command) for command in commands] if isinstance(commands, list) else []


def options_from(section: dict[str, Any]) -> dict[str, list[str]]:
    options = section.get("options", [])
    result: dict[str, list[str]] = {}
    if not isinstance(options, list):
        return result
    for option in options:
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("id", ""))
        option_commands = option.get("commands", [])
        if option_id and isinstance(option_commands, list):
            result[option_id] = [str(command) for command in option_commands]
    return result


def resolve_path(root: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value or fallback))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_xr_policy_integrity(root: Path, command_card: dict[str, Any]) -> dict[str, Any]:
    base = (
        command_card.get("xr_vits_policy_integrity", {})
        if isinstance(command_card.get("xr_vits_policy_integrity"), dict)
        else {}
    )
    policy_path = resolve_path(root, base.get("policy_path"), "docs/resources/xr_vits_replacement_policy.json")
    candidate_audit_path = resolve_path(
        root,
        base.get("candidate_audit"),
        xr_policy.DEFAULT_AUDIT_REL,
    )
    result = dict(base)
    result.update(
        {
            "required": base.get("required") is True,
            "policy_path": str(policy_path),
            "candidate_audit": str(candidate_audit_path),
            "policy_exists": policy_path.exists(),
        }
    )
    if not policy_path.exists():
        result["validation"] = {
            "status": "pending-policy-creation",
            "policy_exists": False,
            "candidate_audit_path": str(candidate_audit_path),
            "errors": [],
            "warnings": ["replacement policy has not been written yet; operator approval is still required"],
        }
        return result

    policy = load_json(policy_path)
    validation = xr_policy.validate_policy_integrity(root, policy)
    validation["policy_exists"] = True
    result["validation"] = validation
    for field in sorted(xr_policy.INTEGRITY_FIELD_NAMES):
        if field in policy:
            result[field] = policy[field]
    return result


def build_packet(root: Path, require_vref_successor_physical_smoke: bool = False) -> dict[str, Any]:
    root = root.resolve()
    resources = root / "docs" / "resources"
    command_card = load_json(resources / f"final_unblock_commands_{DATE_TAG}.json")
    contract = load_json(resources / f"c3b_smoke_result_contract_{DATE_TAG}.json")
    readiness = load_json(resources / f"c3b_board_smoke_readiness_{DATE_TAG}.json")
    transfer = load_json(resources / f"c3b_smoke_transfer_manifest_{DATE_TAG}.json")
    resolution = load_json(resources / f"xr_vits_reference_resolution_{DATE_TAG}.json")
    closure = load_json(resources / f"final_blocker_closure_readiness_{DATE_TAG}.json")
    intake = load_json(resources / f"final_unblock_intake_{DATE_TAG}.json")
    final_audit = load_json(resources / f"final_signoff_audit_{DATE_TAG}.json")
    vref_successor_gate = build_vref_successor_gate(root, require_vref_successor_physical_smoke)
    qkv_uram_successor_gate = build_qkv_uram_successor_gate(root)

    blockers = final_audit.get("blockers", [])
    remaining = [
        str(item.get("name"))
        for item in blockers
        if isinstance(item, dict) and str(item.get("name", ""))
    ]
    if (
        require_vref_successor_physical_smoke
        and vref_successor_gate.get("physical_smoke_status") != "pass"
        and "VREF-P0 successor physical smoke result" not in remaining
    ):
        remaining.append("VREF-P0 successor physical smoke result")
    u0 = section_by_id(command_card, "U0")
    u1 = section_by_id(command_card, "U1")
    u2 = section_by_id(command_card, "U2")
    u4 = section_by_id(command_card, "U4")
    u5 = section_by_id(command_card, "U5")

    required_artifacts = [
        artifact_entry(root, f"docs/resources/final_unblock_commands_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/final_unblock_commands_{DATE_TAG}.md"),
        artifact_entry(root, f"docs/resources/c3b_smoke_result_contract_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/c3b_smoke_result_contract_{DATE_TAG}.md"),
        artifact_entry(root, f"docs/resources/c3b_board_smoke_readiness_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/c3b_smoke_transfer_manifest_{DATE_TAG}.json"),
        artifact_entry(root, "docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"),
        artifact_entry(root, f"docs/resources/xr_vits_reference_resolution_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/final_blocker_closure_readiness_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/final_unblock_intake_{DATE_TAG}.json"),
        artifact_entry(root, f"docs/resources/final_unblock_intake_{DATE_TAG}.md"),
        artifact_entry(root, f"docs/resources/final_signoff_audit_{DATE_TAG}.json"),
    ]
    missing = [entry["relative_path"] for entry in required_artifacts if entry["status"] != "pass"]
    status = "ready-for-operator-unblock" if not missing else "incomplete"

    tar = transfer.get("tar", {}) if isinstance(transfer.get("tar"), dict) else {}
    exact = resolution.get("exact", {}) if isinstance(resolution.get("exact"), dict) else {}
    candidate = resolution.get("candidate", {}) if isinstance(resolution.get("candidate"), dict) else {}
    contract_fields = contract.get("required_fields", {}) if isinstance(contract.get("required_fields"), dict) else {}
    contract_validation = contract.get("validation", {}) if isinstance(contract.get("validation"), dict) else {}
    xr_policy_integrity = build_xr_policy_integrity(root, command_card)
    return {
        "status": status,
        "root": str(root),
        "remaining_blockers": remaining,
        "required_artifacts": required_artifacts,
        "missing_required_artifacts": missing,
        "board_package": {
            "status": readiness.get("status"),
            "tar": tar.get("path"),
            "tar_sha256": tar.get("sha256"),
            "preset": transfer.get("preset"),
            "expected_runtime_state": (transfer.get("board_expected_outputs") or {}).get("expected_runtime_state")
            if isinstance(transfer.get("board_expected_outputs"), dict)
            else None,
            "expected_out_raw": (transfer.get("board_expected_outputs") or {}).get("expected_out_raw")
            if isinstance(transfer.get("board_expected_outputs"), dict)
            else None,
        },
        "c3b_smoke_contract": {
            "status": contract.get("status"),
            "preset": contract.get("preset"),
            "variant": contract.get("variant"),
            "canonical_result_path": contract.get("canonical_result_path"),
            "expected_runtime_state": contract_fields.get("runtime_state"),
            "expected_out_raw": contract_fields.get("out_raw"),
            "validate_command": contract_validation.get("validate_command", ""),
            "dry_run_import_command": contract_validation.get("dry_run_import_command", ""),
            "active_import_command": contract_validation.get("active_import_command", ""),
        },
        "xr_vits_resolution": {
            "status": resolution.get("status"),
            "requested_path": resolution.get("requested_path"),
            "exact_status": exact.get("status"),
            "candidate_status": candidate.get("status"),
            "candidate_replacement_path": candidate.get("replacement_path"),
            "candidate_score": candidate.get("score"),
            "policy_integrity": xr_policy_integrity,
        },
        "closure_readiness": {
            "status": closure.get("status"),
            "current_ready": bool(closure.get("current_ready")),
            "candidate_ready": bool(closure.get("candidate_ready")),
            "final_runner_command": closure.get("final_runner_command", ""),
        },
        "unblock_intake": {
            "status": intake.get("status"),
            "dry_run_final_runner_command": intake.get("dry_run_final_runner_command", ""),
            "active_final_runner_command": intake.get("active_final_runner_command", ""),
        },
        "vref_successor_gate": vref_successor_gate,
        "qkv_uram_successor_gate": qkv_uram_successor_gate,
        "vref_successor_policy": {
            "require_physical_smoke_for_final_signoff": require_vref_successor_physical_smoke,
        },
        "operator_commands": {
            "dry_run_readiness": commands_from(u0),
            "run_board_smoke": commands_from(u1),
            "resolve_xr_vits": options_from(u2),
            "combined_unblock": options_from(u4),
            "qkv_uram_successor": commands_from(u5),
        },
        "expected_canonical_outputs": [
            str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
            str(root / "docs/resources/xr_vits_replacement_policy.json"),
            str(root / f"docs/resources/third_goal_final_signoff_run_{DATE_TAG}.json"),
        ],
        "expected_successor_outputs": [
            vref_successor_gate.get("expected_result_json", ""),
            qkv_uram_successor_gate.get("expected_result_json", ""),
        ],
        "safety": {
            "executes_commands": False,
            "executes_network": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Unblock Closeout Packet",
        "",
        f"- status: `{packet['status']}`",
        f"- root: `{packet['root']}`",
        "",
        "## Remaining Blockers",
        "",
    ]
    if packet["remaining_blockers"]:
        lines.extend(f"- `{item}`" for item in packet["remaining_blockers"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Required Artifacts", "", "| Artifact | Status | SHA256 |", "|---|---|---|"])
    for entry in packet["required_artifacts"]:
        lines.append(f"| `{entry['relative_path']}` | `{entry['status']}` | `{entry.get('sha256', '')}` |")
    lines.extend(["", "## Dry-Run Readiness", "", "```sh"])
    lines.extend(packet["operator_commands"]["dry_run_readiness"])
    lines.extend(["```", "", "## Combined Unblock", ""])
    for option_id, commands in packet["operator_commands"]["combined_unblock"].items():
        lines.extend([f"### {option_id}", "", "```sh"])
        lines.extend(commands)
        lines.extend(["```", ""])
    xr_integrity = packet.get("xr_vits_resolution", {}).get("policy_integrity", {})
    if isinstance(xr_integrity, dict):
        lines.extend(
            [
                "## XR-VITs Policy Integrity",
                "",
                f"- required: `{xr_integrity.get('required', False)}`",
                f"- generator: `{xr_integrity.get('generator', '')}`",
                f"- validator: `{xr_integrity.get('validator', '')}`",
                f"- legacy_policy_clears_final_signoff: `{xr_integrity.get('legacy_policy_clears_final_signoff', True)}`",
                f"- required_policy_fields: `{xr_integrity.get('required_policy_fields', [])}`",
                f"- policy_exists: `{xr_integrity.get('policy_exists', False)}`",
                f"- validation_status: `{(xr_integrity.get('validation', {}) or {}).get('status', 'missing')}`",
                f"- candidate_audit_fingerprint: `{xr_integrity.get('candidate_audit_fingerprint', '')}`",
                f"- policy_fingerprint: `{xr_integrity.get('policy_fingerprint', '')}`",
                "",
            ]
        )
    vref_gate = packet.get("vref_successor_gate", {}) if isinstance(packet.get("vref_successor_gate"), dict) else {}
    vref_policy = packet.get("vref_successor_policy", {}) if isinstance(packet.get("vref_successor_policy"), dict) else {}
    qkv_gate = packet.get("qkv_uram_successor_gate", {}) if isinstance(packet.get("qkv_uram_successor_gate"), dict) else {}
    lines.extend(
        [
            "## VREF Successor Gate",
            "",
            f"- status: `{vref_gate.get('status', 'missing')}`",
            f"- required_for_final_signoff: `{vref_gate.get('required_for_final_signoff', False)}`",
            f"- require_physical_smoke_for_final_signoff: `{vref_policy.get('require_physical_smoke_for_final_signoff', False)}`",
            f"- recommended_resource_variant: `{vref_gate.get('recommended_resource_variant', '')}`",
            f"- projection_status: `{vref_gate.get('projection_status', '')}`",
            f"- physical_smoke_status: `{vref_gate.get('physical_smoke_status', '')}`",
            f"- expected_result_json: `{vref_gate.get('expected_result_json', '')}`",
            "",
        ]
    )
    lines.extend(
        [
            "## QKV URAM Successor Gate",
            "",
            f"- status: `{qkv_gate.get('status', 'missing')}`",
            f"- required_for_final_signoff: `{qkv_gate.get('required_for_final_signoff', False)}`",
            f"- csynth_status: `{qkv_gate.get('csynth_status', '')}`",
            f"- latency_cycles: `{qkv_gate.get('latency_cycles', '')}`",
            f"- resources: `{qkv_gate.get('resources', {})}`",
            f"- routed_wns_ns: `{qkv_gate.get('routed_wns_ns', '')}`",
            f"- route_errors: `{qkv_gate.get('route_errors', '')}`",
            f"- physical_smoke_status: `{qkv_gate.get('physical_smoke_status', '')}`",
            f"- expected_result_json: `{qkv_gate.get('expected_result_json', '')}`",
            "",
        ]
    )
    qkv_commands = packet.get("operator_commands", {}).get("qkv_uram_successor", [])
    if isinstance(qkv_commands, list) and qkv_commands:
        lines.extend(["### QKV URAM Operator Commands", "", "```sh"])
        lines.extend(str(command) for command in qkv_commands)
        lines.extend(["```", ""])
    lines.extend(["## Safety", ""])
    for key, value in packet["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final unblock closeout packet.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument(
        "--require-vref-successor-physical-smoke",
        action="store_true",
        help="Treat VREF-P0 successor physical smoke as a hard final-signoff blocker instead of optional promotion evidence.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    packet = build_packet(args.root, require_vref_successor_physical_smoke=args.require_vref_successor_physical_smoke)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(packet))
    print(
        f"[final-unblock-closeout-packet] status={packet['status']} "
        f"blockers={len(packet['remaining_blockers'])} missing={len(packet['missing_required_artifacts'])}"
    )
    return 0 if packet["status"] == "ready-for-operator-unblock" else 1


if __name__ == "__main__":
    raise SystemExit(main())
