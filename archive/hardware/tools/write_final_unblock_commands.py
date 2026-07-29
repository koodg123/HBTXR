#!/usr/bin/env python3
"""Write exact operator commands for the remaining third-goal unblock steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DATE_TAG = "2026_06_10"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def blocker_names(payload: dict[str, Any]) -> list[str]:
    blockers = payload.get("blockers", [])
    if isinstance(blockers, list):
        names = [str(item.get("name")) for item in blockers if isinstance(item, dict) and item.get("name")]
        if names:
            return sorted(set(names))
    remaining = payload.get("remaining_blockers", [])
    if isinstance(remaining, list):
        return sorted({str(item) for item in remaining if str(item)})
    return []


def build_commands(
    *,
    root: Path,
    final_audit: dict[str, Any],
    readiness: dict[str, Any] | None,
    xr_vits_packet: dict[str, Any] | None,
    c3b_contract: dict[str, Any] | None,
    host_placeholder: str,
    user_placeholder: str,
    approver_placeholder: str,
) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware"
    blockers = blocker_names(final_audit)
    readiness_status = (readiness or {}).get("status", "unknown")
    xr_vits_status = (xr_vits_packet or {}).get("status", "unknown")
    contract = c3b_contract or {}
    contract_validation = contract.get("validation", {}) if isinstance(contract.get("validation"), dict) else {}
    contract_validate_command = str(
        contract_validation.get(
            "validate_command",
            "python3 tools/validate_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
        )
    )
    contract_dry_run_import_command = str(
        contract_validation.get(
            "dry_run_import_command",
            "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --dry-run",
        )
    )
    contract_active_import_command = str(
        contract_validation.get(
            "active_import_command",
            "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
        )
    )

    sections: list[dict[str, Any]] = []
    xr_vits_policy_integrity = {
        "required": True,
        "policy_path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
        "candidate_audit": str(root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
        "generator": "tools/create_xr_vits_replacement_policy.py",
        "validator": "tools/check_final_blocker_closure_readiness.py",
        "required_policy_fields": [
            "candidate_audit_fingerprint",
            "candidate_audit_recommendation_snapshot",
            "candidate_audit_meta",
            "approval_event",
            "policy_fingerprint",
        ],
        "legacy_policy_clears_final_signoff": False,
    }
    if blockers:
        sections.append(
            {
                "id": "U0",
                "title": "Dry-run final blocker closure readiness",
                "status": "advisory-no-side-effects",
                "reason": "Check whether supplied C3b smoke and XR-VITs choices would clear final blockers before writing canonical unblock inputs.",
                "commands": [
                    (
                        "python3 tools/check_final_blocker_closure_readiness.py "
                        f"--root {root} "
                        "--c3b-no-require-paths "
                        "--json-out /tmp/hgtxr_final_blocker_readiness_current.json "
                        "--markdown-out /tmp/hgtxr_final_blocker_readiness_current.md"
                    ),
                    (
                        "python3 tools/check_final_blocker_closure_readiness.py "
                        f"--root {root} "
                        "--c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
                        "--xr-vits-mode exact "
                        "--json-out /tmp/hgtxr_final_blocker_readiness_exact.json "
                        "--markdown-out /tmp/hgtxr_final_blocker_readiness_exact.md"
                    ),
                    (
                        "python3 tools/check_final_blocker_closure_readiness.py "
                        f"--root {root} "
                        "--c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
                        "--xr-vits-mode replacement "
                        f"--approved-by {approver_placeholder} "
                        '--reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                        "--json-out /tmp/hgtxr_final_blocker_readiness_replacement.json "
                        "--markdown-out /tmp/hgtxr_final_blocker_readiness_replacement.md"
                    ),
                ],
                "expected_result": "status=would-clear-with-candidates before running side-effectful final unblock command.",
            }
        )

    if "C3b AXIS/DMA physical smoke result" in blockers:
        sections.append(
            {
                "id": "U1",
                "title": "Run C3b ZCU104 smoke through SSH/SCP",
                "status": "pending-board-run",
                "reason": "Final signoff needs a real board-produced C3b smoke JSON.",
                "precheck": [
                    "python3 tools/check_c3b_board_smoke_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR",
                ],
                "commands": [
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--zcu104-host {host_placeholder} --zcu104-user {user_placeholder} "
                        "--execute-zcu104-smoke --allow-blocked"
                    ),
                    (
                        "python3 tools/run_zcu104_c3b_smoke_remote.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--host {host_placeholder} --user {user_placeholder} --execute"
                    ),
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
                        "--allow-blocked"
                    ),
                    contract_validate_command,
                    contract_dry_run_import_command,
                    contract_active_import_command,
                ],
                "contract": {
                    "status": contract.get("status", "missing"),
                    "preset": contract.get("preset", "axis-c3b-mem16"),
                    "canonical_result_path": contract.get(
                        "canonical_result_path",
                        str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                    ),
                    "expected_out_raw": (contract.get("required_fields") or {}).get("out_raw")
                    if isinstance(contract.get("required_fields"), dict)
                    else [32, -13, 26, -6, 14, -11],
                },
                "evidence_after": [
                    str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                    str(hardware / "generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.json"),
                ],
            }
        )

    if "requested XR-VITs sibling" in blockers:
        sections.append(
            {
                "id": "U2",
                "title": "Resolve XR-VITs reference gate",
                "status": "pending-user-choice",
                "reason": "Final signoff requires the exact XR-VITs sibling or explicit replacement approval.",
                "options": [
                    {
                        "id": "U2a",
                        "title": "Restore exact XR-VITs sibling",
                        "commands": [
                            "test -d /home/kjm26/project/PRJXR/XR-VITs",
                            "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
                        ],
                        "evidence_after": ["/home/kjm26/project/PRJXR/XR-VITs"],
                    },
                    {
                        "id": "U2b",
                        "title": "Approve XR_Accel replacement policy",
                        "commands": [
                            (
                                "python3 tools/run_third_goal_final_signoff.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                "--approve-xr-vits-replacement --dry-run-xr-vits-replacement "
                                f"--xr-vits-approved-by {approver_placeholder} "
                                '--xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                                "--allow-blocked"
                            ),
                            (
                                "python3 tools/create_xr_vits_replacement_policy.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                f"--approve --approved-by {approver_placeholder} "
                                '--reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                                "--dry-run"
                            ),
                            (
                                "python3 tools/create_xr_vits_replacement_policy.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                f"--approve --approved-by {approver_placeholder} "
                                '--reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'
                            ),
                            (
                                "python3 tools/run_third_goal_final_signoff.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                "--approve-xr-vits-replacement "
                                f"--xr-vits-approved-by {approver_placeholder} "
                                '--xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                                "--allow-blocked"
                            ),
                        ],
                        "evidence_after": [str(root / "docs/resources/xr_vits_replacement_policy.json")],
                        "integrity_required": xr_vits_policy_integrity,
                    },
                ],
            }
        )

    if {
        "C3b AXIS/DMA physical smoke result",
        "requested XR-VITs sibling",
    }.issubset(set(blockers)):
        sections.append(
            {
                "id": "U4",
                "title": "Combined one-shot unblock",
                "status": "pending-user-choice",
                "reason": "Use after a real C3b smoke JSON is copied back and the XR-VITs decision is known.",
                "options": [
                    {
                        "id": "U4a",
                        "title": "Import C3b result with restored XR-VITs",
                        "commands": [
                            "test -d /home/kjm26/project/PRJXR/XR-VITs",
                            (
                                "python3 tools/run_third_goal_final_signoff.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json"
                            ),
                        ],
                        "evidence_after": [
                            str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                            "/home/kjm26/project/PRJXR/XR-VITs",
                            str(root / "docs/resources/third_goal_final_signoff_run_2026_06_10.json"),
                        ],
                    },
                    {
                        "id": "U4b",
                        "title": "Import C3b result and approve XR_Accel",
                        "commands": [
                            (
                                "python3 tools/run_third_goal_final_signoff.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
                                "--dry-run-import-c3b-smoke "
                                "--approve-xr-vits-replacement --dry-run-xr-vits-replacement "
                                f"--xr-vits-approved-by {approver_placeholder} "
                                '--xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                                "--allow-blocked"
                            ),
                            (
                                "python3 tools/create_xr_vits_replacement_policy.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                f"--approve --approved-by {approver_placeholder} "
                                '--reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" '
                                "--dry-run"
                            ),
                            (
                                "python3 tools/create_xr_vits_replacement_policy.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                f"--approve --approved-by {approver_placeholder} "
                                '--reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'
                            ),
                            (
                                "python3 tools/run_third_goal_final_signoff.py "
                                "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                                "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
                                "--approve-xr-vits-replacement "
                                f"--xr-vits-approved-by {approver_placeholder} "
                                '--xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff"'
                            ),
                        ],
                        "evidence_after": [
                            str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                            str(root / "docs/resources/xr_vits_replacement_policy.json"),
                            str(root / "docs/resources/third_goal_final_signoff_run_2026_06_10.json"),
                        ],
                        "integrity_required": xr_vits_policy_integrity,
                    },
                ],
            }
        )

    if blockers:
        qkv_smoke_json = "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
        sections.append(
            {
                "id": "U5",
                "title": "Optional QKV URAM successor smoke and promotion check",
                "status": "optional-ready-for-board-run",
                "reason": "QKV URAM is not required for default final signoff, but this path validates the successor before any promotion claim.",
                "commands": [
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--zcu104-host {host_placeholder} --zcu104-user {user_placeholder} "
                        "--require-qkv-uram-physical-smoke --allow-blocked"
                    ),
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--zcu104-host {host_placeholder} --zcu104-user {user_placeholder} "
                        "--execute-qkv-uram-smoke --allow-blocked"
                    ),
                    (
                        "python3 tools/validate_pynq_smoke_result.py "
                        f"/path/to/{qkv_smoke_json} "
                        "--preset axis-vref-p0-softmax-input-x2-qkv-uram"
                    ),
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--import-qkv-uram-smoke-json /path/to/{qkv_smoke_json} "
                        "--dry-run-import-qkv-uram-smoke --allow-blocked"
                    ),
                    (
                        "python3 tools/run_third_goal_final_signoff.py "
                        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
                        f"--import-qkv-uram-smoke-json /path/to/{qkv_smoke_json} "
                        "--require-qkv-uram-physical-smoke --allow-blocked"
                    ),
                ],
                "evidence_after": [
                    str(hardware / "pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"),
                    str(root / "docs/resources/qkv_uram_smoke_import_2026_06_16.json"),
                    str(root / "docs/resources/zcu104_vref_p0_softmax_input_x2_qkv_uram_smoke_remote_run_2026_06_10.json"),
                ],
                "expected_result": "QKV physical smoke validates only optional successor promotion; default final blockers remain C3b smoke and XR-VITs gate.",
            }
        )

    sections.append(
        {
            "id": "U3",
            "title": "Run final signoff",
            "status": "pending" if blockers else "ready",
            "reason": "Run after U1/U2 are cleared.",
            "commands": [
                "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR",
            ],
            "expected_result": "status=pass, final_preflight_failures=0",
        }
    )

    return {
        "status": "pending-unblock" if blockers else "ready-for-final-signoff",
        "root": str(root),
        "safety": {
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "executes_commands": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
            "operator_must_replace_placeholders": True,
        },
        "source_final_audit_status": final_audit.get("status"),
        "readiness_status": readiness_status,
        "xr_vits_packet_status": xr_vits_status,
        "c3b_smoke_contract": {
            "status": contract.get("status", "missing"),
            "preset": contract.get("preset", "axis-c3b-mem16"),
            "canonical_result_path": contract.get(
                "canonical_result_path",
                str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
            ),
            "validate_command": contract_validate_command,
            "dry_run_import_command": contract_dry_run_import_command,
            "active_import_command": contract_active_import_command,
        },
        "xr_vits_policy_integrity": xr_vits_policy_integrity,
        "remaining_blockers": blockers,
        "sections": sections,
    }


def render_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Final Unblock Commands",
        "",
        f"- status: `{card['status']}`",
        f"- root: `{card['root']}`",
        f"- readiness_status: `{card['readiness_status']}`",
        f"- xr_vits_packet_status: `{card['xr_vits_packet_status']}`",
        f"- c3b_smoke_contract_status: `{card['c3b_smoke_contract']['status']}`",
        f"- xr_vits_policy_integrity_required: `{card['xr_vits_policy_integrity']['required']}`",
        "",
        "## Safety",
        "",
        "- This card does not create board smoke results.",
        "- This card does not create an XR-VITs replacement policy.",
        "- This card does not run network commands by itself.",
        "- Replace placeholders before executing commands.",
        "- XR-VITs replacement policy must include candidate audit fingerprint, recommendation snapshot, candidate metadata, and policy fingerprint.",
        "",
    ]
    if card["remaining_blockers"]:
        lines.append("## Remaining Blockers")
        lines.append("")
        for blocker in card["remaining_blockers"]:
            lines.append(f"- `{blocker}`")
        lines.append("")

    lines.append("## Commands")
    lines.append("")
    for section in card["sections"]:
        lines.append(f"### {section['id']} {section['title']}")
        lines.append(f"- status: `{section['status']}`")
        lines.append(f"- why: {section['reason']}")
        if section.get("precheck"):
            lines.append("- precheck:")
            lines.append("```sh")
            lines.extend(section["precheck"])
            lines.append("```")
        if section.get("commands"):
            lines.append("```sh")
            lines.extend(section["commands"])
            lines.append("```")
        if section.get("contract"):
            contract = section["contract"]
            lines.append("- C3b contract:")
            lines.append(f"  - status: `{contract['status']}`")
            lines.append(f"  - preset: `{contract['preset']}`")
            lines.append(f"  - canonical_result_path: `{contract['canonical_result_path']}`")
            lines.append(f"  - expected_out_raw: `{contract['expected_out_raw']}`")
        if section.get("options"):
            for option in section["options"]:
                lines.append(f"- option `{option['id']}`: {option['title']}")
                lines.append("```sh")
                lines.extend(option["commands"])
                lines.append("```")
        if section.get("expected_result"):
            lines.append(f"- expected: {section['expected_result']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final unblock command card.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--final-audit", type=Path, default=None)
    parser.add_argument("--readiness-json", type=Path, default=None)
    parser.add_argument("--xr-vits-packet-json", type=Path, default=None)
    parser.add_argument("--c3b-contract-json", type=Path, default=None)
    parser.add_argument("--host-placeholder", default="<zcu104-ip-or-host>")
    parser.add_argument("--user-placeholder", default="xilinx")
    parser.add_argument("--approver-placeholder", default="<approved-by>")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    resources = root / "docs" / "resources"
    final_audit_path = args.final_audit or resources / f"final_signoff_audit_{DEFAULT_DATE_TAG}.json"
    readiness_path = args.readiness_json or resources / f"c3b_board_smoke_readiness_{DEFAULT_DATE_TAG}.json"
    xr_vits_path = args.xr_vits_packet_json or resources / f"xr_vits_unblock_packet_{DEFAULT_DATE_TAG}.json"
    c3b_contract_path = args.c3b_contract_json or resources / f"c3b_smoke_result_contract_{DEFAULT_DATE_TAG}.json"

    final_audit = load_json(final_audit_path)
    readiness = load_json(readiness_path) if readiness_path.exists() else None
    xr_vits_packet = load_json(xr_vits_path) if xr_vits_path.exists() else None
    c3b_contract = load_json(c3b_contract_path) if c3b_contract_path.exists() else None
    card = build_commands(
        root=root,
        final_audit=final_audit,
        readiness=readiness,
        xr_vits_packet=xr_vits_packet,
        c3b_contract=c3b_contract,
        host_placeholder=args.host_placeholder,
        user_placeholder=args.user_placeholder,
        approver_placeholder=args.approver_placeholder,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(card))
    print(
        f"[final-unblock-commands] status={card['status']} "
        f"blockers={len(card['remaining_blockers'])} sections={len(card['sections'])}"
    )
    return 0 if card["status"] == "ready-for-final-signoff" else 1


if __name__ == "__main__":
    raise SystemExit(main())
