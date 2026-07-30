#!/usr/bin/env python3
"""Write an executable unblock checklist for the remaining third-goal blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def blocker_names(final_audit: dict[str, Any]) -> set[str]:
    blockers = final_audit.get("blockers", [])
    if not isinstance(blockers, list):
        return set()
    return {str(blocker.get("name")) for blocker in blockers if isinstance(blocker, dict)}


def completion_blockers(completion_audit: dict[str, Any]) -> list[dict[str, Any]]:
    items = completion_audit.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and item.get("status") == "blocked"]


def build_checklist(root: Path, final_audit: dict[str, Any], completion_audit: dict[str, Any]) -> dict[str, Any]:
    hardware = root / "hardware"
    names = blocker_names(final_audit)
    steps: list[dict[str, Any]] = []

    if "requested XR-VITs sibling" in names:
        steps.append(
            {
                "id": "B1",
                "title": "Resolve XR-VITs reference gate",
                "status": "pending-user-choice",
                "why": "Final signoff requires the exact XR-VITs sibling or an explicitly approved replacement policy.",
                "options": [
                    {
                        "id": "B1a",
                        "title": "Restore exact XR-VITs checkout",
                        "commands": [
                            "test -d /home/kjm26/project/PRJXR/XR-VITs",
                            "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
                        ],
                    },
                    {
                        "id": "B1b",
                        "title": "Approve XR_Accel as replacement",
                        "commands": [
                            "python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <name> --reason \"Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff\" --dry-run",
                            "python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <name> --reason \"Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff\"",
                            "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
                        ],
                    },
                ],
                "evidence_after": [
                    str(root / "docs/resources/xr_vits_replacement_policy.json"),
                    "/home/kjm26/project/PRJXR/XR-VITs",
                ],
            }
        )

    if "C3b AXIS/DMA physical smoke result" in names:
        steps.append(
            {
                "id": "B2",
                "title": "Run and import C3b ZCU104 physical smoke",
                "status": "pending-board-run",
                "why": "Final signoff requires a real board-produced C3b smoke JSON.",
                "host_artifacts": [
                    str(hardware / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"),
                    str(hardware / "generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md"),
                ],
                "board_commands": [
                    "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                    "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
                    "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                    "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                ],
                "host_commands": [
                    "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --allow-blocked",
                    "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --allow-blocked",
                    "python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                    "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff",
                ],
                "evidence_after": [
                    "generated/signoff/c3b_smoke_import_2026_06_10.json",
                    "generated/signoff/c3b_smoke_import_validation_2026_06_10.json",
                    "generated/signoff/third_goal_final_signoff_run_2026_06_10.json",
                    str(hardware / "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                ],
            }
        )

    steps.append(
        {
            "id": "B3",
            "title": "Run final host signoff after blockers are cleared",
            "status": "pending",
            "commands": [
                "python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out generated/signoff/third_goal_final_signoff_2026_06_10.json",
                "python3 tools/write_final_signoff_audit.py --preflight-json generated/signoff/third_goal_final_signoff_2026_06_10.json --json-out generated/signoff/final_signoff_audit_2026_06_10.json --markdown-out generated/signoff/final_signoff_audit_2026_06_10.md",
                "python3 tools/write_third_goal_completion_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/third_goal_completion_audit_2026_06_10.json --markdown-out generated/signoff/third_goal_completion_audit_2026_06_10.md",
            ],
            "expected_result": "final-signoff fail=0 and completion audit status=pass",
        }
    )

    return {
        "status": "pending-unblock" if names else "ready-for-final-signoff",
        "root": str(root),
        "source_final_audit_status": final_audit.get("status"),
        "source_completion_status": completion_audit.get("status"),
        "final_blockers": sorted(names),
        "completion_blocked_items": [
            {"id": str(item.get("id")), "title": str(item.get("title"))}
            for item in completion_blockers(completion_audit)
        ],
        "step_count": len(steps),
        "steps": steps,
    }


def render_markdown(checklist: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Third Goal Unblock Checklist",
        "",
        f"- status: `{checklist['status']}`",
        f"- root: `{checklist['root']}`",
        f"- final_blockers: `{len(checklist['final_blockers'])}`",
        f"- steps: `{checklist['step_count']}`",
        "",
    ]
    if checklist["final_blockers"]:
        lines.append("## Final Blockers")
        lines.append("")
        for blocker in checklist["final_blockers"]:
            lines.append(f"- `{blocker}`")
        lines.append("")

    lines.append("## Steps")
    lines.append("")
    for step in checklist["steps"]:
        lines.append(f"### {step['id']} {step['title']}")
        lines.append(f"- status: `{step['status']}`")
        if step.get("why"):
            lines.append(f"- why: {step['why']}")
        if step.get("options"):
            for option in step["options"]:
                lines.append(f"- option `{option['id']}`: {option['title']}")
                lines.append("```sh")
                lines.extend(option["commands"])
                lines.append("```")
        if step.get("board_commands"):
            lines.append("- board commands:")
            lines.append("```sh")
            lines.extend(step["board_commands"])
            lines.append("```")
        if step.get("host_commands"):
            lines.append("- host commands:")
            lines.append("```sh")
            lines.extend(step["host_commands"])
            lines.append("```")
        if step.get("commands"):
            lines.append("```sh")
            lines.extend(step["commands"])
            lines.append("```")
        if step.get("expected_result"):
            lines.append(f"- expected: {step['expected_result']}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write third-goal unblock checklist.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--final-audit", type=Path, default=None)
    parser.add_argument("--completion-audit", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    final_audit_path = args.final_audit or root / "docs/resources/final_signoff_audit_2026_06_10.json"
    completion_audit_path = args.completion_audit or root / "docs/resources/third_goal_completion_audit_2026_06_10.json"
    checklist = build_checklist(root, load_json(final_audit_path), load_json(completion_audit_path))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(checklist, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(checklist) + "\n")
    print(
        f"[third-goal-unblock-checklist] status={checklist['status']} "
        f"blockers={len(checklist['final_blockers'])} steps={checklist['step_count']}"
    )
    return 0 if checklist["status"] == "ready-for-final-signoff" else 1


if __name__ == "__main__":
    raise SystemExit(main())
