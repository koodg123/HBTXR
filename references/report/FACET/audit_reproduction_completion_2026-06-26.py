#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/kjm26/project/PRJXR/HBTXR")
REPORT_ROOT = ROOT / "references/report/FACET"

EXPECTED_STATUS_ITEMS = [
    "Gate 0 preflight",
    "Phase 1 subset DeanDataset",
    "Phase 2 U-Net labelled PNG dataset",
    "Phase 3 full DeanDataset_full_unet",
    "Report artifacts",
    "U-Net labelled subset visual samples",
    "Phase 1 EPNet smoke checkpoint",
    "Phase 2 U-Net smoke checkpoint",
    "Phase 2 full U-Net checkpoint",
    "Phase 4 full EPNet checkpoint",
    "Phase 4 full EPNet training completion",
    "Phase 4B full HBTXR checkpoint",
    "Phase 4B full HBTXR training completion",
    "Phase 4 EPNet fpn_dw ablation checkpoint",
    "Phase 4 EPNet fpn_dw ablation completion",
    "Phase 4B HBTXR effective-batch-32 checkpoint",
    "Phase 4B HBTXR effective-batch-32 completion",
    "Phase 4 final evaluation artifacts",
]

PLAN_REQUIREMENT_GROUPS = [
    {
        "name": "Phase 1 subset EPNet baseline",
        "status_items": [
            "Phase 1 subset DeanDataset",
            "Phase 1 EPNet smoke checkpoint",
        ],
    },
    {
        "name": "Phase 2 U-Net relabeling pipeline",
        "status_items": [
            "Phase 2 U-Net labelled PNG dataset",
            "Phase 2 U-Net smoke checkpoint",
            "Phase 2 full U-Net checkpoint",
        ],
    },
    {
        "name": "Phase 3 full DeanDataset expansion",
        "status_items": [
            "Phase 3 full DeanDataset_full_unet",
            "U-Net labelled subset visual samples",
        ],
    },
    {
        "name": "Phase 4 full EPNet reproduction",
        "status_items": [
            "Phase 4 full EPNet checkpoint",
            "Phase 4 full EPNet training completion",
            "Phase 4 EPNet fpn_dw ablation checkpoint",
            "Phase 4 EPNet fpn_dw ablation completion",
            "Phase 4 final evaluation artifacts",
        ],
    },
    {
        "name": "Phase 4B HBTXR parallel comparison",
        "status_items": [
            "Phase 4B full HBTXR checkpoint",
            "Phase 4B full HBTXR training completion",
            "Phase 4B HBTXR effective-batch-32 checkpoint",
            "Phase 4B HBTXR effective-batch-32 completion",
            "Phase 4 final evaluation artifacts",
        ],
    },
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def item_by_name(status: dict):
    return {item.get("name"): item for item in status.get("items", [])}


def summarize_group(group: dict, items: dict):
    rows = []
    group_passed = True
    for name in group["status_items"]:
        item = items.get(name)
        if item is None:
            rows.append(
                {
                    "name": name,
                    "state": "missing_from_status",
                    "missing": ["status item is absent"],
                    "note": "",
                }
            )
            group_passed = False
            continue
        state = item.get("state")
        if state != "passed":
            group_passed = False
        rows.append(
            {
                "name": name,
                "state": state,
                "missing": item.get("missing", []),
                "note": item.get("note", ""),
            }
        )
    return {
        "name": group["name"],
        "state": "passed" if group_passed else "incomplete",
        "items": rows,
    }


def build_audit(plan_path: Path, status_path: Path):
    status = load_json(status_path)
    items = item_by_name(status)
    missing_expected_items = [
        name for name in EXPECTED_STATUS_ITEMS if name not in items
    ]
    non_passed = [
        {
            "name": item.get("name"),
            "state": item.get("state"),
            "missing": item.get("missing", []),
            "note": item.get("note", ""),
        }
        for item in status.get("items", [])
        if item.get("state") != "passed"
    ]
    groups = [
        summarize_group(group, items)
        for group in PLAN_REQUIREMENT_GROUPS
    ]
    can_mark_goal_complete = (
        status.get("overall_status") == "passed"
        and not missing_expected_items
        and not non_passed
        and all(group["state"] == "passed" for group in groups)
        and plan_path.is_file()
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "status_path": str(status_path),
        "plan_exists": plan_path.is_file(),
        "status_overall": status.get("overall_status"),
        "status_counts": status.get("counts"),
        "can_mark_goal_complete": can_mark_goal_complete,
        "completion_decision": "complete" if can_mark_goal_complete else "incomplete",
        "missing_expected_status_items": missing_expected_items,
        "non_passed_status_items": non_passed,
        "requirement_groups": groups,
        "completion_rule": (
            "Goal completion requires the plan file to exist, every expected "
            "status item to be present and passed, grouped Phase 1-4B "
            "requirements to be passed, and final full-validation artifacts "
            "to be accepted by the status checker."
        ),
    }


def make_markdown(audit: dict):
    lines = [
        "# FACET Reproduction Completion Audit",
        "",
        f"Generated at: `{audit['generated_at']}`",
        f"Plan: `{audit['plan_path']}`",
        f"Status JSON: `{audit['status_path']}`",
        f"Status overall: `{audit['status_overall']}`",
        f"Status counts: `{audit['status_counts']}`",
        f"Can mark goal complete: `{audit['can_mark_goal_complete']}`",
        f"Completion decision: `{audit['completion_decision']}`",
        "",
        "## Completion Rule",
        "",
        audit["completion_rule"],
        "",
        "## Requirement Groups",
        "",
        "| Group | State |",
        "|---|---|",
    ]
    for group in audit["requirement_groups"]:
        lines.append(f"| {group['name']} | {group['state']} |")

    lines.extend(["", "## Non-Passed Status Items", ""])
    if not audit["non_passed_status_items"]:
        lines.append("All expected status items are passed.")
    else:
        lines.append("| Status item | State | Missing |")
        lines.append("|---|---|---|")
        for item in audit["non_passed_status_items"]:
            missing = ", ".join(str(value) for value in item.get("missing", []))
            lines.append(f"| {item['name']} | {item['state']} | {missing} |")

    if audit["missing_expected_status_items"]:
        lines.extend(["", "## Missing Expected Status Items", ""])
        for name in audit["missing_expected_status_items"]:
            lines.append(f"- {name}")

    lines.extend(["", "## Group Details", ""])
    for group in audit["requirement_groups"]:
        lines.extend([f"### {group['name']}", "", f"State: `{group['state']}`", ""])
        lines.append("| Status item | State | Missing |")
        lines.append("|---|---|---|")
        for item in group["items"]:
            missing = ", ".join(str(value) for value in item.get("missing", []))
            lines.append(f"| {item['name']} | {item['state']} | {missing} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether the FACET reproduction plan can be marked complete."
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPORT_ROOT / "FACET_reproduction_plan_2026-06-25.md",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=REPORT_ROOT / "FACET_reproduction_status_2026-06-26.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPORT_ROOT / "FACET_reproduction_completion_audit_2026-06-26.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORT_ROOT / "FACET_reproduction_completion_audit_2026-06-26.md",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with code 1 when the audit cannot mark the goal complete.",
    )
    args = parser.parse_args()

    audit = build_audit(args.plan, args.status_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(make_markdown(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if args.fail_on_incomplete and not audit["can_mark_goal_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
