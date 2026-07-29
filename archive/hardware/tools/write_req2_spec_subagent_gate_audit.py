#!/usr/bin/env python3
"""Write Req2 spec-kit and sub-agent usage gate audit."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def rel_or_abs(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def marker_check(text: str, markers: Sequence[str]) -> dict[str, bool]:
    lower = text.lower()
    return {marker: marker.lower() in lower for marker in markers}


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    hardware_docs = {
        "master_plan": hardware / "docs" / "Master-Plan.md",
        "sub_plan": hardware / "docs" / "Sub-Plan.md",
        "spec": hardware / "docs" / "Spec.md",
        "execution": hardware / "docs" / "Execution.md",
        "validation": hardware / "docs" / "Validation.md",
        "progress": hardware / "docs" / "track" / "PROGRESS.md",
        "requirements": hardware / "docs" / "track" / f"THIRD_GOAL_REQUIREMENTS_{DATE_TAG}.md",
    }
    parent_docs = {
        "master_plan": hgtxr / "docs" / "Master-Plan.md",
        "sub_plan": hgtxr / "docs" / "Sub-Plan.md",
        "spec": hgtxr / "docs" / "Spec.md",
        "execution": hgtxr / "docs" / "Execution.md",
        "validation": hgtxr / "docs" / "Validation.md",
        "progress": hgtxr / "docs" / "track" / "PROGRESS.md",
        "log": hgtxr / "docs" / "track" / "log.md",
    }
    generated = {
        "spec_plan_conformance": hardware / "generated" / "signoff" / "spec_plan_conformance_audit_2026_06_10.json",
        "third_goal_requirements_trace": hardware / "generated" / "signoff" / "third_goal_requirements_trace_2026_06_10.json",
    }

    hardware_text = "\n".join(read_text(path) for path in hardware_docs.values())
    parent_text = "\n".join(read_text(path) for path in parent_docs.values())
    combined_text = hardware_text + "\n" + parent_text

    spec_kit_path = shutil.which("spec-kit")
    specify_path = shutil.which("specify")
    doc_presence = {
        **{f"hardware_{name}": path.exists() for name, path in hardware_docs.items()},
        **{f"parent_{name}": path.exists() for name, path in parent_docs.items()},
    }
    required_docs_present = all(
        doc_presence[name]
        for name in [
            "hardware_master_plan",
            "hardware_sub_plan",
            "hardware_spec",
            "hardware_execution",
            "hardware_validation",
            "hardware_progress",
            "hardware_requirements",
            "parent_spec",
            "parent_execution",
            "parent_sub_plan",
        ]
    )
    spec_kit_markers = marker_check(
        combined_text,
        [
            "spec-kit",
            "specify",
            "not available",
            "not found",
            "manual",
            "manual spec",
        ],
    )
    subagent_markers = marker_check(
        combined_text,
        [
            "GPT5.3-Codex-Spark",
            "Spark",
            "GPT5.5",
            "fallback",
            "quota",
            "sub-agent",
            "Task Cards",
        ],
    )
    plan_markers = marker_check(
        combined_text,
        [
            "ZCU104",
            "C3b",
            "VREF",
            "Q4",
            "Q8",
            "HGTXR_PARALLELISM_FACTOR",
            "HGTXR_BUS_WIDTH",
            "HGTXR_FIFO_DEPTH",
        ],
    )
    generated_presence = {name: path.exists() for name, path in generated.items()}
    spec_kit_unavailable = spec_kit_path is None and specify_path is None
    manual_fallback_recorded = (
        spec_kit_markers["spec-kit"]
        and spec_kit_markers["specify"]
        and (spec_kit_markers["not available"] or spec_kit_markers["not found"])
        and spec_kit_markers["manual"]
    )
    subagent_policy_recorded = (
        subagent_markers["GPT5.3-Codex-Spark"]
        and subagent_markers["GPT5.5"]
        and subagent_markers["fallback"]
        and subagent_markers["sub-agent"]
        and subagent_markers["Task Cards"]
    )
    plan_spec_covered = required_docs_present and all(plan_markers.values())
    generated_trace_available = all(generated_presence.values())

    status = (
        "pass-manual-spec-and-model-fallback"
        if plan_spec_covered and spec_kit_unavailable and manual_fallback_recorded and subagent_policy_recorded
        else "partial"
    )
    return {
        "status": status,
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "req_id": "2",
        "gate": "Plan/spec first, spec-kit where available, Spark-first sub-agent routing",
        "spec_kit": {
            "spec_kit_path": spec_kit_path,
            "specify_path": specify_path,
            "available": not spec_kit_unavailable,
            "manual_fallback_recorded": manual_fallback_recorded,
            "markers": spec_kit_markers,
        },
        "subagents": {
            "spark_first_recorded": subagent_markers["GPT5.3-Codex-Spark"] and subagent_markers["Spark"],
            "gpt55_fallback_recorded": subagent_markers["GPT5.5"] and subagent_markers["fallback"],
            "task_cards_recorded": subagent_markers["Task Cards"],
            "quota_or_limit_recorded": subagent_markers["quota"],
            "markers": subagent_markers,
        },
        "plan_spec": {
            "required_docs_present": required_docs_present,
            "plan_spec_covered": plan_spec_covered,
            "markers": plan_markers,
            "documents": {
                **{f"hardware_{name}": rel_or_abs(hardware, path) for name, path in hardware_docs.items()},
                **{f"parent_{name}": str(path) for name, path in parent_docs.items()},
            },
        },
        "generated_evidence": {
            "all_present": generated_trace_available,
            "artifacts": {
                name: {
                    "path": rel_or_abs(hardware, path),
                    "exists": path.exists(),
                }
                for name, path in generated.items()
            },
        },
        "decision": {
            "reflected_when": "manual spec fallback is recorded because spec-kit/specify are unavailable and Spark-first/GPT5.5 fallback routing is documented",
            "does_not_claim": [
                "spec-kit CLI execution",
                "unlimited Spark availability",
            ],
        },
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "executes_network": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Req2 Spec/Sub-Agent Gate Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- req_id: `{audit['req_id']}`",
        f"- spec_kit_available: `{audit['spec_kit']['available']}`",
        f"- manual_fallback_recorded: `{audit['spec_kit']['manual_fallback_recorded']}`",
        f"- spark_first_recorded: `{audit['subagents']['spark_first_recorded']}`",
        f"- gpt55_fallback_recorded: `{audit['subagents']['gpt55_fallback_recorded']}`",
        f"- task_cards_recorded: `{audit['subagents']['task_cards_recorded']}`",
        f"- plan_spec_covered: `{audit['plan_spec']['plan_spec_covered']}`",
        "",
        "## Generated Evidence",
    ]
    for name, artifact in audit["generated_evidence"]["artifacts"].items():
        lines.append(f"- {name}: `{artifact['exists']}` `{artifact['path']}`")
    lines.extend(["", "## Decision"])
    lines.append(f"- {audit['decision']['reflected_when']}")
    lines.extend(["", "## Does Not Claim"])
    for item in audit["decision"]["does_not_claim"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, default=Path(f"generated/signoff/req2_spec_subagent_gate_audit_{DATE_TAG}.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path(f"generated/signoff/req2_spec_subagent_gate_audit_{DATE_TAG}.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    write_outputs(audit, args.json_out, args.markdown_out)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "spec_kit_available": audit["spec_kit"]["available"],
                "manual_fallback_recorded": audit["spec_kit"]["manual_fallback_recorded"],
                "spark_first_recorded": audit["subagents"]["spark_first_recorded"],
                "gpt55_fallback_recorded": audit["subagents"]["gpt55_fallback_recorded"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
