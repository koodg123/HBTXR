#!/usr/bin/env python3
"""Audit that planning/spec artifacts cover the active third-goal contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_10"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SELF_GATED_EVIDENCE_CHECKS = {
    "spec_plan_conformance_pass",
    "spec_plan_current_doc_freshness_contract",
    "third_goal_completion_audit_status_known",
    "third_goal_completion_audit_item_count",
    "third_goal_completion_audit_counts_match_items",
    "third_goal_completion_audit_req12_manifest_pass",
    "third_goal_completion_audit_req11_matches_trace",
    "third_goal_completion_audit_req4_matches_trace",
}

DOCS = {
    "master_plan": "docs/Master-Plan.md",
    "sub_plan": "docs/Sub-Plan.md",
    "spec": "docs/Spec.md",
    "execution": "docs/Execution.md",
    "validation": "docs/Validation.md",
    "progress": "docs/track/PROGRESS.md",
    "current_handover": "docs/track/HANDOVER.md",
    "handover": "docs/track/HANDOVER-2026-06-10-E2E.md",
    "choice": "docs/track/CHOICE.md",
    "log": "docs/track/log.md",
}


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def contains_number_with_context(text: str, value: int, context: str) -> bool:
    return f"`{value}`" in text and context in text


def choice_doc_path(base: Path) -> Path:
    direct = base / "docs" / "CHOICE.md"
    return direct if direct.exists() else base / "docs" / "track" / "CHOICE.md"


def failed_manifest_consistency(evidence_manifest: dict[str, Any] | None) -> list[str]:
    if not isinstance(evidence_manifest, dict):
        return []
    failed = evidence_manifest.get("failed_consistency_checks", [])
    if isinstance(failed, list):
        return [str(name) for name in failed]
    checks = evidence_manifest.get("consistency_checks", [])
    if not isinstance(checks, list):
        return []
    return [
        str(check.get("name", "unknown"))
        for check in checks
        if isinstance(check, dict) and check.get("status") == "fail"
    ]


def evidence_manifest_effectively_passes(evidence_manifest: dict[str, Any] | None) -> bool:
    if not isinstance(evidence_manifest, dict):
        return False
    required_complete = evidence_manifest.get("present_required_count") == evidence_manifest.get("required_count")
    external_failed = [
        name
        for name in failed_manifest_consistency(evidence_manifest)
        if name not in BOOTSTRAP_SELF_GATED_EVIDENCE_CHECKS
    ]
    return required_complete and (evidence_manifest.get("status") == "pass" or not external_failed)


def build_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    checks: list[dict[str, Any]] = []
    docs = {name: root / rel for name, rel in DOCS.items()}
    docs["choice"] = choice_doc_path(root)
    texts = {name: read_text(path) for name, path in docs.items()}

    for name, path in docs.items():
        add_check(checks, f"{name}_exists", path.exists(), str(path))

    spec = texts["spec"]
    master = texts["master_plan"]
    sub = texts["sub_plan"]
    execution = texts["execution"]
    validation = texts["validation"]
    progress = texts["progress"]
    current_handover = texts["current_handover"]
    handover = texts["handover"]
    choice = texts["choice"]
    log = texts["log"]
    current_doc_base = root / "hardware" if (root / "hardware" / "docs" / "Spec.md").exists() else root
    current_doc_paths = {
        "master_plan": current_doc_base / "docs" / "Master-Plan.md",
        "sub_plan": current_doc_base / "docs" / "Sub-Plan.md",
        "spec": current_doc_base / "docs" / "Spec.md",
        "validation": current_doc_base / "docs" / "Validation.md",
        "progress": current_doc_base / "docs" / "track" / "PROGRESS.md",
        "current_handover": current_doc_base / "docs" / "track" / "HANDOVER.md",
        "choice": choice_doc_path(current_doc_base),
        "log": current_doc_base / "docs" / "track" / "log.md",
    }
    current_doc_texts = {name: read_text(path) for name, path in current_doc_paths.items()}

    add_check(checks, "spec_records_spec_kit_unavailable", "spec-kit" in spec and "manual" in spec, "spec-kit/manual")
    add_check(checks, "spec_records_zcu104", "ZCU104" in spec, "ZCU104")
    add_check(checks, "spec_records_q4w_q8a", "Q4W/Q8A" in spec or "Q4W8A" in spec, "Q4W/Q8A")
    add_check(checks, "spec_records_param_knobs", contains_all(spec, ["HGTXR_PARALLELISM_FACTOR", "HGTXR_BUS_WIDTH", "HGTXR_FIFO_DEPTH"]), "parallelism/bus/fifo")
    add_check(checks, "spec_records_a2_a1_c", contains_all(spec, ["A2", "A1", "PAR=16"]), "A2/A1/PAR16")

    add_check(checks, "master_records_selected_paths", contains_all(master, ["Path 1 = A2 then A1", "Path 2 = C", "E pending"]), "selected paths")
    add_check(checks, "sub_plan_records_latest_task", "task_id: T-033" in sub, "T-033")
    add_check(checks, "execution_records_xilinx_root", "/tools/Xilinx" in execution, "/tools/Xilinx")
    add_check(checks, "validation_records_selected_path_audit", "Selected Path Execution Audit" in validation, "Selected Path Execution Audit")
    add_check(checks, "progress_records_selected_path_audit", "Latest Continuation: Selected Path Execution Audit" in progress, "progress T-033")
    add_check(checks, "handover_records_e2e_context", contains_all(handover, ["E2E Q4W/Q8A", "ZCU104", "active196_b6_ff768"]), "handover E2E")
    add_check(checks, "choice_records_e_pending", contains_all(choice, ["E: pending", "Selected Path Execution Audit"]), "choice E pending")
    add_check(checks, "log_records_latest_task", "Selected path execution audit" in log, "log T-033")

    resources = root / "docs" / "resources"
    requirements_trace = load_json_object(resources / f"third_goal_requirements_trace_{DATE_TAG}.json")
    completion_audit = load_json_object(resources / f"third_goal_completion_audit_{DATE_TAG}.json")
    evidence_manifest = load_json_object(resources / f"final_evidence_manifest_{DATE_TAG}.json")
    selected_path = load_json_object(resources / f"selected_path_execution_audit_{DATE_TAG}.json")
    resource_policy = load_json_object(resources / f"e2e_resource_policy_audit_{DATE_TAG}.json")
    source_audit = load_json_object(resources / "third_goal_source_audit_2026_06_16.json")
    current_audit = load_json_object(resources / "third_goal_current_audit_2026_06_16.json")
    operator_handoff_validation = load_json_object(resources / f"final_operator_handoff_validation_{DATE_TAG}.json")
    bundle_validation = load_json_object(resources / f"final_signoff_bundle_validation_{DATE_TAG}.json")

    requirements = requirements_trace.get("requirements", []) if isinstance(requirements_trace, dict) else []
    requirement_ids = {str(item.get("id")) for item in requirements if isinstance(item, dict)}
    add_check(checks, "requirements_trace_status_known", isinstance(requirements_trace, dict) and requirements_trace.get("status") in {"pass", "partial", "blocked"}, str(requirements_trace.get("status") if isinstance(requirements_trace, dict) else "missing"))
    add_check(checks, "requirements_trace_has_12_requirements", len(requirements) == 12, str(len(requirements)))
    add_check(checks, "requirements_trace_ids_0_to_11", requirement_ids == {str(i) for i in range(12)}, ",".join(sorted(requirement_ids)))
    add_check(checks, "requirements_trace_has_manifest_contract", isinstance(requirements_trace, dict) and isinstance(requirements_trace.get("final_evidence_manifest_contract"), dict), str(type(requirements_trace.get("final_evidence_manifest_contract")).__name__ if isinstance(requirements_trace, dict) else "missing"))

    add_check(checks, "completion_audit_status_known", isinstance(completion_audit, dict) and completion_audit.get("status") in {"pass", "partial", "blocked"}, str(completion_audit.get("status") if isinstance(completion_audit, dict) else "missing"))
    add_check(checks, "completion_audit_has_14_items", isinstance(completion_audit, dict) and completion_audit.get("item_count") == 14, str(completion_audit.get("item_count") if isinstance(completion_audit, dict) else "missing"))
    add_check(
        checks,
        "evidence_manifest_pass",
        evidence_manifest_effectively_passes(evidence_manifest),
        (
            f"status={evidence_manifest.get('status')} "
            f"external_failed="
            f"{[name for name in failed_manifest_consistency(evidence_manifest) if name not in BOOTSTRAP_SELF_GATED_EVIDENCE_CHECKS]}"
            if isinstance(evidence_manifest, dict)
            else "missing"
        ),
    )
    add_check(checks, "evidence_manifest_required_complete", isinstance(evidence_manifest, dict) and evidence_manifest.get("present_required_count") == evidence_manifest.get("required_count"), f"{evidence_manifest.get('present_required_count') if isinstance(evidence_manifest, dict) else 'missing'}/{evidence_manifest.get('required_count') if isinstance(evidence_manifest, dict) else 'missing'}")
    add_check(checks, "selected_path_audit_pass", isinstance(selected_path, dict) and selected_path.get("status") == "pass", str(selected_path.get("status") if isinstance(selected_path, dict) else "missing"))
    add_check(checks, "resource_policy_audit_pass", isinstance(resource_policy, dict) and resource_policy.get("status") == "pass", str(resource_policy.get("status") if isinstance(resource_policy, dict) else "missing"))

    consistency_checks = (
        evidence_manifest.get("consistency_checks", [])
        if isinstance(evidence_manifest, dict) and isinstance(evidence_manifest.get("consistency_checks"), list)
        else []
    )
    manifest_required = (
        f"{evidence_manifest.get('present_required_count')}/{evidence_manifest.get('required_count')}"
        if isinstance(evidence_manifest, dict)
        else "missing"
    )
    consistency_count = len(consistency_checks)
    source_count = source_audit.get("source_count") if isinstance(source_audit, dict) else None
    source_required = source_audit.get("required_count") if isinstance(source_audit, dict) else None
    operator_handoff_check_count = (
        operator_handoff_validation.get("check_count")
        if isinstance(operator_handoff_validation, dict)
        else None
    )
    operator_handoff_pass_count = (
        operator_handoff_validation.get("pass_count")
        if isinstance(operator_handoff_validation, dict)
        else None
    )
    bundle_check_count = bundle_validation.get("check_count") if isinstance(bundle_validation, dict) else None
    bundle_pass_count = bundle_validation.get("pass_count") if isinstance(bundle_validation, dict) else None
    operator_handoff_target = (
        f"{operator_handoff_check_count}/{operator_handoff_check_count}"
        if operator_handoff_check_count is not None
        else None
    )
    bundle_target = f"{bundle_check_count}/{bundle_check_count}" if bundle_check_count is not None else None
    current_summary = (
        current_audit.get("summary", {})
        if isinstance(current_audit, dict) and isinstance(current_audit.get("summary"), dict)
        else {}
    )
    current_summary_text = (
        f"reflected `{current_summary.get('reflected')}`, "
        f"partial `{current_summary.get('partial')}`, "
        f"blocked `{current_summary.get('blocked')}`"
    )
    current_docs = current_doc_texts
    for doc_name, text in current_docs.items():
        add_check(
            checks,
            f"{doc_name}_records_live_manifest_required",
            manifest_required in text,
            manifest_required,
        )
        add_check(
            checks,
            f"{doc_name}_records_live_manifest_consistency",
            contains_number_with_context(text, consistency_count, "consistency"),
            str(consistency_count),
        )
        add_check(
            checks,
            f"{doc_name}_records_live_source_count",
            source_count is not None and source_required is not None
            and f"required `{source_required}`" in text
            and f"sources `{source_count}`" in text,
            f"required={source_required} sources={source_count}",
        )
        add_check(
            checks,
            f"{doc_name}_records_live_current_summary",
            current_summary_text in text,
            current_summary_text,
        )
        add_check(
            checks,
            f"{doc_name}_records_live_operator_handoff_validation",
            operator_handoff_target is not None
            and f"operator handoff validation `{operator_handoff_target}`" in text,
            str(operator_handoff_target),
        )
        add_check(
            checks,
            f"{doc_name}_records_live_bundle_validation",
            bundle_target is not None
            and f"bundle validation `{bundle_target}`" in text,
            str(bundle_target),
        )
    handover_text = current_docs.get("current_handover", "")
    add_check(
        checks,
        "current_handover_records_live_operator_handoff_validation_artifact_summary",
        operator_handoff_check_count is not None
        and operator_handoff_pass_count is not None
        and (
            f"final_operator_handoff_validation_{DATE_TAG}.md`, status `pass`, "
            f"checks `{operator_handoff_check_count}`, fail `0`"
        )
        in handover_text,
        f"checks={operator_handoff_check_count} pass={operator_handoff_pass_count}",
    )
    add_check(
        checks,
        "current_handover_records_live_bundle_validation_artifact_summary",
        bundle_check_count is not None
        and bundle_pass_count is not None
        and (
            f"final_signoff_bundle_validation_{DATE_TAG}.md`, status `pass`, "
            f"checks `{bundle_check_count}`, fail `0`"
        )
        in handover_text,
        f"checks={bundle_check_count} pass={bundle_pass_count}",
    )
    stale_current_tokens = [
        "manifest `66/66`",
        "required `66/66`",
        "final manifest `66/66`",
        "consistency count `72`",
        "consistency checks `72`",
        "consistency count `326`",
        "consistency checks `326`",
        "consistency count `332`",
        "consistency checks `332`",
        "consistency count `337`",
        "consistency checks `337`",
        "consistency count `340`",
        "consistency checks `340`",
        "sources `199`",
    ]
    stale_checked_docs = {
        name: text
        for name, text in current_docs.items()
        if name not in {"validation", "choice", "log"}
    }
    stale_hits = [
        f"{doc_name}:{token}"
        for doc_name, text in stale_checked_docs.items()
        for token in stale_current_tokens
        if token in text
    ]
    add_check(checks, "current_docs_have_no_stale_final_evidence_counts", stale_hits == [], str(stale_hits))
    stale_validator_tokens = [
        "operator handoff validation `44/44`",
        "operator handoff validation `46/46`",
        "operator handoff validation `52/52`",
        "operator handoff validation `54/54`",
        "operator handoff validation `58/58`",
        "operator handoff validation `60/60`",
        "operator handoff validation `62/62`",
        "operator handoff validation `66/66`",
        "operator handoff validation `72/72`",
        "operator handoff validation `79/79`",
        "operator handoff validation `85/85`",
        "operator handoff validation `92/92`",
        "operator handoff validation `101/101`",
        "operator handoff validation `108/108`",
        "operator handoff validation `116/116`",
        "operator handoff validation `127/127`",
        "operator handoff validation `129/129`",
        "bundle validation `74/74`",
        "bundle validation `80/80`",
        "bundle validation `87/87`",
        "bundle validation `95/95`",
        "bundle validation `102/102`",
        "bundle validation `110/110`",
        "bundle validation `120/120`",
        "bundle validation `128/128`",
        "bundle validation `137/137`",
        "bundle validation `150/150`",
    ]
    stale_validator_hits = [
        f"{doc_name}:{token}"
        for doc_name, text in stale_checked_docs.items()
        for token in stale_validator_tokens
        if token in text
    ]
    add_check(
        checks,
        "current_docs_have_no_stale_validator_counts",
        stale_validator_hits == [],
        str(stale_validator_hits),
    )
    stale_policy_tokens = [
        "operator_handoff_policy_check_count=7",
        "operator_handoff_policy_check_count=8",
        "operator_handoff_policy_check_count: `7`",
        "operator_handoff_policy_check_count: `8`",
        "policy check count `7`",
        "policy check count `8`",
        "policy check count: `7`",
        "policy check count: `8`",
        "policy check count=7",
        "policy check count=8",
        "policy-check count `7`",
        "policy-check count `8`",
    ]
    stale_policy_hits = [
        f"{doc_name}:{token}"
        for doc_name, text in stale_checked_docs.items()
        for token in stale_policy_tokens
        if token in text
    ]
    add_check(
        checks,
        "current_docs_have_no_stale_xr_policy_check_count",
        stale_policy_hits == [],
        str(stale_policy_hits),
    )

    fail_count = sum(1 for check in checks if check["status"] != "pass")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "root": str(root),
        "date_tag": DATE_TAG,
        "scope": "third-goal planning, spec, execution, validation, and trace conformance",
        "documents": {name: str(path) for name, path in docs.items()},
        "observed": {
            "requirements_trace_status": requirements_trace.get("status") if isinstance(requirements_trace, dict) else "missing",
            "requirements_count": len(requirements),
            "completion_status": completion_audit.get("status") if isinstance(completion_audit, dict) else "missing",
            "evidence_manifest_status": evidence_manifest.get("status") if isinstance(evidence_manifest, dict) else "missing",
            "evidence_required": evidence_manifest.get("required_count") if isinstance(evidence_manifest, dict) else None,
            "evidence_consistency_count": consistency_count,
            "source_required_count": source_required,
            "source_count": source_count,
            "operator_handoff_validation_check_count": operator_handoff_check_count,
            "final_signoff_bundle_validation_check_count": bundle_check_count,
            "current_audit_summary": current_summary,
            "current_doc_base": str(current_doc_base),
            "selected_path_status": selected_path.get("status") if isinstance(selected_path, dict) else "missing",
            "resource_policy_status": resource_policy.get("status") if isinstance(resource_policy, dict) else "missing",
        },
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "checks": checks,
        "safety": {
            "executes_hls_or_vivado": False,
            "executes_board_smoke": False,
            "creates_board_result": False,
            "creates_xr_vits_policy": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# HGTXR Spec/Plan Conformance Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- scope: {audit['scope']}",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        "",
        "## Observed",
        "",
    ]
    for key, value in audit["observed"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Documents", ""])
    for name, path in audit["documents"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Does not run HLS or Vivado.",
            "- Does not run board smoke.",
            "- Does not create board result JSON.",
            "- Does not create XR-VITs replacement policy.",
            "- Does not write canonical unblock inputs.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HGTXR third-goal spec/plan conformance.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(f"[spec-plan-conformance-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
