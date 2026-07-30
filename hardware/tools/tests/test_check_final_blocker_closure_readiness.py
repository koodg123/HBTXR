#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_final_blocker_closure_readiness as readiness_tool  # noqa: E402
import create_xr_vits_replacement_policy  # noqa: E402
from validate_pynq_smoke_result import EXPECTED_RAW  # noqa: E402


def smoke_payload() -> dict[str, object]:
    return {
        "status": "pass",
        "variant": "c3b-mem16",
        "weights_mode": "file",
        "expected_runtime_state": 2,
        "runtime_state": 2,
        "runtime_match": True,
        "expected_out_raw": EXPECTED_RAW,
        "out_raw": EXPECTED_RAW,
        "output_match": True,
        "out_state": EXPECTED_RAW,
        "dma_in_name": "axi_dma_0",
        "dma_out_name": "axi_dma_0",
        "bitfile": "/tmp/hgtxr_e2e_axis_dma_c3b_mem16.bit",
        "hwhfile": "/tmp/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_candidate_audit(root: Path, replacement: Path) -> None:
    write_json(
        root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
        {
            "requested_path": str(create_xr_vits_replacement_policy.default_requested_path(root).resolve()),
            "recommendation": {"path": str(replacement.resolve()), "role": "candidate-xr-accel"},
        },
    )


def write_integrity_policy(root: Path, replacement: Path) -> dict[str, object]:
    audit_path = root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
    audit = create_xr_vits_replacement_policy.load_candidate_audit(audit_path)
    policy = create_xr_vits_replacement_policy.build_policy(
        requested_path=create_xr_vits_replacement_policy.default_requested_path(root).resolve(),
        replacement_path=replacement.resolve(),
        approved_by="unit-test",
        approved_at="2026-06-10T00:00:00Z",
        reason="unit test approval",
        reason_code=create_xr_vits_replacement_policy.DEFAULT_REASON_CODE,
        candidate_audit_rel=create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
        candidate_audit_path=audit_path.resolve(),
        audit=audit,
    )
    write_json(root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL, policy)
    return policy


class CheckFinalBlockerClosureReadinessTests(unittest.TestCase):
    def test_current_missing_reports_blocked_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            root.mkdir(parents=True)
            result = readiness_tool.build_readiness(
                root=root,
                c3b_smoke_json=None,
                c3b_require_paths=True,
                xr_vits_mode="exact",
                approved_by="",
                reason="",
                replacement_path=None,
            )

            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["current_ready"])
            self.assertIn("C3b AXIS/DMA physical smoke result", result["missing_current_blockers"])
            self.assertIn("requested XR-VITs sibling", result["missing_current_blockers"])
            self.assertEqual(result["pynq_discovery"]["c3b"]["preset"], "axis-c3b-mem16")
            self.assertEqual(result["pynq_discovery"]["c3b"]["status"], "missing")
            self.assertEqual(result["pynq_discovery"]["c3b"]["candidate_count"], 0)
            self.assertEqual(result["pynq_discovery"]["vref_p0"]["preset"], "axis-vref-p0-softmax-input-x2-dsp-mixed-stream")
            self.assertEqual(result["pynq_discovery"]["qkv_uram"]["preset"], "axis-vref-p0-softmax-input-x2-qkv-uram")
            self.assertFalse(result["pynq_discovery"]["qkv_uram"]["safety"]["writes_canonical_inputs"])
            plan = result["operator_unblock_plan"]
            self.assertEqual(plan["status"], "needs-external-inputs")
            self.assertEqual(len(plan["required_inputs"]), 2)
            self.assertFalse(plan["required_inputs"][0]["ready"])
            self.assertIn("axis-c3b-mem16", "; ".join(plan["required_inputs"][0]["acceptance"]))
            self.assertIn("USER", plan["dry_run_sequence"][1]["command"])
            self.assertIn("--dry-run-xr-vits-replacement", plan["dry_run_sequence"][1]["command"])
            self.assertFalse(plan["dry_run_sequence"][1]["side_effects"])
            self.assertFalse(plan["safety"]["writes_canonical_inputs"])
            self.assertFalse(result["safety"]["writes_canonical_inputs"])
            self.assertTrue(result["safety"]["discovers_pynq_candidates_only"])

    def test_current_canonical_evidence_ready_to_run_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            xr_vits = Path(tmp) / "XR-VITs"
            xr_vits.mkdir(parents=True)
            write_json(readiness_tool.canonical_c3b_path(root), smoke_payload())

            result = readiness_tool.build_readiness(
                root=root,
                c3b_smoke_json=None,
                c3b_require_paths=True,
                xr_vits_mode="exact",
                approved_by="",
                reason="",
                replacement_path=None,
            )

            self.assertEqual(result["status"], "ready-to-run-final")
            self.assertTrue(result["current_ready"])
            self.assertEqual(result["missing_current_blockers"], [])
            self.assertIn("run_third_goal_final_signoff.py", result["final_runner_command"])
            self.assertEqual(result["dry_run_final_runner_command"], result["final_runner_command"])

    def test_current_canonical_no_require_paths_mode_controls_current_c3b_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            xr_vits = Path(tmp) / "XR-VITs"
            xr_vits.mkdir(parents=True)
            payload = smoke_payload()
            payload.pop("bitfile")
            payload.pop("hwhfile")
            write_json(readiness_tool.canonical_c3b_path(root), payload)

            relaxed = readiness_tool.build_readiness(
                root=root,
                c3b_smoke_json=None,
                c3b_require_paths=False,
                xr_vits_mode="exact",
                approved_by="",
                reason="",
                replacement_path=None,
            )
            strict = readiness_tool.build_readiness(
                root=root,
                c3b_smoke_json=None,
                c3b_require_paths=True,
                xr_vits_mode="exact",
                approved_by="",
                reason="",
                replacement_path=None,
            )

            self.assertEqual(relaxed["current"]["c3b_smoke"]["status"], "pass")
            self.assertTrue(relaxed["current_ready"])
            self.assertEqual(relaxed["status"], "ready-to-run-final")
            self.assertEqual(strict["current"]["c3b_smoke"]["status"], "fail")
            self.assertFalse(strict["current_ready"])
            self.assertIn("C3b AXIS/DMA physical smoke result", strict["missing_current_blockers"])

    def test_candidate_smoke_and_replacement_would_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)
            candidate = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            write_json(candidate, smoke_payload())

            result = readiness_tool.build_readiness(
                root=root,
                c3b_smoke_json=candidate,
                c3b_require_paths=True,
                xr_vits_mode="replacement",
                approved_by="unit-test",
                reason="unit test approval",
                replacement_path=replacement,
            )

            self.assertEqual(result["status"], "would-clear-with-candidates")
            self.assertFalse(result["current_ready"])
            self.assertTrue(result["candidate_ready"])
            plan = result["operator_unblock_plan"]
            self.assertEqual(plan["status"], "ready-with-candidates")
            self.assertTrue(all(item["ready"] for item in plan["required_inputs"]))
            self.assertTrue(plan["dry_run_sequence"][2]["ready_to_execute"])
            self.assertIn("--dry-run-import-c3b-smoke", plan["dry_run_sequence"][2]["command"])
            self.assertIn("--dry-run-xr-vits-replacement", plan["dry_run_sequence"][2]["command"])
            self.assertTrue(plan["active_sequence"][0]["ready_to_execute"])
            self.assertTrue(plan["active_sequence"][0]["side_effects"])
            self.assertIn("--import-c3b-smoke-json", result["final_runner_command"])
            self.assertIn("--approve-xr-vits-replacement", result["final_runner_command"])
            self.assertIn("--xr-vits-replacement-path", result["final_runner_command"])
            self.assertIn(str(replacement), result["final_runner_command"])
            self.assertIn("--dry-run-import-c3b-smoke", result["dry_run_final_runner_command"])
            self.assertIn("--dry-run-xr-vits-replacement", result["dry_run_final_runner_command"])
            self.assertIn("--allow-blocked", result["dry_run_final_runner_command"])
            self.assertIn("--xr-vits-replacement-path", result["dry_run_final_runner_command"])
            self.assertIn(str(replacement), result["dry_run_final_runner_command"])

    def test_integrity_policy_can_clear_current_xr_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)
            write_integrity_policy(root, replacement)

            xr = readiness_tool.check_current_xr_vits(root)

            self.assertEqual(xr["status"], "pass")
            self.assertTrue(xr["would_clear"])
            self.assertEqual(xr["mode"], "replacement-policy")
            self.assertEqual(xr["replacement_policy"]["integrity"]["status"], "pass")

    def test_legacy_policy_does_not_clear_current_xr_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)
            write_json(
                root / create_xr_vits_replacement_policy.DEFAULT_POLICY_REL,
                {
                    "approved_replacement": True,
                    "requested_path": str(create_xr_vits_replacement_policy.default_requested_path(root).resolve()),
                    "replacement_path": str(replacement.resolve()),
                    "replacement_role": "candidate-xr-accel",
                    "candidate_audit": create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL,
                    "approved_by": "unit-test",
                    "approved_at": "2026-06-10T00:00:00Z",
                    "reason": "legacy",
                },
            )

            xr = readiness_tool.check_current_xr_vits(root)

            self.assertEqual(xr["status"], "missing")
            self.assertFalse(xr["would_clear"])
            self.assertEqual(xr["replacement_policy"]["integrity"]["status"], "legacy-warning")
            self.assertIn("regenerate policy", "; ".join(xr["errors"]))

    def test_policy_drift_does_not_clear_current_xr_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            replacement = Path(tmp) / "XR-VIT" / "XR_Accel"
            replacement.mkdir(parents=True)
            write_candidate_audit(root, replacement)
            write_integrity_policy(root, replacement)
            audit_path = root / create_xr_vits_replacement_policy.DEFAULT_AUDIT_REL
            audit = json.loads(audit_path.read_text())
            audit["recommendation"]["role"] = "tampered"
            write_json(audit_path, audit)

            xr = readiness_tool.check_current_xr_vits(root)

            self.assertEqual(xr["status"], "missing")
            self.assertFalse(xr["would_clear"])
            self.assertEqual(xr["replacement_policy"]["integrity"]["status"], "fail")
            self.assertIn("candidate_audit_fingerprint mismatch", "; ".join(xr["errors"]))

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "XR-VIT" / "HGTXR"
            json_out = root / "generated" / "readiness.json"
            markdown_out = root / "generated" / "readiness.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = readiness_tool.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 1)
            self.assertEqual(json.loads(json_out.read_text())["status"], "blocked")
            self.assertIn("HGTXR Final Blocker Closure Readiness", markdown_out.read_text())
            self.assertIn("PYNQ Discovery", markdown_out.read_text())
            self.assertIn("qkv_uram", markdown_out.read_text())
            self.assertIn("Operator Unblock Plan", markdown_out.read_text())
            self.assertIn("Dry-Run Sequence", markdown_out.read_text())
            self.assertIn("Dry-Run Final Runner Command", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
