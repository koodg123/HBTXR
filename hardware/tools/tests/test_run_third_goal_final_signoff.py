#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_third_goal_final_signoff as runner  # noqa: E402


class RunThirdGoalFinalSignoffTests(unittest.TestCase):
    def make_paths(self, root: Path) -> dict[str, Path]:
        return runner.default_paths(root)

    def make_tools(self, *, blocked: bool, calls: list[str]) -> dict[str, runner.ToolMain]:
        def c3b_import(argv: list[str]) -> int:
            preset = argv[argv.index("--preset") + 1]
            if preset == "axis-vref-p0-softmax-input-x2-qkv-uram":
                calls.append("qkv_import")
            elif preset == "axis-vref-p0-softmax-input-x2-dsp-mixed-stream":
                calls.append("vref_import")
            else:
                calls.append("c3b_import")
            json_out = Path(argv[argv.index("--json-out") + 1])
            validation_out = Path(argv[argv.index("--validation-out") + 1])
            dry_run = "--dry-run" in argv
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "preset": preset, "copied": not dry_run, "dry_run": dry_run, "errors": []}) + "\n")
            validation_out.write_text(json.dumps({"status": "pass", "preset": preset, "dry_run": dry_run, "errors": []}) + "\n")
            return 0

        def zcu104_remote(argv: list[str]) -> int:
            profile = argv[argv.index("--profile") + 1] if "--profile" in argv else "c3b-mem16"
            if profile == "vref-p0-softmax-input-x2-qkv-uram":
                calls.append("qkv_remote")
            elif profile == "vref-p0-softmax-input-x2-dsp-mixed-stream":
                calls.append("vref_remote")
            else:
                calls.append("zcu104_remote")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            execute = "--execute" in argv
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass" if execute else "dry-run", "execute": execute, "profile": profile, "errors": []}) + "\n")
            markdown_out.write_text("# zcu104 remote\n")
            return 0

        def c3b_smoke_contract(argv: list[str]) -> int:
            calls.append("c3b_smoke_contract")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "preset": "axis-c3b-mem16",
                        "safety": {
                            "executes_commands": False,
                            "creates_board_result": False,
                            "creates_xr_vits_policy": False,
                            "writes_canonical_inputs": False,
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# c3b smoke contract\n")
            return 0

        def c3b_transfer_manifest(argv: list[str]) -> int:
            calls.append("c3b_transfer_manifest")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            sha256_out = Path(argv[argv.index("--sha256-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            sha256_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "variant": "c3b-mem16",
                        "preset": "axis-c3b-mem16",
                        "tar": {"sha256": "a" * 64},
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# c3b transfer manifest\n")
            sha256_out.write_text(("a" * 64) + "  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n")
            return 0

        def xr_vits_policy(argv: list[str]) -> int:
            self.assertIn("--approve", argv)
            if "--preview-json-out" in argv:
                calls.append("xr_vits_policy_preview")
                preview_json = Path(argv[argv.index("--preview-json-out") + 1])
                preview_md = Path(argv[argv.index("--preview-markdown-out") + 1])
                preview_json.parent.mkdir(parents=True, exist_ok=True)
                preview_json.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "preview_only": True,
                            "active_policy_written": False,
                            "integrity": {"status": "pass"},
                            "safety": {"creates_xr_vits_policy": False, "writes_canonical_inputs": False},
                        }
                    )
                    + "\n"
                )
                preview_md.write_text("# xr vits policy preview\n")
                return 0
            calls.append("xr_vits_policy")
            json_out = Path(argv[argv.index("--json-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"approved_replacement": "--dry-run" not in argv, "dry_run": "--dry-run" in argv}) + "\n")
            return 0

        def xr_vits_packet(argv: list[str]) -> int:
            calls.append("xr_vits_packet")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            status = "pending-user-choice" if blocked else "exact-restored"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": status}) + "\n")
            markdown_out.write_text("# xr vits packet\n")
            return 1 if blocked else 0

        def xr_vits_reference_resolution(argv: list[str]) -> int:
            calls.append("xr_vits_reference_resolution")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            status = "candidate-ready-needs-approval" if blocked else "exact-ready"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": status, "approval_required": blocked}) + "\n")
            markdown_out.write_text("# xr vits reference resolution\n")
            return 0

        def readiness(argv: list[str]) -> int:
            calls.append("readiness")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "ready-for-board" if blocked else "complete"}) + "\n")
            markdown_out.write_text("# readiness\n")
            return 0

        def preflight(argv: list[str]) -> int:
            calls.append("preflight")
            json_out = Path(argv[argv.index("--json-out") + 1])
            fail_count = 2 if blocked else 0
            checks = []
            if blocked:
                checks = [
                    {"status": "fail", "name": "C3b AXIS/DMA physical smoke result", "detail": "missing"},
                    {"status": "fail", "name": "requested XR-VITs sibling", "detail": "missing"},
                ]
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"mode": "final-signoff", "summary": {"ok": 62, "warn": 4, "fail": fail_count}, "checks": checks}) + "\n")
            return 1 if blocked else 0

        def final_audit(argv: list[str]) -> int:
            calls.append("final_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            blockers = []
            if blocked:
                blockers = [
                    {"name": "C3b AXIS/DMA physical smoke result"},
                    {"name": "requested XR-VITs sibling"},
                ]
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "blocked" if blocked else "pass", "blockers": blockers}) + "\n")
            markdown_out.write_text("# final audit\n")
            return 1 if blocked else 0

        def completion(argv: list[str]) -> int:
            calls.append("completion")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "blocked" if blocked else "pass"}) + "\n")
            markdown_out.write_text("# completion\n")
            return 1 if blocked else 0

        def unblock(argv: list[str]) -> int:
            calls.append("unblock")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pending-unblock" if blocked else "ready-for-final-signoff"}) + "\n")
            markdown_out.write_text("# unblock\n")
            return 1 if blocked else 0

        def command_card(argv: list[str]) -> int:
            calls.append("command_card")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pending-unblock" if blocked else "ready-for-final-signoff"}) + "\n")
            markdown_out.write_text("# command card\n")
            return 1 if blocked else 0

        def resource_matrix(argv: list[str]) -> int:
            calls.append("resource_matrix")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "summary": {"recommended_board_smoke_variant": "C3b"}}) + "\n")
            markdown_out.write_text("# resource matrix\n")
            return 0

        def resource_policy(argv: list[str]) -> int:
            calls.append("resource_policy")
            matrix_path = Path(argv[argv.index("--resource-matrix") + 1])
            self.assertTrue(matrix_path.exists())
            self.assertIn("generated/signoff", matrix_path.as_posix())
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "check_count": 22, "pass_count": 22, "fail_count": 0}) + "\n")
            markdown_out.write_text("# resource policy\n")
            return 0

        def selected_path(argv: list[str]) -> int:
            calls.append("selected_path")
            matrix_path = Path(argv[argv.index("--resource-matrix") + 1])
            self.assertTrue(matrix_path.exists())
            self.assertIn("generated/signoff", matrix_path.as_posix())
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "check_count": 22,
                        "pass_count": 22,
                        "fail_count": 0,
                        "selection": {"path_1": "A2 then A1", "path_2": "C", "e": "pending"},
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# selected path\n")
            return 0

        def requirements_trace(argv: list[str]) -> int:
            calls.append("requirements_trace")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            signoff_run = Path(argv[argv.index("--signoff-run") + 1])
            resource_matrix_path = Path(argv[argv.index("--resource-matrix") + 1])
            candidate_audit_path = Path(argv[argv.index("--candidate-audit") + 1])
            xr_vits_resolution_path = Path(argv[argv.index("--xr-vits-resolution") + 1])
            evidence_manifest_path = Path(argv[argv.index("--evidence-manifest") + 1])
            self.assertTrue(signoff_run.exists())
            self.assertTrue(resource_matrix_path.exists())
            self.assertTrue(candidate_audit_path.exists())
            self.assertTrue(xr_vits_resolution_path.exists())
            self.assertIn("generated/signoff", resource_matrix_path.as_posix())
            self.assertIn("generated/signoff", candidate_audit_path.as_posix())
            self.assertIn("generated/signoff", xr_vits_resolution_path.as_posix())
            self.assertIn("docs/resources", evidence_manifest_path.as_posix())
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "blocked" if blocked else "pass"}) + "\n")
            markdown_out.write_text("# requirements trace\n")
            return 1 if blocked else 0

        def spec_plan(argv: list[str]) -> int:
            calls.append("spec_plan")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "check_count": 32, "pass_count": 32, "fail_count": 0}) + "\n")
            markdown_out.write_text("# spec plan\n")
            return 0

        def candidate_audit(argv: list[str]) -> int:
            calls.append("candidate_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            if blocked:
                status = "blocked"
                remaining = ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"]
            else:
                status = "would-clear"
                remaining = []
            json_out.write_text(json.dumps({"status": status, "remaining_blockers": remaining}) + "\n")
            markdown_out.write_text("# candidate audit\n")
            return 1 if blocked else 0

        def operator_handoff(argv: list[str]) -> int:
            calls.append("operator_handoff")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            requirements_trace_path = Path(argv[argv.index("--requirements-trace") + 1])
            candidate_audit_path = Path(argv[argv.index("--candidate-audit") + 1])
            xr_vits_resolution_path = Path(argv[argv.index("--xr-vits-resolution") + 1])
            self.assertTrue(requirements_trace_path.exists())
            self.assertTrue(candidate_audit_path.exists())
            self.assertTrue(xr_vits_resolution_path.exists())
            self.assertIn("generated/signoff", requirements_trace_path.as_posix())
            self.assertIn("generated/signoff", candidate_audit_path.as_posix())
            self.assertIn("generated/signoff", xr_vits_resolution_path.as_posix())
            status = "pending-operator-actions" if blocked else "ready-for-final-signoff"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": status, "remaining_blockers": [] if not blocked else ["requested XR-VITs sibling"]}) + "\n")
            markdown_out.write_text("# operator handoff\n")
            return 1 if blocked else 0

        def operator_handoff_validation(argv: list[str]) -> int:
            calls.append("operator_handoff_validation")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            handoff_path = Path(argv[argv.index("--handoff") + 1])
            self.assertTrue(handoff_path.exists())
            self.assertIn("generated/signoff", handoff_path.as_posix())
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "fail_count": 0}) + "\n")
            markdown_out.write_text("# operator handoff validation\n")
            return 0

        def bundle_validation(argv: list[str]) -> int:
            calls.append("bundle_validation")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "fail_count": 0}) + "\n")
            markdown_out.write_text("# bundle validation\n")
            return 0

        def c3b_smoke_discovery(argv: list[str]) -> int:
            calls.append("c3b_smoke_discovery")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            status = "missing" if blocked else "found"
            json_out.write_text(json.dumps({"status": status, "pass_count": 0 if blocked else 1}) + "\n")
            markdown_out.write_text("# c3b discovery\n")
            return 1 if blocked else 0

        def pynq_smoke_discovery(argv: list[str]) -> int:
            preset = argv[argv.index("--preset") + 1]
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            if preset == "axis-c3b-mem16":
                calls.append("pynq_c3b_smoke_discovery")
            elif preset == "axis-vref-p0-softmax-input-x2-qkv-uram":
                calls.append("qkv_smoke_discovery")
            elif "pynq_smoke_candidate_discovery_vref_p0" in json_out.name:
                calls.append("pynq_vref_smoke_discovery")
            else:
                calls.append("vref_smoke_discovery")
            self.assertIn(
                preset,
                {
                    "axis-c3b-mem16",
                    "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "axis-vref-p0-softmax-input-x2-qkv-uram",
                },
            )
            json_out.parent.mkdir(parents=True, exist_ok=True)
            status = "missing" if blocked else "found"
            json_out.write_text(json.dumps({"status": status, "pass_count": 0 if blocked else 1}) + "\n")
            markdown_out.write_text("# vref discovery\n")
            return 1 if blocked else 0

        def blocker_closure(argv: list[str]) -> int:
            calls.append("blocker_closure")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            status = "blocked" if blocked else "ready-to-run-final"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": status, "current_ready": not blocked, "candidate_ready": False}) + "\n")
            markdown_out.write_text("# blocker closure\n")
            return 1 if blocked else 0

        def unblock_intake(argv: list[str]) -> int:
            calls.append("unblock_intake")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            status = "blocked" if blocked else "ready-for-active-unblock"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": status,
                        "dry_run_final_runner_command": "",
                        "active_final_runner_command": "",
                        "safety": {
                            "executes_commands": False,
                            "creates_board_result": False,
                            "creates_xr_vits_policy": False,
                            "writes_canonical_inputs": False,
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# unblock intake\n")
            return 1 if blocked else 0

        def evidence_manifest(argv: list[str]) -> int:
            calls.append("evidence_manifest")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "required_count": 1, "present_required_count": 1}) + "\n")
            markdown_out.write_text("# evidence manifest\n")
            return 0

        def source_audit(argv: list[str]) -> int:
            calls.append("source_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "required_count": 66, "source_count": 175}) + "\n")
            markdown_out.write_text("# source audit\n")
            return 0

        def current_audit(argv: list[str]) -> int:
            calls.append("current_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            status = "blocked-external" if blocked else "pass"
            json_out.write_text(json.dumps({"status": status, "summary": {"blocked": 1 if blocked else 0}}) + "\n")
            markdown_out.write_text("# current audit\n")
            return 0

        def qkv_uram_successor(argv: list[str]) -> int:
            calls.append("qkv_uram_successor")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "csynth": {"resources": {"bram_18k": 114, "dsp": 128, "lut": 43236, "uram": 40}},
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# qkv uram successor\n")
            return 0

        def req5_q4q8_swhw(argv: list[str]) -> int:
            calls.append("req5_q4q8_swhw")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "precision": {"weight_bits": 4, "activation_bits": 8},
                        "pass_count": 6,
                        "fail_count": 0,
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# req5 q4q8\n")
            return 0

        def p2_vit_scale_calibration(argv: list[str]) -> int:
            calls.append("p2_vit_scale_calibration")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "policy": {"decision": "keep_current_pot_scales_for_c3b"},
                        "check_count": 10,
                        "pass_count": 10,
                        "fail_count": 0,
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# p2 vit scale calibration\n")
            return 0

        def vref_p0_pot_scale_audit(argv: list[str]) -> int:
            calls.append("vref_p0_pot_scale_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps({"status": "pass", "check_count": 14, "pass_count": 14, "fail_count": 0}) + "\n"
            )
            markdown_out.write_text("# vref p0 pot scale audit\n")
            return 0

        def vref_p0_pot_scale_sweep(argv: list[str]) -> int:
            calls.append("vref_p0_pot_scale_sweep")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "spec_count": 3,
                        "summary": {
                            "total_candidate_count": 45,
                            "total_fail_count": 0,
                            "specs_recommending_current": 3,
                            "next_action": "keep current PoT scales for C3b",
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# vref p0 pot scale sweep\n")
            return 0

        def req6_parameterization(argv: list[str]) -> int:
            calls.append("req6_parameterization")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "knobs": {
                            "config_macros": {
                                "HGTXR_TILING_FACTOR": 1,
                                "HGTXR_PARALLELISM_FACTOR": 8,
                                "HGTXR_BUS_WIDTH": 256,
                                "HGTXR_BIT_WIDTH": 8,
                                "HGTXR_WEIGHT_BIT_WIDTH": 4,
                                "HGTXR_BUFFER_SIZE": 256,
                                "HGTXR_FIFO_DEPTH": 128,
                            }
                        },
                        "pass_count": 41,
                        "fail_count": 0,
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# req6 parameterization\n")
            return 0

        def req1_environment(argv: list[str]) -> int:
            calls.append("req1_environment")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "check_count": 13,
                        "pass_count": 13,
                        "fail_count": 0,
                        "checks": [
                            {"name": "platform_linux", "status": "pass", "detail": "Linux"},
                            {"name": "ubuntu_id", "status": "pass", "detail": "ubuntu"},
                            {"name": "ubuntu_version_22_04", "status": "pass", "detail": "22.04"},
                            {"name": "not_wsl_kernel", "status": "pass", "detail": "generic"},
                            {"name": "hardware_root_expected_prefix", "status": "pass", "detail": "/home/kjm26/project"},
                            {"name": "no_legacy_wsl_or_xilinx_paths", "status": "pass", "detail": []},
                            {"name": "vitis_hls_tools_xilinx_executable", "status": "pass", "detail": "/tools/Xilinx"},
                            {"name": "vivado_tools_xilinx_executable", "status": "pass", "detail": "/tools/Xilinx"},
                        ],
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# req1 environment\n")
            return 0

        def req9_deit_image(argv: list[str]) -> int:
            calls.append("req9_deit_image")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "check_count": 11,
                        "pass_count": 11,
                        "fail_count": 0,
                        "requested_image": {"sha256": "a" * 64, "exists": True},
                        "checks": [
                            {"name": "requested_image_exists", "status": "pass", "detail": "ok"},
                            {"name": "requested_image_png_magic", "status": "pass", "detail": "89504e470d0a1a0a"},
                            {"name": "hardware_docs_copy_matches_requested_sha256", "status": "pass", "detail": "ok"},
                        ],
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# req9 deit image\n")
            return 0

        def xr_vits_gate(argv: list[str]) -> int:
            calls.append("xr_vits_gate")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "resolution_mode": "candidate-ready-needs-approval",
                        "exact": {"exists": False, "path": "/tmp/XR-VITs"},
                        "replacement_candidate": {"exists": True, "path": "/tmp/XR_Accel"},
                        "active_policy_summary": {"status": "missing"},
                        "candidate_audit": {"recommendation_matches": True},
                        "safety": {
                            "executes_commands": False,
                            "executes_network": False,
                            "creates_xr_vits_policy": False,
                            "writes_canonical_inputs": False,
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# xr vits gate\n")
            return 0

        def c3b_physical_smoke_gate(argv: list[str]) -> int:
            calls.append("c3b_physical_smoke_gate")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "blocked_missing_canonical_physical_smoke_result",
                        "ready_for_board": True,
                        "canonical_result": {"status": "missing"},
                        "canonical_validation": {"status": "missing"},
                        "bundle": {"status": "pass"},
                        "session": {"status": "pass"},
                        "safety": {
                            "executes_commands": False,
                            "executes_network": False,
                            "creates_board_result": False,
                            "writes_canonical_inputs": False,
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# c3b physical smoke gate\n")
            return 0

        def vref_p0_buffer_lifetime(argv: list[str]) -> int:
            calls.append("vref_p0_buffer_lifetime")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "check_count": 36, "pass_count": 36, "fail_count": 0}) + "\n")
            markdown_out.write_text("# vref p0 buffer lifetime\n")
            return 0

        def hgpipe_operator_audit(argv: list[str]) -> int:
            calls.append("hgpipe_operator_audit")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "contract_summary": {
                            "passed_ref_checks": 97,
                            "total_checked_samples": 5_899_008,
                            "total_ref_checks": 97,
                        },
                        "property_summary": {
                            "check_count": 211,
                            "pass_count": 211,
                            "fail_count": 0,
                            "status": "pass",
                        },
                        "operators": [
                            {"operator": "LayerNorm", "status": "pass", "property_fail_count": 0},
                            {"operator": "GeLU", "status": "pass", "property_fail_count": 0},
                            {"operator": "Softmax", "status": "pass", "property_fail_count": 0},
                            {"operator": "Quantization", "status": "pass", "property_fail_count": 0},
                        ],
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# hgpipe operator audit\n")
            return 0

        def closeout_packet(argv: list[str]) -> int:
            calls.append("closeout_packet")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(
                json.dumps(
                    {
                        "status": "ready-for-operator-unblock",
                        "vref_successor_policy": {
                            "require_physical_smoke_for_final_signoff": "--require-vref-successor-physical-smoke" in argv
                        },
                    }
                )
                + "\n"
            )
            markdown_out.write_text("# closeout packet\n")
            return 0

        def closeout_validation(argv: list[str]) -> int:
            calls.append("closeout_validation")
            json_out = Path(argv[argv.index("--json-out") + 1])
            markdown_out = Path(argv[argv.index("--markdown-out") + 1])
            packet_path = Path(argv[argv.index("--packet") + 1])
            self.assertTrue(packet_path.exists())
            self.assertIn("generated/signoff", packet_path.as_posix())
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps({"status": "pass", "check_count": 22, "pass_count": 22, "fail_count": 0}) + "\n")
            markdown_out.write_text("# closeout validation\n")
            return 0

        return {
            "c3b_import": c3b_import,
            "zcu104_remote": zcu104_remote,
            "c3b_transfer_manifest": c3b_transfer_manifest,
            "c3b_smoke_contract": c3b_smoke_contract,
            "xr_vits_policy": xr_vits_policy,
            "xr_vits_packet": xr_vits_packet,
            "xr_vits_reference_resolution": xr_vits_reference_resolution,
            "readiness": readiness,
            "preflight": preflight,
            "final_audit": final_audit,
            "completion": completion,
            "unblock": unblock,
            "command_card": command_card,
            "resource_matrix": resource_matrix,
            "resource_policy": resource_policy,
            "selected_path": selected_path,
            "requirements_trace": requirements_trace,
            "spec_plan": spec_plan,
            "candidate_audit": candidate_audit,
            "operator_handoff": operator_handoff,
            "operator_handoff_validation": operator_handoff_validation,
            "bundle_validation": bundle_validation,
            "c3b_smoke_discovery": c3b_smoke_discovery,
            "pynq_smoke_discovery": pynq_smoke_discovery,
            "blocker_closure": blocker_closure,
            "unblock_intake": unblock_intake,
            "closeout_packet": closeout_packet,
            "closeout_validation": closeout_validation,
            "evidence_manifest": evidence_manifest,
            "source_audit": source_audit,
            "current_audit": current_audit,
            "qkv_uram_successor": qkv_uram_successor,
            "vref_p0_pot_scale_audit": vref_p0_pot_scale_audit,
            "vref_p0_pot_scale_sweep": vref_p0_pot_scale_sweep,
            "req5_q4q8_swhw": req5_q4q8_swhw,
            "p2_vit_scale_calibration": p2_vit_scale_calibration,
            "req1_environment": req1_environment,
            "req6_parameterization": req6_parameterization,
            "req9_deit_image": req9_deit_image,
            "xr_vits_gate": xr_vits_gate,
            "c3b_physical_smoke_gate": c3b_physical_smoke_gate,
            "vref_p0_buffer_lifetime": vref_p0_buffer_lifetime,
            "hgpipe_operator_audit": hgpipe_operator_audit,
        }

    def test_blocked_run_writes_summary_and_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=False,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
            )

            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "blocked")
            self.assertEqual(
                calls,
                [
                    "zcu104_remote",
                    "c3b_transfer_manifest",
                    "c3b_smoke_contract",
                    "xr_vits_policy_preview",
                    "xr_vits_packet",
                    "xr_vits_reference_resolution",
                    "readiness",
                    "preflight",
                    "final_audit",
                    "completion",
                    "unblock",
                    "command_card",
                    "resource_matrix",
                    "resource_policy",
                    "selected_path",
                    "candidate_audit",
                    "requirements_trace",
                    "spec_plan",
                    "operator_handoff",
                    "operator_handoff_validation",
                    "bundle_validation",
                    "c3b_smoke_discovery",
                    "pynq_c3b_smoke_discovery",
                    "vref_smoke_discovery",
                    "pynq_vref_smoke_discovery",
                    "qkv_smoke_discovery",
                    "qkv_uram_successor",
                    "vref_p0_pot_scale_audit",
                    "vref_p0_pot_scale_sweep",
                    "req5_q4q8_swhw",
                    "p2_vit_scale_calibration",
                    "req1_environment",
                    "req6_parameterization",
                    "req9_deit_image",
                    "xr_vits_gate",
                    "c3b_physical_smoke_gate",
                    "vref_p0_buffer_lifetime",
                    "hgpipe_operator_audit",
                    "blocker_closure",
                    "unblock_intake",
                    "closeout_packet",
                    "closeout_validation",
                    "source_audit",
                    "current_audit",
                    "source_audit",
                    "evidence_manifest",
                    "requirements_trace",
                    "operator_handoff",
                    "operator_handoff_validation",
                    "bundle_validation",
                    "evidence_manifest",
                    "spec_plan",
                    "evidence_manifest",
                    "completion",
                    "requirements_trace",
                    "operator_handoff",
                    "operator_handoff_validation",
                    "bundle_validation",
                    "evidence_manifest",
                    "spec_plan",
                    "evidence_manifest",
                    "requirements_trace",
                    "operator_handoff",
                    "operator_handoff_validation",
                    "bundle_validation",
                    "evidence_manifest",
                    "spec_plan",
                    "evidence_manifest",
                ],
            )
            final_manifest_index = calls.index("evidence_manifest")
            self.assertEqual(calls[final_manifest_index + 1 : final_manifest_index + 23], [
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
                "completion",
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
            ])
            self.assertEqual(summary["zcu104_remote_status"], "dry-run")
            self.assertEqual(summary["c3b_transfer_manifest_status"], "pass")
            self.assertEqual(summary["c3b_smoke_contract_status"], "pass")
            self.assertEqual(summary["c3b_import_status"], "skipped")
            self.assertFalse(summary["zcu104_remote_execute"])
            self.assertEqual(summary["xr_vits_policy_status"], "skipped")
            self.assertEqual(summary["xr_vits_policy_preview_status"], "pass")
            self.assertEqual(summary["xr_vits_packet_status"], "pending-user-choice")
            self.assertEqual(summary["xr_vits_reference_resolution_status"], "candidate-ready-needs-approval")
            self.assertEqual(summary["resource_matrix_status"], "pass")
            self.assertEqual(summary["resource_policy_status"], "pass")
            self.assertEqual(summary["selected_path_status"], "pass")
            self.assertEqual(summary["requirements_trace_status"], "blocked")
            self.assertEqual(summary["spec_plan_status"], "pass")
            self.assertEqual(summary["candidate_audit_status"], "blocked")
            self.assertEqual(summary["operator_handoff_status"], "pending-operator-actions")
            self.assertEqual(summary["operator_handoff_validation_status"], "pass")
            self.assertEqual(summary["final_bundle_validation_status"], "pass")
            self.assertEqual(summary["c3b_smoke_discovery_status"], "missing")
            self.assertEqual(summary["pynq_c3b_smoke_discovery_status"], "missing")
            self.assertEqual(summary["vref_smoke_discovery_status"], "missing")
            self.assertEqual(summary["pynq_vref_smoke_discovery_status"], "missing")
            self.assertEqual(summary["qkv_uram_smoke_discovery_status"], "missing")
            self.assertEqual(summary["qkv_uram_import_status"], "skipped")
            self.assertEqual(summary["qkv_uram_remote_status"], "skipped")
            self.assertFalse(summary["qkv_uram_remote_execute"])
            self.assertEqual(summary["p2_vit_scale_calibration_status"], "pass")
            self.assertEqual(summary["final_blocker_closure_status"], "blocked")
            self.assertEqual(summary["final_unblock_intake_status"], "blocked")
            self.assertEqual(summary["final_unblock_closeout_packet_status"], "ready-for-operator-unblock")
            self.assertEqual(summary["final_unblock_closeout_validation_status"], "pass")
            self.assertEqual(summary["third_goal_source_audit_status"], "pass")
            self.assertEqual(summary["third_goal_current_audit_status"], "blocked-external")
            self.assertEqual(summary["final_evidence_manifest_status"], "pass")
            self.assertEqual(summary["blocker_count"], len(summary["remaining_blockers"]))
            self.assertEqual(summary["blocker_count"], 2)
            self.assertEqual(
                summary["remaining_blocker_detail_count"],
                len(summary["remaining_blocker_details"]),
            )
            self.assertIn("requested XR-VITs sibling", summary["remaining_blockers"])
            self.assertIn("C3b AXIS/DMA physical smoke result", summary["remaining_blocker_input_paths"])
            self.assertIn("requested XR-VITs sibling", summary["remaining_blocker_input_paths"])
            self.assertEqual(
                set(summary["remaining_blockers"]),
                set(summary["remaining_blocker_input_paths"]),
            )
            self.assertEqual(
                set(summary["remaining_blockers"]),
                set(summary["remaining_blocker_details"]),
            )
            self.assertEqual(
                summary["remaining_blocker_input_paths"]["C3b AXIS/DMA physical smoke result"],
                [str(root / "hardware" / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json")],
            )
            self.assertIn(
                str(root.parent.parent / "XR-VITs"),
                summary["remaining_blocker_input_paths"]["requested XR-VITs sibling"],
            )
            self.assertIn(
                str(root / "docs" / "resources" / "xr_vits_replacement_policy.json"),
                summary["remaining_blocker_input_paths"]["requested XR-VITs sibling"],
            )
            self.assertEqual(
                summary["remaining_blocker_details"]["requested XR-VITs sibling"]["kind"],
                "exact-source-or-approved-replacement-policy",
            )
            self.assertEqual(
                summary["remaining_blocker_details"]["C3b AXIS/DMA physical smoke result"]["kind"],
                "canonical-pynq-smoke-json",
            )
            self.assertTrue((root / "docs" / "resources" / "third_goal_final_signoff_run_2026_06_10.json").exists())
            self.assertTrue((root / "docs" / "resources" / "c3b_smoke_transfer_manifest_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256").exists())
            self.assertEqual(len(summary["mirrored_artifacts"]), len(set(summary["mirrored_artifacts"])))
            self.assertEqual(summary["mirrored_artifact_count"], len(summary["mirrored_artifacts"]))
            self.assertEqual(summary["mirrored_artifact_unique_count"], len(set(summary["mirrored_artifacts"])))
            self.assertEqual(summary["mirrored_artifact_duplicate_count"], 0)
            self.assertEqual(
                summary["mirrored_artifacts"].count(
                    str(root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.json")
                ),
                1,
            )
            self.assertTrue((root / "docs" / "resources" / "c3b_smoke_result_contract_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "xr_vits_replacement_policy_preview_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "xr_vits_reference_resolution_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_unblock_commands_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "e2e_resource_matrix_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "e2e_resource_policy_audit_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "selected_path_execution_audit_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "third_goal_requirements_trace_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "spec_plan_conformance_audit_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_unblock_candidate_audit_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_operator_handoff_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_operator_handoff_validation_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_signoff_bundle_validation_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "c3b_smoke_candidate_discovery_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "pynq_smoke_candidate_discovery_c3b_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "vref_successor_smoke_candidate_discovery_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "pynq_smoke_candidate_discovery_vref_p0_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "qkv_uram_smoke_candidate_discovery_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "vref_p0_qkv_uram_cache_successor_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "vref_p0_pot_scale_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "vref_p0_pot_scale_sweep_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "p2_vit_scale_calibration_report_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "req1_environment_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "req9_deit_image_reference_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "xr_vits_gate_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "c3b_physical_smoke_gate_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_blocker_closure_readiness_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_unblock_intake_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_unblock_closeout_packet_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_unblock_closeout_packet_validation_2026_06_10.md").exists())
            self.assertTrue((root / "docs" / "resources" / "third_goal_source_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "third_goal_current_audit_2026_06_16.md").exists())
            self.assertTrue((root / "docs" / "resources" / "final_evidence_manifest_2026_06_10.md").exists())

    def test_allow_blocked_keeps_evidence_but_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "blocked")
            self.assertTrue(summary["allow_blocked"])

    def test_execute_zcu104_smoke_forwards_remote_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            identity = Path(tmp) / "id_ed25519"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                zcu104_host="192.0.2.10",
                zcu104_user="xilinx",
                zcu104_port=2222,
                zcu104_identity_file=identity,
                zcu104_remote_dir="/home/xilinx/custom_hgtxr",
                execute_zcu104_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["zcu104_remote_status"], "pass")
            self.assertTrue(summary["zcu104_remote_execute"])
            zcu104_step = summary["steps"][0]
            self.assertEqual(zcu104_step["name"], "zcu104-remote-execute")
            self.assertIn("--execute", zcu104_step["argv"])
            self.assertIn("192.0.2.10", zcu104_step["argv"])
            self.assertIn("2222", zcu104_step["argv"])
            self.assertIn(str(identity), zcu104_step["argv"])
            self.assertIn("/home/xilinx/custom_hgtxr", zcu104_step["argv"])

    def test_require_vref_successor_smoke_runs_profile_dry_run_without_remote_dir_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                zcu104_remote_dir="/home/xilinx/c3b_only",
                require_vref_successor_physical_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertTrue(summary["vref_successor_required_for_final_signoff"])
            self.assertEqual(summary["vref_successor_remote_status"], "dry-run")
            self.assertFalse(summary["vref_successor_remote_execute"])
            self.assertIn("vref_remote", calls)
            vref_step = next(step for step in summary["steps"] if step["name"] == "vref-successor-remote-dry-run")
            self.assertIn("--profile", vref_step["argv"])
            self.assertIn("vref-p0-softmax-input-x2-dsp-mixed-stream", vref_step["argv"])
            self.assertNotIn("/home/xilinx/c3b_only", vref_step["argv"])
            closeout_step = next(step for step in summary["steps"] if step["name"] == "final-unblock-closeout-packet")
            self.assertIn("--require-vref-successor-physical-smoke", closeout_step["argv"])

    def test_execute_vref_successor_smoke_forwards_remote_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            identity = Path(tmp) / "id_ed25519"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                zcu104_host="192.0.2.11",
                zcu104_user="xilinx",
                zcu104_port=2223,
                zcu104_identity_file=identity,
                vref_successor_remote_dir="/home/xilinx/vref_hgtxr",
                execute_vref_successor_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertFalse(summary["vref_successor_required_for_final_signoff"])
            self.assertEqual(summary["vref_successor_remote_status"], "pass")
            self.assertTrue(summary["vref_successor_remote_execute"])
            self.assertIn("vref_remote", calls)
            vref_step = next(step for step in summary["steps"] if step["name"] == "vref-successor-remote-execute")
            self.assertIn("--execute", vref_step["argv"])
            self.assertIn("192.0.2.11", vref_step["argv"])
            self.assertIn("2223", vref_step["argv"])
            self.assertIn(str(identity), vref_step["argv"])
            self.assertIn("/home/xilinx/vref_hgtxr", vref_step["argv"])

    def test_require_qkv_uram_smoke_runs_profile_dry_run_and_blocks_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                zcu104_remote_dir="/home/xilinx/c3b_only",
                require_qkv_uram_physical_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertTrue(summary["qkv_uram_required_for_final_signoff"])
            self.assertEqual(summary["qkv_uram_remote_status"], "dry-run")
            self.assertFalse(summary["qkv_uram_remote_execute"])
            self.assertIn("qkv_remote", calls)
            self.assertIn("VREF-P0-02 QKV URAM physical smoke result", summary["remaining_blockers"])
            qkv_step = next(step for step in summary["steps"] if step["name"] == "qkv-uram-remote-dry-run")
            self.assertIn("--profile", qkv_step["argv"])
            self.assertIn("vref-p0-softmax-input-x2-qkv-uram", qkv_step["argv"])
            self.assertNotIn("/home/xilinx/c3b_only", qkv_step["argv"])

    def test_execute_qkv_uram_smoke_forwards_remote_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            identity = Path(tmp) / "id_ed25519"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                zcu104_host="192.0.2.12",
                zcu104_user="xilinx",
                zcu104_port=2224,
                zcu104_identity_file=identity,
                qkv_uram_remote_dir="/home/xilinx/qkv_hgtxr",
                execute_qkv_uram_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertFalse(summary["qkv_uram_required_for_final_signoff"])
            self.assertEqual(summary["qkv_uram_remote_status"], "pass")
            self.assertTrue(summary["qkv_uram_remote_execute"])
            self.assertIn("qkv_remote", calls)
            qkv_step = next(step for step in summary["steps"] if step["name"] == "qkv-uram-remote-execute")
            self.assertIn("--execute", qkv_step["argv"])
            self.assertIn("192.0.2.12", qkv_step["argv"])
            self.assertIn("2224", qkv_step["argv"])
            self.assertIn(str(identity), qkv_step["argv"])
            self.assertIn("/home/xilinx/qkv_hgtxr", qkv_step["argv"])

    def test_import_c3b_smoke_json_runs_before_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            board_result = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            board_result.parent.mkdir(parents=True)
            board_result.write_text("{}\n")
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                import_c3b_smoke_json=board_result,
                import_c3b_no_require_paths=True,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["c3b_import_status"], "pass")
            self.assertEqual(calls[0], "c3b_import")
            import_step = summary["steps"][0]
            self.assertEqual(import_step["name"], "c3b-smoke-import")
            self.assertIn(str(board_result), import_step["argv"])
            self.assertIn("--no-require-paths", import_step["argv"])
            self.assertLess(calls.index("c3b_import"), calls.index("readiness"))
            candidate_step = next(step for step in summary["steps"] if step["name"] == "final-unblock-candidate-audit")
            self.assertIn(str(board_result), candidate_step["argv"])
            self.assertIn("--c3b-no-require-paths", candidate_step["argv"])
            self.assertTrue((root / "docs" / "resources" / "c3b_smoke_import_2026_06_10.json").exists())
            self.assertTrue((root / "docs" / "resources" / "c3b_smoke_import_validation_2026_06_10.json").exists())

    def test_dry_run_import_c3b_smoke_json_validates_without_copy_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            board_result = Path(tmp) / "board" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
            board_result.parent.mkdir(parents=True)
            board_result.write_text("{}\n")
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                import_c3b_smoke_json=board_result,
                dry_run_import_c3b_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["c3b_import_status"], "dry-run-pass")
            import_step = summary["steps"][0]
            self.assertIn("--dry-run", import_step["argv"])

    def test_import_vref_successor_smoke_json_runs_before_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            board_result = Path(tmp) / "board" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
            board_result.parent.mkdir(parents=True)
            board_result.write_text("{}\n")
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                import_vref_successor_smoke_json=board_result,
                import_vref_successor_no_require_paths=True,
                dry_run_import_vref_successor_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["vref_successor_import_status"], "dry-run-pass")
            self.assertEqual(calls[0], "vref_import")
            import_step = summary["steps"][0]
            self.assertEqual(import_step["name"], "vref-successor-smoke-import")
            self.assertIn(str(board_result), import_step["argv"])
            self.assertIn("axis-vref-p0-softmax-input-x2-dsp-mixed-stream", import_step["argv"])
            self.assertIn("--no-require-paths", import_step["argv"])
            self.assertIn("--dry-run", import_step["argv"])
            self.assertLess(calls.index("vref_import"), calls.index("zcu104_remote"))
            self.assertTrue((root / "docs" / "resources" / "vref_successor_smoke_import_2026_06_10.json").exists())
            self.assertTrue((root / "docs" / "resources" / "vref_successor_smoke_import_validation_2026_06_10.json").exists())

    def test_import_qkv_uram_smoke_json_runs_before_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            board_result = Path(tmp) / "board" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
            board_result.parent.mkdir(parents=True)
            board_result.write_text("{}\n")
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                import_qkv_uram_smoke_json=board_result,
                import_qkv_uram_no_require_paths=True,
                dry_run_import_qkv_uram_smoke=True,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["qkv_uram_import_status"], "dry-run-pass")
            self.assertEqual(calls[0], "qkv_import")
            import_step = summary["steps"][0]
            self.assertEqual(import_step["name"], "qkv-uram-smoke-import")
            self.assertIn(str(board_result), import_step["argv"])
            self.assertIn("axis-vref-p0-softmax-input-x2-qkv-uram", import_step["argv"])
            self.assertIn("--no-require-paths", import_step["argv"])
            self.assertIn("--dry-run", import_step["argv"])
            self.assertLess(calls.index("qkv_import"), calls.index("zcu104_remote"))
            self.assertTrue((root / "docs" / "resources" / "qkv_uram_smoke_import_2026_06_16.json").exists())
            self.assertTrue((root / "docs" / "resources" / "qkv_uram_smoke_import_validation_2026_06_16.json").exists())

    def test_approve_xr_vits_replacement_forwards_policy_args_before_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            replacement = Path(tmp) / "XR_Accel"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                approve_xr_vits_replacement=True,
                xr_vits_approved_by="unit-test",
                xr_vits_replacement_reason="unit test approval",
                xr_vits_replacement_path=replacement,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["xr_vits_policy_status"], "pass")
            self.assertLess(calls.index("xr_vits_policy"), calls.index("xr_vits_packet"))
            policy_step = next(step for step in summary["steps"] if step["name"] == "xr-vits-replacement-policy")
            self.assertEqual(policy_step["name"], "xr-vits-replacement-policy")
            self.assertIn("--approved-by", policy_step["argv"])
            self.assertIn("unit-test", policy_step["argv"])
            self.assertIn("--replacement-path", policy_step["argv"])
            self.assertIn(str(replacement), policy_step["argv"])
            candidate_step = next(step for step in summary["steps"] if step["name"] == "final-unblock-candidate-audit")
            self.assertIn("--xr-vits-mode", candidate_step["argv"])
            self.assertIn("replacement", candidate_step["argv"])
            self.assertIn("--approved-by", candidate_step["argv"])
            self.assertIn("unit-test", candidate_step["argv"])

    def test_dry_run_xr_vits_replacement_validates_without_policy_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                approve_xr_vits_replacement=True,
                dry_run_xr_vits_replacement=True,
                xr_vits_approved_by="unit-test",
                xr_vits_replacement_reason="unit test approval",
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["xr_vits_policy_status"], "dry-run-pass")
            policy_step = next(step for step in summary["steps"] if step["name"] == "xr-vits-replacement-policy-dry-run")
            self.assertEqual(policy_step["name"], "xr-vits-replacement-policy-dry-run")
            self.assertIn("--dry-run", policy_step["argv"])
            self.assertLess(calls.index("xr_vits_policy"), calls.index("xr_vits_packet"))

    def test_dry_run_xr_vits_replacement_allows_placeholder_approver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=self.make_tools(blocked=True, calls=calls),
                paths=self.make_paths(root),
                approve_xr_vits_replacement=True,
                dry_run_xr_vits_replacement=True,
                xr_vits_approved_by="<approved-by>",
                xr_vits_replacement_reason="unit test approval",
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["xr_vits_policy_status"], "dry-run-pass")
            policy_step = next(step for step in summary["steps"] if step["name"] == "xr-vits-replacement-policy-dry-run")
            self.assertIn("<approved-by>", policy_step["argv"])
            self.assertIn("--dry-run", policy_step["argv"])

    def test_active_xr_vits_replacement_policy_failure_blocks_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            tools = self.make_tools(blocked=False, calls=calls)
            base_policy = tools["xr_vits_policy"]

            def rejecting_policy(argv: list[str]) -> int:
                if "--preview-json-out" in argv:
                    return base_policy(argv)
                calls.append("xr_vits_policy")
                approved_by = argv[argv.index("--approved-by") + 1]
                if "--dry-run" not in argv and approved_by == "<approved-by>":
                    return 1
                return base_policy(argv)

            tools["xr_vits_policy"] = rejecting_policy
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=tools,
                paths=self.make_paths(root),
                approve_xr_vits_replacement=True,
                xr_vits_approved_by="<approved-by>",
                xr_vits_replacement_reason="unit test approval",
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["xr_vits_policy_status"], "fail")
            self.assertEqual(summary["status"], "blocked")

    def test_final_refresh_uses_fresh_evidence_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            paths = self.make_paths(root)
            paths["doc_evidence_manifest_json"].parent.mkdir(parents=True, exist_ok=True)
            paths["doc_evidence_manifest_json"].write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "required_count": 1,
                        "present_required_count": 1,
                        "consistency_checks": [],
                        "failed_consistency_checks": [],
                        "safety": {
                            "executes_commands": False,
                            "creates_board_result": False,
                            "creates_xr_vits_policy": False,
                            "writes_canonical_inputs": False,
                        },
                    }
                )
                + "\n"
            )
            calls: list[str] = []
            tools = self.make_tools(blocked=True, calls=calls)

            def contract_from_manifest(manifest_path: Path) -> dict:
                manifest = json.loads(manifest_path.read_text())
                consistency_checks = manifest.get("consistency_checks", [])
                return {
                    "status": manifest.get("status"),
                    "source": "manifest-json",
                    "path": str(paths["doc_evidence_manifest_json"]),
                    "required_count": manifest.get("required_count"),
                    "present_required_count": manifest.get("present_required_count"),
                    "consistency_count": len(consistency_checks) if isinstance(consistency_checks, list) else 0,
                    "failed_consistency_checks": manifest.get("failed_consistency_checks", []),
                    "safety": manifest.get("safety", {}),
                }

            def evidence_manifest(argv: list[str]) -> int:
                calls.append("evidence_manifest")
                json_out = Path(argv[argv.index("--json-out") + 1])
                markdown_out = Path(argv[argv.index("--markdown-out") + 1])
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "required_count": 66,
                            "present_required_count": 66,
                            "consistency_checks": [{"name": f"check_{index}", "status": "pass"} for index in range(170)],
                            "failed_consistency_checks": [],
                            "safety": {
                                "executes_commands": False,
                                "creates_board_result": False,
                                "creates_xr_vits_policy": False,
                                "writes_canonical_inputs": False,
                            },
                        }
                    )
                    + "\n"
                )
                markdown_out.write_text("# evidence manifest\n")
                return 0

            def requirements_trace(argv: list[str]) -> int:
                calls.append("requirements_trace")
                evidence_path = Path(argv[argv.index("--evidence-manifest") + 1])
                contract = contract_from_manifest(evidence_path)
                if "evidence_manifest" in calls:
                    self.assertEqual(contract["required_count"], 66)
                    self.assertEqual(contract["consistency_count"], 170)
                json_out = Path(argv[argv.index("--json-out") + 1])
                markdown_out = Path(argv[argv.index("--markdown-out") + 1])
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(
                    json.dumps(
                        {
                            "status": "blocked",
                            "final_evidence_manifest_contract": contract,
                        }
                    )
                    + "\n"
                )
                markdown_out.write_text("# requirements trace\n")
                return 1

            def operator_handoff(argv: list[str]) -> int:
                calls.append("operator_handoff")
                requirements_trace_path = Path(argv[argv.index("--requirements-trace") + 1])
                trace = json.loads(requirements_trace_path.read_text())
                contract = trace.get("final_evidence_manifest_contract", {})
                if "evidence_manifest" in calls:
                    self.assertEqual(contract.get("required_count"), 66)
                    self.assertEqual(contract.get("consistency_count"), 170)
                json_out = Path(argv[argv.index("--json-out") + 1])
                markdown_out = Path(argv[argv.index("--markdown-out") + 1])
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(
                    json.dumps(
                        {
                            "status": "pending-operator-actions",
                            "remaining_blockers": ["requested XR-VITs sibling"],
                            "final_evidence_manifest_contract": contract,
                        }
                    )
                    + "\n"
                )
                markdown_out.write_text("# operator handoff\n")
                return 1

            def bundle_validation(argv: list[str]) -> int:
                calls.append("bundle_validation")
                trace = json.loads(paths["doc_requirements_trace_json"].read_text())
                handoff = json.loads(paths["doc_operator_handoff_json"].read_text())
                if "evidence_manifest" in calls:
                    self.assertEqual(trace["final_evidence_manifest_contract"]["required_count"], 66)
                    self.assertEqual(handoff["final_evidence_manifest_contract"], trace["final_evidence_manifest_contract"])
                json_out = Path(argv[argv.index("--json-out") + 1])
                markdown_out = Path(argv[argv.index("--markdown-out") + 1])
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(json.dumps({"status": "pass", "fail_count": 0}) + "\n")
                markdown_out.write_text("# bundle validation\n")
                return 0

            tools.update(
                {
                    "evidence_manifest": evidence_manifest,
                    "requirements_trace": requirements_trace,
                    "operator_handoff": operator_handoff,
                    "bundle_validation": bundle_validation,
                }
            )
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=True,
                tools=tools,
                paths=paths,
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["final_bundle_validation_status"], "pass")
            final_manifest_index = calls.index("evidence_manifest")
            self.assertEqual(calls[final_manifest_index + 1 : final_manifest_index + 23], [
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
                "completion",
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
                "requirements_trace",
                "operator_handoff",
                "operator_handoff_validation",
                "bundle_validation",
                "evidence_manifest",
                "spec_plan",
                "evidence_manifest",
            ])
            final_handoff = json.loads(paths["operator_handoff_json"].read_text())
            self.assertEqual(final_handoff["final_evidence_manifest_contract"]["required_count"], 66)
            self.assertEqual(final_handoff["final_evidence_manifest_contract"]["consistency_count"], 170)

    def test_pass_run_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            calls: list[str] = []
            code, summary = runner.run_pipeline(
                root=root,
                allow_blocked=False,
                tools=self.make_tools(blocked=False, calls=calls),
                paths=self.make_paths(root),
            )

            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "pass")
            self.assertEqual(summary["c3b_smoke_contract_status"], "pass")
            self.assertEqual(summary["xr_vits_packet_status"], "exact-restored")
            self.assertEqual(summary["xr_vits_reference_resolution_status"], "exact-ready")
            self.assertEqual(summary["xr_vits_policy_status"], "skipped")
            self.assertEqual(summary["xr_vits_policy_preview_status"], "pass")
            self.assertEqual(summary["resource_matrix_status"], "pass")
            self.assertEqual(summary["resource_policy_status"], "pass")
            self.assertEqual(summary["selected_path_status"], "pass")
            self.assertEqual(summary["requirements_trace_status"], "pass")
            self.assertEqual(summary["spec_plan_status"], "pass")
            self.assertEqual(summary["candidate_audit_status"], "would-clear")
            self.assertEqual(summary["operator_handoff_status"], "ready-for-final-signoff")
            self.assertEqual(summary["operator_handoff_validation_status"], "pass")
            self.assertEqual(summary["final_bundle_validation_status"], "pass")
            self.assertEqual(summary["c3b_smoke_discovery_status"], "found")
            self.assertEqual(summary["pynq_c3b_smoke_discovery_status"], "found")
            self.assertEqual(summary["vref_smoke_discovery_status"], "found")
            self.assertEqual(summary["pynq_vref_smoke_discovery_status"], "found")
            self.assertEqual(summary["qkv_uram_smoke_discovery_status"], "found")
            self.assertEqual(summary["final_blocker_closure_status"], "ready-to-run-final")
            self.assertEqual(summary["final_unblock_intake_status"], "ready-for-active-unblock")
            self.assertEqual(summary["final_unblock_closeout_packet_status"], "ready-for-operator-unblock")
            self.assertEqual(summary["final_unblock_closeout_validation_status"], "pass")
            self.assertEqual(summary["third_goal_source_audit_status"], "pass")
            self.assertEqual(summary["third_goal_current_audit_status"], "pass")
            self.assertEqual(summary["final_evidence_manifest_status"], "pass")
            self.assertEqual(summary["remaining_blockers"], [])
            self.assertEqual(summary["blocker_count"], 0)
            self.assertEqual(summary["remaining_blocker_detail_count"], 0)
            self.assertEqual(summary["remaining_blocker_input_paths"], {})
            self.assertEqual(summary["remaining_blocker_details"], {})
            self.assertEqual(summary["mirrored_artifact_count"], len(summary["mirrored_artifacts"]))
            self.assertEqual(summary["mirrored_artifact_unique_count"], len(set(summary["mirrored_artifacts"])))
            self.assertEqual(summary["mirrored_artifact_duplicate_count"], 0)
            self.assertTrue((root / "docs" / "resources" / "third_goal_final_signoff_2026_06_10.json").exists())


if __name__ == "__main__":
    unittest.main()
