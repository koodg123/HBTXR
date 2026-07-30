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

import write_third_goal_current_audit as audit_tool  # noqa: E402


class ThirdGoalCurrentAuditTests(unittest.TestCase):
    def write(self, root: Path, rel: str, text: str = "fixture") -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def write_json(self, root: Path, rel: str, payload: dict[str, object]) -> None:
        self.write(root, rel, json.dumps(payload))

    def make_hardware_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        hardware = Path(tmp.name) / "XR-VIT" / "HGTXR" / "hardware"
        for rel in [
            "docs/Master-Plan.md",
            "docs/Sub-Plan.md",
            "docs/Spec.md",
            "docs/Execution.md",
            "docs/Validation.md",
            "docs/track/PROGRESS.md",
            "docs/track/THIRD_GOAL_REQUIREMENTS_2026_06_16.md",
            "docs/legacy/legacy_experiment_analysis_2026_06_12.md",
            "docs/SRC_CASE_MODULE_GUIDE.md",
            "docs/DeiT-Tiny C-Syn Results.png",
            "analysis/vit-accel/manifest.json",
            "configs/zcu104_e2e_q4w8a_defines.h",
            "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml",
            "hls/tb/tb_hgtxr_e2e_axis_top.cpp",
            "hls/tb/tb_hgtxr_e2e_m_axi_top.cpp",
            "hls/include/hgtxr_cyclic_math.hpp",
            "hls/include/hgtxr_e2e_vit.hpp",
            "pynq/hgtxr/run_e2e_axis_dma_smoke.py",
            "tools/validate_hgpipe_lut_math.py",
        ]:
            self.write(hardware, rel)
        self.write(
            hardware,
            "tools/check_third_goal_preflight.py",
            'XILINX_ROOT = "/tools/Xilinx"\n',
        )
        self.write(
            hardware,
            "tools/run_third_goal_final_signoff.py",
            "--require-vref-successor-physical-smoke --vref-successor-remote-dir\n",
        )
        self.write_json(
            hardware,
            "generated/signoff/final_unblock_closeout_packet_2026_06_16.json",
            {
                "status": "ready-for-operator-unblock",
                "remaining_blockers": [
                    "requested XR-VITs sibling",
                    "C3b AXIS/DMA physical smoke result",
                ],
                "qkv_uram_successor_gate": {
                    "status": "ready-for-physical-smoke",
                    "required_for_final_signoff": False,
                    "resources": {"bram_18k": 114, "dsp": 128, "ff": 19664, "lut": 43236, "uram": 40},
                },
                "operator_commands": {
                    "qkv_uram_successor": [
                        "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
                        "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
                    ],
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_unblock_closeout_packet_vref_required_2026_06_16.json",
            {
                "status": "ready-for-operator-unblock",
                "remaining_blockers": [
                    "requested XR-VITs sibling",
                    "C3b AXIS/DMA physical smoke result",
                    "VREF-P0 successor physical smoke result",
                ],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json",
            {
                "status": "pass",
                "candidate": "softmax_input_x2",
                "recommended_resource_variant": "dsp_mixed_stream",
                "expected_raw": [58, -51, 42, -28, 36, -41],
                "recommended_c3b_projection": {
                    "status": "pending",
                    "pending": ["physical_smoke_available"],
                },
                "physical_smoke_result": {
                    "status": "not_captured",
                    "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "result_json": "",
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.json",
            {
                "status": "missing",
                "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                "candidate_count": 0,
                "pass_count": 0,
                "recommended_candidate": None,
                "dry_run_import_command": "",
                "import_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json",
            {
                "status": "missing",
                "preset": "axis-c3b-mem16",
                "candidate_count": 0,
                "pass_count": 0,
                "recommended_candidate": None,
                "dry_run_import_command": "",
                "import_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json",
            {
                "status": "missing",
                "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                "candidate_count": 0,
                "pass_count": 0,
                "recommended_candidate": None,
                "dry_run_import_command": "",
                "import_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json",
            {
                "status": "missing",
                "preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
                "candidate_count": 0,
                "pass_count": 0,
                "recommended_candidate": None,
                "dry_run_import_command": "",
                "import_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.json",
            {
                "status": "pass",
                "candidate": "VREF-P0-02-qkv-weight-cache-uram",
                "macro": "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1",
                "resource_policy": "dsp_mixed_stream",
                "csynth": {
                    "estimated_clock_ns": 4.069,
                    "latency_cycles": 498485,
                    "resources": {"bram_18k": 114, "dsp": 128, "ff": 19664, "lut": 43236, "uram": 40},
                },
                "promotion": {
                    "status": "hls-resource-pass-routed-pending",
                    "remaining": ["routed_overlay_timing", "physical_smoke_json"],
                },
                "physical_smoke": {
                    "status": "missing",
                    "result_json": "",
                },
                "pynq_plumbing": {
                    "status": "ready-for-artifacts",
                    "dry_run_import_command": (
                        "python3 tools/import_pynq_smoke_result.py /path/to/"
                        "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json "
                        "--preset axis-vref-p0-softmax-input-x2-qkv-uram --dry-run"
                    ),
                    "remote_dry_run_command": (
                        "python3 tools/run_zcu104_c3b_smoke_remote.py "
                        "--profile vref-p0-softmax-input-x2-qkv-uram --host <zcu104-ip-or-host> --user xilinx"
                    ),
                    "expected_result_json": (
                        "pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
                    ),
                },
                "pass_count": 9,
                "check_count": 9,
                "fail_count": 0,
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/third_goal_final_signoff_run_2026_06_10.json",
            {
                "status": "blocked",
                "qkv_uram_required_for_final_signoff": False,
                "qkv_uram_import_status": "skipped",
                "qkv_uram_remote_status": "skipped",
                "qkv_uram_remote_execute": False,
                "pynq_c3b_smoke_discovery_status": "not-run",
                "pynq_vref_smoke_discovery_status": "not-run",
                "qkv_uram_smoke_discovery_status": "not-run",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/third_goal_source_audit_2026_06_16.json",
            {
                "status": "pass",
                "required_count": 22,
                "source_count": 140,
                "missing_required": [],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_blocker_closure_readiness_2026_06_10.json",
            {
                "status": "blocked",
                "current_ready": False,
                "candidate_ready": False,
                "missing_current_blockers": [
                    "C3b AXIS/DMA physical smoke result",
                    "requested XR-VITs sibling",
                ],
                "dry_run_final_runner_command": "",
                "final_runner_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_unblock_intake_2026_06_10.json",
            {
                "status": "blocked",
                "candidate_audit": {
                    "would_clear_all": False,
                    "remaining_blockers": [
                        "C3b AXIS/DMA physical smoke result",
                        "requested XR-VITs sibling",
                    ],
                },
                "closure_readiness": {
                    "candidate_ready": False,
                },
                "next_action": "Supply inputs for: C3b AXIS/DMA physical smoke result, requested XR-VITs sibling",
                "dry_run_final_runner_command": "",
                "active_final_runner_command": "",
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_unblock_commands_2026_06_10.json",
            {
                "status": "pending-unblock",
                "readiness_status": "blocked",
                "xr_vits_packet_status": "pending-user-choice",
                "xr_vits_policy_integrity": {
                    "required": True,
                    "policy_path": str(hardware.parent / "docs/resources/xr_vits_replacement_policy.json"),
                    "candidate_audit": str(hardware.parent / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
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
                },
                "remaining_blockers": [
                    "C3b AXIS/DMA physical smoke result",
                    "requested XR-VITs sibling",
                ],
                "sections": [
                    {"id": "U0"},
                    {"id": "U1"},
                    {"id": "U2"},
                    {"id": "U4"},
                    {"id": "U5"},
                    {"id": "U3"},
                ],
                "c3b_smoke_contract": {
                    "status": "pass",
                    "preset": "axis-c3b-mem16",
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_unblock_candidate_audit_2026_06_10.json",
            {
                "status": "blocked",
                "would_clear_all": False,
                "remaining_blockers": [
                    "C3b AXIS/DMA physical smoke result",
                    "requested XR-VITs sibling",
                ],
                "c3b_smoke": {"status": "missing", "would_clear": False},
                "xr_vits": {"status": "fail", "mode": "replacement", "would_clear": False},
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_operator_handoff_2026_06_10.json",
            {
                "status": "pending-operator-actions",
                "remaining_blockers": [
                    "C3b AXIS/DMA physical smoke result",
                    "requested XR-VITs sibling",
                ],
                "board_smoke": {"status": "ready-for-board", "ready": True},
                "xr_vits": {
                    "candidate_status": "fail",
                    "would_clear": False,
                    "reference_resolution": {
                        "policy_integrity": {
                            "required": True,
                            "policy_path": str(hardware.parent / "docs/resources/xr_vits_replacement_policy.json"),
                            "candidate_audit": str(hardware.parent / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"),
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
                            "policy_exists": False,
                            "validation": {"status": "pending-policy-creation"},
                        },
                    },
                },
                "final_evidence_manifest_contract": {
                    "status": "pass",
                    "required_count": 50,
                    "present_required_count": 50,
                },
                "safety": {"requires_operator_action": True},
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/final_operator_handoff_validation_2026_06_10.json",
            {
                "status": "pass",
                "pass_count": 52,
                "fail_count": 0,
                "check_count": 52,
                "checks": [
                    {"name": name, "status": "pass", "detail": "ok"}
                    for name in [
                        "xr_policy_integrity_required",
                        "xr_policy_integrity_fields",
                        "xr_policy_integrity_generator",
                        "xr_legacy_policy_not_accepted",
                        "xr_policy_validation_status",
                        "xr_policy_validation_not_legacy",
                        "xr_policy_validation_embedded_matches_live",
                    ]
                ],
                "safety": {
                    "executes_commands": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/xr_vits_unblock_packet_2026_06_10.json",
            {
                "status": "pending-user-choice",
                "requested_path": str(Path(tmp.name) / "XR-VITs"),
                "requested_path_exists": False,
                "active_policy_approved": False,
                "next_required_action": "Restore exact XR-VITs or explicitly approve XR_Accel replacement.",
                "options": [
                    {"id": "restore-exact-xr-vits"},
                    {"id": "approve-xr-accel-replacement"},
                ],
                "recommendation": {
                    "path": str(Path(tmp.name) / "XR-VIT" / "XR_Accel"),
                    "role": "candidate-xr-accel",
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/hgpipe_operator_audit_2026_06_16.json",
            {
                "status": "pass",
                "contract_summary": {
                    "total_ref_checks": 97,
                    "total_checked_samples": 5899008,
                },
                "operators": [
                    {"operator": "LayerNorm", "status": "pass"},
                    {"operator": "GeLU", "status": "pass"},
                    {"operator": "Softmax", "status": "pass"},
                    {"operator": "Quantization", "status": "pass"},
                ],
                "residual_risk": [
                    "local sampled/reference-vector equivalence, not formal exhaustive proof"
                ],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/xr_vits_gate_audit_2026_06_16.json",
            {
                "status": "blocked",
                "resolution_mode": "candidate-ready-needs-approval",
                "requested_path": str(Path(tmp.name) / "XR-VITs"),
                "remaining_blockers": ["requested XR-VITs sibling"],
                "candidate_audit": {
                    "status": "candidate-found",
                    "recommendation_matches": True,
                },
                "active_policy": {"status": "missing"},
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.json",
            {
                "status": "blocked",
                "canonical_result": {"status": "missing", "exists": False},
                "bundle": {"status": "pass"},
                "session": {"status": "pass"},
                "remaining_blockers": ["C3b AXIS/DMA physical smoke result"],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.json",
            {
                "status": "pass-manual-spec-and-model-fallback",
                "spec_kit": {
                    "available": False,
                    "manual_fallback_recorded": True,
                },
                "subagents": {
                    "spark_first_recorded": True,
                    "gpt55_fallback_recorded": True,
                },
                "plan_spec": {"plan_spec_covered": True},
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/req1_environment_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 13,
                "pass_count": 13,
                "fail_count": 0,
                "environment": {
                    "os_release": {"PRETTY_NAME": "Ubuntu 22.04.5 LTS"},
                    "vitis_hls": "/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls",
                    "vivado": "/tools/Xilinx/Vivado/2023.2/bin/vivado",
                },
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.json",
            {
                "status": "pass",
                "precision": {"weight_bits": 4, "activation_bits": 8},
                "manifest": {"status": "pass"},
                "pass_count": 6,
                "fail_count": 0,
                "claim_supported": "fixture",
                "remaining_scope": ["physical smoke separate"],
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/req6_parameterization_audit_2026_06_16.json",
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
                "check_count": 41,
                "pass_count": 41,
                "fail_count": 0,
            },
        )
        self.write_json(
            hardware,
            "generated/signoff/req9_deit_image_reference_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 11,
                "pass_count": 11,
                "fail_count": 0,
                "requested_user_path": f"{hardware.parent.parent}\\PAPER_PRJXR\\05_RESOURCES\\DeiT-Tiny C-Syn Results.png",
                "normalized_requested_path": str(
                    hardware.parent.parent / "PAPER_PRJXR" / "05_RESOURCES" / "DeiT-Tiny C-Syn Results.png"
                ),
                "requested_image": {"sha256": "a" * 64},
                "hardware_docs_copy": {"sha256": "a" * 64},
            },
        )
        return tmp, hardware

    def test_build_audit_marks_external_blockers(self) -> None:
        tmp, hardware = self.make_hardware_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(hardware)

        self.assertEqual(audit["status"], "blocked-external")
        self.assertEqual(audit["summary"]["requirements"], 12)
        self.assertEqual(audit["summary"]["default_closeout_blockers"], 2)
        self.assertEqual(audit["summary"]["vref_required_closeout_blockers"], 3)
        by_id = {item["id"]: item for item in audit["items"]}
        self.assertEqual(by_id["0"]["status"], "reflected")
        self.assertEqual(by_id["1"]["status"], "reflected")
        self.assertIn("req1_environment_audit", audit)
        self.assertEqual(audit["req1_environment_audit"]["status"], "pass")
        self.assertIn("Req1 environment audit status=pass", by_id["1"]["note"])
        self.assertEqual(by_id["2"]["status"], "reflected")
        self.assertEqual(by_id["4"]["status"], "partial")
        self.assertEqual(audit["summary"]["reflected"], 10)
        self.assertEqual(audit["summary"]["partial"], 1)
        self.assertIn("c3b_physical_smoke_gate_audit", audit)
        self.assertIn("req2_spec_subagent_gate_audit", audit)
        self.assertEqual(by_id["5"]["status"], "reflected")
        self.assertIn("req5_q4q8_swhw_match_audit", audit)
        self.assertEqual(audit["req5_q4q8_swhw_match_audit"]["status"], "pass")
        self.assertEqual(by_id["6"]["status"], "reflected")
        self.assertIn("req6_parameterization_audit", audit)
        self.assertEqual(audit["req6_parameterization_audit"]["status"], "pass")
        self.assertIn("req9_deit_image_reference_audit", audit)
        self.assertEqual(audit["req9_deit_image_reference_audit"]["status"], "pass")
        self.assertIn("Req9 audit status=pass", by_id["9"]["note"])
        self.assertEqual(by_id["8"]["status"], "reflected")
        self.assertEqual(by_id["11"]["evidence_items"][-1]["path"], str(Path(tmp.name) / "XR-VITs"))
        self.assertIn("validated-by-tool", audit["summary"]["evidence_classes"])
        self.assertIn("external-blocker", audit["summary"]["evidence_classes"])
        self.assertEqual(by_id["11"]["status"], "blocked")
        self.assertEqual(audit["gate_modes"]["vref_successor_status"], "pending-physical-smoke")
        self.assertEqual(audit["gate_modes"]["qkv_uram_successor"]["status"], "pass")
        self.assertEqual(audit["gate_modes"]["qkv_uram_successor"]["resources"]["uram"], 40)
        self.assertEqual(audit["gate_modes"]["qkv_uram_successor"]["promotion_status"], "hls-resource-pass-routed-pending")
        self.assertEqual(audit["gate_modes"]["qkv_uram_successor"]["physical_smoke_status"], "missing")
        self.assertEqual(audit["gate_modes"]["qkv_uram_successor"]["pynq_plumbing_status"], "ready-for-artifacts")
        self.assertIn(
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram",
            audit["gate_modes"]["qkv_uram_successor"]["dry_run_import_command"],
        )
        self.assertEqual(audit["gate_modes"]["qkv_uram_runner"]["qkv_uram_import_status"], "skipped")
        self.assertEqual(audit["gate_modes"]["qkv_uram_runner"]["qkv_uram_remote_status"], "skipped")
        self.assertEqual(audit["gate_modes"]["qkv_uram_runner"]["pynq_c3b_smoke_discovery_status"], "missing")
        self.assertEqual(audit["gate_modes"]["qkv_uram_runner"]["pynq_vref_smoke_discovery_status"], "missing")
        self.assertEqual(audit["gate_modes"]["qkv_uram_runner"]["qkv_uram_smoke_discovery_status"], "missing")
        self.assertFalse(audit["gate_modes"]["qkv_uram_runner"]["qkv_uram_required_for_final_signoff"])
        self.assertEqual(audit["gate_modes"]["vref_smoke_discovery"]["status"], "missing")
        self.assertEqual(audit["gate_modes"]["vref_smoke_discovery"]["candidate_count"], 0)
        discovery = audit["gate_modes"]["smoke_candidate_discovery"]
        self.assertEqual(discovery["c3b_generic"]["status"], "missing")
        self.assertEqual(discovery["c3b_generic"]["preset"], "axis-c3b-mem16")
        self.assertEqual(discovery["vref_generic"]["status"], "missing")
        self.assertEqual(discovery["vref_runner"]["status"], "missing")
        self.assertEqual(discovery["qkv_runner"]["status"], "missing")
        self.assertEqual(discovery["qkv_runner"]["preset"], "axis-vref-p0-softmax-input-x2-qkv-uram")
        self.assertIn("generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json", audit["pending_optional_or_internal_evidence"])
        self.assertIn("generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json", audit["pending_optional_or_internal_evidence"])
        self.assertEqual(audit["gate_modes"]["final_blocker_closure"]["status"], "blocked")
        self.assertFalse(audit["gate_modes"]["final_blocker_closure"]["current_ready"])
        self.assertEqual(audit["gate_modes"]["final_unblock_intake"]["status"], "blocked")
        self.assertFalse(audit["gate_modes"]["final_unblock_intake"]["would_clear_all"])
        self.assertIn("generated/signoff/final_blocker_closure_readiness_2026_06_10.json", audit["pending_optional_or_internal_evidence"])
        self.assertEqual(audit["gate_modes"]["final_unblock_commands"]["status"], "pending-unblock")
        self.assertEqual(audit["gate_modes"]["final_unblock_commands"]["section_count"], 6)
        self.assertTrue(audit["gate_modes"]["final_unblock_commands"]["has_qkv_uram_u5"])
        self.assertTrue(audit["gate_modes"]["final_unblock_commands"]["xr_vits_policy_integrity"]["required"])
        self.assertTrue(
            audit["gate_modes"]["final_unblock_commands"]["xr_vits_policy_integrity"][
                "required_policy_fields_complete"
            ]
        )
        self.assertEqual(audit["gate_modes"]["default"]["qkv_uram_successor_gate"]["status"], "ready-for-physical-smoke")
        self.assertEqual(audit["gate_modes"]["default"]["qkv_uram_command_count"], 2)
        self.assertEqual(audit["gate_modes"]["final_unblock_candidate_audit"]["status"], "blocked")
        self.assertFalse(audit["gate_modes"]["final_unblock_candidate_audit"]["would_clear_all"])
        self.assertEqual(audit["gate_modes"]["final_operator_handoff"]["status"], "pending-operator-actions")
        self.assertTrue(audit["gate_modes"]["final_operator_handoff"]["requires_operator_action"])
        self.assertEqual(
            audit["gate_modes"]["final_operator_handoff"]["xr_vits_policy_integrity"]["validation_status"],
            "pending-policy-creation",
        )
        self.assertEqual(audit["gate_modes"]["final_operator_handoff_validation"]["status"], "pass")
        self.assertEqual(audit["gate_modes"]["final_operator_handoff_validation"]["fail_count"], 0)
        self.assertEqual(audit["gate_modes"]["final_operator_handoff_validation"]["policy_check_count"], 7)
        self.assertTrue(audit["gate_modes"]["xr_vits_policy_integrity"]["consistent"])
        self.assertEqual(audit["gate_modes"]["xr_vits_unblock_packet"]["status"], "pending-user-choice")
        self.assertEqual(audit["gate_modes"]["xr_vits_unblock_packet"]["option_count"], 2)
        self.assertIn("generated/signoff/final_unblock_commands_2026_06_10.json", audit["pending_optional_or_internal_evidence"])
        self.assertIn("generated/signoff/final_operator_handoff_2026_06_10.json", audit["pending_optional_or_internal_evidence"])
        self.assertEqual(
            audit["external_blocker_names"],
            ["C3b AXIS/DMA physical smoke result", "requested XR-VITs sibling"],
        )
        self.assertEqual(len(audit["blocked_external_inputs"]), 2)
        self.assertIn(
            "e2e_axis_dma_c3b_mem16_file_smoke.json",
            audit["blocked_external_input_paths_by_blocker"]["C3b AXIS/DMA physical smoke result"],
        )
        self.assertEqual(
            audit["blocked_external_input_paths_by_blocker"]["requested XR-VITs sibling"],
            str(Path(tmp.name) / "XR-VITs"),
        )
        self.assertEqual(audit["blocker_schema"]["remaining_external_inputs_kind"], "absolute paths for missing or optional external inputs")
        self.assertIn(str(Path(tmp.name) / "XR-VITs"), audit["remaining_external_inputs"])

    def test_write_outputs_creates_json_and_markdown(self) -> None:
        tmp, hardware = self.make_hardware_root()
        self.addCleanup(tmp.cleanup)
        audit = audit_tool.build_audit(hardware)
        json_out = hardware / "generated/signoff/third_goal_current_audit_2026_06_16.json"
        md_out = hardware / "generated/signoff/third_goal_current_audit_2026_06_16.md"

        audit_tool.write_outputs(audit, json_out, md_out)

        self.assertEqual(json.loads(json_out.read_text())["status"], "blocked-external")
        markdown = md_out.read_text()
        self.assertIn("Third Goal Current Audit", markdown)
        self.assertIn("vref_smoke_discovery", markdown)
        self.assertIn("qkv_uram_runner", markdown)
        self.assertIn("qkv_uram_physical_smoke", markdown)
        self.assertIn("smoke_candidate_discovery.c3b_generic", markdown)
        self.assertIn("smoke_candidate_discovery.vref_generic", markdown)
        self.assertIn("smoke_candidate_discovery.qkv_runner", markdown)
        self.assertIn("final_blocker_closure", markdown)
        self.assertIn("final_unblock_intake", markdown)
        self.assertIn("final_unblock_commands", markdown)
        self.assertIn("has_qkv_u5", markdown)
        self.assertIn("Blocked external input paths by blocker", markdown)
        self.assertIn("final_unblock_candidate_audit", markdown)
        self.assertIn("final_operator_handoff", markdown)
        self.assertIn("final_operator_handoff_validation", markdown)
        self.assertIn("xr_vits_policy_integrity", markdown)
        self.assertIn("xr_vits_unblock_packet", markdown)
        self.assertIn("req1_environment", markdown)
        self.assertIn("req5_q4q8_swhw_match", markdown)
        self.assertIn("Pending Optional Or Internal Evidence", markdown)


if __name__ == "__main__":
    unittest.main()
