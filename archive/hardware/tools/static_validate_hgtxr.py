#!/usr/bin/env python3
"""Static validation checklist for HGTXR generated files and handoff artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_FILES = [
    "hardware/hls/include/hgtxr_cyclic_transformer_params.hpp",
    "hardware/hls/include/hgtxr_cyclic_mac.hpp",
    "hardware/hls/include/hgtxr_cyclic_scheduler.hpp",
    "hardware/hls/include/hgtxr_cyclic_math.hpp",
    "hardware/hls/include/hgtxr_cyclic_norm.hpp",
    "hardware/hls/include/hgtxr_cyclic_attention.hpp",
    "hardware/hls/include/hgtxr_cyclic_mlp.hpp",
    "hardware/hls/include/hgtxr_cyclic_transformer_block.hpp",
    "hardware/hls/include/hgtxr_cyclic_weight_layout.hpp",
    "hardware/hls/include/hgtxr_cyclic_s2_projection.hpp",
    "hardware/hls/include/hgtxr_e2e_vit.hpp",
    "hardware/hls/src/hgtxr_e2e_axis_top.cpp",
    "hardware/hls/src/hgtxr_e2e_m_axi_top.cpp",
    "hardware/hls/tb/e2e_axis_vector_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active8_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active16_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active16_ff64_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active16_ff128_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active16_ff256_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active16_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active32_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active64_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active128_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active196_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_active196_b6_ff768_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_hgpipe_math_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp",
    "hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active16_golden.hpp",
    "hardware/hls/tb/tb_cyclic_primitives.cpp",
    "hardware/hls/tb/tb_cyclic_s2_projection.cpp",
    "hardware/hls/tb/tb_cyclic_head_attention.cpp",
    "hardware/hls/tb/tb_cyclic_s2_block_vector.cpp",
    "hardware/hls/tb/tb_cyclic_s2_block_vector_hls_main.cpp",
    "hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp",
    "hardware/hls/tb/tb_hgtxr_e2e_m_axi_top.cpp",
    "software/tests/test_e2e_axis_vector_csim.py",
    "hardware/configs/sweeps/zcu104_cyclic_transformer_sweep.yaml",
    "hardware/vivado/scripts/run_cyclic_s0_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s1_weight_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_first_step_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_qkv_first_step_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_attn_first_step_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_mlp_first_step_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_block_first_step_csynth.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_block_q4w8a_csim.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_block_vector_csim.tcl",
    "hardware/vivado/scripts/run_cyclic_s2_block_q4w8a_csynth.tcl",
    "hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl",
    "hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl",
    "hardware/vivado/scripts/run_e2e_m_axi_q4w8a_csim.tcl",
    "hardware/vivado/scripts/run_e2e_m_axi_q4w8a_csynth.tcl",
    "hardware/vivado/scripts/package_e2e_axis_ip.tcl",
    "hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl",
    "hardware/vivado/scripts/package_e2e_m_axi_ip.tcl",
    "hardware/vivado/scripts/build_e2e_m_axi_bitstream.tcl",
    "hardware/pynq/hgtxr/e2e_axis_dma_overlay.py",
    "hardware/pynq/hgtxr/e2e_m_axi_overlay.py",
    "hardware/pynq/hgtxr/e2e_m_axi_weights.py",
    "hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py",
    "hardware/pynq/hgtxr/run_e2e_m_axi_smoke.py",
    "hardware/configs/zcu104_cyclic_s0_defines.h",
    "hardware/configs/zcu104_cyclic_s1_weight_defines.h",
    "hardware/configs/zcu104_cyclic_s2_first_step_defines.h",
    "hardware/configs/zcu104_cyclic_s2_qkv_first_step_defines.h",
    "hardware/configs/zcu104_cyclic_s2_attn_first_step_defines.h",
    "hardware/configs/zcu104_cyclic_s2_mlp_first_step_defines.h",
    "hardware/configs/zcu104_cyclic_s2_block_first_step_defines.h",
    "hardware/configs/zcu104_cyclic_s2_block_q4w8a_defines.h",
    "hardware/configs/zcu104_e2e_q4w8a_defines.h",
    "hardware/tools/repair_hls_to_pynq.py",
    "hardware/tools/hgtxr_sweep_expand.py",
    "hardware/tools/extract_resource_metrics.py",
    "hardware/tools/pack_cyclic_weights.py",
    "hardware/tools/validate_s2_block_preln.py",
    "hardware/tools/validate_s2_block_full.py",
    "hardware/tools/export_s2_pytorch_golden.py",
    "hardware/tools/validate_hgpipe_lut_math.py",
    "hardware/tools/validate_e2e_axis_vector.py",
    "hardware/tools/check_third_goal_preflight.py",
    "hardware/tools/export_e2e_m_axi_weights.py",
    "hardware/tools/package_e2e_m_axi_pynq_bundle.py",
    "hardware/tools/package_e2e_axis_dma_pynq_bundle.py",
    "hardware/tools/validate_pynq_smoke_result.py",
    "hardware/tools/validate_pynq_bundle_package.py",
    "hardware/tools/import_pynq_smoke_result.py",
    "hardware/tools/prepare_zcu104_smoke_session.py",
    "hardware/tools/write_final_signoff_audit.py",
    "hardware/tools/audit_reference_inputs.py",
    "hardware/tools/audit_xr_vits_candidates.py",
    "hardware/tools/create_xr_vits_replacement_policy.py",
    "hardware/tools/write_third_goal_completion_audit.py",
    "hardware/tools/write_third_goal_unblock_checklist.py",
    "hardware/tools/write_c3b_smoke_transfer_manifest.py",
    "hardware/tools/check_c3b_board_smoke_readiness.py",
    "hardware/tools/check_final_blocker_closure_readiness.py",
    "hardware/tools/check_xr_vits_reference_resolution.py",
    "hardware/tools/discover_c3b_smoke_candidates.py",
    "hardware/tools/run_third_goal_final_signoff.py",
    "hardware/tools/run_zcu104_c3b_smoke_remote.py",
    "hardware/tools/write_c3b_smoke_result_contract.py",
    "hardware/tools/write_xr_vits_unblock_packet.py",
    "hardware/tools/write_final_unblock_commands.py",
    "hardware/tools/write_final_unblock_intake.py",
    "hardware/tools/write_final_unblock_closeout_packet.py",
    "hardware/tools/validate_final_unblock_closeout_packet.py",
    "hardware/tools/write_e2e_resource_matrix.py",
    "hardware/tools/write_selected_path_execution_audit.py",
    "hardware/tools/write_e2e_resource_policy_audit.py",
    "hardware/tools/write_third_goal_requirements_trace.py",
    "hardware/tools/write_spec_plan_conformance_audit.py",
    "hardware/tools/audit_final_unblock_candidates.py",
    "hardware/tools/write_final_operator_handoff.py",
    "hardware/tools/validate_final_operator_handoff.py",
    "hardware/tools/validate_final_signoff_bundle.py",
    "hardware/tools/write_final_evidence_manifest.py",
    "hardware/tests/test_check_third_goal_preflight.py",
    "hardware/tests/test_package_e2e_m_axi_pynq_bundle.py",
    "hardware/tests/test_package_e2e_axis_dma_pynq_bundle.py",
    "hardware/tests/test_validate_pynq_smoke_result.py",
    "hardware/tests/test_validate_pynq_bundle_package.py",
    "hardware/tests/test_import_pynq_smoke_result.py",
    "hardware/tests/test_prepare_zcu104_smoke_session.py",
    "hardware/tests/test_write_final_signoff_audit.py",
    "hardware/tests/test_audit_reference_inputs.py",
    "hardware/tests/test_audit_xr_vits_candidates.py",
    "hardware/tests/test_create_xr_vits_replacement_policy.py",
    "hardware/tests/test_write_third_goal_completion_audit.py",
    "hardware/tests/test_write_third_goal_unblock_checklist.py",
    "hardware/tests/test_write_c3b_smoke_transfer_manifest.py",
    "hardware/tests/test_check_c3b_board_smoke_readiness.py",
    "hardware/tests/test_check_final_blocker_closure_readiness.py",
    "hardware/tests/test_check_xr_vits_reference_resolution.py",
    "hardware/tests/test_discover_c3b_smoke_candidates.py",
    "hardware/tests/test_run_third_goal_final_signoff.py",
    "hardware/tests/test_run_zcu104_c3b_smoke_remote.py",
    "hardware/tests/test_write_c3b_smoke_result_contract.py",
    "hardware/tests/test_write_xr_vits_unblock_packet.py",
    "hardware/tests/test_write_final_unblock_commands.py",
    "hardware/tests/test_write_final_unblock_intake.py",
    "hardware/tests/test_write_final_unblock_closeout_packet.py",
    "hardware/tests/test_validate_final_unblock_closeout_packet.py",
    "hardware/tests/test_write_e2e_resource_matrix.py",
    "hardware/tests/test_write_selected_path_execution_audit.py",
    "hardware/tests/test_write_e2e_resource_policy_audit.py",
    "hardware/tests/test_write_third_goal_requirements_trace.py",
    "hardware/tests/test_write_spec_plan_conformance_audit.py",
    "hardware/tests/test_audit_final_unblock_candidates.py",
    "hardware/tests/test_write_final_operator_handoff.py",
    "hardware/tests/test_validate_final_operator_handoff.py",
    "hardware/tests/test_validate_final_signoff_bundle.py",
    "hardware/tests/test_write_final_evidence_manifest.py",
    "hardware/refs/hgpipe_lut_math_contract.json",
    "hardware/refs/e2e_axis_vector_spec.json",
    "hardware/refs/e2e_axis_vector_blocks2_spec.json",
    "hardware/refs/e2e_axis_vector_active8_spec.json",
    "hardware/refs/e2e_axis_vector_active16_spec.json",
    "hardware/refs/e2e_axis_vector_active16_ff64_spec.json",
    "hardware/refs/e2e_axis_vector_active16_ff128_spec.json",
    "hardware/refs/e2e_axis_vector_active16_ff256_spec.json",
    "hardware/refs/e2e_axis_vector_active16_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_active32_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_active64_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_active128_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_active196_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_active196_b6_ff768_spec.json",
    "hardware/refs/e2e_axis_vector_hgpipe_math_spec.json",
    "hardware/refs/e2e_axis_vector_hgpipe_math_lnq_spec.json",
    "hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json",
    "hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active16_spec.json",
]

DOC_FILES = [
    "docs/Spec-2026-06-05-ZCU104-Cyclic-Transformer.md",
    "docs/Architecture-2026-06-05-Cyclic-Transformer.md",
    "docs/Experiment-Matrix-2026-06-05-ZCU104.md",
    "docs/QA-Gates-2026-06-05-HLS-to-PYNQ.md",
    "docs/Resource-Extraction-Protocol-2026-06-06.md",
    "docs/resources/resource_extraction_log.md",
    "docs/resources/hls_csynth_summary_2026_06_06.csv",
    "docs/resources/cyclic_weight_layout_2026_06_06.md",
    "docs/resources/cyclic_weight_packing_2026_06_06.md",
    "docs/resources/hgpipe_reference_analysis_2026_06_08.md",
    "docs/resources/e2e_axis_baseline_2026_06_09.md",
    "docs/resources/s2_block_preln_validation_2026_06_06.json",
    "docs/resources/s2_block_full_validation_2026_06_06.json",
    "docs/resources/s2_block_full_validation_software_initial_2026_06_06.json",
    "docs/resources/s2_block_full_validation_software_initial_expected_2026_06_06.json",
    "docs/resources/s2_block_full_outputs_software_initial_2026_06_06.npz",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial.bin",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial.npz",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_manifest.json",
    "docs/resources/s2_pytorch_golden_software_initial_2026_06_08.json",
    "docs/resources/s2_pytorch_golden_software_initial_2026_06_08.npz",
    "docs/resources/q4w8a_s2_block_quantization_2026_06_08.json",
    "docs/resources/hgpipe_lut_math_validation_2026_06_09.json",
    "docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json",
    "docs/resources/third_goal_preflight_2026_06_10.json",
    "docs/resources/final_signoff_audit_2026_06_10.json",
    "docs/resources/final_signoff_audit_2026_06_10.md",
    "docs/resources/reference_input_audit_2026_06_10.json",
    "docs/resources/reference_input_audit_2026_06_10.md",
    "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
    "docs/resources/xr_vits_candidate_audit_2026_06_10.md",
    "docs/resources/xr_vits_replacement_policy.template.json",
    "docs/resources/third_goal_completion_audit_2026_06_10.json",
    "docs/resources/third_goal_completion_audit_2026_06_10.md",
    "docs/resources/third_goal_unblock_checklist_2026_06_10.json",
    "docs/resources/third_goal_unblock_checklist_2026_06_10.md",
    "docs/resources/c3b_smoke_transfer_manifest_2026_06_10.json",
    "docs/resources/c3b_smoke_transfer_manifest_2026_06_10.md",
    "docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
    "docs/resources/c3b_board_smoke_readiness_2026_06_10.json",
    "docs/resources/c3b_board_smoke_readiness_2026_06_10.md",
    "docs/resources/third_goal_final_signoff_2026_06_10.json",
    "docs/resources/third_goal_final_signoff_run_2026_06_10.json",
    "docs/resources/third_goal_final_signoff_run_2026_06_10.md",
    "docs/resources/zcu104_c3b_smoke_remote_run_2026_06_10.json",
    "docs/resources/zcu104_c3b_smoke_remote_run_2026_06_10.md",
    "docs/resources/c3b_smoke_result_contract_2026_06_10.json",
    "docs/resources/c3b_smoke_result_contract_2026_06_10.md",
    "docs/resources/xr_vits_unblock_packet_2026_06_10.json",
    "docs/resources/xr_vits_unblock_packet_2026_06_10.md",
    "docs/resources/xr_vits_reference_resolution_2026_06_10.json",
    "docs/resources/xr_vits_reference_resolution_2026_06_10.md",
    "docs/resources/final_unblock_commands_2026_06_10.json",
    "docs/resources/final_unblock_commands_2026_06_10.md",
    "docs/resources/final_unblock_intake_2026_06_10.json",
    "docs/resources/final_unblock_intake_2026_06_10.md",
    "docs/resources/final_unblock_closeout_packet_2026_06_10.json",
    "docs/resources/final_unblock_closeout_packet_2026_06_10.md",
    "docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json",
    "docs/resources/final_unblock_closeout_packet_validation_2026_06_10.md",
    "docs/resources/e2e_resource_matrix_2026_06_10.json",
    "docs/resources/e2e_resource_matrix_2026_06_10.md",
    "docs/resources/selected_path_execution_audit_2026_06_10.json",
    "docs/resources/selected_path_execution_audit_2026_06_10.md",
    "docs/resources/e2e_resource_policy_audit_2026_06_10.json",
    "docs/resources/e2e_resource_policy_audit_2026_06_10.md",
    "docs/resources/third_goal_requirements_trace_2026_06_10.json",
    "docs/resources/third_goal_requirements_trace_2026_06_10.md",
    "docs/resources/spec_plan_conformance_audit_2026_06_10.json",
    "docs/resources/spec_plan_conformance_audit_2026_06_10.md",
    "docs/resources/final_unblock_candidate_audit_2026_06_10.json",
    "docs/resources/final_unblock_candidate_audit_2026_06_10.md",
    "docs/resources/final_operator_handoff_2026_06_10.json",
    "docs/resources/final_operator_handoff_2026_06_10.md",
    "docs/resources/final_operator_handoff_validation_2026_06_10.json",
    "docs/resources/final_operator_handoff_validation_2026_06_10.md",
    "docs/resources/final_signoff_bundle_validation_2026_06_10.json",
    "docs/resources/final_signoff_bundle_validation_2026_06_10.md",
    "docs/resources/c3b_smoke_candidate_discovery_2026_06_10.json",
    "docs/resources/c3b_smoke_candidate_discovery_2026_06_10.md",
    "docs/resources/final_blocker_closure_readiness_2026_06_10.json",
    "docs/resources/final_blocker_closure_readiness_2026_06_10.md",
    "docs/resources/final_evidence_manifest_2026_06_10.json",
    "docs/resources/final_evidence_manifest_2026_06_10.md",
    "docs/track/GOAL-AUDIT-2026-06-10-THIRD-GOAL.md",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4.bin",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4.npz",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_manifest.json",
    "docs/resources/s2_pytorch_golden_software_initial_q4w8a_head64_2026_06_10.npz",
    "docs/resources/s2_pytorch_golden_software_initial_q4w8a_head64_2026_06_10.json",
    "docs/resources/s2_block_full_q4w8a_head64_outputs_2026_06_10.npz",
    "docs/resources/s2_block_full_q4w8a_head64_outputs_hls_q8_2026_06_10.json",
    "docs/resources/s2_block_full_q4w8a_head64_outputs_hls_q8_2026_06_10.f32bin",
    "docs/resources/s2_block_full_q4w8a_head64_outputs_2026_06_10.f32bin",
    "docs/resources/s2_block_full_q4w8a_head64_validation_2026_06_10.json",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64_manifest.json",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.npz",
    "hardware/refs/weights/cyclic_weights_s2_block_software_initial_q4_head64.bin",
]

EXPECTED_ARTIFACTS = [
    "hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.runs/impl_1/hgtxr_system_wrapper.bit",
    "hardware/generated/build/vivado/hgtxr_overlay/hgtxr_overlay.gen/sources_1/bd/hgtxr_system/hw_handoff/hgtxr_system.hwh",
    "hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.bit",
    "hardware/generated/build/vivado/overlay/hgtxr_overlay/hgtxr.hwh",
    "hardware/pynq/hgtxr/hgtxr.bit",
    "hardware/pynq/hgtxr/hgtxr.hwh",
]

AXI_MARKERS = [
    "#pragma HLS INTERFACE m_axi",
    "#pragma HLS INTERFACE s_axilite",
    "port=return",
]

E2E_AXIS_MARKERS = [
    "void hgtxr_e2e_axis_top",
    "#pragma HLS INTERFACE axis",
    "#pragma HLS INTERFACE m_axi",
    "#pragma HLS INTERFACE s_axilite",
]

E2E_M_AXI_MARKERS = [
    "void hgtxr_e2e_m_axi_top",
    "#pragma HLS INTERFACE m_axi",
    "#pragma HLS INTERFACE s_axilite",
    "port=return",
]

A1_AXIS_DMA_TCL_MARKERS = [
    "xilinx.com:hls:hgtxr_e2e_axis_top:1.0",
    "xilinx.com:ip:axi_dma",
    "hgtxr_e2e_axis_dma",
    "${artifact_name}.bit",
    "${artifact_name}.hwh",
]

C_PAR_ISOLATION_TCL_MARKERS = [
    "HGTXR_E2E_RUN_TAG",
    "HGTXR_E2E_PROJECT_NAME",
    "HGTXR_E2E_MEM_BANK_PAR",
    "hgtxr_e2e_project_name",
]

C3_DSP_LUT_CLEANUP_MARKERS = [
    "HGTXR_E2E_MEM_BANK_PAR",
    "HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH",
]

C_PACKAGE_ISOLATION_TCL_MARKERS = [
    "-artifact_name",
    "${artifact_name}.bit",
    "${artifact_name}.hwh",
]


def check_exists(root: Path, rels: list[str], required: bool) -> int:
    failures = 0
    for rel in rels:
        path = root / rel
        if path.exists():
            print(f"[ok] {rel}")
        elif required:
            print(f"[missing] {rel}")
            failures += 1
        else:
            print(f"[not-yet] {rel}")
    return failures


def check_markers(root: Path, rel: str, markers: list[str], label: str, required: bool = True) -> int:
    path = root / rel
    if not path.exists():
        status = "[missing]" if required else "[not-yet]"
        print(f"{status} {rel}")
        return 1 if required else 0
    text = path.read_text(errors="ignore")
    failures = 0
    for marker in markers:
        if marker in text:
            print(f"[ok] {label} marker: {marker}")
        else:
            print(f"[missing] {label} marker: {marker}")
            failures += 1
    return failures


def check_axi_markers(root: Path) -> int:
    failures = 0
    failures += check_markers(
        root,
        "hardware/hls/src/hgtxr_top.cpp",
        AXI_MARKERS,
        "legacy top",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/hls/src/hgtxr_e2e_axis_top.cpp",
        E2E_AXIS_MARKERS,
        "E2E AXIS top",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/hls/src/hgtxr_e2e_m_axi_top.cpp",
        E2E_M_AXI_MARKERS,
        "E2E m_axi top",
        required=True,
    )
    return failures


def check_a1_axis_dma_markers(root: Path) -> int:
    return check_markers(
        root,
        "hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl",
        A1_AXIS_DMA_TCL_MARKERS,
        "A1 AXIS/DMA BD",
        required=True,
    )


def check_c_par_isolation_markers(root: Path) -> int:
    failures = 0
    failures += check_markers(
        root,
        "hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl",
        C_PAR_ISOLATION_TCL_MARKERS,
        "C PAR CSim isolation",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl",
        C_PAR_ISOLATION_TCL_MARKERS,
        "C PAR CSynth isolation",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/vivado/scripts/package_e2e_axis_ip.tcl",
        ["HGTXR_E2E_PROJECT_NAME", "hgtxr_e2e_axis_hls"],
        "C package project override",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl",
        C_PACKAGE_ISOLATION_TCL_MARKERS,
        "C board artifact isolation",
        required=True,
    )
    failures += check_markers(
        root,
        "hardware/hls/include/hgtxr_e2e_vit.hpp",
        C3_DSP_LUT_CLEANUP_MARKERS,
        "C3 DSP/LUT cleanup",
        required=True,
    )
    return failures


def check_spec_kit_status() -> int:
    spec_kit = shutil.which("spec-kit")
    specify = shutil.which("specify")
    if spec_kit:
        print(f"[ok] spec-kit on PATH: {spec_kit}")
    elif specify:
        print(f"[ok] specify on PATH: {specify}")
    else:
        print("[warn] spec-kit/specify not on PATH; project spec maintained manually")
    return 0


def check_tcl_patch(root: Path) -> int:
    script = root / "hardware" / "vivado" / "scripts" / "build_bitstream.tcl"
    if not script.exists():
        print("[missing] hardware/vivado/scripts/build_bitstream.tcl")
        return 1
    text = script.read_text(errors="ignore")
    if "glob -nocomplain -recursive -directory $build_dir *.hwh" in text:
        print("[missing] Vivado HWH copy Tcl patch")
        return 1
    print("[ok] Vivado HWH copy Tcl patch marker")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures = 0
    print(f"[root] {root}")
    failures += check_exists(root, REQUIRED_FILES, required=True)
    failures += check_exists(root, DOC_FILES, required=True)
    failures += check_axi_markers(root)
    failures += check_a1_axis_dma_markers(root)
    failures += check_c_par_isolation_markers(root)
    failures += check_tcl_patch(root)
    failures += check_spec_kit_status()
    failures += check_exists(root, EXPECTED_ARTIFACTS, required=args.require_artifacts)

    if failures:
        print(f"[fail] {failures} static checks failed")
        return 1
    print("[pass] static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
