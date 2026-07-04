"""HG-PIPE quantization reconstruction package."""

from .api import HgPipeQuantizationPackage, RunnerPairResult
from .artifact_imagenet import evaluate_artifact_image_batch_npy, write_artifact_image_batch_report
from .artifact_patch import evaluate_artifact_patch_batch_npy, evaluate_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest, write_artifact_patch_batch_report, write_artifact_patch_matrix_manifest_template, write_artifact_patch_matrix_manifest_validation_json, write_artifact_patch_matrix_manifest_validation_markdown, write_artifact_patch_matrix_report
from .completion_audit import audit_completion, write_completion_audit_markdown
from .fake_quant import AffineFakeQuantizer, FakeQuantGraphRunner, HGTableFakeQuantizer, insert_output_fake_quantizers
from .int_infer import TorchIntCaseRunner, TorchIntGraphRunner
from .lut_calibration import apply_table, calibrate_gelu_requant, calibrate_lut_from_array, calibrate_requant, calibrate_rsqrt, calibrate_softmax, cursor_for, write_hgpipe_txt_artifacts, write_lut_payload_json
from .paper_equivalence import scan_paper_equivalence_assets, validate_artifact_imagenet_report, write_artifact_imagenet_validation_json, write_artifact_imagenet_validation_markdown, write_paper_equivalence_assets_json, write_paper_equivalence_assets_markdown
from .pipeline import discover_cases, verify_all
from .quant_params import AffineQuantParams, LutQuantParams, OpQuantContract, QuantParamStore, TensorDTypeSpec, TensorRangeSpec
from .quantization_scheme import calibrate_dyadic_scale_kl, hardware_lut_index, quantize_group_vector, quantize_nonlinear_lut
from .run_result import InferenceComparison, InferenceResult, compare_inference_results, make_inference_result

__all__ = [
    "audit_completion",
    "write_completion_audit_markdown",
    "AffineFakeQuantizer",
    "AffineQuantParams",
    "FakeQuantGraphRunner",
    "HGTableFakeQuantizer",
    "HgPipeQuantizationPackage",
    "InferenceComparison",
    "InferenceResult",
    "LutQuantParams",
    "OpQuantContract",
    "QuantParamStore",
    "RunnerPairResult",
    "TensorDTypeSpec",
    "TensorRangeSpec",
    "TorchIntCaseRunner",
    "TorchIntGraphRunner",
    "apply_table",
    "calibrate_gelu_requant",
    "calibrate_dyadic_scale_kl",
    "calibrate_lut_from_array",
    "calibrate_requant",
    "calibrate_rsqrt",
    "calibrate_softmax",
    "compare_inference_results",
    "cursor_for",
    "discover_cases",
    "evaluate_artifact_image_batch_npy",
    "evaluate_artifact_patch_batch_npy",
    "evaluate_artifact_patch_matrix_manifest",
    "validate_artifact_patch_matrix_manifest",
    "insert_output_fake_quantizers",
    "hardware_lut_index",
    "make_inference_result",
    "quantize_group_vector",
    "quantize_nonlinear_lut",
    "scan_paper_equivalence_assets",
    "validate_artifact_imagenet_report",
    "write_artifact_imagenet_validation_json",
    "write_artifact_imagenet_validation_markdown",
    "write_paper_equivalence_assets_json",
    "write_paper_equivalence_assets_markdown",
    "verify_all",
    "write_artifact_image_batch_report",
    "write_artifact_patch_batch_report",
    "write_artifact_patch_matrix_manifest_template",
    "write_artifact_patch_matrix_manifest_validation_json",
    "write_artifact_patch_matrix_manifest_validation_markdown",
    "write_artifact_patch_matrix_report",
    "write_hgpipe_txt_artifacts",
    "write_lut_payload_json",
]

from .artifact_patch import ingest_artifact_patch_matrix_assets, load_artifact_patch_asset_source_manifest

__all__.extend([
    "ingest_artifact_patch_matrix_assets",
    "load_artifact_patch_asset_source_manifest",
])

from .artifact_patch import run_artifact_patch_matrix_pipeline

__all__.extend([
    "run_artifact_patch_matrix_pipeline",
])

from .artifact_patch import write_artifact_patch_asset_source_manifest_from_directory

__all__.extend([
    "write_artifact_patch_asset_source_manifest_from_directory",
])
