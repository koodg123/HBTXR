"""HG-PIPE quantization reconstruction package."""

from .api import HgPipeQuantizationPackage, RunnerPairResult
from .artifact_imagenet import evaluate_artifact_image_batch_npy, write_artifact_image_batch_report
from .artifact_patch import evaluate_artifact_patch_batch_npy, evaluate_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest, write_artifact_patch_batch_report, write_artifact_patch_matrix_manifest_template, write_artifact_patch_matrix_manifest_validation_json, write_artifact_patch_matrix_manifest_validation_markdown, write_artifact_patch_matrix_report
from .completion_audit import audit_completion, write_completion_audit_markdown
from .fake_quant import AffineFakeQuantizer, FakeQuantGraphRunner, HGTableFakeQuantizer, insert_output_fake_quantizers
from .int_infer import TorchIntCaseRunner, TorchIntGraphRunner
from .paper_equivalence import scan_paper_equivalence_assets, validate_artifact_imagenet_report, write_artifact_imagenet_validation_json, write_artifact_imagenet_validation_markdown, write_paper_equivalence_assets_json, write_paper_equivalence_assets_markdown
from .pipeline import discover_cases, verify_all
from .quant_params import AffineQuantParams, LutQuantParams, OpQuantContract, QuantParamStore, TensorDTypeSpec, TensorRangeSpec
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
    "compare_inference_results",
    "discover_cases",
    "evaluate_artifact_image_batch_npy",
    "evaluate_artifact_patch_batch_npy",
    "evaluate_artifact_patch_matrix_manifest",
    "validate_artifact_patch_matrix_manifest",
    "insert_output_fake_quantizers",
    "make_inference_result",
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
