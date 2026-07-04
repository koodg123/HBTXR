# HG-PIPE Quantization

This directory reconstructs the quantization path used by ICCAD24-HG-PIPE. It treats the original repository as a read-only evidence source and implements bit-exact Python equivalents of the HLS quantization kernels.

## What Is Implemented

- Table-based ReQuant: cursor = (x + b) >> s, clamp, table lookup.
- GeLU-ReQuant fusion with the same table lookup kernel.
- LayerNorm quantization with integer mean, rsqrt table, affine shift, and signed clamp.
- Softmax quantization with inverse-exp table, segmented reciprocal tables, and unsigned clamp.
- Automatic discovery of refs under ../ICCAD24-HG-PIPE/case/refs.
- Input/output statistics attachment from statistics/type.npy and statistics/range.npy.
- End-to-end verification against all discovered golden input/output reference files.
- Artifact graph verification for patch embedding, attention blocks, MLP blocks, and head.
- Single-input end-to-end graph verification from patch embedding input through head.
- Experimental ImageNet tensor-to-patch input bridge with explicit-scale int8 quantization.
- FakeQuantizer modules for affine and HG-PIPE LUT contracts plus FX output insertion utility.
- Torch integer end-to-end runner for patch input through head.
- FakeQuantizer-inserted artifact graph runner with comparable final logits/top-k output.
- Structured quant parameter loading for scalar, LUT, dtype, range, and affine placeholders.
- Paper-equivalence asset preflight and artifact ImageNet 3 x 3 validator reports.

## Usage

From this directory:

~~~bash
python3 -m hgpipe_quantization.cli audit-completion --refresh
python3 -m hgpipe_quantization.cli audit-completion
python3 -m hgpipe_quantization.cli list --limit 10
python3 -m hgpipe_quantization.cli verify
python3 -m hgpipe_quantization.cli verify-graph
python3 -m hgpipe_quantization.cli verify-e2e
python3 -m hgpipe_quantization.cli verify-int
python3 -m hgpipe_quantization.cli verify-fakequant
python3 -m hgpipe_quantization.cli verify-fakequant-graph
python3 -m hgpipe_quantization.cli export-contracts
python3 -m hgpipe_quantization.cli run-int
python3 -m hgpipe_quantization.cli run-fakequant-graph
python3 -m hgpipe_quantization.cli run-compare
python3 -m hgpipe_quantization.cli estimate-input-scale-npy --images-npy reports/smoke_scale_images.npy
python3 -m hgpipe_quantization.cli run-compare-image-npy --image-npy reports/smoke_zero_image.npy --scale 1.0
python3 -m hgpipe_quantization.cli run-artifact-image-batch-npy --images-npy reports/smoke_artifact_images.npy --labels-npy reports/smoke_artifact_labels.npy --scale 1.0
python3 -m hgpipe_quantization.cli run-artifact-patch-batch-npy --patch-inputs-npy reports/smoke_patch_inputs.npy --labels-npy reports/smoke_patch_labels.npy --model deit_tiny_patch16_224 --precision int8
python3 -m hgpipe_quantization.cli write-artifact-patch-matrix-template --json configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli validate-artifact-patch-matrix-manifest --manifest configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli run-artifact-patch-matrix-npy --manifest reports/artifact_patch_smoke_manifest.json --json reports/artifact_patch_smoke_matrix_accuracy.json
python3 -m hgpipe_quantization.cli validate-artifact-imagenet-report --report reports/artifact_imagenet_accuracy.json
python3 -m hgpipe_quantization.cli check-paper-equivalence-assets
python3 -m hgpipe_quantization.cli compare-run-results --left reports/run_fakequant_graph_result.json --right reports/run_int_result.json
python3 -m hgpipe_quantization.cli trace-fakequant-graph
python3 -m hgpipe_quantization.cli trace-int-cases
python3 -m hgpipe_quantization.cli trace-fakequant
python3 -m hgpipe_quantization.cli compare-traces --left reports/fakequant_graph_trace.json --right reports/torch_int_case_trace.json
~~~

The verify and run commands write reports under reports, including:

- reports/verification.json and reports/verification.md
- reports/torch_int_verification.json
- reports/fakequant_graph_verification.json
- reports/quant_contracts.json
- reports/requant_contracts_with_tables.json
- reports/run_int_result.json
- reports/run_fakequant_graph_result.json
- reports/run_result_comparison.json
- reports/run_compare_result.json and reports/run_compare_result.md
- reports/run_compare_image_result.json and reports/run_compare_image_result.md
- reports/input_scale_estimate.json
- reports/artifact_imagenet_accuracy.json
- reports/artifact_patch_smoke_accuracy.json and reports/artifact_patch_smoke_validation.json
- reports/artifact_patch_smoke_manifest.json, reports/artifact_patch_smoke_matrix_accuracy.json, and reports/artifact_patch_smoke_matrix_validation.json
- reports/artifact_patch_matrix_manifest_validation.json and reports/artifact_patch_matrix_manifest_validation.md
- configs/artifact_patch_matrix_manifest.template.json
- reports/artifact_imagenet_validation.json and reports/artifact_imagenet_validation.md
- reports/paper_equivalence_assets.json and reports/paper_equivalence_assets.md
- reports/completion_audit.json and reports/completion_audit.md

## Evidence Sources

- Paper: ../Vision Transformer Acceleration with Hybrid-Grained Pipeline.pdf
- HLS kernels: ../ICCAD24-HG-PIPE/src/quant.h, layernorm.h, softmax.h, gelu.h
- Generated cases and references: ../ICCAD24-HG-PIPE/case/
- Quantization statistics: ../ICCAD24-HG-PIPE/statistics/type.npy and range.npy

## Contract Export Note

HG-PIPE public artifacts encode table quantization with scalar tuples (offset b, shift-scale s, bound) and LUT tables. The package exports these as scalars, offset, shift_scale, effective_divisor, bound, and table_sizes or full tables. For these LUT contracts zero_point is None because no affine zero-point artifact is present. AffineFakeQuantizer still supports caller-supplied scale and zero_point for ordinary affine fake quantization.

## Experimental Artifact Image-Batch Command

Command: python3 -m hgpipe_quantization.cli run-artifact-image-batch-npy --images-npy reports/smoke_artifact_images.npy --labels-npy reports/smoke_artifact_labels.npy --scale 1.0
python3 -m hgpipe_quantization.cli run-artifact-patch-batch-npy --patch-inputs-npy reports/smoke_patch_inputs.npy --labels-npy reports/smoke_patch_labels.npy --model deit_tiny_patch16_224 --precision int8
python3 -m hgpipe_quantization.cli write-artifact-patch-matrix-template --json configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli validate-artifact-patch-matrix-manifest --manifest configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli run-artifact-patch-matrix-npy --manifest reports/artifact_patch_smoke_manifest.json --json reports/artifact_patch_smoke_matrix_accuracy.json

This writes reports/artifact_imagenet_accuracy.json. The report is artifact-backed and explicit-scale, but paper_equivalent remains false.


## Artifact Patch-Input Batch Command

Command: python3 -m hgpipe_quantization.cli run-artifact-patch-batch-npy --patch-inputs-npy reports/smoke_patch_inputs.npy --labels-npy reports/smoke_patch_labels.npy --model deit_tiny_patch16_224 --precision int8
python3 -m hgpipe_quantization.cli write-artifact-patch-matrix-template --json configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli validate-artifact-patch-matrix-manifest --manifest configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli run-artifact-patch-matrix-npy --manifest reports/artifact_patch_smoke_manifest.json --json reports/artifact_patch_smoke_matrix_accuracy.json

This writes an artifact graph ImageNet-style row from already-quantized HG-PIPE patch input tensors. Use --paper-equivalent-inputs only when the .npy patch inputs were generated by the original or accepted replacement HG-PIPE preprocessing/calibration flow. With --append, this command can fill or replace one model/precision row in reports/artifact_imagenet_accuracy.json.


## Artifact Patch Matrix Manifest

Command: python3 -m hgpipe_quantization.cli write-artifact-patch-matrix-template --json configs/artifact_patch_matrix_manifest.template.json
python3 -m hgpipe_quantization.cli validate-artifact-patch-matrix-manifest --manifest configs/artifact_patch_matrix_manifest.template.json

Command: python3 -m hgpipe_quantization.cli run-artifact-patch-matrix-npy --manifest reports/artifact_patch_smoke_manifest.json --json reports/artifact_patch_smoke_matrix_accuracy.json

The template contains the expected DeiT-tiny, DeiT-small, and ViT-tiny rows across int8, int4, and w4a8. Replace the placeholder patch input and label paths with official already-quantized HG-PIPE patch tensors, then run the matrix command to produce reports/artifact_imagenet_accuracy.json or another report path.

## Artifact ImageNet Paper-Equivalence Validator

Command: python3 -m hgpipe_quantization.cli validate-artifact-imagenet-report --report reports/artifact_imagenet_accuracy.json

This writes reports/artifact_imagenet_validation.json and reports/artifact_imagenet_validation.md. The current report is expected to fail because artifact_imagenet_accuracy.json is an experimental explicit-scale smoke report with rows=1, expected_rows=9, missing_pairs=9, and paper_equivalent_rows=0.

## Paper-Equivalence Asset Preflight

Command: python3 -m hgpipe_quantization.cli check-paper-equivalence-assets

This writes reports/paper_equivalence_assets.json and reports/paper_equivalence_assets.md. The current preflight is incomplete because original preprocessing, QAT or calibration flow, quantized checkpoints, and full paper-equivalent artifact ImageNet matrix are not present.

## Native ViT/DeiT ImageNet Models

The ImageNet evaluation path now instantiates deit_tiny_patch16_224, deit_small_patch16_224, and vit_tiny_patch16_224 from pure torch.nn modules under hgpipe_quantization/models/vision_transformer.py. timm is no longer required to construct these three models. Optional checkpoint loading is available through hgpipe_quantization.eval.model_registry.create_paper_model and python -m hgpipe_quantization.eval.imagenet_eval --checkpoint PATH; without a checkpoint, evaluation runs with deterministic random initialization and reports pretrained=false.
