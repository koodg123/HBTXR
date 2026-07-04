# Changelog

## 0.1.0

- Added bit-exact Python reconstruction of HG-PIPE quantization kernels.
- Added refs discovery, statistics metadata, CLI verification, reports, and tests.

## 2026-06-09 - Artifact-backed graph verification

- Added hgpipe_quantization.graph.ArtifactGraphRunner for scale/table-based graph reconstruction from ICCAD24-HG-PIPE refs.
- Added verify-graph CLI and reports/graph_verification.json.
- Verified 256/256 graph-level cases with 0 mismatches.

## 2026-06-09 - Complete attention O-input reconstruction

- Reconstructed the attention A head-merge layout feeding gen_o_matmul from generated aq tensors.
- Removed the graph runner dependence on saved attn_*_gen_o_matmul_input.txt as the O matmul source; it is now used only as a verification target.
- Verified 268/268 graph-level cases with 0 mismatches.

## 2026-06-09 - Single-input end-to-end graph reconstruction

- Added forward_from_patch_input and verify_end_to_end graph execution.
- Added verify-e2e CLI and reports/e2e_graph_verification.json.
- Added graph_chain_boundary checks for all 25 stage transitions.
- Added experimental ImageNet tensor-to-patch input bridge and unit tests.
- Verified 293/293 end-to-end graph cases with 0 mismatches.

## 2026-06-09 - Quant parameter, FakeQuantizer, and torch integer runtime

- Added quant_params.py for structured scalar, LUT, dtype, range, and affine parameter contracts.
- Added fake_quant package with AffineFakeQuantizer, HGTableFakeQuantizer, and FX output insertion.
- Added int_infer package with torch integer kernels and TorchIntGraphRunner.
- Added verify-int CLI and reports/torch_int_verification.json.
- Verified torch_int_cases=293/293 with 0 mismatches.

## 2026-06-09 - FakeQuantizer verification CLI

- Added FakeQuantRunner for supported LUT-backed artifact cases.
- Added verify-fakequant CLI and reports/fakequant_verification.json.
- Verified fakequant_cases=60/60 with 0 mismatches.

## 2026-06-09 - Trace comparison reports

- Added shared TensorTrace schema.
- Added TorchIntCaseRunner LUT case traces.
- Added trace-fakequant, trace-int-cases, and compare-traces CLI commands.
- Generated reports/fakequant_trace.json, reports/torch_int_case_trace.json, reports/fakequant_vs_int.json, and reports/fakequant_vs_int.md.
- Verified trace_comparisons=60/60 with 0 mismatches.

## 2026-06-09 - FakeQuant graph runner

- Added FakeQuantGraphRunner with HGTableFakeQuantizer insertion at graph LUT quantization points.
- Added verify-fakequant-graph and trace-fakequant-graph CLI commands.
- Generated reports/fakequant_graph_verification.json, reports/fakequant_graph_trace.json, reports/fakequant_graph_vs_int.json, and reports/fakequant_graph_vs_int.md.
- Verified fakequant_graph_cases=293/293 and fakequant graph-vs-int trace_comparisons=60/60 with 0 mismatches.

## 2026-06-09 - Final inference result comparison

- Added run_result.py with final logits/top-k result serialization and exact comparison.
- Added run-int, run-fakequant-graph, and compare-run-results CLI commands.
- Generated reports/run_int_result.json, reports/run_fakequant_graph_result.json, and reports/run_result_comparison.json.
- Verified run_result_comparison=passed with mismatches=0 and top1_equal=True.

## 2026-06-09 - Package API and contract export

- Added HgPipeQuantizationPackage high-level API and top-level package exports.
- Added export-contracts CLI and JSON reports for scalar/LUT contract metadata.
- Added explicit offset, shift_scale, effective_divisor, bound, and zero_point fields for LUT contract export.
- Added CLI-level tests and fixed grouped range metadata handling in QuantParamStore.

## 2026-06-09 - Combined runner CLI

- Added run-compare command for one-step torch.int versus FakeQuant graph execution and comparison.
- Added reports/run_compare_result.json.
- Added focused CLI test coverage for run-compare.

## 2026-06-09 - Human-readable runner comparison report

- Added Markdown report generation for run-compare.
- Generated reports/run_compare_result.md with PASS summary and top-k output table for torch.int and FakeQuant graph runners.
- Added CLI test assertions for the Markdown report.

## 2026-06-09 - Experimental image-npy bridge runner

- Added run-compare-image-npy CLI for normalized NumPy image tensors with explicit scale.
- Added input bridge metadata to reports/run_compare_image_result.json, including paper_equivalent=false.
- Generated reports/run_compare_image_result.md and added CLI test coverage.

## 2026-06-10 - Experimental input scale estimate

- Added estimate-input-scale-npy CLI for normalized single-image or batch .npy tensors.
- Added reports/input_scale_estimate.json with symmetric max-absolute patch scale and paper_equivalent=false metadata.
- Added input bridge and CLI tests for the scale estimate path.

## 2026-06-10 - Completion audit CLI

- Added audit-completion CLI to summarize generated reports against package completion requirements.
- Generated reports/completion_audit.json and reports/completion_audit.md.
- Added completion audit tests.

## 2026-06-10 - Refreshable completion audit

- Added audit-completion --refresh with optional --device for refreshed torch graph reports.
- Stored refresh file provenance in completion audit JSON and Markdown reports.
- Fixed completion audit missing-report detail formatting.
- Verified focused CLI/completion audit tests and refreshed package audit.


## 2026-06-10 - ImageNet audit hardening

- Strengthened PyTorch timm ImageNet fake-quant report validation in completion_audit.py.
- Added regression coverage for missing ImageNet model/precision pairs.
- Regenerated completion audit reports with stricter ImageNet evidence detail.

## 2026-06-10 - ImageNet provenance hardening

- Added provenance fields to ImageNet fake-quant reports and audit checks.
- Added W4A8 to configs/imagenet_eval.yaml.
- Updated ImageNet documentation to state the timm fake-quant versus HG-PIPE artifact-backed boundary.

## 2026-06-10 - Artifact-backed ImageNet audit split

- Added an explicit partial audit item for artifact-backed HG-PIPE ImageNet paper-equivalence.
- Completion audit now reports PARTIAL with complete=14/15 while keeping operational report checks runnable.
- Regenerated completion_audit JSON and Markdown reports with the partial item.

## 2026-06-10 - Artifact image-batch experimental report

- Added run-artifact-image-batch-npy CLI and hgpipe_quantization.artifact_imagenet helpers.
- Generated reports/artifact_imagenet_accuracy.json from smoke_artifact_images.npy and smoke_artifact_labels.npy.
- Completion audit now reads an experimental artifact-backed image report, while preserving the paper-equivalence item as partial.

## 2026-06-10 - Experimental artifact ImageNet audit split

- Added a complete audit item for the experimental artifact-backed image-batch report.
- Kept artifact-backed HG-PIPE ImageNet paper-equivalence as the only partial ImageNet item.
- Exposed evaluate_artifact_image_batch_npy through the package API and top-level package exports.

## 2026-06-10 - Paper-equivalence asset preflight

- Added hgpipe_quantization.paper_equivalence scanner and check-paper-equivalence-assets CLI.
- Generated reports/paper_equivalence_assets.json and reports/paper_equivalence_assets.md.
- Completion audit now includes a complete preflight evidence item while preserving paper-equivalent ImageNet as partial.

## Artifact ImageNet Paper-Equivalence Validator - 2026-06-10

Added validate-artifact-imagenet-report. It checks that reports/artifact_imagenet_accuracy.json contains the full paper-equivalent artifact-backed matrix: DeiT-tiny, DeiT-small, and ViT-tiny across int8, int4, and w4a8. Each row must use evaluation_mode=hgpipe_artifact_graph, quantization_flow=torch_int or fakequant_graph, paper_equivalent=true, valid samples, and valid top1/top5 metrics.

Latest command:

- .venv/bin/python -m hgpipe_quantization.cli validate-artifact-imagenet-report --report reports/artifact_imagenet_accuracy.json --json reports/artifact_imagenet_validation.json --markdown reports/artifact_imagenet_validation.md

Latest result:

- artifact_imagenet_validation status=failed rows=1 expected=9 missing_pairs=9 paper_equivalent_rows=0.
- completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=16.

This is expected for the current checkout: the existing artifact ImageNet report is an experimental explicit-scale bridge smoke report, not a full paper-equivalent HG-PIPE ImageNet matrix.

## Artifact Patch-Input Batch Path - 2026-06-10

Added run-artifact-patch-batch-npy and HgPipeQuantizationPackage.evaluate_artifact_patch_batch_npy. This path accepts already-quantized HG-PIPE patch input .npy batches plus labels, runs the artifact-backed torch.int and FakeQuant graph paths, and emits one ImageNet-style row for a requested model and precision.

Latest smoke command:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-artifact-patch-batch-npy --patch-inputs-npy reports/smoke_patch_inputs.npy --labels-npy reports/smoke_patch_labels.npy --model deit_tiny_patch16_224 --precision int8 --json reports/artifact_patch_smoke_accuracy.json --topk 5

Latest smoke result:

- artifact_patch_batch model=deit_tiny_patch16_224 precision=int8 samples=1 top1=100.0 top5=100.0 paper_equivalent=False flow=torch_int runner_comparison_passed=True.
- artifact_patch_smoke_validation remains failed with rows=1 expected=9 missing_pairs=8 because it is a single-row smoke report, not the full 3 model by 3 precision matrix.

Boundary: --paper-equivalent-inputs is an explicit caller assertion. The package can now consume official patch-input artifacts when available, but it still cannot infer that raw ImageNet images are paper-equivalent without the missing preprocessing/calibration/QAT/export assets.

## Artifact Patch-Input Hardening - 2026-06-10

Added focused tests for the already-quantized patch-input path: 1D and batched patch input handling, scalar rejection, label-count mismatch, invalid quantization flow rejection, fakequant_graph selection, paper-equivalent flag propagation, and append/replace report behavior.

Latest focused result:

- .venv/bin/python -m unittest tests.test_artifact_patch tests.test_cli tests.test_package_api tests.test_completion_audit
- Ran 23 tests in 89.846s: OK.

## Artifact Patch Matrix Manifest - 2026-06-10

Added run-artifact-patch-matrix-npy and write-artifact-patch-matrix-template. The template at configs/artifact_patch_matrix_manifest.template.json enumerates the expected 3 model by 3 precision matrix entries. The matrix runner evaluates each manifest row from already-quantized HG-PIPE patch input .npy files and writes an ImageNet-style report list.

Latest smoke commands:

- .venv/bin/python -m hgpipe_quantization.cli write-artifact-patch-matrix-template --json configs/artifact_patch_matrix_manifest.template.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-artifact-patch-matrix-npy --manifest reports/artifact_patch_smoke_manifest.json --json reports/artifact_patch_smoke_matrix_accuracy.json --topk 5
- .venv/bin/python -m hgpipe_quantization.cli validate-artifact-imagenet-report --report reports/artifact_patch_smoke_matrix_accuracy.json --json reports/artifact_patch_smoke_matrix_validation.json --markdown reports/artifact_patch_smoke_matrix_validation.md

Latest smoke result:

- artifact_patch_matrix rows=1 runner_passed=1 paper_equivalent_rows=0.
- artifact_patch_smoke_matrix_validation remains failed with rows=1 expected=9 missing_pairs=8 because it is a one-row smoke manifest, not the full paper-equivalent matrix.

Boundary: this closes the package-side matrix execution path, but not the missing source-side requirement for official preprocessing/calibration/QAT/checkpoint assets.

## Artifact Patch Matrix Manifest Preflight - 2026-06-10

Added validate-artifact-patch-matrix-manifest. It checks that a manifest covers the expected DeiT-tiny, DeiT-small, and ViT-tiny rows across int8, int4, and w4a8, has no duplicate or unexpected pairs, uses valid quantization flows, marks rows as paper_equivalent when appropriate, and points to existing patch input and label .npy files.

Latest command:

- .venv/bin/python -m hgpipe_quantization.cli validate-artifact-patch-matrix-manifest --manifest configs/artifact_patch_matrix_manifest.template.json --json reports/artifact_patch_matrix_manifest_validation.json --markdown reports/artifact_patch_matrix_manifest_validation.md

Latest result:

- artifact_patch_matrix_manifest_validation status=failed rows=9 expected=9 missing_pairs=0 existing_patch_inputs=0 existing_labels=0.

This is expected for the template: coverage is complete, but the official patch input and label files have not been supplied yet.

## Artifact Patch Matrix Manifest Diagnostics - 2026-06-10

Enhanced validate-artifact-patch-matrix-manifest with schema diagnostics plus missing_patch_input_files and missing_label_files. The template preflight now reports complete 9/9 pair coverage while listing the exact 9 patch input files and 9 label files that still need to be supplied under configs/patch_inputs.

Latest focused result:

- .venv/bin/python -m unittest tests.test_artifact_patch tests.test_cli tests.test_package_api tests.test_completion_audit
- Ran 32 tests in 87.714s: OK.

- 2026-06-10: Strengthened artifact patch matrix manifest preflight to inspect .npy artifacts for loadability, patch width 150528, integer dtype, and matching patch/label sample counts.

- 2026-06-10: Added canonical artifact patch matrix asset ingest command with optional copy, paper-equivalent assertion, and validation-report generation.

- 2026-06-10: Added final artifact patch matrix pipeline gate command with optional ingest and validation chaining for real paper-equivalent asset runs.

- 2026-06-10: Added artifact patch source-manifest generator and CLI command for drop-in asset discovery when real paper-equivalent assets are staged locally.
- Replaced timm model construction for deit_tiny_patch16_224, deit_small_patch16_224, and vit_tiny_patch16_224 with native torch.nn Vision Transformer implementations.
- Added optional torch state_dict checkpoint loading for the native ImageNet evaluation registry while keeping deterministic random initialization when no checkpoint is supplied.
