# Progress

- [x] Inspect HG-PIPE quantization code and paper evidence.
- [x] Implement artifact loading.
- [x] Implement table ReQuant and GeLU-ReQuant.
- [x] Implement LayerNorm rsqrt-table quantization.
- [x] Implement Softmax segmented-table quantization.
- [x] Run full verification and archive report.
- [x] Completion audit against original objective.

## 2026-06-09

- Completed artifact-backed graph-level verification for patch embedding, 12 attention blocks, 12 MLP blocks, and head.
- Generated reports/graph_verification.json with 268/268 passed cases and zero mismatches.
- Recovered attention A head-merge layout so O matmul input is generated from aq output rather than loaded as a saved source.
- Next: implement and validate the ImageNet image-to-integer patch input bridge.

## 2026-06-09 - Single-input end-to-end graph verification

- Added verify-e2e path that starts from patch_embed input and chains patch_embed, 12 attention blocks, 12 MLP blocks, and head.
- Verified reports/e2e_graph_verification.json with 293/293 passed cases and zero mismatches.
- Added hgpipe_quantization/input_bridge.py for experimental normalized-image tensor to signed-int8 patch input conversion.
- Added docs/Input-Bridge.md and docs/ImageNet-Accuracy-Summary.md.
- Remaining: artifact-backed arbitrary ImageNet accuracy requires original or approved replacement calibration policy for patch_embed.input.

## 2026-06-09 - Quant parameter and torch integer runtime

- Added QuantParamStore and typed quant contract dataclasses.
- Added torch integer kernels and TorchIntGraphRunner.
- Added verify-int CLI and generated reports/torch_int_verification.json.
- Added FakeQuantizer modules and FX output insertion utility.
- Added FakeQuantRunner and verify-fakequant for LUT-backed artifact cases.
- Verified kernel refs, component graph, NumPy e2e graph, torch.int e2e graph, fakequant LUT cases, and 16 unit tests.
- Added LUT-backed fakequant-vs-int trace comparison reports with 60/60 exact matches.
- Added FakeQuantGraphRunner so LUT FakeQuantizers are inserted into the artifact graph and verified end-to-end.
- Added fakequant graph traces and fakequant graph-vs-int comparison reports with 60/60 exact matches.
- Remaining: implement a pure timm/FX floating model rewrite for arbitrary pretrained PyTorch models if paper-equivalent external model evaluation is required.

## 2026-06-09 - Final logits run-result comparison

- Added run_result.py for final inference result serialization with dtype, shape, statistics, full values, and top-k entries.
- Added run-int, run-fakequant-graph, and compare-run-results CLI commands.
- Generated reports/run_int_result.json, reports/run_fakequant_graph_result.json, and reports/run_result_comparison.json.
- Verified FakeQuantGraphRunner final logits against TorchIntGraphRunner final logits with mismatches=0 and top1_equal=True.
- Re-ran 22 unit tests and full CLI verification: 97/97 kernel cases, 268/268 graph cases, 293/293 e2e cases, 293/293 torch.int cases, 60/60 fakequant LUT cases, and 293/293 fakequant graph cases all passed.

## 2026-06-09 - Package API and contract export

- Added hgpipe_quantization.api.HgPipeQuantizationPackage as a high-level package entry point.
- Exported core runners, FakeQuantizers, quant parameter dataclasses, and run-result helpers from hgpipe_quantization.__init__.
- Added export_contracts API and export-contracts CLI for scalar/LUT/zero-point-policy metadata.
- Added reports/quant_contracts.json and reports/requant_contracts_with_tables.json.
- Added CLI-level tests for command exposure, export-contracts, and compare-run-results.
- Fixed QuantParamStore range handling for grouped input/output range metadata.
- Verified 34 unit tests pass.

## 2026-06-09 - Combined runner CLI

- Added run-compare CLI to execute torch.int and FakeQuant graph runners in one command.
- Generated reports/run_compare_result.json with both final logits summaries and comparison metadata.
- Added CLI test coverage for run-compare.
- Verified run_compare=passed with mismatches=0 and top1_equal=True.

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

- Added audit-completion --refresh to regenerate core evidence reports before auditing.
- Refresh covers kernel refs, artifact graph, e2e graph, torch.int graph, FakeQuant LUT cases, FakeQuant graph, quant contracts, and final run comparison.
- Generated reports/completion_audit.json and reports/completion_audit.md with refresh provenance.
- Verified completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=16.


## 2026-06-10 - ImageNet audit hardening

- Strengthened completion_audit.py so ImageNet/timm reports must cover 3 models x 3 precisions: int8, int4, and w4a8.
- Added checks for required metric fields, 50000 samples, CUDA device metadata, pretrained=true, and valid top1/top5 ranges.
- Added a negative test proving incomplete model/precision coverage is rejected.
- Verified tests.test_completion_audit passes with 3 tests.

## 2026-06-10 - ImageNet provenance hardening

- Added W4A8 to the standard ImageNet eval config.
- Backfilled ImageNet report rows with evaluation_mode, quantization_flow, paper_equivalent, dataset, script, command, and model provenance.
- Strengthened completion audit to require provenance=9/9 for timm fake-quant ImageNet reports.

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

- 2026-06-10: Upgraded artifact patch matrix manifest preflight from existence-only checks to .npy integrity validation: loadability, width 150528, integer dtype, and patch/label sample-count matching.

- 2026-06-10: Added ingest-artifact-patch-matrix-assets to map source patch assets into canonical template destinations and validate readiness after ingest.

- 2026-06-10: Added run-artifact-patch-matrix-pipeline to chain optional ingest, manifest preflight, matrix execution, and artifact ImageNet validation as the final gate.

- 2026-06-10: Added write-artifact-patch-source-manifest to scan drop-in asset directories for expected patch/label filename patterns and emit the source manifest used by ingest/pipeline commands.
