# Validation

## Planned Checks

- Unit tests for core integer kernels.
- Smoke tests against selected refs: `attn_0_qq`, `attn_0_lnq`, `attn_0_softmaxq`, `mlp_0_geluq`, `head_lnq`.
- Full verification over all discovered refs.

## Pass Criteria

- `python3 -m unittest discover -s tests` passes.
- `python3 -m hgpipe_quantization.cli verify` reports all discovered cases passed with zero mismatches.

## Artifact-backed graph verification - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-graph --json reports/graph_verification.json
- .venv/bin/python -m unittest discover -s tests

Results:

- graph_cases=268/268 passed mismatches=0
- Ran 6 tests in 8.480s: OK

Remaining risk: arbitrary ImageNet images still need a verified image-to-integer patch embedding input bridge. The internal attn_*_aq_output to attn_*_gen_o_matmul_input bridge is now regenerated and verified for all 12 attention blocks.

## Single-input end-to-end graph verification - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-e2e --json reports/e2e_graph_verification.json
- .venv/bin/python -m unittest discover -s tests

Results:

- e2e_graph_cases=293/293 passed mismatches=0
- Ran 11 tests in 18.728s: OK

Remaining risk: arbitrary ImageNet images still need a verified calibration policy for the signed int8 patch_embed input.

## Torch integer and FakeQuantizer validation - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-int --json reports/torch_int_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant --json reports/fakequant_verification.json
- .venv/bin/python -m unittest discover -s tests

Results:

- torch_int_cases=293/293 passed mismatches=0
- fakequant_cases=60/60 passed mismatches=0
- Ran 16 tests in 34.648s: OK

Coverage:

- QuantParamStore loads scalar, table, dtype, and range contracts from ICCAD24-HG-PIPE artifacts.
- TorchIntGraphRunner verifies patch input through head using torch integer tensors.
- HGTableFakeQuantizer matches the HG-PIPE LUT cursor formula on a small case.
- AffineFakeQuantizer and FX output insertion are covered by unit tests.

## FakeQuant versus torch integer trace comparison - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE trace-fakequant --json reports/fakequant_trace.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE trace-int-cases --json reports/torch_int_case_trace.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE compare-traces --left reports/fakequant_trace.json --right reports/torch_int_case_trace.json --json reports/fakequant_vs_int.json --markdown reports/fakequant_vs_int.md

Results:

- fakequant_traces=60
- torch_int_traces=60
- trace_comparisons=60/60 passed mismatches=0
- Ran 18 tests in 20.714s: OK

Coverage: exact value comparison for LUT-backed ReQuant and GeLU-ReQuant FakeQuantizer outputs against torch integer outputs.

Remaining risk: full graph FakeQuant tracing for LayerNorm, Softmax, MatMul, residual, and attention composition remains future work because those fake-quant graph modules require additional floating-graph wrappers beyond the LUT output fakequantizers.

## FakeQuant graph validation - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant-graph --json reports/fakequant_graph_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE trace-fakequant-graph --json reports/fakequant_graph_trace.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE compare-traces --left reports/fakequant_graph_trace.json --right reports/torch_int_case_trace.json --json reports/fakequant_graph_vs_int.json --markdown reports/fakequant_graph_vs_int.md
- .venv/bin/python -m unittest discover -s tests

Results:

- fakequant_graph_cases=293/293 passed mismatches=0
- fakequant_graph_traces=60
- trace_comparisons=60/60 passed mismatches=0
- Ran 20 tests in 35.254s: OK

Coverage: end-to-end artifact graph execution with HGTableFakeQuantizer inserted at LUT quantization points, plus exact trace comparison against torch integer LUT case outputs.

Remaining risk: the graph still uses artifact integer MatMul, LayerNorm, Softmax, residual, and layout operators around the inserted FakeQuantizers. A pure timm/FX floating model rewrite for arbitrary pretrained PyTorch models remains separate from this artifact-backed graph runner.

## Final inference result comparison - 2026-06-09

Commands:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-int --json reports/run_int_result.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-fakequant-graph --json reports/run_fakequant_graph_result.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE compare-run-results --left reports/run_fakequant_graph_result.json --right reports/run_int_result.json --json reports/run_result_comparison.json
- .venv/bin/python -m unittest discover -s tests
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-graph --json reports/graph_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-e2e --json reports/e2e_graph_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-int --json reports/torch_int_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant --json reports/fakequant_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant-graph --json reports/fakequant_graph_verification.json

Results:

- torch_int_result wrote=reports/run_int_result.json top1=0
- fakequant_graph_result wrote=reports/run_fakequant_graph_result.json top1=0
- run_result_comparison=passed mismatches=0 top1_equal=True
- Ran 22 tests in 63.925s: OK
- cases=97/97 passed mismatches=0
- graph_cases=268/268 passed mismatches=0
- e2e_graph_cases=293/293 passed mismatches=0
- torch_int_cases=293/293 passed mismatches=0
- fakequant_cases=60/60 passed mismatches=0
- fakequant_graph_cases=293/293 passed mismatches=0

Coverage: final head logits from the FakeQuantizer-inserted artifact graph and the torch.int artifact graph are serialized with top-k summaries and compared exactly.

Remaining risk: this is an artifact-backed HG-PIPE graph path. A separate pure PyTorch/timm FX rewrite is still needed if the target is arbitrary pretrained model graph rewriting outside the recovered HG-PIPE artifacts.

## Package API and contract export validation - 2026-06-09

Commands:

- .venv/bin/python -m unittest tests.test_package_api tests.test_cli
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE export-contracts --json reports/quant_contracts.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE export-contracts --kind requant_table --include-tables --json reports/requant_contracts_with_tables.json
- .venv/bin/python -m unittest discover -s tests

Results:

- Package/API/CLI focused tests: Ran 7 tests: OK
- quant_contracts=97 wrote=reports/quant_contracts.json
- quant_contracts=48 wrote=reports/requant_contracts_with_tables.json
- Full test suite: Ran 34 tests in 72.550s: OK

Coverage: top-level package exports, high-level HgPipeQuantizationPackage, LUT contract JSON export, scalar/shift-scale/effective-divisor/bound fields, zero_point=None policy for source-backed LUT contracts, parser command exposure, export-contracts CLI, and compare-run-results CLI.

Remaining risk: no source-backed affine scale/zero-point loader exists because ICCAD24-HG-PIPE public artifacts used here do not provide affine zero-point files for graph quantization points. The package exposes this explicitly instead of inventing values.

## Combined runner CLI validation - 2026-06-09

Commands:

- .venv/bin/python -m unittest tests.test_cli
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-compare --json reports/run_compare_result.json --markdown reports/run_compare_result.md --topk 5

Results:

- CLI focused tests: Ran 4 tests: OK
- run_compare=passed mismatches=0 top1_int=0 top1_fakequant=0 top1_equal=True
- reports/run_compare_result.md contains a human-readable PASS summary and top-k table for both runners.

Coverage: one command now runs both artifact-backed graph paths, serializes torch.int and FakeQuant graph final logits/top-k outputs, embeds the exact comparison result in JSON, and writes a human-readable Markdown report.

## Final full artifact verification - 2026-06-09

Commands:

- .venv/bin/python -m unittest discover -s tests
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-graph --json reports/graph_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-e2e --json reports/e2e_graph_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-int --json reports/torch_int_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant --json reports/fakequant_verification.json
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE verify-fakequant-graph --json reports/fakequant_graph_verification.json

Results:

- Unit tests: Ran 34 tests in 72.550s: OK
- cases=97/97 passed mismatches=0
- graph_cases=268/268 passed mismatches=0
- e2e_graph_cases=293/293 passed mismatches=0
- torch_int_cases=293/293 passed mismatches=0
- fakequant_cases=60/60 passed mismatches=0
- fakequant_graph_cases=293/293 passed mismatches=0

## Experimental image-npy bridge validation - 2026-06-09

Commands:

- .venv/bin/python -m unittest tests.test_cli
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-compare-image-npy --image-npy reports/smoke_zero_image.npy --scale 1.0 --json reports/run_compare_image_result.json --markdown reports/run_compare_image_result.md --topk 3

Results:

- CLI focused tests: Ran 5 tests: OK
- run_compare_image_npy=passed mismatches=0 scale=1.0 paper_equivalent=False

Coverage: normalized NumPy image tensors can be converted with an explicit scale into HG-PIPE patch input and used to run/compare the torch.int and FakeQuant graph paths. This is intentionally marked experimental and not paper-equivalent.

## Experimental input scale estimate validation - 2026-06-10

Commands:

- .venv/bin/python -m unittest tests.test_input_bridge tests.test_cli
- .venv/bin/python -m hgpipe_quantization.cli estimate-input-scale-npy --images-npy reports/smoke_scale_images.npy --json reports/input_scale_estimate.json

Results:

- Focused tests: Ran 12 tests: OK
- input_scale=1.0 images=2 paper_equivalent=False wrote=reports/input_scale_estimate.json

Coverage: single-image and batch .npy tensors are accepted, a symmetric max-absolute patch scale is estimated, and the output records the contract plus paper_equivalent=false.

## Completion audit report validation - 2026-06-10

Commands:

- .venv/bin/python -m unittest tests.test_completion_audit tests.test_cli
- .venv/bin/python -m hgpipe_quantization.cli audit-completion --json reports/completion_audit.json --markdown reports/completion_audit.md

Results:

- Focused tests: Ran 8 tests: OK
- completion_audit=partial complete=14/15 incomplete=0 missing=0

Coverage: generated reports are checked as evidence for scalar/LUT/statistics contracts, core kernels, graph/e2e execution, torch.int inference, FakeQuant graph inference, runner comparison, experimental image bridge, and timm ImageNet fake-quant reports.

## Refreshable completion audit validation - 2026-06-10

Commands:

- .venv/bin/python -m py_compile hgpipe_quantization/cli.py hgpipe_quantization/completion_audit.py
- .venv/bin/python -m unittest tests.test_cli tests.test_completion_audit
- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE audit-completion --refresh --json reports/completion_audit.json --markdown reports/completion_audit.md

Results:

- Syntax check passed.
- Focused tests: Ran 8 tests: OK.
- Full test suite: Ran 37 tests: OK.
- Refresh regenerated 10 core report files.
- cases=97/97 passed mismatches=0.
- graph_cases=268/268 passed mismatches=0.
- e2e_graph_cases=293/293 passed mismatches=0.
- torch_int_cases=293/293 passed mismatches=0.
- fakequant_cases=60/60 passed mismatches=0.
- fakequant_graph_cases=293/293 passed mismatches=0.
- quant_contracts=97 wrote=reports/quant_contracts.json.
- run_compare=passed mismatches=0 top1_int=0 top1_fakequant=0 top1_equal=True.
- completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=16.

Coverage: audit-completion --refresh now regenerates the core evidence reports before the completion audit. It intentionally does not rerun external ImageNet/timm evaluation or the experimental image-npy bridge unless those reports are regenerated by their dedicated commands.

## ImageNet report audit hardening - 2026-06-10

Commands:

- .venv/bin/python -m py_compile hgpipe_quantization/completion_audit.py
- .venv/bin/python -m unittest tests.test_completion_audit
- .venv/bin/python -m hgpipe_quantization.cli audit-completion --json reports/completion_audit.json --markdown reports/completion_audit.md

Results:

- Syntax check passed.
- Focused completion audit tests: Ran 3 tests: OK.
- completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=0.
- ImageNet timm report detail: rows=9 expected=9 required_fields=9/9 full_val=9/9 cuda=9/9 pretrained=9/9 valid_metrics=9/9 missing_pairs=[].

Coverage: completion audit no longer treats ImageNet report existence alone as sufficient. It verifies all three paper model names across int8, int4, and w4a8; each row must include samples, top1/top5, throughput, pretrained, and device metadata, with 50000-sample CUDA validation rows and valid metric ranges.

## ImageNet provenance audit hardening - 2026-06-10

Commands:

- .venv/bin/python -m py_compile hgpipe_quantization/eval/imagenet_eval.py hgpipe_quantization/completion_audit.py
- .venv/bin/python -m unittest tests.test_completion_audit
- .venv/bin/python -m hgpipe_quantization.cli audit-completion --json reports/completion_audit.json --markdown reports/completion_audit.md

Results:

- Focused completion audit tests: Ran 3 tests: OK.
- ImageNet eval default precisions: fp32, int8, int4, w4a8.
- completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=0.
- ImageNet report detail includes provenance=9/9 and missing_pairs=[].

Coverage: ImageNet fake-quant reports now require explicit provenance fields and paper_equivalent=false. This strengthens the sanity-report evidence while preserving the residual limit that artifact-backed arbitrary ImageNet accuracy is not paper-equivalent.

## Artifact-backed ImageNet Audit Split - 2026-06-10

The completion audit now separates PyTorch/timm fake-quant ImageNet sanity reports from artifact-backed HG-PIPE ImageNet paper-equivalence. The timm reports remain complete, while artifact-backed arbitrary ImageNet accuracy is a partial item because reports/artifact_imagenet_accuracy.json is currently an experimental explicit-scale smoke report, not a full 3 model x 3 precision paper-equivalent report, and the original calibration/QAT/export flow is not available in the checkout. Latest refresh result: completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=16.

## Experimental Artifact Image-Batch Report - 2026-06-10

Added run-artifact-image-batch-npy. It accepts a normalized image batch .npy, a labels .npy, and an explicit patch-input scale, then runs the artifact-backed torch.int and FakeQuant graph paths for each image. The generated report is reports/artifact_imagenet_accuracy.json. It records evaluation_mode=hgpipe_artifact_graph_experimental, quantization_flow=input_bridge_explicit_scale, runner_comparison_passed, top1/top5 over the supplied labels, and paper_equivalent=false.

Latest smoke command:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-artifact-image-batch-npy --images-npy reports/smoke_artifact_images.npy --labels-npy reports/smoke_artifact_labels.npy --scale 1.0 --json reports/artifact_imagenet_accuracy.json --topk 5

Latest smoke result:

- artifact_image_batch samples=1 top1=0.0 top5=0.0 paper_equivalent=False runner_comparison_passed=True.

This moves the artifact-backed ImageNet item from absent evidence to an experimental explicit-scale bridge report. It remains partial because it is not the original HG-PIPE paper-equivalent calibration/QAT/export path and does not cover the 3 model x 3 precision matrix.

## Experimental Artifact Image-Batch Audit Split - 2026-06-10

The completion audit now has three artifact ImageNet rows: experimental artifact-backed image-batch report is complete, the paper-equivalence validator report is complete, and artifact-backed HG-PIPE ImageNet paper-equivalence remains partial. Latest audit result: completion_audit=partial complete=14/15 incomplete=0 missing=0 refreshed=0.

## Paper-Equivalence Asset Preflight - 2026-06-10

Added check-paper-equivalence-assets. It scans the ICCAD24-HG-PIPE source tree and the quantization package for assets required to claim paper-equivalent artifact-backed ImageNet accuracy. It writes reports/paper_equivalence_assets.json and reports/paper_equivalence_assets.md.

Latest result:

- paper_equivalence_assets status=incomplete present=1/5 missing=4 ready=False.

The preflight currently confirms that the package has not found the original image-to-patch quantization policy, QAT or calibration flow, quantized checkpoints, or a full 3 model by 3 precision paper-equivalent artifact ImageNet matrix. This report is now part of completion audit evidence.

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

- 2026-06-10: Artifact patch matrix manifest preflight now validates .npy loadability with read-only inspection, enforces patch width 150528, requires integer patch/label dtypes, and checks patch-sample versus label-count matching.

- 2026-06-10: Added canonical artifact patch matrix asset ingest/readiness gate with source-manifest mapping, optional copy into template destinations, and post-ingest manifest validation.

- 2026-06-10: Added a final artifact patch matrix pipeline gate command that chains optional ingest, manifest preflight, matrix execution, and artifact report validation once real paper-equivalent assets are available.

- 2026-06-10: Added write-artifact-patch-source-manifest for drop-in asset discovery. Expected filenames include flat {model}_{precision}_patch_inputs.npy/{model}_{precision}_labels.npy and nested {model}/{precision}/patch_inputs.npy, labels.npy, inputs.npy, targets.npy.

## Native ViT/DeiT Registry

Validation now includes torch-only construction and CPU forward-pass coverage for deit_tiny_patch16_224, deit_small_patch16_224, and vit_tiny_patch16_224, plus a checkpoint reload smoke test. The ImageNet eval helper no longer requires timm to build these three paper models.
