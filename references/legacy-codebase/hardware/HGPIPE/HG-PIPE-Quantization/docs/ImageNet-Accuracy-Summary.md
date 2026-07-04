# ImageNet Accuracy Summary

## Dataset

Dataset path provided by the user: G:/dataset/imagenet-val. The previous PyTorch evaluation scripts consumed the corresponding ImageFolder layout through torchvision.

## PyTorch Fake-Quant Results

These results are from reports/imagenet_accuracy_int8_int4.json and reports/imagenet_accuracy_w4a8.json. They use timm pretrained models with PyTorch fake quantization, not the artifact-backed HG-PIPE integer graph.

| Model | Precision | Samples | Top-1 | Top-5 | Images/s |
|---|---:|---:|---:|---:|---:|
| deit_tiny_patch16_224 | int8 | 50000 | 71.308 | 90.614 | 86.217 |
| deit_tiny_patch16_224 | int4 | 50000 | 0.140 | 0.656 | 78.649 |
| deit_tiny_patch16_224 | w4a8 | 50000 | 4.472 | 12.188 | 25.917 |
| deit_small_patch16_224 | int8 | 50000 | 78.934 | 94.522 | 54.099 |
| deit_small_patch16_224 | int4 | 50000 | 0.084 | 0.544 | 69.821 |
| deit_small_patch16_224 | w4a8 | 50000 | 0.992 | 3.264 | 30.727 |
| vit_tiny_patch16_224 | int8 | 50000 | 55.370 | 79.840 | 77.250 |
| vit_tiny_patch16_224 | int4 | 50000 | 0.108 | 0.582 | 44.807 |
| vit_tiny_patch16_224 | w4a8 | 50000 | 0.176 | 0.790 | 45.823 |

## Interpretation

The int8 rows are useful environment sanity checks. The int4 and w4a8 rows are not HG-PIPE paper-equivalent because they use naive PyTorch fake quantization without the original HG-PIPE QAT/calibration path.

## Artifact-Backed Accuracy Status

The artifact graph now runs bit-exact from patch_embed input through head for the saved reference input. Arbitrary ImageNet accuracy still requires a verified image-to-int8 patch calibration policy. The current input bridge provides tensor reshaping and explicit-scale quantization only.

## Provenance Fields

The JSON reports now include evaluation_mode, quantization_flow, paper_equivalent, dataset_path, dataset_split, eval_script, command, paper_model, and timm_model_name. These fields make the method boundary explicit: the rows are timm fake-quant ImageNet sanity runs and are not HG-PIPE artifact-backed paper-equivalent accuracy.

## Experimental Artifact Image-Batch Report - 2026-06-10

Added run-artifact-image-batch-npy. It accepts a normalized image batch .npy, a labels .npy, and an explicit patch-input scale, then runs the artifact-backed torch.int and FakeQuant graph paths for each image. The generated report is reports/artifact_imagenet_accuracy.json. It records evaluation_mode=hgpipe_artifact_graph_experimental, quantization_flow=input_bridge_explicit_scale, runner_comparison_passed, top1/top5 over the supplied labels, and paper_equivalent=false.

Latest smoke command:

- .venv/bin/python -m hgpipe_quantization.cli --source ../ICCAD24-HG-PIPE run-artifact-image-batch-npy --images-npy reports/smoke_artifact_images.npy --labels-npy reports/smoke_artifact_labels.npy --scale 1.0 --json reports/artifact_imagenet_accuracy.json --topk 5

Latest smoke result:

- artifact_image_batch samples=1 top1=0.0 top5=0.0 paper_equivalent=False runner_comparison_passed=True.

This moves the artifact-backed ImageNet item from absent evidence to an experimental explicit-scale bridge report. It remains partial because it is not the original HG-PIPE paper-equivalent calibration/QAT/export path and does not cover the 3 model x 3 precision matrix.

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
