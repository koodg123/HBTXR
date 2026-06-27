# HG-PIPE Quantization Completion Audit

## Summary

- Status: PARTIAL
- Complete: 14/15
- Incomplete: 0
- Partial: 1
- Missing: 0

## Requirement Matrix

| Requirement | Status | Evidence | Detail |
|---|---|---|---|
| scale/scalar, dtype, range, zero-point policy contract export | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/quant_contracts.json | contracts=97 has_required_fields=True |
| core quantization kernels | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/verification.json | count=97 expected=97 passed=97 mismatches=0 |
| artifact component graph | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/graph_verification.json | count=268 expected=268 passed=268 mismatches=0 |
| single-input end-to-end graph | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/e2e_graph_verification.json | count=293 expected=293 passed=293 mismatches=0 |
| torch.int end-to-end graph | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/torch_int_verification.json | count=293 expected=293 passed=293 mismatches=0 |
| FakeQuant LUT cases | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/fakequant_verification.json | count=60 expected=60 passed=60 mismatches=0 |
| FakeQuant graph | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/fakequant_graph_verification.json | count=293 expected=293 passed=293 mismatches=0 |
| torch.int versus FakeQuant graph final-output comparison | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/run_compare_result.json | passed=True mismatches=0 top1_equal=True |
| experimental explicit-scale image bridge | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/input_scale_estimate.json, /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/run_compare_image_result.json | scale=1.0 scale_paper_equivalent=False comparison_passed=True bridge_paper_equivalent=False |
| PyTorch timm ImageNet fake-quant evaluation reports | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/imagenet_accuracy_int8_int4.json, /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/imagenet_accuracy_w4a8.json | rows=9 expected=9 required_fields=9/9 full_val=9/9 cuda=9/9 pretrained=9/9 provenance=9/9 valid_metrics=9/9 missing_pairs=[] |
| paper-equivalence asset preflight report | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/paper_equivalence_assets.json | status=incomplete present=1/5 missing=4 ready=False |
| artifact-backed image-batch report | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/artifact_imagenet_accuracy.json | rows=1 required_fields=1/1 experimental_rows=1 artifact_patch_rows=0 covered_rows=1/1 valid_metrics=1/1 |
| artifact ImageNet paper-equivalence validator report | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/artifact_imagenet_validation.json | status=failed rows=1/9 missing_pairs=9 paper_equivalent_rows=0 errors=['evaluation_mode', 'missing_pairs', 'paper_equivalent', 'quantization_flow', 'row_count', 'unexpected_pairs'] |
| artifact patch matrix manifest preflight report | complete | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/artifact_patch_matrix_manifest_validation.json | status=failed rows=9/9 missing_pairs=0 existing_patch_inputs=0 existing_labels=0 errors=['integer_label_dtype_rows', 'integer_patch_dtype_rows', 'label_files', 'loadable_label_files', 'loadable_patch_input_files', 'matching_sample_count_rows', 'patch_input_files', 'valid_label_shape_rows', 'valid_patch_shape_rows'] |
| artifact-backed HG-PIPE ImageNet paper-equivalence report | partial | /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports/artifact_imagenet_accuracy.json | rows=1 expected=9 artifact_rows=0 missing_pairs=[('deit_small_patch16_224', 'int4'), ('deit_small_patch16_224', 'int8'), ('deit_small_patch16_224', 'w4a8'), ('deit_tiny_patch16_224', 'int4'), ('deit_tiny_patch16_224', 'int8'), ('deit_tiny_patch16_224', 'w4a8'), ('vit_tiny_patch16_224', 'int4'), ('vit_tiny_patch16_224', 'int8'), ('vit_tiny_patch16_224', 'w4a8')] |

## Refresh

- Reports dir: /home/user/project/PRJXR/impl_repos/HGPIPE/HG-PIPE-Quantization/reports
- Device: default
- Files:
- verification.json
- verification.md
- graph_verification.json
- e2e_graph_verification.json
- torch_int_verification.json
- fakequant_verification.json
- fakequant_graph_verification.json
- quant_contracts.json
- run_compare_result.json
- run_compare_result.md
- paper_equivalence_assets.json
- paper_equivalence_assets.md
- artifact_imagenet_validation.json
- artifact_imagenet_validation.md
- artifact_patch_matrix_manifest_validation.json
- artifact_patch_matrix_manifest_validation.md

## Residual Limits

- Source-backed HG-PIPE LUT contracts expose zero_point=None because affine zero-point artifacts are absent.
- Artifact-backed arbitrary ImageNet accuracy is not paper-equivalent without the original preprocessing/calibration/QAT export flow.
