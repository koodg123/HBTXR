# HG-PIPE Paper-Equivalence Asset Preflight

## Summary

- Status: incomplete
- Paper-equivalent ready: False
- Present: 1/5
- Missing: 4

## Requirements

- original_image_to_patch_quantization_policy: status=missing, match_count=0, required_for=Converting ImageNet images into paper-equivalent HG-PIPE patch input tensors.
- qat_or_calibration_flow: status=missing, match_count=0, required_for=Reproducing paper-equivalent activation scales and tables beyond saved reference inputs.
- quantized_model_checkpoints: status=missing, match_count=0, required_for=Evaluating DeiT-tiny, DeiT-small, and ViT-tiny under int8, int4, and W4A8 without synthetic weights.
- model_export_or_ref_generation_flow: status=present, match_count=1, required_for=Regenerating HG-PIPE artifacts from model checkpoints instead of only replaying checked-in refs.
- full_artifact_imagenet_accuracy_matrix: status=missing, match_count=0, required_for=Closing the remaining completion-audit partial item.

## Missing Items

- original_image_to_patch_quantization_policy: Original preprocessing or image-to-int8 patch input quantization policy.
- qat_or_calibration_flow: Original QAT or calibration scripts and configs.
- quantized_model_checkpoints: Quantized model checkpoints or exported model state for all paper models and precisions.
- full_artifact_imagenet_accuracy_matrix: Artifact-backed ImageNet report covering three models by int8, int4, and W4A8 with paper_equivalent true.
