"""Experimental artifact-backed image-batch evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .input_bridge import iter_images_from_npy, patch_input_contract


def _topk_hit(topk_entries, label: int, k: int) -> bool:
    return any(int(entry.index) == int(label) for entry in topk_entries[:k])


def evaluate_artifact_image_batch_npy(
    source,
    *,
    images_npy: str | Path,
    labels_npy: str | Path,
    scale: float,
    device: str | None = None,
    topk: int = 5,
) -> dict[str, object]:
    images_npy = Path(images_npy)
    labels_npy = Path(labels_npy)
    images_array = np.load(images_npy)
    labels_array = np.asarray(np.load(labels_npy))
    labels = labels_array.reshape(-1)
    images = list(iter_images_from_npy(images_array))
    if len(images) != len(labels):
        raise ValueError("image count {} does not match label count {}".format(len(images), len(labels)))

    from .api import HgPipeQuantizationPackage

    package = HgPipeQuantizationPackage(source, device=device)
    samples: list[dict[str, object]] = []
    top1_hits = 0
    top5_hits = 0
    comparison_mismatches = 0
    for index, (image, label) in enumerate(zip(images, labels)):
        result = package.compare_graph_runners_from_image(image, scale=scale, topk=max(topk, 5))
        top1_hit = _topk_hit(result.torch_int.topk, int(label), 1)
        top5_hit = _topk_hit(result.torch_int.topk, int(label), 5)
        top1_hits += int(top1_hit)
        top5_hits += int(top5_hit)
        comparison_mismatches += int(result.comparison.mismatches)
        samples.append(
            {
                "index": index,
                "label": int(label),
                "torch_int_top1": result.torch_int.topk[0].index if result.torch_int.topk else None,
                "fakequant_top1": result.fakequant_graph.topk[0].index if result.fakequant_graph.topk else None,
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
                "comparison_passed": result.comparison.passed,
                "comparison_mismatches": result.comparison.mismatches,
            }
        )

    count = len(images)
    return {
        "model": "hgpipe_artifact_reference_graph",
        "precision": "explicit_scale_patch_input",
        "samples": count,
        "top1": (100.0 * top1_hits / count) if count else 0.0,
        "top5": (100.0 * top5_hits / count) if count else 0.0,
        "evaluation_mode": "hgpipe_artifact_graph_experimental",
        "quantization_flow": "input_bridge_explicit_scale",
        "paper_equivalent": False,
        "scale": scale,
        "device": device or "default",
        "images_npy": str(images_npy),
        "labels_npy": str(labels_npy),
        "image_shape": list(images_array.shape),
        "labels_shape": list(labels_array.shape),
        "contract": patch_input_contract(),
        "runner_comparison_passed": comparison_mismatches == 0,
        "runner_comparison_mismatches": comparison_mismatches,
        "samples_detail": samples,
        "provenance_note": "Experimental artifact-backed image bridge; not paper-equivalent without original calibration/QAT/export flow.",
    }


def write_artifact_image_batch_report(payload: dict[str, object], path: str | Path) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([payload], indent=2, sort_keys=True))
