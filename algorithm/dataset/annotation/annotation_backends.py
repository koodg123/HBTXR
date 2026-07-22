from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dataset.annotation.annotation_groundedsam import (
    expand_eye_region_from_bbox,
    mask_bbox_xywh,
    mask_to_ellipse_xywht,
)


DEFAULT_ANNOTATION_BACKEND = "groundedsam"
SUPPORTED_ANNOTATION_BACKENDS = ("groundedsam", "ultralytics_sam3", "groundedsam2")


def resolve_annotation_backend(annotation_backend: str | None) -> str:
    text = str(annotation_backend or DEFAULT_ANNOTATION_BACKEND).strip().lower()
    if text not in SUPPORTED_ANNOTATION_BACKENDS:
        raise ValueError(f"Unsupported annotation backend: {annotation_backend!r}")
    return text


def _prepend_sys_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def _resolve_existing_candidate(candidates: Iterable[Path], *, fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return fallback.resolve()


def resolve_default_ultralytics_sam3_checkpoint(repo_root: Path) -> Path:
    return _resolve_existing_candidate(
        [
            repo_root / "weights" / "sam3_b.pt",
            repo_root / "weights" / "sam3_l.pt",
            repo_root / "checkpoints" / "sam3_b.pt",
            repo_root / "checkpoints" / "sam3_l.pt",
            repo_root / "sam3_b.pt",
            repo_root / "sam3_l.pt",
        ],
        fallback=repo_root / "weights" / "sam3_b.pt",
    )


def resolve_default_groundedsam2_config(repo_root: Path) -> Path:
    return _resolve_existing_candidate(
        [
            repo_root / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
            repo_root / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml",
            repo_root / "sam2" / "configs" / "sam2" / "sam2_hiera_l.yaml",
        ],
        fallback=repo_root / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
    )


def resolve_default_groundedsam2_checkpoint(repo_root: Path) -> Path:
    return _resolve_existing_candidate(
        [
            repo_root / "checkpoints" / "sam2.1_hiera_large.pt",
            repo_root / "checkpoints" / "sam2.1_hiera_base_plus.pt",
            repo_root / "checkpoints" / "sam2_hiera_large.pt",
        ],
        fallback=repo_root / "checkpoints" / "sam2.1_hiera_large.pt",
    )


def resolve_default_groundedsam2_groundingdino_config(repo_root: Path) -> Path:
    return _resolve_existing_candidate(
        [
            repo_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
            repo_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
        ],
        fallback=repo_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    )


def resolve_default_groundedsam2_groundingdino_checkpoint(repo_root: Path) -> Path:
    return _resolve_existing_candidate(
        [
            repo_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth",
            repo_root / "gdino_checkpoints" / "groundingdino_swinb_cogcoor.pth",
        ],
        fallback=repo_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth",
    )


def _choose_mask_candidate(
    candidates: Iterable[dict[str, Any]],
    *,
    image_shape: tuple[int, int, int],
    min_mask_area: int,
) -> dict[str, Any] | None:
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    image_area = float(max(image_h * image_w, 1))
    center_x = 0.5 * float(image_w)
    center_y = 0.5 * float(image_h)
    best_key: tuple[float, float, float] | None = None
    best_item: dict[str, Any] | None = None

    for item in candidates:
        mask = np.asarray(item["mask"], dtype=np.uint8)
        area = int(mask.sum())
        if area < int(min_mask_area):
            continue
        bbox = item.get("bbox_xywh") or mask_bbox_xywh(mask)
        ellipse = item.get("ellipse_xywht") or mask_to_ellipse_xywht(mask)
        if bbox is None or ellipse is None:
            continue
        bbox = [float(v) for v in bbox]
        ellipse = [float(v) for v in ellipse]
        cx = float(bbox[0] + 0.5 * bbox[2])
        cy = float(bbox[1] + 0.5 * bbox[3])
        center_dist = float(np.hypot(cx - center_x, cy - center_y) / max(np.hypot(center_x, center_y), 1.0))
        area_ratio = float(area) / image_area
        area_bias = abs(area_ratio - 0.08)
        score = float(item.get("score", item.get("predicted_iou", item.get("stability_score", 0.0))))
        key = (center_dist + 0.35 * area_bias, -score, abs(area_ratio - 0.12))
        if best_key is None or key < best_key:
            best_key = key
            best_item = {
                "mask": mask,
                "bbox_xywh": bbox,
                "ellipse_xywht": ellipse,
                "box_xyxy": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[0] + bbox[2]),
                    float(bbox[1] + bbox[3]),
                ],
                "box_confidence": score,
                "mask_score": float(item.get("mask_score", score)),
                "class_name": str(item.get("class_name", "eye")),
            }
    if best_item is None:
        return None
    best_item["eye_region_xywh"] = expand_eye_region_from_bbox(
        best_item["bbox_xywh"],
        image_size_wh=(image_w, image_h),
    )
    return best_item


class _UltralyticsSam3Runtime:
    def __init__(
        self,
        *,
        ultralytics_root: Path,
        checkpoint_path: Path,
        min_mask_area: int,
        device: str | None = None,
    ) -> None:
        repo_root = Path(ultralytics_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Ultralytics root not found: {repo_root}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Ultralytics SAM3 checkpoint not found: {checkpoint_path}")
        _prepend_sys_path(repo_root)
        try:
            import torch
            from ultralytics import SAM
        except Exception as exc:  # pragma: no cover - import path is environment-dependent
            raise RuntimeError(
                f"Failed to import Ultralytics SAM3 from {repo_root}. "
                "Check the repository path and its Python dependencies."
            ) from exc
        self.torch = torch
        self.device = torch.device(str(device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.model = SAM(str(Path(checkpoint_path).resolve()))
        self.min_mask_area = int(min_mask_area)

    def annotate_image(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        results = self.model.predict(
            image_rgb,
            stream=False,
            verbose=False,
            device=str(self.device),
        )
        if not results:
            return None
        result = results[0]
        masks = getattr(getattr(result, "masks", None), "data", None)
        boxes = getattr(result, "boxes", None)
        if masks is None:
            return None
        masks_np = masks.detach().cpu().numpy()
        box_xyxy = None if boxes is None else boxes.xyxy.detach().cpu().numpy()
        scores = None if boxes is None else boxes.conf.detach().cpu().numpy()
        candidates: list[dict[str, Any]] = []
        for idx, mask in enumerate(masks_np):
            binary = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
            if binary.ndim == 3:
                binary = binary.squeeze(0)
            bbox = None
            if box_xyxy is not None and idx < len(box_xyxy):
                x0, y0, x1, y1 = [float(v) for v in box_xyxy[idx].tolist()]
                bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
            candidates.append(
                {
                    "mask": binary,
                    "bbox_xywh": bbox,
                    "score": 0.0 if scores is None or idx >= len(scores) else float(scores[idx]),
                    "class_name": "eye",
                }
            )
        return _choose_mask_candidate(candidates, image_shape=image_rgb.shape, min_mask_area=self.min_mask_area)


class _GroundedSam2Runtime:
    def __init__(
        self,
        *,
        groundedsam2_root: Path,
        model_config: Path,
        checkpoint_path: Path,
        groundingdino_config: Path,
        groundingdino_checkpoint: Path,
        classes: Iterable[str],
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        min_mask_area: int,
        device: str | None = None,
    ) -> None:
        repo_root = Path(groundedsam2_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Grounded-SAM-2 root not found: {repo_root}")
        for path in (model_config, checkpoint_path, groundingdino_config, groundingdino_checkpoint):
            if not Path(path).exists():
                raise FileNotFoundError(f"Grounded-SAM-2 dependency path not found: {path}")
        _prepend_sys_path(repo_root)
        try:
            import torch
            from torchvision.ops import box_convert
            from grounding_dino.groundingdino.util.inference import load_model, predict
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # pragma: no cover - import path is environment-dependent
            raise RuntimeError(
                f"Failed to import Grounded-SAM-2 from {repo_root}. "
                "Check the repository path and its Python dependencies."
            ) from exc
        self.torch = torch
        self.box_convert = box_convert
        self.predict = predict
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = tuple(str(item).strip() for item in classes if str(item).strip()) or ("pupil", "iris", "eye")
        self.priority_map = {name: idx for idx, name in enumerate(self.classes)}
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_mask_area = int(min_mask_area)
        self.prompt_text = " ".join(f"{name}." for name in self.classes)
        self.sam2_predictor = SAM2ImagePredictor(
            build_sam2(str(Path(model_config)), str(Path(checkpoint_path)), device=self.device),
        )
        self.grounding_model = load_model(
            model_config_path=str(Path(groundingdino_config)),
            model_checkpoint_path=str(Path(groundingdino_checkpoint)),
            device=self.device,
        )

    def annotate_image(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        self.sam2_predictor.set_image(image_rgb)
        image_for_gdino = image_rgb[:, :, ::-1].copy()
        boxes, confidences, labels = self.predict(
            model=self.grounding_model,
            image=image_for_gdino,
            caption=self.prompt_text,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        if len(confidences) == 0:
            return None
        h, w = image_rgb.shape[:2]
        boxes = boxes * self.torch.tensor([w, h, w, h], dtype=self.torch.float32)
        input_boxes = self.box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        candidates: list[dict[str, Any]] = []
        confidences_list = confidences.cpu().numpy().tolist()
        labels_list = list(labels)
        for idx, mask in enumerate(np.asarray(masks)):
            class_name = str(labels_list[idx]) if idx < len(labels_list) else "eye"
            confidence = float(confidences_list[idx]) if idx < len(confidences_list) else 0.0
            x0, y0, x1, y1 = [float(v) for v in input_boxes[idx].tolist()]
            bbox = [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]
            candidates.append(
                {
                    "mask": (np.asarray(mask, dtype=np.float32) > 0.0).astype(np.uint8),
                    "bbox_xywh": bbox,
                    "score": confidence,
                    "mask_score": 0.0 if idx >= len(scores) else float(np.asarray(scores[idx]).max()),
                    "class_name": class_name,
                }
            )
        chosen = _choose_mask_candidate(candidates, image_shape=image_rgb.shape, min_mask_area=self.min_mask_area)
        if chosen is None:
            return None
        class_priority = self.priority_map.get(str(chosen["class_name"]).strip().lower(), len(self.priority_map) + 1)
        chosen["box_confidence"] = float(max(chosen["box_confidence"], 1.0 - 0.05 * class_priority))
        return chosen


def build_annotation_runtime(
    *,
    annotation_backend: str,
    groundedsam_root: Path | None = None,
    groundingdino_config: Path | None = None,
    groundingdino_checkpoint: Path | None = None,
    sam_checkpoint: Path | None = None,
    ultralytics_root: Path | None = None,
    groundedsam2_root: Path | None = None,
    ultralytics_sam3_checkpoint: Path | None = None,
    groundedsam2_config: Path | None = None,
    groundedsam2_checkpoint: Path | None = None,
    groundedsam2_groundingdino_config: Path | None = None,
    groundedsam2_groundingdino_checkpoint: Path | None = None,
    sam_encoder_version: str = "vit_h",
    classes: Iterable[str] = (),
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    nms_threshold: float = 0.8,
    min_mask_area: int = 16,
    device: str | None = None,
) -> Any:
    resolved_backend = resolve_annotation_backend(annotation_backend)
    if resolved_backend == "groundedsam":
        from dataset.annotation.groundedsam_build import build_groundedsam_runtime

        if groundedsam_root is None:
            raise ValueError("groundedsam_root is required for annotation_backend='groundedsam'")
        return build_groundedsam_runtime(
            groundedsam_root=groundedsam_root,
            groundingdino_config=groundingdino_config,
            groundingdino_checkpoint=groundingdino_checkpoint,
            sam_checkpoint=sam_checkpoint,
            sam_encoder_version=sam_encoder_version,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
            min_mask_area=min_mask_area,
            device=device,
        )
    if resolved_backend == "ultralytics_sam3":
        if ultralytics_root is None:
            raise ValueError("ultralytics_root is required for annotation_backend='ultralytics_sam3'")
        repo_root = Path(ultralytics_root).resolve()
        return _UltralyticsSam3Runtime(
            ultralytics_root=repo_root,
            checkpoint_path=resolve_default_ultralytics_sam3_checkpoint(repo_root)
            if ultralytics_sam3_checkpoint is None
            else Path(ultralytics_sam3_checkpoint).resolve(),
            min_mask_area=min_mask_area,
            device=device,
        )
    if groundedsam2_root is None:
        raise ValueError("groundedsam2_root is required for annotation_backend='groundedsam2'")
    repo_root = Path(groundedsam2_root).resolve()
    return _GroundedSam2Runtime(
        groundedsam2_root=repo_root,
        model_config=resolve_default_groundedsam2_config(repo_root)
        if groundedsam2_config is None
        else Path(groundedsam2_config).resolve(),
        checkpoint_path=resolve_default_groundedsam2_checkpoint(repo_root)
        if groundedsam2_checkpoint is None
        else Path(groundedsam2_checkpoint).resolve(),
        groundingdino_config=resolve_default_groundedsam2_groundingdino_config(repo_root)
        if groundedsam2_groundingdino_config is None
        else Path(groundedsam2_groundingdino_config).resolve(),
        groundingdino_checkpoint=resolve_default_groundedsam2_groundingdino_checkpoint(repo_root)
        if groundedsam2_groundingdino_checkpoint is None
        else Path(groundedsam2_groundingdino_checkpoint).resolve(),
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        min_mask_area=min_mask_area,
        device=device,
    )


__all__ = [
    "DEFAULT_ANNOTATION_BACKEND",
    "SUPPORTED_ANNOTATION_BACKENDS",
    "build_annotation_runtime",
    "resolve_annotation_backend",
    "resolve_default_groundedsam2_checkpoint",
    "resolve_default_groundedsam2_config",
    "resolve_default_groundedsam2_groundingdino_checkpoint",
    "resolve_default_groundedsam2_groundingdino_config",
    "resolve_default_ultralytics_sam3_checkpoint",
]
