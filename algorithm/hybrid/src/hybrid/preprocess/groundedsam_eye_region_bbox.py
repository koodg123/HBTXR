from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from hybrid.preprocess.groundedsam_build import (
    _patch_groundingdino_ms_deform_attn_compat,
    _patch_transformers_groundingdino_compat,
    _resolve_default_groundingdino_config,
    _resolve_runtime_device,
)
from hybrid.preprocess.io_utils import code_to_session_dir, collect_frame_records, collect_users, discover_session_layout, session_dir_to_code
from hybrid.utils.io import read_jsonl, write_json, write_jsonl


@dataclass
class GroundedSamEyeRegionBboxConfig:
    raw_root: Path
    annotation_root: Path
    preview_root: Path
    groundedsam_root: Path
    groundingdino_config: Path
    groundingdino_checkpoint: Path
    user_id: int = 1
    session_code: str = "101"
    eye: str = "both"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_threshold: float = 0.8
    max_box_area_ratio: float = 0.85
    border_margin_px: int = 3
    max_border_touches: int = 3
    device: str | None = None
    overwrite: bool = False


def _resolve_requested_eyes(eye: str) -> tuple[str, ...]:
    value = str(eye).strip().lower()
    if value == "both":
        return ("left", "right")
    if value in {"left", "right"}:
        return (value,)
    raise ValueError(f"Unsupported eye selector: {eye}")


def _normalize_session_code(session_code: str) -> str:
    value = str(session_code).strip()
    if value.startswith("session_"):
        return session_dir_to_code(value)
    if value.startswith("session"):
        value = value[len("session") :]
    value = value.replace("_", "")
    if len(value) != 3 or not value.isdigit():
        raise ValueError(f"Unsupported session code: {session_code}")
    return value


def _resolve_raw_user_dir(raw_root: str | Path, user_id: int) -> Path:
    raw_root = Path(raw_root).resolve()
    for user_dir in collect_users(raw_root):
        try:
            parsed_user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        if parsed_user_id == int(user_id):
            return user_dir
    return raw_root / f"user{int(user_id):02d}"


def _xyxy_to_xywh_sensor(
    box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    image_size_wh: tuple[int, int],
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    image_w, image_h = [int(v) for v in image_size_wh]
    x0 = min(max(0.0, x0), float(image_w))
    y0 = min(max(0.0, y0), float(image_h))
    x1 = min(max(x0, x1), float(image_w))
    y1 = min(max(y0, y1), float(image_h))
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]


def _box_area_ratio_xyxy(
    box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    image_size_wh: tuple[int, int],
) -> float:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    image_w, image_h = [int(v) for v in image_size_wh]
    image_area = float(max(1, image_w * image_h))
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    return max(0.0, (width * height) / image_area)


def _count_touched_borders(
    box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    border_margin_px: int = 3,
) -> int:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    image_w, image_h = [int(v) for v in image_size_wh]
    margin = max(0, int(border_margin_px))
    touches = 0
    if x0 <= float(margin):
        touches += 1
    if y0 <= float(margin):
        touches += 1
    if x1 >= float(max(0, image_w - margin)):
        touches += 1
    if y1 >= float(max(0, image_h - margin)):
        touches += 1
    return touches


def _eye_bbox_rejection_reason(
    box_xyxy: list[float] | tuple[float, float, float, float] | np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    max_box_area_ratio: float,
    border_margin_px: int,
    max_border_touches: int,
) -> str | None:
    area_ratio = _box_area_ratio_xyxy(box_xyxy, image_size_wh=image_size_wh)
    if area_ratio > float(max_box_area_ratio):
        return f"rejected_large_box(area_ratio={area_ratio:.3f})"
    border_touches = _count_touched_borders(
        box_xyxy,
        image_size_wh=image_size_wh,
        border_margin_px=border_margin_px,
    )
    if border_touches >= int(max_border_touches):
        return f"rejected_border_touch(border_touches={border_touches})"
    return None


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _save_overlay_preview(
    *,
    frame_path: Path,
    output_path: Path,
    eye_region_bbox_xywh_sensor: list[float] | None,
    caption: str,
) -> Path:
    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if eye_region_bbox_xywh_sensor:
        _draw_xywh(draw, eye_region_bbox_xywh_sensor, outline="#00ff88", width=2)
    draw.text((6, 6), caption, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


class _GroundedSamEyeRegionRuntime:
    def __init__(self, cfg: GroundedSamEyeRegionBboxConfig) -> None:
        self.cfg = cfg
        repo_root = cfg.groundedsam_root.resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Grounded-SAM root not found: {repo_root}")
        if not cfg.groundingdino_config.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {cfg.groundingdino_config}")
        if not cfg.groundingdino_checkpoint.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {cfg.groundingdino_checkpoint}")

        extra_paths = [repo_root, repo_root / "GroundingDINO"]
        for extra in extra_paths:
            text = str(extra)
            if text not in sys.path:
                sys.path.insert(0, text)

        import torch  # noqa: WPS433
        import torchvision  # noqa: WPS433

        _patch_transformers_groundingdino_compat()
        _patch_groundingdino_ms_deform_attn_compat()
        from groundingdino.util.inference import Model  # noqa: WPS433

        self._torch = torch
        self._torchvision = torchvision
        self.device, device_warning = _resolve_runtime_device(torch, cfg.device)
        if device_warning:
            warnings.warn(device_warning, stacklevel=2)
        self.model = Model(
            model_config_path=str(cfg.groundingdino_config),
            model_checkpoint_path=str(cfg.groundingdino_checkpoint),
            device=str(self.device),
        )

    def _pick_eye_detection(self, detections: Any, image_shape: tuple[int, int, int]) -> tuple[int | None, str | None]:
        xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)))
        conf = np.asarray(getattr(detections, "confidence", np.empty((0,), dtype=np.float32)))
        if xyxy.size == 0 or conf.size == 0:
            return None, "no_eye_detections"
        keep = self._torchvision.ops.nms(
            self._torch.from_numpy(xyxy),
            self._torch.from_numpy(conf.astype(np.float32)),
            float(self.cfg.nms_threshold),
        ).cpu().numpy().tolist()
        image_size_wh = (image_shape[1], image_shape[0])
        best_key = None
        best_idx = None
        best_rejected_key = None
        best_rejected_reason = None
        for idx in keep:
            area_ratio = _box_area_ratio_xyxy(xyxy[idx], image_size_wh=image_size_wh)
            border_touches = _count_touched_borders(
                xyxy[idx],
                image_size_wh=image_size_wh,
                border_margin_px=self.cfg.border_margin_px,
            )
            key = (-float(conf[idx]), border_touches, area_ratio)
            rejection_reason = _eye_bbox_rejection_reason(
                xyxy[idx],
                image_size_wh=image_size_wh,
                max_box_area_ratio=self.cfg.max_box_area_ratio,
                border_margin_px=self.cfg.border_margin_px,
                max_border_touches=self.cfg.max_border_touches,
            )
            if rejection_reason is not None:
                if best_rejected_key is None or key < best_rejected_key:
                    best_rejected_key = key
                    best_rejected_reason = rejection_reason
                continue
            if best_key is None or key < best_key:
                best_key = key
                best_idx = int(idx)
        if best_idx is None:
            return None, best_rejected_reason or "no_valid_eye_box"
        return best_idx, None

    def detect_eye_region_bbox(self, image_rgb: np.ndarray) -> tuple[dict[str, Any] | None, str | None]:
        image_bgr = image_rgb[:, :, ::-1].copy()
        detections = self.model.predict_with_classes(
            image=image_bgr,
            classes=["eye"],
            box_threshold=float(self.cfg.box_threshold),
            text_threshold=float(self.cfg.text_threshold),
        )
        idx, rejection_reason = self._pick_eye_detection(detections, image_rgb.shape)
        if idx is None:
            return None, rejection_reason
        xyxy = np.asarray(detections.xyxy[idx], dtype=np.float32)
        confidence = float(detections.confidence[idx])
        image_size_wh = (image_rgb.shape[1], image_rgb.shape[0])
        bbox_xywh = _xyxy_to_xywh_sensor(
            xyxy,
            image_size_wh=image_size_wh,
        )
        return {
            "eye_region_bbox_xywh_sensor": bbox_xywh,
            "gsam_box_xyxy": [float(v) for v in xyxy.tolist()],
            "gsam_box_confidence": confidence,
            "gsam_class_name": "eye",
            "gsam_box_area_ratio": _box_area_ratio_xyxy(xyxy, image_size_wh=image_size_wh),
            "gsam_border_touches": _count_touched_borders(
                xyxy,
                image_size_wh=image_size_wh,
                border_margin_px=self.cfg.border_margin_px,
            ),
        }, None


def export_groundedsam_eye_region_bbox_overlays(
    *,
    raw_root: str | Path,
    annotation_root: str | Path,
    preview_root: str | Path,
    groundedsam_root: str | Path,
    groundingdino_config: str | Path | None = None,
    groundingdino_checkpoint: str | Path | None = None,
    user_id: int = 1,
    session_code: str = "101",
    eye: str = "both",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    nms_threshold: float = 0.8,
    max_box_area_ratio: float = 0.85,
    border_margin_px: int = 3,
    max_border_touches: int = 3,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_root = Path(raw_root).resolve()
    annotation_root = Path(annotation_root).resolve()
    preview_root = Path(preview_root).resolve()
    groundedsam_root = Path(groundedsam_root).resolve()
    groundingdino_config_path = (
        Path(groundingdino_config).resolve()
        if groundingdino_config
        else _resolve_default_groundingdino_config(groundedsam_root)
    )
    if groundingdino_checkpoint is None:
        groundingdino_checkpoint = groundedsam_root / "groundingdino_swint_ogc.pth"

    cfg = GroundedSamEyeRegionBboxConfig(
        raw_root=raw_root,
        annotation_root=annotation_root,
        preview_root=preview_root,
        groundedsam_root=groundedsam_root,
        groundingdino_config=Path(groundingdino_config_path).resolve(),
        groundingdino_checkpoint=Path(groundingdino_checkpoint).resolve(),
        user_id=int(user_id),
        session_code=_normalize_session_code(str(session_code)),
        eye=str(eye),
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        nms_threshold=float(nms_threshold),
        max_box_area_ratio=float(max_box_area_ratio),
        border_margin_px=max(0, int(border_margin_px)),
        max_border_touches=max(1, int(max_border_touches)),
        device=device,
        overwrite=bool(overwrite),
    )
    runtime = _GroundedSamEyeRegionRuntime(cfg)

    bbox_root = annotation_root / "eye_region_bbox_only"
    preview_base = preview_root / "groundedsam_eye_region_bbox_only"
    session_dir_name = code_to_session_dir(cfg.session_code)
    requested_eyes = _resolve_requested_eyes(cfg.eye)
    raw_user_dir = _resolve_raw_user_dir(raw_root, cfg.user_id)
    session_summaries: list[dict[str, Any]] = []
    total_frames = 0
    total_detected = 0
    found_any_session = False
    started_at = time.perf_counter()

    for eye_name in requested_eyes:
        raw_session_dir = raw_user_dir / eye_name / session_dir_name
        session_key = f"user{cfg.user_id:02d}/{eye_name}/session_{cfg.session_code}"
        bbox_session_dir = bbox_root / "sessions" / f"user{cfg.user_id:02d}" / eye_name / f"session_{cfg.session_code}"
        preview_session_dir = preview_base / "sessions" / f"user{cfg.user_id:02d}" / eye_name / f"session_{cfg.session_code}" / "overlays"
        store_path = bbox_session_dir / "eye_region_bboxes.jsonl"

        if not raw_session_dir.exists():
            session_summaries.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(raw_session_dir),
                    "bbox_store_path": str(store_path),
                    "overlay_dir": str(preview_session_dir),
                    "n_frames": 0,
                    "n_detected": 0,
                    "message": "raw_session_missing",
                }
            )
            continue

        found_any_session = True
        if store_path.exists() and not cfg.overwrite:
            rows = read_jsonl(store_path)
            detected = sum(1 for row in rows if row.get("eye_region_bbox_xywh_sensor"))
            session_summaries.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(raw_session_dir),
                    "bbox_store_path": str(store_path),
                    "overlay_dir": str(preview_session_dir),
                    "n_frames": len(rows),
                    "n_detected": int(detected),
                    "message": "existing_store_reused",
                }
            )
            total_frames += len(rows)
            total_detected += int(detected)
            continue

        layout = discover_session_layout(raw_session_dir, user_id=cfg.user_id)
        if layout.frames_dir is None:
            session_summaries.append(
                {
                    "session_key": session_key,
                    "raw_session_dir": str(raw_session_dir),
                    "bbox_store_path": str(store_path),
                    "overlay_dir": str(preview_session_dir),
                    "n_frames": 0,
                    "n_detected": 0,
                    "message": "frames_dir_missing",
                }
            )
            continue

        frame_records = collect_frame_records(layout.frames_dir)
        rows: list[dict[str, Any]] = []
        detected = 0

        for rec in frame_records:
            frame_path = layout.frames_dir / rec.filename
            image_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            result, detection_status = runtime.detect_eye_region_bbox(image_rgb)
            bbox_xywh = None if result is None else result["eye_region_bbox_xywh_sensor"]
            if result is not None:
                detected += 1
            caption = f"{session_key} | {Path(rec.filename).stem}"
            if result is None:
                caption = f"{caption} | {detection_status or 'no_eye_box'}"
            else:
                caption = (
                    f"{caption} | conf={result['gsam_box_confidence']:.3f} "
                    f"| area={result['gsam_box_area_ratio']:.3f} "
                    f"| borders={result['gsam_border_touches']}"
                )
            _save_overlay_preview(
                frame_path=frame_path,
                output_path=preview_session_dir / rec.filename,
                eye_region_bbox_xywh_sensor=bbox_xywh,
                caption=caption,
            )
            row = {
                "session_key": session_key,
                "user_id": int(cfg.user_id),
                "subject_id": int(cfg.user_id),
                "eye": eye_name,
                "session_code": str(cfg.session_code),
                "frame_filename": rec.filename,
                "frame_idx": rec.frame_idx,
                "timestamp_us": int(rec.timestamp_us),
                "frame_path": str(frame_path),
                "gsam_detection_status": str(detection_status or "detected"),
                "eye_region_bbox_xywh_sensor": bbox_xywh,
                "gsam_box_xyxy": None if result is None else result["gsam_box_xyxy"],
                "gsam_box_confidence": None if result is None else float(result["gsam_box_confidence"]),
                "gsam_class_name": None if result is None else result["gsam_class_name"],
                "gsam_box_area_ratio": None if result is None else float(result["gsam_box_area_ratio"]),
                "gsam_border_touches": None if result is None else int(result["gsam_border_touches"]),
                "annotation_source": "gsa_eye_bbox_only",
                "heuristic_eye_region_used": False,
            }
            rows.append(row)

        rows.sort(key=lambda row: (int(row.get("frame_idx") or -1), int(row.get("timestamp_us") or 0), str(row["frame_filename"])))
        write_jsonl(rows, store_path)
        session_summary = {
            "session_key": session_key,
            "raw_session_dir": str(raw_session_dir),
            "frames_dir": str(layout.frames_dir),
            "bbox_store_path": str(store_path),
            "overlay_dir": str(preview_session_dir),
            "n_frames": len(rows),
            "n_detected": int(detected),
            "max_box_area_ratio": float(cfg.max_box_area_ratio),
            "border_margin_px": int(cfg.border_margin_px),
            "max_border_touches": int(cfg.max_border_touches),
            "message": "groundedsam_eye_bbox_complete",
        }
        write_json(session_summary, bbox_session_dir / "session_summary.json")
        session_summaries.append(session_summary)
        total_frames += len(rows)
        total_detected += int(detected)

    if not found_any_session:
        raise FileNotFoundError(
            f"No raw session directories found for user{cfg.user_id:02d} session_{cfg.session_code}. "
            f"Expected under {raw_user_dir}"
        )

    run_summary = {
        "raw_root": str(raw_root),
        "annotation_root": str(annotation_root),
        "preview_root": str(preview_root),
        "groundedsam_root": str(groundedsam_root),
        "groundingdino_config": str(cfg.groundingdino_config),
        "groundingdino_checkpoint": str(cfg.groundingdino_checkpoint),
        "user_id": int(cfg.user_id),
        "session_code": str(cfg.session_code),
        "eyes": list(requested_eyes),
        "device": str(runtime.device),
        "n_sessions": len(session_summaries),
        "n_frames": int(total_frames),
        "n_detected": int(total_detected),
        "max_box_area_ratio": float(cfg.max_box_area_ratio),
        "border_margin_px": int(cfg.border_margin_px),
        "max_border_touches": int(cfg.max_border_touches),
        "heuristic_eye_region_used": False,
        "bbox_root": str(bbox_root),
        "preview_output_root": str(preview_base),
        "elapsed_sec": float(time.perf_counter() - started_at),
    }
    write_json(
        run_summary,
        bbox_root / f"user{cfg.user_id:02d}__session_{cfg.session_code}__summary.json",
    )
    write_jsonl(
        session_summaries,
        bbox_root / f"user{cfg.user_id:02d}__session_{cfg.session_code}__sessions.jsonl",
    )
    return run_summary
