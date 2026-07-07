from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
for extra in (SCRIPT_ROOT, SRC_ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from _bootstrap import ensure_project_src_on_path  # noqa: E402

ensure_project_src_on_path()

from hbtxr.preprocess.groundedsam_build import build_groundedsam_runtime  # noqa: E402
from hbtxr.preprocess.io_utils import collect_frame_records  # noqa: E402
from hbtxr.preprocess.path_utils import resolve_paths  # noqa: E402
from hbtxr.utils.io import read_json, write_json  # noqa: E402
import run_roi_crop_prompted_preview_batch as roi_runner  # noqa: E402


def _session_dir(raw_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return raw_root / f"user{int(user_id)}" / str(eye) / f"session_{session_code[0]}_{session_code[1]}_{session_code[2]}"


def _select_indices(events_total: list[int], *, per_group: int) -> dict[str, list[int]]:
    total = len(events_total)
    if total <= 0:
        return {"uniform": [], "burst": [], "sparse": []}
    n = min(int(per_group), total)
    uniform = sorted({int(round(v)) for v in np.linspace(0, total - 1, num=n)})
    burst = sorted(idx for idx, _ in sorted(enumerate(events_total), key=lambda item: item[1], reverse=True)[:n])
    sparse = sorted(idx for idx, _ in sorted(enumerate(events_total), key=lambda item: item[1])[:n])
    return {"uniform": uniform, "burst": burst, "sparse": sparse}


def _clip_xywh_to_rect(xywh: list[float], *, image_size_wh: tuple[int, int]) -> list[int]:
    return roi_runner._clip_crop_rect([float(v) for v in xywh], image_size_wh)


def _run_pupil_chain(
    *,
    crop_rgb: np.ndarray,
    runtime: Any,
    pupil_option_chain: list[dict[str, Any]],
    default_resize_wh: tuple[int, int] | None,
    optional_post_filters: dict[str, Any],
    retry_pupil_on_fail_like: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    accepted = None
    best_fail_like = None
    attempted_stage_names: list[str] = []
    crop_source_size_wh = (int(crop_rgb.shape[1]), int(crop_rgb.shape[0]))

    for pupil_options in pupil_option_chain:
        stage_name = str(pupil_options.get("name") or pupil_options.get("classes", ["pupil"])[0])
        attempted_stage_names.append(stage_name)
        candidates = roi_runner._annotate_crop_with_pupil_options(
            source_rgb=crop_rgb,
            runtime=runtime,
            pupil_options=pupil_options,
            default_resize_wh=default_resize_wh,
        )
        if not candidates:
            continue

        stage_best_fail_like = None
        for raw_candidate in candidates:
            candidate = roi_runner._project_candidate_to_source(raw_candidate, source_size_wh=crop_source_size_wh)
            if candidate is None:
                continue
            local_bbox = [float(v) for v in candidate["bbox_xywh"]]
            crop_mask = np.asarray(candidate["mask"]).astype(np.uint8)
            crop_area = float(max(1, crop_rgb.shape[0] * crop_rgb.shape[1]))
            crop_mask_area_ratio = float(crop_mask.sum()) / crop_area
            crop_bbox_fill_width_ratio = float(local_bbox[2]) / float(max(1, crop_rgb.shape[1]))
            crop_bbox_fill_height_ratio = float(local_bbox[3]) / float(max(1, crop_rgb.shape[0]))
            fail_like_mask_ratio = crop_mask_area_ratio > float(optional_post_filters.get("crop_mask_area_ratio_max", 0.30))
            fail_like_fill_ratio = max(crop_bbox_fill_width_ratio, crop_bbox_fill_height_ratio) > float(
                optional_post_filters.get("crop_bbox_fill_max", 0.85),
            )
            fail_like_union = bool(fail_like_mask_ratio or fail_like_fill_ratio)
            candidate["crop_mask_area_ratio"] = crop_mask_area_ratio
            candidate["crop_bbox_fill_width_ratio"] = crop_bbox_fill_width_ratio
            candidate["crop_bbox_fill_height_ratio"] = crop_bbox_fill_height_ratio
            candidate["fail_like_mask_ratio"] = fail_like_mask_ratio
            candidate["fail_like_fill_ratio"] = fail_like_fill_ratio
            candidate["fail_like_union"] = fail_like_union
            acceptance_filters = dict(pupil_options.get("acceptance_filters") or {})
            stage_gate_passed, stage_gate_reasons = roi_runner._passes_geometry_gate(
                crop_mask_area_ratio=crop_mask_area_ratio,
                crop_bbox_fill_width_ratio=crop_bbox_fill_width_ratio,
                crop_bbox_fill_height_ratio=crop_bbox_fill_height_ratio,
                filters=acceptance_filters,
            )
            candidate["stage_gate_passed"] = bool(stage_gate_passed)
            candidate["stage_gate_reasons"] = list(stage_gate_reasons)
            if acceptance_filters and not stage_gate_passed:
                continue
            if not fail_like_union:
                accepted = candidate
                break
            if stage_best_fail_like is None or roi_runner._fail_like_preference_key(candidate) < roi_runner._fail_like_preference_key(stage_best_fail_like):
                stage_best_fail_like = candidate

        if accepted is not None:
            break
        if stage_best_fail_like is not None and (
            best_fail_like is None or roi_runner._fail_like_preference_key(stage_best_fail_like) < roi_runner._fail_like_preference_key(best_fail_like)
        ):
            best_fail_like = stage_best_fail_like
        if not retry_pupil_on_fail_like:
            break

    return accepted or best_fail_like, attempted_stage_names


def _detect_from_crop(
    *,
    frame_rgb: np.ndarray,
    crop_xywh: list[float],
    runtime: Any,
    pupil_option_chain: list[dict[str, Any]],
    default_resize_wh: tuple[int, int] | None,
    optional_post_filters: dict[str, Any],
    retry_pupil_on_fail_like: bool,
) -> dict[str, Any]:
    rect = _clip_xywh_to_rect(crop_xywh, image_size_wh=(int(frame_rgb.shape[1]), int(frame_rgb.shape[0])))
    x0, y0, x1, y1 = rect
    crop_rgb = frame_rgb[y0:y1, x0:x1].copy()
    result, attempted_stage_names = _run_pupil_chain(
        crop_rgb=crop_rgb,
        runtime=runtime,
        pupil_option_chain=pupil_option_chain,
        default_resize_wh=default_resize_wh,
        optional_post_filters=optional_post_filters,
        retry_pupil_on_fail_like=retry_pupil_on_fail_like,
    )
    payload: dict[str, Any] = {
        "crop_rect_xyxy_sensor": rect,
        "crop_xywh_sensor": [float(v) for v in crop_xywh],
        "attempted_pupil_stage_names": attempted_stage_names,
        "status": "pupil_failed",
        "predicted_class_name": None,
        "predicted_pupil_stage_name": None,
        "predicted_box_confidence": None,
        "predicted_candidate_rank": None,
        "predicted_max_candidate_boxes": None,
        "sensor_pupil_bbox_xywh": None,
        "sensor_pupil_ellipse_xywht": None,
        "crop_mask_area_ratio": None,
        "crop_bbox_fill_width_ratio": None,
        "crop_bbox_fill_height_ratio": None,
        "fail_like_union": False,
        "fail_like_mask_ratio": False,
        "fail_like_fill_ratio": False,
        "crop_mask": None,
        "crop_mask_input": None,
        "crop_pupil_bbox_xywh_local": None,
        "crop_pupil_ellipse_xywht_local": None,
        "crop_input_size_wh": [int(crop_rgb.shape[1]), int(crop_rgb.shape[0])],
    }
    if result is None:
        return payload

    local_bbox = [float(v) for v in result["bbox_xywh"]]
    local_ellipse = [float(v) for v in result["ellipse_xywht"]]
    sensor_bbox = roi_runner._sensor_bbox_from_local(local_bbox, rect)
    sensor_ellipse = roi_runner._sensor_ellipse_from_local(local_ellipse, rect)
    payload.update(
        {
            "status": "completed",
            "predicted_class_name": str(result["class_name"]),
            "predicted_pupil_stage_name": str(result.get("stage_name") or result["class_name"]),
            "predicted_box_confidence": float(result["box_confidence"]),
            "predicted_candidate_rank": int(result.get("candidate_rank", 1)),
            "predicted_max_candidate_boxes": int(result.get("max_candidate_boxes", 1)),
            "sensor_pupil_bbox_xywh": sensor_bbox,
            "sensor_pupil_ellipse_xywht": sensor_ellipse,
            "crop_mask_area_ratio": float(result["crop_mask_area_ratio"]),
            "crop_bbox_fill_width_ratio": float(result["crop_bbox_fill_width_ratio"]),
            "crop_bbox_fill_height_ratio": float(result["crop_bbox_fill_height_ratio"]),
            "fail_like_union": bool(result["fail_like_union"]),
            "fail_like_mask_ratio": bool(result["fail_like_mask_ratio"]),
            "fail_like_fill_ratio": bool(result["fail_like_fill_ratio"]),
            "crop_mask": np.asarray(result["mask"]).astype(np.uint8),
            "crop_mask_input": np.asarray(result.get("mask_input", result["mask"])).astype(np.uint8),
            "crop_pupil_bbox_xywh_local": local_bbox,
            "crop_pupil_ellipse_xywht_local": local_ellipse,
            "crop_input_size_wh": [int(v) for v in result.get("input_size_wh", [crop_rgb.shape[1], crop_rgb.shape[0]])],
        }
    )
    return payload


def _status_rank(method_row: dict[str, Any]) -> int:
    if method_row.get("status") != "completed":
        return 0
    if bool(method_row.get("fail_like_union")):
        return 1
    return 2


def _outcome_label(*, full_eye: dict[str, Any], event_roi: dict[str, Any]) -> str:
    full_rank = _status_rank(full_eye)
    event_rank = _status_rank(event_roi)
    if event_rank > full_rank:
        return "event_roi_better"
    if event_rank < full_rank:
        return "full_eye_better"
    return "tie"


def _draw_xywh(draw: ImageDraw.ImageDraw, xywh: list[float], *, outline: str, width: int = 2) -> None:
    x, y, w, h = [float(v) for v in xywh]
    draw.rectangle((x, y, x + w, y + h), outline=outline, width=width)


def _draw_rotated_ellipse(draw: ImageDraw.ImageDraw, ellipse_xywht: list[float], *, outline: str, width: int = 2) -> None:
    roi_runner._draw_rotated_ellipse(draw, ellipse_xywht, outline=outline, width=width)


def _overlay_mask(base: Image.Image, mask: np.ndarray | None, *, offset_xy: tuple[int, int], color: tuple[int, int, int], alpha: int) -> Image.Image:
    if mask is None:
        return base
    return roi_runner._apply_mask_overlay(base, mask, color=color, alpha=alpha, offset_xy=offset_xy)


def _compose_compare_panel(
    *,
    frame_path: Path,
    eye_bbox_xywh: list[float],
    event_roi_xywh: list[float],
    full_eye: dict[str, Any],
    event_roi: dict[str, Any],
    title: str,
) -> Image.Image:
    frame = Image.open(frame_path).convert("RGB")
    full_overlay = frame.copy()
    full_overlay = _overlay_mask(
        full_overlay,
        full_eye.get("crop_mask"),
        offset_xy=(int(full_eye["crop_rect_xyxy_sensor"][0]), int(full_eye["crop_rect_xyxy_sensor"][1])),
        color=(255, 96, 64),
        alpha=72,
    )
    full_overlay = _overlay_mask(
        full_overlay,
        event_roi.get("crop_mask"),
        offset_xy=(int(event_roi["crop_rect_xyxy_sensor"][0]), int(event_roi["crop_rect_xyxy_sensor"][1])),
        color=(64, 176, 255),
        alpha=72,
    )
    draw = ImageDraw.Draw(full_overlay)
    _draw_xywh(draw, eye_bbox_xywh, outline="#00ff88", width=2)
    _draw_xywh(draw, event_roi_xywh, outline="#56b7ff", width=2)
    if full_eye.get("sensor_pupil_bbox_xywh"):
        _draw_xywh(draw, full_eye["sensor_pupil_bbox_xywh"], outline="#ffd84d", width=2)
    if full_eye.get("sensor_pupil_ellipse_xywht"):
        _draw_rotated_ellipse(draw, full_eye["sensor_pupil_ellipse_xywht"], outline="#ff6840", width=2)
    if event_roi.get("sensor_pupil_bbox_xywh"):
        _draw_xywh(draw, event_roi["sensor_pupil_bbox_xywh"], outline="#72c3ff", width=2)
    if event_roi.get("sensor_pupil_ellipse_xywht"):
        _draw_rotated_ellipse(draw, event_roi["sensor_pupil_ellipse_xywht"], outline="#5d8cff", width=2)
    draw.text((6, 6), title, fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((6, 18), f"full={full_eye['status']}{' fail_like' if full_eye.get('fail_like_union') else ''}", fill=(255, 255, 255), font=ImageFont.load_default())
    draw.text((6, 30), f"event={event_roi['status']}{' fail_like' if event_roi.get('fail_like_union') else ''}", fill=(255, 255, 255), font=ImageFont.load_default())

    def _crop_panel(method_row: dict[str, Any], label: str) -> Image.Image:
        rect = method_row["crop_rect_xyxy_sensor"]
        crop = frame.crop((rect[0], rect[1], rect[2], rect[3]))
        mask = method_row.get("crop_mask_input")
        if mask is not None:
            crop = roi_runner._apply_mask_overlay(crop, mask, color=(255, 64, 64), alpha=96) if label == "full-eye crop" else roi_runner._apply_mask_overlay(crop, mask, color=(64, 176, 255), alpha=96)
        draw = ImageDraw.Draw(crop)
        bbox = method_row.get("crop_pupil_bbox_xywh_local")
        ellipse = method_row.get("crop_pupil_ellipse_xywht_local")
        if bbox is not None:
            _draw_xywh(draw, bbox, outline="#ffd84d" if label == "full-eye crop" else "#72c3ff", width=2)
        if ellipse is not None:
            _draw_rotated_ellipse(draw, ellipse, outline="#ff4040" if label == "full-eye crop" else "#5d8cff", width=2)
        draw.text((6, 6), label, fill=(255, 255, 255), font=ImageFont.load_default())
        draw.text((6, 18), f"status={method_row['status']}", fill=(255, 255, 255), font=ImageFont.load_default())
        return crop

    full_crop_panel = _crop_panel(full_eye, "full-eye crop")
    event_crop_panel = _crop_panel(event_roi, "event ROI crop")
    full_crop_panel.thumbnail((frame.width, frame.height))
    event_crop_panel.thumbnail((frame.width, frame.height))
    canvas = Image.new("RGB", (frame.width * 3, frame.height), color=(12, 12, 12))
    canvas.paste(full_overlay, (0, 0))
    canvas.paste(full_crop_panel, (frame.width, 0))
    canvas.paste(event_crop_panel, (frame.width * 2, 0))
    return canvas


def _thumb_with_title(image: Image.Image, *, label: str, thumb_size_wh: tuple[int, int]) -> Image.Image:
    thumb_w, thumb_h = thumb_size_wh
    title_h = 18
    out = Image.new("RGB", (thumb_w, thumb_h + title_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(out)
    draw.text((4, 2), label, fill=(230, 230, 230), font=ImageFont.load_default())
    thumb = image.copy()
    thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    paste_x = (thumb_w - thumb.width) // 2
    paste_y = title_h + (thumb_h - thumb.height) // 2
    out.paste(thumb, (paste_x, paste_y))
    return out


def _save_pages(
    *,
    items: list[tuple[str, Image.Image]],
    out_dir: Path,
    title_prefix: str,
    cols: int = 2,
    rows: int = 3,
    thumb_size_wh: tuple[int, int] = (540, 190),
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_paths: list[str] = []
    per_page = cols * rows
    for page_idx in range(0, len(items), per_page):
        chunk = items[page_idx : page_idx + per_page]
        thumb_w, thumb_h = thumb_size_wh
        tile_h = thumb_h + 18
        title_h = 28
        page = Image.new("RGB", (cols * thumb_w, rows * tile_h + title_h), color=(12, 12, 12))
        draw = ImageDraw.Draw(page)
        draw.text((8, 6), f"{title_prefix} page={page_idx // per_page + 1}", fill=(240, 240, 240), font=ImageFont.load_default())
        for local_idx, (label, image) in enumerate(chunk):
            r = local_idx // cols
            c = local_idx % cols
            x = c * thumb_w
            y = title_h + r * tile_h
            page.paste(_thumb_with_title(image, label=label, thumb_size_wh=thumb_size_wh), (x, y))
        out_path = out_dir / f"page_{page_idx // per_page + 1:03d}.png"
        page.save(out_path)
        page_paths.append(str(out_path))
    return page_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare full-eye crop vs event-guided ROI crop using the v10 pupil detector.")
    parser.add_argument("--paths-config", type=str, default="configs/paths/ev_eye_groundedsam_paths.json")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--session-code", type=str, default="201")
    parser.add_argument("--eyes", nargs="+", default=["left", "right"])
    parser.add_argument("--detector-config", type=Path, default=PROJECT_ROOT / "exps" / "configs" / "groundedsam" / "roi_crop_prompted_pupil_v10_eye_prompted_native_then_crop128_rescue.json")
    parser.add_argument("--event-guided-root", type=Path, default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_event_guided_pupil_roi")
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "workspace_event_voxel_analysis" / "user01_session201_event_guided_pupil_detector_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_args = argparse.Namespace(
        paths_config=args.paths_config,
        project_root=None,
        raw_root=None,
        canonical_root=None,
        indexes_root=None,
        manifests_root=None,
        annotation_root=None,
        groundedsam_root=None,
        timelens_root=None,
        preview_root=None,
    )
    paths = resolve_paths(path_args, need_raw=True, need_canonical=False)
    raw_root = Path(paths.raw_root).resolve()
    groundedsam_root = Path(paths.groundedsam_root).resolve()
    detector_cfg = read_json(args.detector_config)
    pupil_stage = dict(detector_cfg["pupil_stage"])
    pupil_fallbacks = [dict(item) for item in detector_cfg.get("pupil_stage_fallbacks") or []]
    pupil_option_chain = [pupil_stage] + pupil_fallbacks
    optional_post_filters = dict(detector_cfg.get("optional_post_filters") or {})
    retry_pupil_on_fail_like = bool(detector_cfg.get("retry_pupil_on_fail_like", True))
    default_resize_wh = roi_runner._normalize_size_wh(detector_cfg.get("crop_resize_wh"))

    runtime = build_groundedsam_runtime(
        groundedsam_root=groundedsam_root,
        classes=tuple(pupil_stage["classes"]),
        box_threshold=float(pupil_stage["box_threshold"]),
        text_threshold=float(pupil_stage["text_threshold"]),
        nms_threshold=float(pupil_stage["nms_threshold"]),
        min_mask_area=int(pupil_stage["min_mask_area"]),
        device=args.device,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for eye in args.eyes:
        stats_path = args.event_guided_root / eye / "interval_event_guided_roi_stats.json"
        rows = json.loads(stats_path.read_text(encoding="utf-8"))
        selected = _select_indices([int(row["events_total"]) for row in rows], per_group=int(args.per_group))
        selected_union = sorted({idx for group in selected.values() for idx in group})
        frames_dir = _session_dir(raw_root, user_id=int(args.user_id), eye=str(eye), session_code=str(args.session_code)) / "frames"
        frame_records = collect_frame_records(frames_dir)
        eye_root = output_root / str(eye)
        eye_root.mkdir(parents=True, exist_ok=True)

        result_rows: list[dict[str, Any]] = []
        preview_by_group: dict[str, list[tuple[str, Image.Image]]] = {name: [] for name in selected}
        preview_by_outcome: dict[str, list[tuple[str, Image.Image]]] = {"event_roi_better": [], "full_eye_better": [], "tie": []}

        for idx in selected_union:
            interval_row = rows[idx]
            frame_filename = str(interval_row["frame_from"])
            frame_path = frames_dir / frame_filename
            frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            eye_bbox = [float(v) for v in interval_row["eye_bbox_xywh_sensor"]]
            event_roi_xywh = [float(v) for v in interval_row["event_guided_roi_xywh_sensor"]]

            full_eye = _detect_from_crop(
                frame_rgb=frame_rgb,
                crop_xywh=eye_bbox,
                runtime=runtime,
                pupil_option_chain=pupil_option_chain,
                default_resize_wh=default_resize_wh,
                optional_post_filters=optional_post_filters,
                retry_pupil_on_fail_like=retry_pupil_on_fail_like,
            )
            event_roi = _detect_from_crop(
                frame_rgb=frame_rgb,
                crop_xywh=event_roi_xywh,
                runtime=runtime,
                pupil_option_chain=pupil_option_chain,
                default_resize_wh=default_resize_wh,
                optional_post_filters=optional_post_filters,
                retry_pupil_on_fail_like=retry_pupil_on_fail_like,
            )
            outcome = _outcome_label(full_eye=full_eye, event_roi=event_roi)
            interval_groups = [name for name, indices in selected.items() if idx in indices]
            record = {
                "interval_index": int(interval_row["interval_index"]),
                "frame_from": frame_filename,
                "events_total": int(interval_row["events_total"]),
                "event_guided_source": interval_row["event_guided_source"],
                "groups": interval_groups,
                "eye_bbox_xywh_sensor": eye_bbox,
                "event_guided_roi_xywh_sensor": event_roi_xywh,
                "full_eye": {k: v for k, v in full_eye.items() if k not in {"crop_mask", "crop_mask_input"}},
                "event_roi": {k: v for k, v in event_roi.items() if k not in {"crop_mask", "crop_mask_input"}},
                "comparison_outcome": outcome,
            }
            result_rows.append(record)

            panel = _compose_compare_panel(
                frame_path=frame_path,
                eye_bbox_xywh=eye_bbox,
                event_roi_xywh=event_roi_xywh,
                full_eye=full_eye,
                event_roi=event_roi,
                title=f"{eye} interval={interval_row['interval_index']} events={interval_row['events_total']}",
            )
            label = f"{int(interval_row['interval_index']):04d} | {outcome}"
            for group_name in interval_groups:
                preview_by_group[group_name].append((label, panel))
            preview_by_outcome[outcome].append((label, panel))

        stats_out_path = eye_root / "compare_stats.json"
        stats_out_path.write_text(json.dumps(result_rows, indent=2), encoding="utf-8")

        group_pages: dict[str, list[str]] = {}
        for group_name, items in preview_by_group.items():
            group_pages[group_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{group_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} compare {group_name}",
            )

        outcome_pages: dict[str, list[str]] = {}
        for outcome_name, items in preview_by_outcome.items():
            outcome_pages[outcome_name] = _save_pages(
                items=items,
                out_dir=eye_root / f"{outcome_name}_pages",
                title_prefix=f"user{int(args.user_id):02d} {eye} session_{args.session_code} compare {outcome_name}",
            )

        def _count(method: str, *, status: str | None = None, fail_like: bool | None = None) -> int:
            total = 0
            for row in result_rows:
                item = row[method]
                if status is not None and item["status"] != status:
                    continue
                if fail_like is not None and bool(item.get("fail_like_union")) != bool(fail_like):
                    continue
                total += 1
            return total

        summary_rows.append(
            {
                "user_id": int(args.user_id),
                "eye": str(eye),
                "session_code": str(args.session_code),
                "n_selected": len(result_rows),
                "selection_groups": {name: len(indices) for name, indices in selected.items()},
                "full_eye_completed": _count("full_eye", status="completed"),
                "full_eye_fail_like": _count("full_eye", status="completed", fail_like=True),
                "full_eye_pupil_failed": _count("full_eye", status="pupil_failed"),
                "event_roi_completed": _count("event_roi", status="completed"),
                "event_roi_fail_like": _count("event_roi", status="completed", fail_like=True),
                "event_roi_pupil_failed": _count("event_roi", status="pupil_failed"),
                "comparison_outcomes": dict(sorted(Counter(row["comparison_outcome"] for row in result_rows).items())),
                "group_pages": group_pages,
                "outcome_pages": outcome_pages,
                "compare_stats_path": str(stats_out_path),
            }
        )
        print(f"[done] detector compare finished for {eye} selected={len(result_rows)}")

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    write_json(summary_rows, summary_json_path)
    lines = [
        "# Event-Guided ROI Detector Compare Summary",
        "",
        f"- user_id: {int(args.user_id)}",
        f"- session_code: {str(args.session_code)}",
        f"- eyes: {', '.join(str(v) for v in args.eyes)}",
        f"- detector_config: {str(Path(args.detector_config).resolve())}",
        "- comparison: anchor eye crop vs event-guided ROI crop on the same frame",
        "",
    ]
    for row in summary_rows:
        lines.extend(
            [
                f"## {row['eye']}",
                f"- n_selected: {row['n_selected']}",
                f"- selection_groups: {row['selection_groups']}",
                f"- full_eye_completed/fail_like/pupil_failed: {row['full_eye_completed']} / {row['full_eye_fail_like']} / {row['full_eye_pupil_failed']}",
                f"- event_roi_completed/fail_like/pupil_failed: {row['event_roi_completed']} / {row['event_roi_fail_like']} / {row['event_roi_pupil_failed']}",
                f"- comparison_outcomes: {row['comparison_outcomes']}",
                f"- uniform_first_page: {row['group_pages']['uniform'][0] if row['group_pages']['uniform'] else ''}",
                f"- burst_first_page: {row['group_pages']['burst'][0] if row['group_pages']['burst'] else ''}",
                f"- sparse_first_page: {row['group_pages']['sparse'][0] if row['group_pages']['sparse'] else ''}",
                f"- event_roi_better_first_page: {row['outcome_pages']['event_roi_better'][0] if row['outcome_pages']['event_roi_better'] else ''}",
                f"- full_eye_better_first_page: {row['outcome_pages']['full_eye_better'][0] if row['outcome_pages']['full_eye_better'] else ''}",
                "",
            ]
        )
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote outputs under {output_root}")


if __name__ == "__main__":
    main()
