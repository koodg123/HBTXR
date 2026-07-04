from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from src.preprocess.annotation_groundedsam import GroundedSamAnnotation, groundedsam_session_dir, save_mask
from src.preprocess.io_utils import collect_frame_records, discover_session_layout
from src.preprocess.raw_ellipse_blink import apply_groundedsam_store_blink_metadata
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


class GroundedSamRuntimeFactory:
    def __init__(
        self,
        *,
        config_cls: type[Any],
        runtime_cls: type[Any],
        default_classes: tuple[str, ...],
        validate_shard_spec: Callable[[int | None, int | None], tuple[int, int]],
    ) -> None:
        self.config_cls = config_cls
        self.runtime_cls = runtime_cls
        self.default_classes = default_classes
        self.validate_shard_spec = validate_shard_spec

    @staticmethod
    def resolve_default_groundingdino_config(repo_root: Path) -> Path:
        return repo_root / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"

    @staticmethod
    def resolve_default_groundingdino_checkpoint(repo_root: Path) -> Path:
        candidates = [
            repo_root / "groundingdino_swint_ogc.pth",
            repo_root / "model_zoo" / "groundingdino_swint_ogc.pth",
            repo_root.parent / "model_zoo" / "groundingdino_swint_ogc.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def resolve_default_sam_checkpoint(repo_root: Path) -> Path:
        candidates = [
            repo_root / "sam_vit_h_4b8939.pth",
            repo_root / "model_zoo" / "sam_vit_h_4b8939.pth",
            repo_root.parent / "model_zoo" / "sam_vit_h_4b8939.pth",
            repo_root / "sam_hq_vit_h.pth",
            repo_root / "model_zoo" / "sam_hq_vit_h.pth",
            repo_root.parent / "model_zoo" / "sam_hq_vit_h.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def build_config(
        self,
        *,
        raw_root: str | Path,
        annotation_root: str | Path,
        groundedsam_root: str | Path,
        groundingdino_config: str | Path | None = None,
        groundingdino_checkpoint: str | Path | None = None,
        sam_checkpoint: str | Path | None = None,
        sam_encoder_version: str = "vit_h",
        classes: Iterable[str] = (),
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.8,
        min_mask_area: int = 16,
        frame_step: int = 1,
        max_frames_per_session: int | None = None,
        max_sessions: int | None = None,
        user_id: int | None = None,
        eye: str | None = None,
        session_codes: Iterable[str] | None = None,
        include_nonstandard_sessions: bool = False,
        num_shards: int = 1,
        shard_index: int = 0,
        device: str | None = None,
        overwrite: bool = False,
    ) -> Any:
        raw_root = Path(raw_root).resolve()
        annotation_root = Path(annotation_root).resolve()
        groundedsam_root = Path(groundedsam_root).resolve()
        config_path = (
            Path(groundingdino_config).resolve()
            if groundingdino_config
            else self.resolve_default_groundingdino_config(groundedsam_root)
        )
        if groundingdino_checkpoint is None:
            groundingdino_checkpoint = self.resolve_default_groundingdino_checkpoint(groundedsam_root)
        if sam_checkpoint is None:
            sam_checkpoint = self.resolve_default_sam_checkpoint(groundedsam_root)
        resolved_num_shards, resolved_shard_index = self.validate_shard_spec(num_shards, shard_index)
        return self.config_cls(
            raw_root=raw_root,
            annotation_root=annotation_root,
            groundedsam_root=groundedsam_root,
            groundingdino_config=Path(config_path).resolve(),
            groundingdino_checkpoint=Path(groundingdino_checkpoint).resolve(),
            sam_checkpoint=Path(sam_checkpoint).resolve(),
            sam_encoder_version=str(sam_encoder_version),
            classes=tuple(str(item).strip() for item in classes if str(item).strip()) or self.default_classes,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            nms_threshold=float(nms_threshold),
            min_mask_area=int(min_mask_area),
            frame_step=max(1, int(frame_step)),
            max_frames_per_session=None if max_frames_per_session is None else max(1, int(max_frames_per_session)),
            max_sessions=None if max_sessions is None else max(1, int(max_sessions)),
            user_id=None if user_id is None else int(user_id),
            eye=None if eye is None else str(eye),
            session_codes=None if session_codes is None else tuple(str(code) for code in session_codes),
            include_nonstandard_sessions=bool(include_nonstandard_sessions),
            num_shards=resolved_num_shards,
            shard_index=resolved_shard_index,
            device=device,
            overwrite=bool(overwrite),
        )

    def build_from_config(self, cfg: Any) -> Any:
        return self.runtime_cls(cfg)

    def build_runtime(
        self,
        *,
        groundedsam_root: str | Path,
        groundingdino_config: str | Path | None = None,
        groundingdino_checkpoint: str | Path | None = None,
        sam_checkpoint: str | Path | None = None,
        sam_encoder_version: str = "vit_h",
        classes: Iterable[str] = (),
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.8,
        min_mask_area: int = 16,
        device: str | None = None,
    ) -> Any:
        repo_root = Path(groundedsam_root).resolve()
        cfg = self.build_config(
            raw_root=repo_root,
            annotation_root=repo_root,
            groundedsam_root=repo_root,
            groundingdino_config=groundingdino_config,
            groundingdino_checkpoint=groundingdino_checkpoint,
            sam_checkpoint=sam_checkpoint,
            sam_encoder_version=sam_encoder_version,
            classes=classes or self.default_classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
            min_mask_area=min_mask_area,
            device=device,
        )
        return self.build_from_config(cfg)


class GroundedSamSummaryWriter:
    def __init__(self, annotation_root: str | Path) -> None:
        self.annotation_root = Path(annotation_root).resolve()

    @staticmethod
    def _validate_shard_spec(num_shards: int, shard_index: int) -> tuple[int, int]:
        resolved_num_shards = int(num_shards)
        resolved_shard_index = int(shard_index)
        if resolved_num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {resolved_num_shards}")
        if resolved_shard_index < 0 or resolved_shard_index >= resolved_num_shards:
            raise ValueError(
                f"shard_index must be in [0, {resolved_num_shards}), got {resolved_shard_index}",
            )
        return resolved_num_shards, resolved_shard_index

    def summary_paths(self, *, num_shards: int, shard_index: int) -> tuple[Path, Path]:
        resolved_num_shards, resolved_shard_index = self._validate_shard_spec(num_shards, shard_index)
        if resolved_num_shards == 1:
            return self.annotation_root / "annotation_summary.json", self.annotation_root / "annotation_sessions.jsonl"
        suffix = f".shard_{resolved_shard_index}_of_{resolved_num_shards}"
        return (
            self.annotation_root / f"annotation_summary{suffix}.json",
            self.annotation_root / f"annotation_sessions{suffix}.jsonl",
        )

    def build_session_summary_payload(
        self,
        *,
        session_key: str,
        raw_session_dir: Path,
        store_path: Path,
        n_annotations: int,
        classes: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        frame_step: int,
        max_frames_per_session: int | None,
        blink_candidate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "session_key": session_key,
            "raw_session_dir": str(raw_session_dir),
            "annotation_store_path": str(store_path),
            "n_annotations": int(n_annotations),
            "classes": list(classes),
            "box_threshold": float(box_threshold),
            "text_threshold": float(text_threshold),
            "nms_threshold": float(nms_threshold),
            "frame_step": int(frame_step),
            "max_frames_per_session": max_frames_per_session,
            "blink_candidate_report": dict(blink_candidate_report or {}),
        }

    def build_session_index_row(
        self,
        *,
        session_key: str,
        raw_session_dir: Path,
        store_path: Path,
        rows: list[dict[str, Any]],
        skipped: bool,
        message: str,
        blink_candidate_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "session_key": session_key,
            "raw_session_dir": str(raw_session_dir),
            "annotation_store_path": str(store_path),
            "n_annotations": len(rows),
            "skipped": bool(skipped),
            "message": str(message),
            "blink_candidate_report": dict(blink_candidate_report or {}),
        }

    def write_session_summary(self, session_summary_path: Path, payload: dict[str, Any]) -> None:
        write_json(payload, session_summary_path)

    def write_shard_outputs(
        self,
        *,
        summary: dict[str, Any],
        session_summaries: list[dict[str, Any]],
        num_shards: int,
        shard_index: int,
    ) -> None:
        self.annotation_root.mkdir(parents=True, exist_ok=True)
        summary_path, sessions_path = self.summary_paths(num_shards=num_shards, shard_index=shard_index)
        write_json(summary, summary_path)
        write_jsonl(session_summaries, sessions_path)

    def aggregate_shards(
        self,
        *,
        num_shards: int,
        devices: Sequence[str] | None = None,
        elapsed_sec: float | None = None,
    ) -> dict[str, Any]:
        resolved_num_shards, _ = self._validate_shard_spec(num_shards, 0)
        shard_summaries: list[dict[str, Any]] = []
        session_rows: list[dict[str, Any]] = []
        for shard_index in range(resolved_num_shards):
            summary_path, sessions_path = self.summary_paths(
                num_shards=resolved_num_shards,
                shard_index=shard_index,
            )
            shard_summary = read_json(summary_path)
            shard_summaries.append(shard_summary)
            session_rows.extend(read_jsonl(sessions_path))

        if not shard_summaries:
            raise ValueError("No Grounded-SAM shard summaries were produced.")

        reference = shard_summaries[0]
        merged_summary = {
            "raw_root": reference["raw_root"],
            "annotation_root": str(self.annotation_root),
            "annotation_backend": reference.get("annotation_backend", "groundedsam"),
            "groundedsam_root": reference["groundedsam_root"],
            "ultralytics_root": reference.get("ultralytics_root"),
            "groundedsam2_root": reference.get("groundedsam2_root"),
            "groundingdino_config": reference["groundingdino_config"],
            "groundingdino_checkpoint": reference["groundingdino_checkpoint"],
            "sam_checkpoint": reference["sam_checkpoint"],
            "ultralytics_sam3_checkpoint": reference.get("ultralytics_sam3_checkpoint"),
            "groundedsam2_config": reference.get("groundedsam2_config"),
            "groundedsam2_checkpoint": reference.get("groundedsam2_checkpoint"),
            "groundedsam2_groundingdino_config": reference.get("groundedsam2_groundingdino_config"),
            "groundedsam2_groundingdino_checkpoint": reference.get("groundedsam2_groundingdino_checkpoint"),
            "classes": list(reference["classes"]),
            "device": None,
            "devices": list(devices) if devices is not None else [str(item.get("device")) for item in shard_summaries],
            "num_shards": resolved_num_shards,
            "shard_index": None,
            "sharded": True,
            "n_sessions": int(sum(int(item["n_sessions"]) for item in shard_summaries)),
            "n_annotations": int(sum(int(item["n_annotations"]) for item in shard_summaries)),
            "elapsed_sec": float(
                elapsed_sec
                if elapsed_sec is not None
                else max(float(item.get("elapsed_sec") or 0.0) for item in shard_summaries)
            ),
            "user_id_filter": reference.get("user_id_filter"),
            "eye_filter": list(reference.get("eye_filter") or []),
            "session_code_filter": list(reference.get("session_code_filter") or []),
            "include_nonstandard_sessions": bool(reference.get("include_nonstandard_sessions", False)),
        }
        session_rows.sort(key=lambda row: str(row.get("session_key") or ""))
        write_json(merged_summary, self.annotation_root / "annotation_summary.json")
        write_jsonl(session_rows, self.annotation_root / "annotation_sessions.jsonl")
        return merged_summary


class GroundedSamSessionProcessor:
    def __init__(self, *, cfg: Any, runtime: Any, summary_writer: GroundedSamSummaryWriter) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self.summary_writer = summary_writer

    @staticmethod
    def annotation_row_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
        frame_idx = row.get("frame_idx")
        return (
            int(frame_idx) if frame_idx is not None else -1,
            int(row.get("timestamp_us") or 0),
            str(row.get("ann_id") or row.get("frame_filename") or ""),
        )

    def _annotate_frame_records(
        self,
        *,
        frame_records: list[Any],
        layout: Any,
        session_key: str,
        session_user_id: int,
        session_eye: str,
        session_code: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rec in frame_records:
            frame_path = layout.frames_dir / rec.filename
            image_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            result = self.runtime.annotate_image(image_rgb)
            if result is None:
                continue
            mask_name = Path(rec.filename).with_suffix(".png").name
            mask_rel = (
                Path("sessions")
                / f"user{session_user_id:02d}"
                / session_eye
                / f"session_{session_code}"
                / "masks"
                / mask_name
            )
            save_mask(result["mask"] * 255, self.cfg.annotation_root / mask_rel)
            quality = max(
                0.0,
                min(1.0, 0.5 * result["box_confidence"] + 0.5 * result["mask_score"]),
            )
            ann = GroundedSamAnnotation(
                ann_id=f"{session_key.replace('/', '__')}__{Path(rec.filename).stem}",
                frame_filename=rec.filename,
                timestamp_us=int(rec.timestamp_us),
                eye_region_bbox_xywh_sensor=[float(v) for v in result["eye_region_xywh"]],
                pupil_mask_path=str(mask_rel),
                pupil_region_bbox_xywh_sensor=[float(v) for v in result["bbox_xywh"]],
                pupil_ellipse_xywht_sensor=[float(v) for v in result["ellipse_xywht"]],
                annotation_source="gsa_auto",
                annotation_quality=float(quality),
                closed_eye_flag=False,
                mask_valid=True,
            )
            row = ann.to_row()
            row["frame_idx"] = rec.frame_idx
            row["session_key"] = session_key
            row["user_id"] = int(session_user_id)
            row["subject_id"] = int(session_user_id)
            row["eye"] = str(session_eye)
            row["session_code"] = str(session_code)
            row["gsam_class_name"] = result["class_name"]
            row["gsam_box_xyxy"] = [float(v) for v in result["box_xyxy"]]
            row["gsam_box_confidence"] = float(result["box_confidence"])
            row["gsam_mask_score"] = float(result["mask_score"])
            rows.append(row)
        return rows

    def process_session(
        self,
        *,
        session_user_id: int,
        session_eye: str,
        raw_session_dir: Path,
    ) -> tuple[dict[str, Any], int] | None:
        layout = discover_session_layout(raw_session_dir, user_id=session_user_id)
        if layout.frames_dir is None:
            return None
        if (not self.cfg.include_nonstandard_sessions) and (not layout.is_official_session):
            return None
        session_code = layout.session_code
        session_key = f"user{session_user_id:02d}/{session_eye}/session_{session_code}"
        session_out = groundedsam_session_dir(
            self.cfg.annotation_root,
            user_id=session_user_id,
            eye=session_eye,
            session_code=session_code,
        )
        store_path = session_out / "frame_annotations.jsonl"
        session_summary_path = session_out / "session_summary.json"

        if store_path.exists() and not self.cfg.overwrite:
            rows = read_jsonl(store_path)
            rows.sort(key=self.annotation_row_sort_key)
            rows, blink_candidate_report = apply_groundedsam_store_blink_metadata(rows, overwrite_existing=False)
            reuse_message = "existing_store_reused"
            if int(blink_candidate_report.get("n_rows_updated") or 0) > 0:
                write_jsonl(rows, store_path)
                reuse_message = "existing_store_reused_blink_backfilled"
            self.summary_writer.write_session_summary(
                session_summary_path,
                self.summary_writer.build_session_summary_payload(
                    session_key=session_key,
                    raw_session_dir=raw_session_dir,
                    store_path=store_path,
                    n_annotations=len(rows),
                    classes=self.cfg.classes,
                    box_threshold=self.cfg.box_threshold,
                    text_threshold=self.cfg.text_threshold,
                    nms_threshold=self.cfg.nms_threshold,
                    frame_step=self.cfg.frame_step,
                    max_frames_per_session=self.cfg.max_frames_per_session,
                    blink_candidate_report=blink_candidate_report,
                ),
            )
            return (
                self.summary_writer.build_session_index_row(
                    session_key=session_key,
                    raw_session_dir=raw_session_dir,
                    store_path=store_path,
                    rows=rows,
                    skipped=True,
                    message=reuse_message,
                    blink_candidate_report=blink_candidate_report,
                ),
                len(rows),
            )

        frame_records = collect_frame_records(layout.frames_dir)
        frame_records = frame_records[:: self.cfg.frame_step]
        if self.cfg.max_frames_per_session is not None:
            frame_records = frame_records[: self.cfg.max_frames_per_session]
        rows = self._annotate_frame_records(
            frame_records=frame_records,
            layout=layout,
            session_key=session_key,
            session_user_id=session_user_id,
            session_eye=session_eye,
            session_code=session_code,
        )
        rows.sort(key=self.annotation_row_sort_key)
        rows, blink_candidate_report = apply_groundedsam_store_blink_metadata(rows, overwrite_existing=True)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, store_path)
        self.summary_writer.write_session_summary(
            session_summary_path,
            self.summary_writer.build_session_summary_payload(
                session_key=session_key,
                raw_session_dir=raw_session_dir,
                store_path=store_path,
                n_annotations=len(rows),
                classes=self.cfg.classes,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                nms_threshold=self.cfg.nms_threshold,
                frame_step=self.cfg.frame_step,
                max_frames_per_session=self.cfg.max_frames_per_session,
                blink_candidate_report=blink_candidate_report,
            ),
        )
        return (
            self.summary_writer.build_session_index_row(
                session_key=session_key,
                raw_session_dir=raw_session_dir,
                store_path=store_path,
                rows=rows,
                skipped=False,
                message="groundedsam_annotation_complete",
                blink_candidate_report=blink_candidate_report,
            ),
            len(rows),
        )


class GroundedSamShardRunner:
    def __init__(
        self,
        *,
        summary_writer: GroundedSamSummaryWriter,
        worker_target: Callable[[dict[str, Any]], None],
    ) -> None:
        self.summary_writer = summary_writer
        self.worker_target = worker_target

    def run(
        self,
        *,
        devices: Sequence[str],
        shared_kwargs: dict[str, Any],
        single_device_runner: Callable[[str | None], dict[str, Any]],
    ) -> dict[str, Any]:
        import multiprocessing as mp

        device_list = list(devices)
        if not device_list:
            return single_device_runner(None)
        if len(device_list) == 1:
            return single_device_runner(device_list[0])

        started_at = time.perf_counter()
        ctx = mp.get_context("spawn")
        processes = []
        for shard_index, device in enumerate(device_list):
            worker_kwargs = dict(
                shared_kwargs,
                device=device,
                num_shards=len(device_list),
                shard_index=shard_index,
            )
            process = ctx.Process(
                target=self.worker_target,
                kwargs={"kwargs": worker_kwargs},
                name=f"groundedsam-shard-{shard_index}",
            )
            process.start()
            processes.append(process)

        failures: list[str] = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append(f"{process.name}(exit={process.exitcode})")
        if failures:
            raise RuntimeError("Grounded-SAM shard workers failed: " + ", ".join(failures))

        return self.summary_writer.aggregate_shards(
            num_shards=len(device_list),
            devices=device_list,
            elapsed_sec=float(time.perf_counter() - started_at),
        )


__all__ = [
    "GroundedSamRuntimeFactory",
    "GroundedSamSessionProcessor",
    "GroundedSamShardRunner",
    "GroundedSamSummaryWriter",
]
