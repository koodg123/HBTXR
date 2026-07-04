from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from src.preprocess.annotation_groundedsam import (
    expand_eye_region_from_bbox,
    mask_bbox_xywh,
    mask_to_ellipse_xywht,
)
from src.preprocess.annotation_backends import (
    DEFAULT_ANNOTATION_BACKEND,
    SUPPORTED_ANNOTATION_BACKENDS,
    build_annotation_runtime,
    resolve_annotation_backend,
    resolve_default_groundedsam2_checkpoint,
    resolve_default_groundedsam2_config,
    resolve_default_groundedsam2_groundingdino_checkpoint,
    resolve_default_groundedsam2_groundingdino_config,
    resolve_default_ultralytics_sam3_checkpoint,
)
from src.preprocess.groundedsam_pipeline import (
    GroundedSamRuntimeFactory,
    GroundedSamSessionProcessor,
    GroundedSamShardRunner,
    GroundedSamSummaryWriter,
)
from src.preprocess.io_utils import collect_users, discover_session_layout


DEFAULT_CLASSES = ("pupil", "iris", "eye")


def _patch_transformers_groundingdino_compat() -> None:
    """Restore legacy BERT helper APIs expected by GroundingDINO."""

    import torch  # noqa: WPS433
    from transformers.modeling_utils import ModuleUtilsMixin, PreTrainedModel  # noqa: WPS433

    if hasattr(PreTrainedModel, "get_head_mask"):
        needs_head_mask_patch = False
    else:
        needs_head_mask_patch = True

    current_get_extended_attention_mask = ModuleUtilsMixin.get_extended_attention_mask
    if getattr(current_get_extended_attention_mask, "_hbtxr_groundingdino_compat", False):
        needs_extended_mask_patch = False
    else:
        needs_extended_mask_patch = True

    if needs_head_mask_patch:
        def _get_head_mask(
            self: Any,
            head_mask: Any | None,
            num_hidden_layers: int,
            is_attention_chunked: bool = False,
        ) -> Any:
            if head_mask is None:
                return [None] * int(num_hidden_layers)

            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(int(num_hidden_layers), -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                raise ValueError(
                    "head_mask must have dim 1 or 2; "
                    f"got dim={head_mask.dim()} for GroundingDINO compatibility.",
                )

            head_mask = head_mask.to(dtype=self.dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        setattr(PreTrainedModel, "get_head_mask", _get_head_mask)

    if needs_extended_mask_patch:
        def _get_extended_attention_mask_compat(
            self: Any,
            attention_mask: Any,
            input_shape: tuple[int, ...],
            dtype: Any | None = None,
        ) -> Any:
            # Older GroundingDINO passes `device` as the third positional argument.
            if isinstance(dtype, (torch.device, str)):
                dtype = None
            return current_get_extended_attention_mask(
                self,
                attention_mask,
                input_shape,
                dtype=dtype,
            )

        setattr(_get_extended_attention_mask_compat, "_hbtxr_groundingdino_compat", True)
        setattr(ModuleUtilsMixin, "get_extended_attention_mask", _get_extended_attention_mask_compat)


def _patch_groundingdino_ms_deform_attn_compat() -> None:
    """Fallback to the PyTorch deformable attention path when the C++ op is unavailable."""

    import torch  # noqa: WPS433
    from groundingdino.models.GroundingDINO import ms_deform_attn as msda  # noqa: WPS433

    current_forward = msda.MultiScaleDeformableAttention.forward
    if getattr(current_forward, "_hbtxr_groundingdino_compat", False):
        return

    def _forward_compat(
        self: Any,
        query: torch.Tensor,
        key: Any = None,
        value: Any = None,
        query_pos: Any = None,
        key_padding_mask: Any = None,
        reference_points: Any = None,
        spatial_shapes: Any = None,
        level_start_index: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if value is None:
            value = query

        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, "
                f"but got {reference_points.shape[-1]} instead.",
            )

        use_custom_op = (
            value.is_cuda
            and torch.cuda.is_available()
            and getattr(msda, "_C", None) is not None
        )
        if use_custom_op:
            halffloat = False
            if value.dtype == torch.float16:
                halffloat = True
                value = value.float()
                sampling_locations = sampling_locations.float()
                attention_weights = attention_weights.float()

            output = msda.MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )

            if halffloat:
                output = output.half()
        else:
            output = msda.multi_scale_deformable_attn_pytorch(
                value,
                spatial_shapes,
                sampling_locations,
                attention_weights,
            )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    setattr(_forward_compat, "_hbtxr_groundingdino_compat", True)
    setattr(msda.MultiScaleDeformableAttention, "forward", _forward_compat)


@dataclass
class GroundedSamBuildConfig:
    raw_root: Path
    annotation_root: Path
    annotation_backend: str = DEFAULT_ANNOTATION_BACKEND
    groundedsam_root: Path | None = None
    ultralytics_root: Path | None = None
    groundedsam2_root: Path | None = None
    groundingdino_config: Path | None = None
    groundingdino_checkpoint: Path | None = None
    sam_checkpoint: Path | None = None
    ultralytics_sam3_checkpoint: Path | None = None
    groundedsam2_config: Path | None = None
    groundedsam2_checkpoint: Path | None = None
    groundedsam2_groundingdino_config: Path | None = None
    groundedsam2_groundingdino_checkpoint: Path | None = None
    sam_encoder_version: str = "vit_h"
    classes: tuple[str, ...] = DEFAULT_CLASSES
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    nms_threshold: float = 0.8
    min_mask_area: int = 16
    frame_step: int = 1
    max_frames_per_session: int | None = None
    max_sessions: int | None = None
    user_id: int | None = None
    eye: str | None = None
    session_codes: tuple[str, ...] | None = None
    include_nonstandard_sessions: bool = False
    num_shards: int = 1
    shard_index: int = 0
    device: str | None = None
    overwrite: bool = False


def parse_groundedsam_devices(device_text: str | None) -> list[str]:
    requested_text = str(device_text).strip() if device_text is not None else ""
    if not requested_text:
        return []

    devices: list[str] = []
    seen: set[str] = set()
    for part in [item.strip() for item in requested_text.split(",") if item.strip()]:
        normalized = f"cuda:{part}" if part.isdigit() else part
        if normalized in seen:
            raise ValueError(f"Duplicate Grounded-SAM device entry: '{normalized}'")
        seen.add(normalized)
        devices.append(normalized)
    return devices


def _validate_shard_spec(num_shards: int | None, shard_index: int | None) -> tuple[int, int]:
    resolved_num_shards = 1 if num_shards is None else int(num_shards)
    resolved_shard_index = 0 if shard_index is None else int(shard_index)
    if resolved_num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {resolved_num_shards}")
    if resolved_shard_index < 0 or resolved_shard_index >= resolved_num_shards:
        raise ValueError(
            f"shard_index must be in [0, {resolved_num_shards}), got {resolved_shard_index}",
        )
    return resolved_num_shards, resolved_shard_index


def _session_belongs_to_shard(session_index: int, *, num_shards: int, shard_index: int) -> bool:
    resolved_num_shards, resolved_shard_index = _validate_shard_spec(num_shards, shard_index)
    return int(session_index) % resolved_num_shards == resolved_shard_index


def _normalize_groundedsam_eye_filter(eye: str | None) -> tuple[str, ...]:
    text = str(eye or "both").strip().lower()
    if text in {"both", "all", "*"}:
        return ("left", "right")
    if text in {"left", "right"}:
        return (text,)
    raise ValueError(f"Unsupported Grounded-SAM eye filter: {eye!r}")


def _normalize_groundedsam_session_filters(session_codes: Iterable[str] | None) -> set[str]:
    normalized: set[str] = set()
    for session_code in session_codes or []:
        text = str(session_code).strip().lower()
        if not text:
            raise ValueError("session_code must not be empty")
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) != 3:
            raise ValueError(f"Unsupported Grounded-SAM session_code filter: {session_code!r}")
        normalized.add(digits)
    return normalized


def _resolve_runtime_device(torch: Any, device_text: str | None) -> tuple[Any, str | None]:
    requested_text = str(device_text).strip() if device_text is not None else ""
    if not requested_text:
        requested_text = "cuda" if torch.cuda.is_available() else "cpu"

    requested_devices = parse_groundedsam_devices(requested_text)
    normalized_text = requested_text
    warning_message = None

    if len(requested_devices) > 1:
        normalized_text = requested_devices[0]
        warning_message = (
            "Each Grounded-SAM worker currently uses a single torch device. "
            f"Received '{requested_text}' and will use '{normalized_text}'."
        )
    elif len(requested_devices) == 1:
        normalized_text = requested_devices[0]

    try:
        return torch.device(normalized_text), warning_message
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported Grounded-SAM device value '{requested_text}'. "
            "Pass a single torch device such as 'cpu', 'cuda', or 'cuda:0'. "
            "Comma-separated lists are accepted only to select the first entry.",
        ) from exc


def _annotate_groundedsam_worker(kwargs: dict[str, Any]) -> None:
    annotate_dataset_with_groundedsam(**kwargs)


def annotate_dataset_with_groundedsam_multi(
    *,
    devices: Sequence[str],
    raw_root: str | Path,
    annotation_root: str | Path,
    annotation_backend: str = DEFAULT_ANNOTATION_BACKEND,
    groundedsam_root: str | Path | None = None,
    ultralytics_root: str | Path | None = None,
    groundedsam2_root: str | Path | None = None,
    groundingdino_config: str | Path | None = None,
    groundingdino_checkpoint: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    ultralytics_sam3_checkpoint: str | Path | None = None,
    groundedsam2_config: str | Path | None = None,
    groundedsam2_checkpoint: str | Path | None = None,
    groundedsam2_groundingdino_config: str | Path | None = None,
    groundedsam2_groundingdino_checkpoint: str | Path | None = None,
    sam_encoder_version: str = "vit_h",
    classes: Iterable[str] = DEFAULT_CLASSES,
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
    overwrite: bool = False,
) -> dict[str, Any]:
    device_list = list(devices)
    shared_kwargs = {
        "raw_root": raw_root,
        "annotation_root": annotation_root,
        "annotation_backend": annotation_backend,
        "groundedsam_root": groundedsam_root,
        "ultralytics_root": ultralytics_root,
        "groundedsam2_root": groundedsam2_root,
        "groundingdino_config": groundingdino_config,
        "groundingdino_checkpoint": groundingdino_checkpoint,
        "sam_checkpoint": sam_checkpoint,
        "ultralytics_sam3_checkpoint": ultralytics_sam3_checkpoint,
        "groundedsam2_config": groundedsam2_config,
        "groundedsam2_checkpoint": groundedsam2_checkpoint,
        "groundedsam2_groundingdino_config": groundedsam2_groundingdino_config,
        "groundedsam2_groundingdino_checkpoint": groundedsam2_groundingdino_checkpoint,
        "sam_encoder_version": sam_encoder_version,
        "classes": tuple(classes),
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "nms_threshold": nms_threshold,
        "min_mask_area": min_mask_area,
        "frame_step": frame_step,
        "max_frames_per_session": max_frames_per_session,
        "max_sessions": max_sessions,
        "user_id": user_id,
        "eye": eye,
        "session_codes": None if session_codes is None else tuple(session_codes),
        "include_nonstandard_sessions": include_nonstandard_sessions,
        "overwrite": overwrite,
    }
    runner = GroundedSamShardRunner(
        summary_writer=GroundedSamSummaryWriter(annotation_root),
        worker_target=_annotate_groundedsam_worker,
    )
    return runner.run(
        devices=device_list,
        shared_kwargs=shared_kwargs,
        single_device_runner=lambda device_text: annotate_dataset_with_groundedsam(
            raw_root=raw_root,
            annotation_root=annotation_root,
            annotation_backend=annotation_backend,
            groundedsam_root=groundedsam_root,
            ultralytics_root=ultralytics_root,
            groundedsam2_root=groundedsam2_root,
            groundingdino_config=groundingdino_config,
            groundingdino_checkpoint=groundingdino_checkpoint,
            sam_checkpoint=sam_checkpoint,
            ultralytics_sam3_checkpoint=ultralytics_sam3_checkpoint,
            groundedsam2_config=groundedsam2_config,
            groundedsam2_checkpoint=groundedsam2_checkpoint,
            groundedsam2_groundingdino_config=groundedsam2_groundingdino_config,
            groundedsam2_groundingdino_checkpoint=groundedsam2_groundingdino_checkpoint,
            sam_encoder_version=sam_encoder_version,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            nms_threshold=nms_threshold,
            min_mask_area=min_mask_area,
            frame_step=frame_step,
            max_frames_per_session=max_frames_per_session,
            max_sessions=max_sessions,
            user_id=user_id,
            eye=eye,
            session_codes=session_codes,
            include_nonstandard_sessions=include_nonstandard_sessions,
            device=device_text,
            overwrite=overwrite,
        ),
    )


class _GroundedSamRuntime:
    def __init__(self, cfg: GroundedSamBuildConfig) -> None:
        self.cfg = cfg
        repo_root = cfg.groundedsam_root.resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Grounded-SAM root not found: {repo_root}")
        if not cfg.groundingdino_config.exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {cfg.groundingdino_config}")
        if not cfg.groundingdino_checkpoint.exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {cfg.groundingdino_checkpoint}")
        if not cfg.sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {cfg.sam_checkpoint}")

        # Grounded-SAM forks are not layout-stable. Some keep the actual SAM package at:
        #   <repo>/segment_anything/segment_anything/__init__.py
        # In that case, adding only <repo> to sys.path resolves `segment_anything` as a
        # namespace package and `from segment_anything import SamPredictor` fails.
        sam_pkg_root = repo_root / "segment_anything"
        extra_paths = [repo_root, repo_root / "GroundingDINO"]
        if (sam_pkg_root / "segment_anything" / "__init__.py").exists():
            extra_paths.append(sam_pkg_root)

        for extra in extra_paths:
            text = str(extra)
            if text not in sys.path:
                sys.path.insert(0, text)

        import torch  # noqa: WPS433
        import torchvision  # noqa: WPS433
        _patch_transformers_groundingdino_compat()
        _patch_groundingdino_ms_deform_attn_compat()
        from groundingdino.util.inference import Model  # noqa: WPS433
        from segment_anything import SamPredictor, sam_hq_model_registry, sam_model_registry  # noqa: WPS433

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
        registry = sam_hq_model_registry if _use_sam_hq_checkpoint(cfg.sam_checkpoint) else sam_model_registry
        sam = registry[str(cfg.sam_encoder_version)](checkpoint=str(cfg.sam_checkpoint))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def _segment(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, float]:
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(box=box_xyxy.astype(np.float32), multimask_output=True)
        index = int(np.argmax(scores))
        return masks[index].astype(np.uint8), float(scores[index])

    def _pick_detection(self, detections: Any, image_shape: tuple[int, int, int]) -> int | None:
        xyxy = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)))
        conf = np.asarray(getattr(detections, "confidence", np.empty((0,), dtype=np.float32)))
        class_id = np.asarray(getattr(detections, "class_id", np.empty((0,), dtype=np.int64)))
        if xyxy.size == 0:
            return None
        keep = self._torchvision.ops.nms(
            self._torch.from_numpy(xyxy),
            self._torch.from_numpy(conf.astype(np.float32)),
            float(self.cfg.nms_threshold),
        ).cpu().numpy().tolist()
        priority_map = {name: idx for idx, name in enumerate(self.cfg.classes)}
        image_area = float(max(1, image_shape[0] * image_shape[1]))
        best_key = None
        best_idx = None
        for idx in keep:
            raw_class_id = int(class_id[idx]) if idx < class_id.shape[0] else 0
            class_name = self.cfg.classes[raw_class_id] if 0 <= raw_class_id < len(self.cfg.classes) else "unknown"
            area = max(0.0, float((xyxy[idx][2] - xyxy[idx][0]) * (xyxy[idx][3] - xyxy[idx][1]))) / image_area
            key = (priority_map.get(class_name, 999), -float(conf[idx]), area)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = int(idx)
        return best_idx

    def annotate_image(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        image_bgr = image_rgb[:, :, ::-1].copy()
        detections = self.model.predict_with_classes(
            image=image_bgr,
            classes=list(self.cfg.classes),
            box_threshold=float(self.cfg.box_threshold),
            text_threshold=float(self.cfg.text_threshold),
        )
        idx = self._pick_detection(detections, image_rgb.shape)
        if idx is None:
            return None
        xyxy = np.asarray(detections.xyxy[idx], dtype=np.float32)
        confidence = float(detections.confidence[idx])
        class_idx = int(detections.class_id[idx]) if idx < len(detections.class_id) else 0
        class_name = self.cfg.classes[class_idx] if 0 <= class_idx < len(self.cfg.classes) else "unknown"
        mask, mask_score = self._segment(image_rgb, xyxy)
        if int(mask.sum()) < int(self.cfg.min_mask_area):
            return None
        bbox = mask_bbox_xywh(mask)
        ellipse = mask_to_ellipse_xywht(mask)
        if bbox is None or ellipse is None:
            return None
        eye_region = expand_eye_region_from_bbox(
            bbox,
            image_size_wh=(image_rgb.shape[1], image_rgb.shape[0]),
        )
        return {
            "mask": mask,
            "bbox_xywh": bbox,
            "ellipse_xywht": ellipse,
            "eye_region_xywh": eye_region,
            "class_name": class_name,
            "box_xyxy": [float(v) for v in xyxy.tolist()],
            "box_confidence": confidence,
            "mask_score": mask_score,
        }


def _iter_raw_sessions(raw_root: Path) -> Iterable[tuple[int, str, Path]]:
    for user_dir in collect_users(raw_root):
        try:
            user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        for eye in ("left", "right"):
            eye_dir = user_dir / eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted(eye_dir.glob("session_*_*_*")):
                yield user_id, eye, session_dir


def _resolve_default_groundingdino_config(repo_root: Path) -> Path:
    return GroundedSamRuntimeFactory.resolve_default_groundingdino_config(Path(repo_root).resolve())


def _resolve_default_groundingdino_checkpoint(repo_root: Path) -> Path:
    return GroundedSamRuntimeFactory.resolve_default_groundingdino_checkpoint(Path(repo_root).resolve())


def _resolve_default_sam_checkpoint(repo_root: Path) -> Path:
    return GroundedSamRuntimeFactory.resolve_default_sam_checkpoint(Path(repo_root).resolve())


def _use_sam_hq_checkpoint(checkpoint_path: str | Path) -> bool:
    stem = Path(checkpoint_path).stem.strip().lower()
    return "sam_hq" in stem or stem.startswith("hqsam")


def _build_annotation_cfg(
    *,
    raw_root: str | Path,
    annotation_root: str | Path,
    annotation_backend: str,
    groundedsam_root: str | Path | None,
    ultralytics_root: str | Path | None,
    groundedsam2_root: str | Path | None,
    groundingdino_config: str | Path | None,
    groundingdino_checkpoint: str | Path | None,
    sam_checkpoint: str | Path | None,
    ultralytics_sam3_checkpoint: str | Path | None,
    groundedsam2_config: str | Path | None,
    groundedsam2_checkpoint: str | Path | None,
    groundedsam2_groundingdino_config: str | Path | None,
    groundedsam2_groundingdino_checkpoint: str | Path | None,
    sam_encoder_version: str,
    classes: Iterable[str],
    box_threshold: float,
    text_threshold: float,
    nms_threshold: float,
    min_mask_area: int,
    frame_step: int,
    max_frames_per_session: int | None,
    max_sessions: int | None,
    user_id: int | None,
    eye: str | None,
    session_codes: Iterable[str] | None,
    include_nonstandard_sessions: bool,
    num_shards: int,
    shard_index: int,
    device: str | None,
    overwrite: bool,
) -> GroundedSamBuildConfig:
    resolved_backend = resolve_annotation_backend(annotation_backend)
    raw_root_path = Path(raw_root).resolve()
    annotation_root_path = Path(annotation_root).resolve()
    resolved_num_shards, resolved_shard_index = _validate_shard_spec(num_shards, shard_index)
    resolved_groundedsam_root = None if groundedsam_root is None else Path(groundedsam_root).resolve()
    resolved_ultralytics_root = None if ultralytics_root is None else Path(ultralytics_root).resolve()
    resolved_groundedsam2_root = None if groundedsam2_root is None else Path(groundedsam2_root).resolve()

    cfg_kwargs: dict[str, Any] = {
        "raw_root": raw_root_path,
        "annotation_root": annotation_root_path,
        "annotation_backend": resolved_backend,
        "groundedsam_root": resolved_groundedsam_root,
        "ultralytics_root": resolved_ultralytics_root,
        "groundedsam2_root": resolved_groundedsam2_root,
        "sam_encoder_version": str(sam_encoder_version),
        "classes": tuple(str(item).strip() for item in classes if str(item).strip()) or DEFAULT_CLASSES,
        "box_threshold": float(box_threshold),
        "text_threshold": float(text_threshold),
        "nms_threshold": float(nms_threshold),
        "min_mask_area": int(min_mask_area),
        "frame_step": max(1, int(frame_step)),
        "max_frames_per_session": None if max_frames_per_session is None else max(1, int(max_frames_per_session)),
        "max_sessions": None if max_sessions is None else max(1, int(max_sessions)),
        "user_id": None if user_id is None else int(user_id),
        "eye": None if eye is None else str(eye),
        "session_codes": None if session_codes is None else tuple(str(code) for code in session_codes),
        "include_nonstandard_sessions": bool(include_nonstandard_sessions),
        "num_shards": int(resolved_num_shards),
        "shard_index": int(resolved_shard_index),
        "device": device,
        "overwrite": bool(overwrite),
    }

    if resolved_backend == "groundedsam":
        if resolved_groundedsam_root is None:
            raise ValueError("groundedsam_root is required for annotation_backend='groundedsam'")
        cfg_kwargs.update(
            {
                "groundingdino_config": (
                    Path(groundingdino_config).resolve()
                    if groundingdino_config
                    else _resolve_default_groundingdino_config(resolved_groundedsam_root)
                ),
                "groundingdino_checkpoint": (
                    Path(groundingdino_checkpoint).resolve()
                    if groundingdino_checkpoint
                    else _resolve_default_groundingdino_checkpoint(resolved_groundedsam_root)
                ),
                "sam_checkpoint": (
                    Path(sam_checkpoint).resolve()
                    if sam_checkpoint
                    else _resolve_default_sam_checkpoint(resolved_groundedsam_root)
                ),
            }
        )
    elif resolved_backend == "ultralytics_sam3":
        if resolved_ultralytics_root is None:
            raise ValueError("ultralytics_root is required for annotation_backend='ultralytics_sam3'")
        cfg_kwargs["ultralytics_sam3_checkpoint"] = (
            Path(ultralytics_sam3_checkpoint).resolve()
            if ultralytics_sam3_checkpoint
            else resolve_default_ultralytics_sam3_checkpoint(resolved_ultralytics_root)
        )
    else:
        if resolved_groundedsam2_root is None:
            raise ValueError("groundedsam2_root is required for annotation_backend='groundedsam2'")
        cfg_kwargs.update(
            {
                "groundedsam2_config": (
                    Path(groundedsam2_config).resolve()
                    if groundedsam2_config
                    else resolve_default_groundedsam2_config(resolved_groundedsam2_root)
                ),
                "groundedsam2_checkpoint": (
                    Path(groundedsam2_checkpoint).resolve()
                    if groundedsam2_checkpoint
                    else resolve_default_groundedsam2_checkpoint(resolved_groundedsam2_root)
                ),
                "groundedsam2_groundingdino_config": (
                    Path(groundedsam2_groundingdino_config).resolve()
                    if groundedsam2_groundingdino_config
                    else resolve_default_groundedsam2_groundingdino_config(resolved_groundedsam2_root)
                ),
                "groundedsam2_groundingdino_checkpoint": (
                    Path(groundedsam2_groundingdino_checkpoint).resolve()
                    if groundedsam2_groundingdino_checkpoint
                    else resolve_default_groundedsam2_groundingdino_checkpoint(resolved_groundedsam2_root)
                ),
            }
        )
    return GroundedSamBuildConfig(**cfg_kwargs)


def build_groundedsam_runtime(
    *,
    groundedsam_root: str | Path,
    groundingdino_config: str | Path | None = None,
    groundingdino_checkpoint: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    sam_encoder_version: str = "vit_h",
    classes: Iterable[str] = DEFAULT_CLASSES,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    nms_threshold: float = 0.8,
    min_mask_area: int = 16,
    device: str | None = None,
) -> _GroundedSamRuntime:
    factory = GroundedSamRuntimeFactory(
        config_cls=GroundedSamBuildConfig,
        runtime_cls=_GroundedSamRuntime,
        default_classes=DEFAULT_CLASSES,
        validate_shard_spec=_validate_shard_spec,
    )
    return factory.build_runtime(
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


def annotate_dataset_with_groundedsam(
    *,
    raw_root: str | Path,
    annotation_root: str | Path,
    annotation_backend: str = DEFAULT_ANNOTATION_BACKEND,
    groundedsam_root: str | Path | None = None,
    ultralytics_root: str | Path | None = None,
    groundedsam2_root: str | Path | None = None,
    groundingdino_config: str | Path | None = None,
    groundingdino_checkpoint: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    ultralytics_sam3_checkpoint: str | Path | None = None,
    groundedsam2_config: str | Path | None = None,
    groundedsam2_checkpoint: str | Path | None = None,
    groundedsam2_groundingdino_config: str | Path | None = None,
    groundedsam2_groundingdino_checkpoint: str | Path | None = None,
    sam_encoder_version: str = "vit_h",
    classes: Iterable[str] = DEFAULT_CLASSES,
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
) -> dict[str, Any]:
    cfg = _build_annotation_cfg(
        raw_root=raw_root,
        annotation_root=annotation_root,
        annotation_backend=annotation_backend,
        groundedsam_root=groundedsam_root,
        ultralytics_root=ultralytics_root,
        groundedsam2_root=groundedsam2_root,
        groundingdino_config=groundingdino_config,
        groundingdino_checkpoint=groundingdino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        ultralytics_sam3_checkpoint=ultralytics_sam3_checkpoint,
        groundedsam2_config=groundedsam2_config,
        groundedsam2_checkpoint=groundedsam2_checkpoint,
        groundedsam2_groundingdino_config=groundedsam2_groundingdino_config,
        groundedsam2_groundingdino_checkpoint=groundedsam2_groundingdino_checkpoint,
        sam_encoder_version=sam_encoder_version,
        classes=classes,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        min_mask_area=min_mask_area,
        frame_step=frame_step,
        max_frames_per_session=max_frames_per_session,
        max_sessions=max_sessions,
        user_id=user_id,
        eye=eye,
        session_codes=session_codes,
        include_nonstandard_sessions=include_nonstandard_sessions,
        num_shards=num_shards,
        shard_index=shard_index,
        device=device,
        overwrite=overwrite,
    )
    runtime = build_annotation_runtime(
        annotation_backend=cfg.annotation_backend,
        groundedsam_root=cfg.groundedsam_root,
        groundingdino_config=cfg.groundingdino_config,
        groundingdino_checkpoint=cfg.groundingdino_checkpoint,
        sam_checkpoint=cfg.sam_checkpoint,
        ultralytics_root=cfg.ultralytics_root,
        groundedsam2_root=cfg.groundedsam2_root,
        ultralytics_sam3_checkpoint=cfg.ultralytics_sam3_checkpoint,
        groundedsam2_config=cfg.groundedsam2_config,
        groundedsam2_checkpoint=cfg.groundedsam2_checkpoint,
        groundedsam2_groundingdino_config=cfg.groundedsam2_groundingdino_config,
        groundedsam2_groundingdino_checkpoint=cfg.groundedsam2_groundingdino_checkpoint,
        sam_encoder_version=cfg.sam_encoder_version,
        classes=cfg.classes,
        box_threshold=cfg.box_threshold,
        text_threshold=cfg.text_threshold,
        nms_threshold=cfg.nms_threshold,
        min_mask_area=cfg.min_mask_area,
        device=cfg.device,
    )
    resolved_user_id = cfg.user_id
    allowed_eyes = _normalize_groundedsam_eye_filter(cfg.eye)
    session_filter = _normalize_groundedsam_session_filters(cfg.session_codes)
    started_at = time.perf_counter()
    session_summaries: list[dict[str, Any]] = []
    total_rows = 0
    summary_writer = GroundedSamSummaryWriter(cfg.annotation_root)
    session_processor = GroundedSamSessionProcessor(
        cfg=cfg,
        runtime=runtime,
        summary_writer=summary_writer,
    )

    for session_index, (session_user_id, session_eye, raw_session_dir) in enumerate(_iter_raw_sessions(cfg.raw_root)):
        if cfg.max_sessions is not None and session_index >= cfg.max_sessions:
            break
        if not _session_belongs_to_shard(session_index, num_shards=cfg.num_shards, shard_index=cfg.shard_index):
            continue
        if resolved_user_id is not None and session_user_id != resolved_user_id:
            continue
        if session_eye not in allowed_eyes:
            continue
        layout = discover_session_layout(raw_session_dir, user_id=session_user_id)
        if layout.frames_dir is None:
            continue
        if (not cfg.include_nonstandard_sessions) and (not layout.is_official_session):
            continue
        session_code = layout.session_code
        if session_filter and session_code not in session_filter:
            continue
        processed = session_processor.process_session(
            session_user_id=session_user_id,
            session_eye=session_eye,
            raw_session_dir=raw_session_dir,
        )
        if processed is None:
            continue
        session_summary_row, n_rows = processed
        total_rows += int(n_rows)
        session_summaries.append(session_summary_row)

    summary = {
        "raw_root": str(cfg.raw_root),
        "annotation_root": str(cfg.annotation_root),
        "annotation_backend": str(cfg.annotation_backend),
        "groundedsam_root": None if cfg.groundedsam_root is None else str(cfg.groundedsam_root),
        "ultralytics_root": None if cfg.ultralytics_root is None else str(cfg.ultralytics_root),
        "groundedsam2_root": None if cfg.groundedsam2_root is None else str(cfg.groundedsam2_root),
        "groundingdino_config": None if cfg.groundingdino_config is None else str(cfg.groundingdino_config),
        "groundingdino_checkpoint": None if cfg.groundingdino_checkpoint is None else str(cfg.groundingdino_checkpoint),
        "sam_checkpoint": None if cfg.sam_checkpoint is None else str(cfg.sam_checkpoint),
        "ultralytics_sam3_checkpoint": None if cfg.ultralytics_sam3_checkpoint is None else str(cfg.ultralytics_sam3_checkpoint),
        "groundedsam2_config": None if cfg.groundedsam2_config is None else str(cfg.groundedsam2_config),
        "groundedsam2_checkpoint": None if cfg.groundedsam2_checkpoint is None else str(cfg.groundedsam2_checkpoint),
        "groundedsam2_groundingdino_config": None if cfg.groundedsam2_groundingdino_config is None else str(cfg.groundedsam2_groundingdino_config),
        "groundedsam2_groundingdino_checkpoint": None if cfg.groundedsam2_groundingdino_checkpoint is None else str(cfg.groundedsam2_groundingdino_checkpoint),
        "classes": list(cfg.classes),
        "device": str(getattr(runtime, "device", cfg.device)),
        "devices": [str(getattr(runtime, "device", cfg.device))],
        "num_shards": int(cfg.num_shards),
        "shard_index": int(cfg.shard_index),
        "sharded": bool(cfg.num_shards > 1),
        "n_sessions": len(session_summaries),
        "n_annotations": int(total_rows),
        "elapsed_sec": float(time.perf_counter() - started_at),
        "user_id_filter": resolved_user_id,
        "eye_filter": list(allowed_eyes),
        "session_code_filter": sorted(session_filter),
        "include_nonstandard_sessions": bool(cfg.include_nonstandard_sessions),
    }
    summary_writer.write_shard_outputs(
        summary=summary,
        session_summaries=session_summaries,
        num_shards=cfg.num_shards,
        shard_index=cfg.shard_index,
    )
    return summary
