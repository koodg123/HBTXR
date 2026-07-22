from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

_HEAD_KEYS = ("eye", "search", "event", "track", "mask", "aux")
_HEAD_LOSS_KEYS = {
    "eye": (
        "eye_weight",
        "eye_conf_weight",
        "eye_box_iou_weight",
        "eye_box_l1_weight",
        "eye_point_cls_weight",
        "eye_point_box_weight",
        "eye_point_dfl_weight",
        "eye_center_focal_weight",
        "eye_center_box_weight",
        "eye_corner_focal_weight",
        "eye_corner_box_weight",
    ),
    "search": (
        "search_xy_weight",
        "search_ab_weight",
        "search_trig_weight",
        "search_geo_weight",
        "search_conf_weight",
        "search_bbox_aux_weight",
        "search_bbox_aux_conf_weight",
        "search_obb_aux_weight",
        "search_obb_aux_angle_weight",
        "search_obb_aux_conf_weight",
    ),
    "event": (
        "event_xy_weight",
        "event_ab_weight",
        "event_trig_weight",
        "event_geo_weight",
        "event_conf_weight",
        "event_bbox_aux_weight",
        "event_bbox_aux_conf_weight",
        "event_obb_aux_weight",
        "event_obb_aux_angle_weight",
        "event_obb_aux_conf_weight",
    ),
    "track": ("track_xy_weight", "track_ab_weight", "track_trig_weight", "track_geo_weight", "track_conf_weight", "track_quality_weight"),
    "mask": ("mask_weight",),
    "aux": ("aux_weight",),
}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, parsed)


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(1e-6, parsed)


def _clamp_width(value: Any, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(1e-6, min(1.0, parsed))


def _largest_valid_num_heads(embed_dim: int, preferred: int, fallback: int) -> int:
    preferred = _positive_int(preferred, fallback)
    if embed_dim % preferred == 0:
        return preferred
    for candidate in range(preferred, 0, -1):
        if embed_dim % candidate == 0:
            return candidate
    fallback = _positive_int(fallback, 1)
    if embed_dim % fallback == 0:
        return fallback
    for candidate in range(min(embed_dim, fallback), 0, -1):
        if embed_dim % candidate == 0:
            return candidate
    return 1


def _normalize_head_selection(normalized: dict[str, Any]) -> None:
    model_cfg = normalized.setdefault("model", {})
    heads_cfg = model_cfg.setdefault("heads", {})
    loss_cfg = normalized.setdefault("loss", {})

    heads_cfg.setdefault("active", "all")
    heads_cfg.setdefault("eye_variant", "legacy")
    heads_cfg.setdefault("eye_reg_max", 1)
    heads_cfg.setdefault("search_bbox_aux", False)
    heads_cfg.setdefault("event_bbox_aux", False)
    heads_cfg.setdefault("search_obb_aux", False)
    heads_cfg.setdefault("event_obb_aux", False)
    for key in _HEAD_KEYS:
        heads_cfg.setdefault(key, True)

    active = str(heads_cfg.get("active", "all")).strip().lower()
    if active in {"", "all", "default"}:
        if not bool(heads_cfg.get("search", True)):
            heads_cfg["search_bbox_aux"] = False
            heads_cfg["search_obb_aux"] = False
        if not bool(heads_cfg.get("event", True)):
            heads_cfg["event_bbox_aux"] = False
            heads_cfg["event_obb_aux"] = False
        return
    if active not in _HEAD_KEYS:
        raise ValueError(
            "Unsupported model.heads.active value: "
            f"{active!r}. Expected one of: all, eye, search, event, track, mask, aux."
        )

    for key in _HEAD_KEYS:
        heads_cfg[key] = key == active
    heads_cfg["search_bbox_aux"] = active == "search"
    heads_cfg["search_obb_aux"] = active == "search"
    heads_cfg["event_bbox_aux"] = active == "event"
    heads_cfg["event_obb_aux"] = active == "event"

    for head_name, loss_keys in _HEAD_LOSS_KEYS.items():
        if head_name == active:
            continue
        for loss_key in loss_keys:
            loss_cfg[loss_key] = 0.0

    if active not in {"search", "track"}:
        loss_cfg["constraint_center_weight"] = 0.0
    if active != "all":
        loss_cfg["consistency_weight"] = 0.0


@dataclass
class StudentModelSpec:
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    patch_size: int
    adapter_hidden_dim: int
    prev_state_hidden_dim: int
    head_hidden_dim: int
    track_head_hidden_dim: int
    mask_hidden_dim: int
    structural_width_ratio: float

    def as_model_kwargs(self) -> dict[str, Any]:
        return asdict(self)


def normalize_compression_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(cfg)
    model_cfg = normalized.setdefault("model", {})
    heads_cfg = model_cfg.setdefault("heads", {})
    pruning_cfg = normalized.setdefault("pruning", {})
    old_ssl_cfg = normalized.get("ssl") or {}

    distillation_cfg = normalized.setdefault("distillation", {})
    regularization_ssl_cfg = normalized.setdefault("regularization_ssl", {})

    old_ssl_enabled = bool(old_ssl_cfg.get("enabled", False))
    old_teacher_student = bool(old_ssl_cfg.get("teacher_student", True))

    if "enabled" not in distillation_cfg:
        distillation_cfg["enabled"] = old_ssl_enabled and old_teacher_student
    distillation_cfg.setdefault("teacher_student", True)
    distillation_cfg.setdefault("ema_decay", float(old_ssl_cfg.get("ema_decay", 0.996)))
    distillation_cfg.setdefault("feature_similarity", bool(old_ssl_cfg.get("feature_similarity", True)))
    distillation_cfg.setdefault("feature_weight", float(old_ssl_cfg.get("feature_weight", 0.1)))
    distillation_cfg.setdefault("state_similarity", bool(old_ssl_cfg.get("state_similarity", True)))
    distillation_cfg.setdefault("state_weight", float(old_ssl_cfg.get("state_weight", 0.1)))
    distillation_cfg.setdefault("prediction_similarity", bool(old_ssl_cfg.get("prediction_similarity", True)))
    distillation_cfg.setdefault("prediction_weight", float(old_ssl_cfg.get("prediction_weight", 0.05)))
    distillation_cfg.setdefault("mask_similarity", True)
    distillation_cfg.setdefault("mask_weight", 0.05)
    distillation_cfg.setdefault("teacher_init_checkpoint", None)
    distillation_cfg.setdefault("force_teacher_from_student", True)
    distillation_cfg.setdefault("kd", deepcopy(old_ssl_cfg.get("kd") or {"enabled": False, "temperature": 1.0, "weight": 0.05}))
    distillation_cfg.setdefault("rkd", deepcopy(old_ssl_cfg.get("rkd") or {"enabled": False, "distance_weight": 0.05}))

    if "enabled" not in regularization_ssl_cfg:
        regularization_ssl_cfg["enabled"] = old_ssl_enabled and not old_teacher_student
    regularization_ssl_cfg.setdefault("force_with_teacher", False)
    regularization_ssl_cfg.setdefault("feature_similarity", bool(old_ssl_cfg.get("feature_similarity", False)))
    regularization_ssl_cfg.setdefault("feature_weight", float(old_ssl_cfg.get("feature_weight", 0.1)))
    regularization_ssl_cfg.setdefault("state_similarity", bool(old_ssl_cfg.get("state_similarity", True)))
    regularization_ssl_cfg.setdefault("state_weight", float(old_ssl_cfg.get("state_weight", 0.1)))
    regularization_ssl_cfg.setdefault("prediction_similarity", bool(old_ssl_cfg.get("prediction_similarity", False)))
    regularization_ssl_cfg.setdefault("prediction_weight", float(old_ssl_cfg.get("prediction_weight", 0.05)))
    regularization_ssl_cfg.setdefault("cross_modal_contrastive", bool(old_ssl_cfg.get("cross_modal_contrastive", True)))
    regularization_ssl_cfg.setdefault("contrastive_weight", float(old_ssl_cfg.get("contrastive_weight", 0.05)))
    regularization_ssl_cfg.setdefault("self_consistency", False)
    regularization_ssl_cfg.setdefault("self_consistency_weight", 0.0)

    old_mode = str(pruning_cfg.get("scheme") or pruning_cfg.get("mode") or pruning_cfg.get("method") or "").strip().lower()
    if "scheme" not in pruning_cfg:
        pruning_cfg["scheme"] = "implicit_masking_legacy" if old_mode == "implicit_width_slicing" else "structural_width"
    pruning_cfg.setdefault("enabled", False)
    pruning_cfg.setdefault("student", {})
    pruning_cfg.setdefault("export", {})
    pruning_cfg["student"].setdefault("width_ratio", pruning_cfg.get("student_width", 1.0))
    pruning_cfg["export"].setdefault("enabled", bool(pruning_cfg.get("enabled", False)) and str(pruning_cfg.get("scheme", "")).strip().lower() == "structural_width")
    pruning_cfg["export"].setdefault("filename", "student_export.pt")
    pruning_cfg["export"].setdefault("report_filename", "student_export_report.json")

    model_cfg.setdefault("student", {})
    _normalize_head_selection(normalized)
    return normalized


def is_structural_pruning_enabled(cfg: dict[str, Any]) -> bool:
    pruning_cfg = (cfg.get("pruning") or {})
    return bool(pruning_cfg.get("enabled", False)) and str(pruning_cfg.get("scheme", "")).strip().lower() == "structural_width"


def is_legacy_masking_enabled(cfg: dict[str, Any]) -> bool:
    pruning_cfg = (cfg.get("pruning") or {})
    return bool(pruning_cfg.get("enabled", False)) and str(pruning_cfg.get("scheme", "")).strip().lower() in {
        "implicit_masking_legacy",
        "implicit_width_slicing",
        "legacy_masking",
    }


def resolve_student_model_spec(cfg: dict[str, Any]) -> StudentModelSpec:
    normalized = normalize_compression_cfg(cfg)
    model_cfg = normalized.get("model") or {}
    student_cfg = dict(model_cfg.get("student") or {})
    pruning_cfg = normalized.get("pruning") or {}
    student_pruning_cfg = pruning_cfg.get("student") or {}

    base_embed_dim = _positive_int(model_cfg.get("embed_dim", 192), 192)
    base_depth = _positive_int(model_cfg.get("depth", 6), 6)
    base_num_heads = _positive_int(model_cfg.get("num_heads", 3), 3)
    base_mlp_ratio = _positive_float(model_cfg.get("mlp_ratio", 4.0), 4.0)
    base_patch_size = _positive_int(model_cfg.get("patch_size", 16), 16)

    width_ratio = _clamp_width(
        student_cfg.get("width_ratio", student_pruning_cfg.get("width_ratio", pruning_cfg.get("student_width", 1.0))),
        1.0,
    )
    structural_enabled = is_structural_pruning_enabled(normalized)
    default_embed_dim = max(1, int(round(base_embed_dim * width_ratio))) if structural_enabled else base_embed_dim
    embed_dim = _positive_int(student_cfg.get("embed_dim", default_embed_dim), default_embed_dim)
    depth = _positive_int(student_cfg.get("depth", base_depth), base_depth)
    requested_num_heads = _positive_int(student_cfg.get("num_heads", base_num_heads), base_num_heads)
    num_heads = _largest_valid_num_heads(embed_dim, requested_num_heads, base_num_heads)
    mlp_ratio = _positive_float(student_cfg.get("mlp_ratio", base_mlp_ratio), base_mlp_ratio)
    patch_size = _positive_int(student_cfg.get("patch_size", base_patch_size), base_patch_size)

    adapter_hidden_dim = _positive_int(student_cfg.get("adapter_hidden_dim", embed_dim), embed_dim)
    prev_state_hidden_dim = _positive_int(student_cfg.get("prev_state_hidden_dim", embed_dim), embed_dim)
    head_hidden_dim = _positive_int(student_cfg.get("head_hidden_dim", embed_dim), embed_dim)
    track_head_hidden_dim = _positive_int(student_cfg.get("track_head_hidden_dim", embed_dim * 2), embed_dim * 2)
    mask_hidden_dim = _positive_int(student_cfg.get("mask_hidden_dim", max(embed_dim // 2, 32)), max(embed_dim // 2, 32))

    return StudentModelSpec(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        adapter_hidden_dim=adapter_hidden_dim,
        prev_state_hidden_dim=prev_state_hidden_dim,
        head_hidden_dim=head_hidden_dim,
        track_head_hidden_dim=track_head_hidden_dim,
        mask_hidden_dim=mask_hidden_dim,
        structural_width_ratio=float(embed_dim) / float(max(1, base_embed_dim)),
    )


def resolve_model_role_cfg(cfg: dict[str, Any], *, role: str = "student") -> dict[str, Any]:
    normalized = normalize_compression_cfg(cfg)
    model_cfg = deepcopy(normalized.get("model") or {})
    role_override = deepcopy(model_cfg.get(role) or {}) if role != "student" else {}
    if role == "teacher" or not is_structural_pruning_enabled(normalized):
        if role_override:
            model_cfg.update(role_override)
        model_cfg.setdefault("adapter_hidden_dim", model_cfg.get("embed_dim", 192))
        model_cfg.setdefault("prev_state_hidden_dim", model_cfg.get("embed_dim", 192))
        model_cfg.setdefault("head_hidden_dim", model_cfg.get("embed_dim", 192))
        model_cfg.setdefault("track_head_hidden_dim", int(model_cfg.get("embed_dim", 192)) * 2)
        model_cfg.setdefault("mask_hidden_dim", max(int(model_cfg.get("embed_dim", 192)) // 2, 32))
        model_cfg["structural_width_ratio"] = 1.0
        return model_cfg

    student_spec = resolve_student_model_spec(normalized)
    role_cfg = deepcopy(model_cfg)
    role_cfg.update(student_spec.as_model_kwargs())
    return role_cfg


def build_export_report_payload(*, cfg: dict[str, Any], stage: str, checkpoint_path: str, report_path: str) -> dict[str, Any]:
    normalized = normalize_compression_cfg(cfg)
    student_spec = resolve_student_model_spec(normalized)
    return {
        "stage": str(stage),
        "scheme": str((normalized.get("pruning") or {}).get("scheme", "")),
        "student_spec": student_spec.as_model_kwargs(),
        "checkpoint_path": str(checkpoint_path),
        "report_path": str(report_path),
    }


__all__ = [
    "StudentModelSpec",
    "build_export_report_payload",
    "is_legacy_masking_enabled",
    "is_structural_pruning_enabled",
    "normalize_compression_cfg",
    "resolve_model_role_cfg",
    "resolve_student_model_spec",
]
