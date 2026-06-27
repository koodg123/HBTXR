"""Model registry for HG-PIPE paper evaluation targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..models.vision_transformer import VisionTransformerConfig, build_vision_transformer

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PaperModel:
    name: str
    timm_name: str
    paper_role: str
    notes: str
    config: VisionTransformerConfig
    input_size: tuple[int, int, int] = (3, 224, 224)
    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD
    crop_pct: float = 224 / 256
    interpolation: str = "bicubic"


PAPER_MODELS: dict[str, PaperModel] = {
    "deit_tiny_patch16_224": PaperModel(
        name="deit_tiny_patch16_224",
        timm_name="deit_tiny_patch16_224",
        paper_role="HG-PIPE primary model; Table 2 reports A4W4 and A3W3.",
        notes="Uses the classifier-token logits path only; no distillation head is required by the current eval path.",
        config=VisionTransformerConfig(model_name="deit_tiny_patch16_224", embed_dim=192, depth=12, num_heads=3),
    ),
    "deit_small_patch16_224": PaperModel(
        name="deit_small_patch16_224",
        timm_name="deit_small_patch16_224",
        paper_role="HG-PIPE larger model; Table 2 includes A3W3 throughput.",
        notes="Uses the classifier-token logits path only; no distillation head is required by the current eval path.",
        config=VisionTransformerConfig(model_name="deit_small_patch16_224", embed_dim=384, depth=12, num_heads=6),
    ),
    "vit_tiny_patch16_224": PaperModel(
        name="vit_tiny_patch16_224",
        timm_name="vit_tiny_patch16_224",
        paper_role="Related-work comparison model in Table 2.",
        notes="Torch-native ViT-tiny classifier used for evaluation compatibility without timm.",
        config=VisionTransformerConfig(model_name="vit_tiny_patch16_224", embed_dim=192, depth=12, num_heads=3),
    ),
}


def resolve_models(names: list[str] | None) -> list[PaperModel]:
    if not names:
        return list(PAPER_MODELS.values())
    missing = [name for name in names if name not in PAPER_MODELS]
    if missing:
        raise KeyError(f"Unknown paper model(s): {missing}. Available: {sorted(PAPER_MODELS)}")
    return [PAPER_MODELS[name] for name in names]


def resolve_model(name: str) -> PaperModel:
    return resolve_models([name])[0]


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a state_dict or a dict containing state_dict/model")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        normalized = key[7:] if key.startswith("module.") else key
        if normalized in {"dist_token", "head_dist.weight", "head_dist.bias"}:
            continue
        state_dict[normalized] = value
    return state_dict


def create_paper_model(
    name: str,
    *,
    checkpoint_path: str | Path | None = None,
    pretrained: bool = False,
    map_location: str | torch.device = "cpu",
):
    spec = resolve_model(name)
    model = build_vision_transformer(spec.config)
    metadata: dict[str, Any] = {
        "paper_model": spec.name,
        "timm_model_name": spec.timm_name,
        "requested_pretrained": bool(pretrained),
        "pretrained": False,
        "checkpoint_path": None,
        "checkpoint_loaded": False,
        "model_backend": "torch_native_vit",
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        state_dict = _extract_state_dict(torch.load(checkpoint_path, map_location=map_location))
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"head.weight", "head.bias"}
        allowed_unexpected = set()
        missing = set(incompatible.missing_keys) - allowed_missing
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint incompatibility for {}: missing_keys={} unexpected_keys={}".format(
                    spec.name,
                    sorted(missing),
                    sorted(unexpected),
                )
            )
        metadata["pretrained"] = True
        metadata["checkpoint_path"] = str(checkpoint_path)
        metadata["checkpoint_loaded"] = True
    model.pretrained = bool(metadata["pretrained"])
    model.checkpoint_path = metadata["checkpoint_path"]
    model.paper_model_name = spec.name
    model.timm_model_name = spec.timm_name
    model.eval()
    return model, metadata
