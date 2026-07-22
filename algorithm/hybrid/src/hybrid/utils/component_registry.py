from __future__ import annotations

from typing import Any


def component_spec(
    component_cfg: dict[str, Any],
    key: str,
    *,
    default_variant: str = "split_v1",
) -> tuple[str, dict[str, Any]]:
    raw = component_cfg.get(key) or {}
    if isinstance(raw, str):
        return raw.strip().lower() or default_variant, {}
    if not isinstance(raw, dict):
        return default_variant, {}
    variant = str(raw.get("variant", default_variant)).strip().lower() or default_variant
    kwargs = dict(raw.get("kwargs") or {})
    return variant, kwargs


def resolve_component(
    component_cfg: dict[str, Any],
    *,
    key: str,
    variants: dict[str, dict[str, type]],
    group_name: str,
    default_variant: str = "split_v1",
) -> tuple[type, dict[str, Any]]:
    variant, kwargs = component_spec(component_cfg, key, default_variant=default_variant)
    choices = variants[key]
    if variant not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(
            f"Unsupported {group_name} component variant for {key!r}: {variant!r}. "
            f"Expected one of: {supported}"
        )
    return choices[variant], kwargs


__all__ = [
    "component_spec",
    "resolve_component",
]
